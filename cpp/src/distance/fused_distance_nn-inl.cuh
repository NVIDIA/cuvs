/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __FUSED_DISTANCE_NN_H
#define __FUSED_DISTANCE_NN_H

#pragma once

#include "detail/fused_distance_nn.cuh"
#include "fused_distance_nn_helpers.cuh"
#include <raft/core/resources.hpp>
#include <raft/linalg/contractions.cuh>
#include <raft/util/cuda_utils.cuh>

#include <cub/util_type.cuh>

#include <stdint.h>

#include <limits>
#include <type_traits>

namespace cuvs {
namespace distance {

/**
 * \ingroup fused_l2_nn
 * @{
 */

template <typename DataT, typename IdxT, typename ReduceOpT, typename KVPReduceOpT>
void fusedDistanceNN(IdxT* nearest_idx,
                     DataT* nearest_dist,
                     const DataT* x,
                     const DataT* y,
                     const DataT* xn,
                     const DataT* yn,
                     IdxT m,
                     IdxT n,
                     IdxT k,
                     void* workspace,
                     ReduceOpT redOp,
                     KVPReduceOpT pairRedOp,
                     bool sqrt,
                     bool initOutBuffer,
                     bool isRowMajor,
                     cuvs::distance::DistanceType metric,
                     float metric_arg,
                     raft::KeyValuePair<IdxT, DataT>* cutlass_kvp_scratch,
                     cudaStream_t stream)
{
  ASSERT(isRowMajor, "fusedDistanceNN only supports row major inputs");
  bool is_skinny = k < 32;

  size_t bytes = sizeof(DataT) * k;
  auto px      = reinterpret_cast<uintptr_t>(x);
  auto py      = reinterpret_cast<uintptr_t>(y);
  if (16 % sizeof(DataT) == 0 && bytes % 16 == 0 && px % 16 == 0 && py % 16 == 0) {
    if (is_skinny) {
      detail::fusedDistanceNNImpl<
        DataT,
        IdxT,
        typename raft::linalg::Policy4x4Skinny<DataT, 16 / sizeof(DataT)>::Policy,
        ReduceOpT>(nearest_idx,
                   nearest_dist,
                   x,
                   y,
                   xn,
                   yn,
                   m,
                   n,
                   k,
                   (int*)workspace,
                   redOp,
                   pairRedOp,
                   sqrt,
                   initOutBuffer,
                   isRowMajor,
                   metric,
                   metric_arg,
                   cutlass_kvp_scratch,
                   stream);
    } else {
      detail::fusedDistanceNNImpl<
        DataT,
        IdxT,
        typename raft::linalg::Policy4x4<DataT, 16 / sizeof(DataT)>::Policy,
        ReduceOpT>(nearest_idx,
                   nearest_dist,
                   x,
                   y,
                   xn,
                   yn,
                   m,
                   n,
                   k,
                   (int*)workspace,
                   redOp,
                   pairRedOp,
                   sqrt,
                   initOutBuffer,
                   isRowMajor,
                   metric,
                   metric_arg,
                   cutlass_kvp_scratch,
                   stream);
    }
  } else if (8 % sizeof(DataT) == 0 && bytes % 8 == 0 && px % 8 == 0 && py % 8 == 0) {
    if (is_skinny) {
      detail::fusedDistanceNNImpl<
        DataT,
        IdxT,
        typename raft::linalg::Policy4x4Skinny<DataT, 8 / sizeof(DataT)>::Policy,
        ReduceOpT>(nearest_idx,
                   nearest_dist,
                   x,
                   y,
                   xn,
                   yn,
                   m,
                   n,
                   k,
                   (int*)workspace,
                   redOp,
                   pairRedOp,
                   sqrt,
                   initOutBuffer,
                   isRowMajor,
                   metric,
                   metric_arg,
                   cutlass_kvp_scratch,
                   stream);
    } else {
      detail::fusedDistanceNNImpl<
        DataT,
        IdxT,
        typename raft::linalg::Policy4x4<DataT, 8 / sizeof(DataT)>::Policy,
        ReduceOpT>(nearest_idx,
                   nearest_dist,
                   x,
                   y,
                   xn,
                   yn,
                   m,
                   n,
                   k,
                   (int*)workspace,
                   redOp,
                   pairRedOp,
                   sqrt,
                   initOutBuffer,
                   isRowMajor,
                   metric,
                   metric_arg,
                   cutlass_kvp_scratch,
                   stream);
    }
  } else {
    if (is_skinny) {
      detail::fusedDistanceNNImpl<DataT,
                                  IdxT,
                                  typename raft::linalg::Policy4x4Skinny<DataT, 1>::Policy,
                                  ReduceOpT>(nearest_idx,
                                             nearest_dist,
                                             x,
                                             y,
                                             xn,
                                             yn,
                                             m,
                                             n,
                                             k,
                                             (int*)workspace,
                                             redOp,
                                             pairRedOp,
                                             sqrt,
                                             initOutBuffer,
                                             isRowMajor,
                                             metric,
                                             metric_arg,
                                             cutlass_kvp_scratch,
                                             stream);
    } else {
      detail::fusedDistanceNNImpl<DataT,
                                  IdxT,
                                  typename raft::linalg::Policy4x4<DataT, 1>::Policy,
                                  ReduceOpT>(nearest_idx,
                                             nearest_dist,
                                             x,
                                             y,
                                             xn,
                                             yn,
                                             m,
                                             n,
                                             k,
                                             (int*)workspace,
                                             redOp,
                                             pairRedOp,
                                             sqrt,
                                             initOutBuffer,
                                             isRowMajor,
                                             metric,
                                             metric_arg,
                                             cutlass_kvp_scratch,
                                             stream);
    }
  }
}

/**
 * @brief Fused GEMM + 1-NN minimum reduction.
 *
 * @param[out] nearest_idx   Nearest neighbor index per row, length `m` (required).
 * @param[out] nearest_dist  Minimum distance per row, length `m` (optional, may be null).
 * @param[in]  cutlass_kvp_scratch  Temp KVP buffer, length `m`; required when CUTLASS/SIMT runs.
 *                                    Unused when cuTile handles the launch.
 */
template <typename DataT, typename IdxT>
void fusedDistanceNNMinReduce(IdxT* nearest_idx,
                              DataT* nearest_dist,
                              const DataT* x,
                              const DataT* y,
                              const DataT* xn,
                              const DataT* yn,
                              IdxT m,
                              IdxT n,
                              IdxT k,
                              void* workspace,
                              bool sqrt,
                              bool initOutBuffer,
                              bool isRowMajor,
                              cuvs::distance::DistanceType metric,
                              float metric_arg,
                              raft::KeyValuePair<IdxT, DataT>* cutlass_kvp_scratch,
                              cudaStream_t stream)
{
  MinAndDistanceReduceOp<IdxT, DataT> redOp;
  redOp.out_idx  = nearest_idx;
  redOp.out_dist = nearest_dist;
  KVPMinReduce<IdxT, DataT> pairRedOp;

  fusedDistanceNN<DataT, IdxT>(nearest_idx,
                               nearest_dist,
                               x,
                               y,
                               xn,
                               yn,
                               m,
                               n,
                               k,
                               workspace,
                               redOp,
                               pairRedOp,
                               sqrt,
                               initOutBuffer,
                               isRowMajor,
                               metric,
                               metric_arg,
                               cutlass_kvp_scratch,
                               stream);
}

/** @} */

}  // namespace distance
}  // namespace cuvs

#endif
