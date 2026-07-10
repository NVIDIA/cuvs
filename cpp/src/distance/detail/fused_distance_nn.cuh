/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "distance_ops/l2_exp.cuh"  // ops::l2_exp_distance_op
#include "fused_distance_nn/cutile/fused_1nn_tile.hpp"
#include "fused_distance_nn/cutlass_base.cuh"
#include "fused_distance_nn/fused_cosine_nn.cuh"
#include "fused_distance_nn/fused_l2_nn.cuh"
#include "fused_distance_nn/helper_structs.cuh"
#include "fused_distance_nn/simt_kernel.cuh"
#include "pairwise_distance_base.cuh"  // PairwiseDistances
#include <cuvs/distance/distance.hpp>
#include <raft/core/error.hpp>
#include <raft/core/kvp.hpp>             // raft::KeyValuePair
#include <raft/core/operators.hpp>       // raft::identity_op
#include <raft/linalg/contractions.cuh>  // Policy
#include <raft/util/arch.cuh>            // raft::util::arch::SM_*
#include <raft/util/cuda_utils.cuh>      // raft::ceildiv, raft::shfl

#include <cstddef>  // size_t
#include <limits>   // std::numeric_limits

namespace cuvs {
namespace distance {

namespace detail {

template <typename DataT, typename IdxT, typename Policy, typename ReduceOpT, typename KVPReduceOpT>
void fusedDistanceNNImpl(IdxT* nearest_idx,
                         DataT* nearest_dist,
                         const DataT* x,
                         const DataT* y,
                         const DataT* xn,
                         const DataT* yn,
                         IdxT m,
                         IdxT n,
                         IdxT k,
                         int* workspace,
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
  typedef Policy P;
  typedef raft::KeyValuePair<IdxT, DataT> KVP;
  constexpr auto maxVal = std::numeric_limits<DataT>::max();

  if constexpr (is_fused_1nn_cutile_data_v<DataT>) {
    if constexpr (cuvs::detail::jit_lto::library_built_with_cutile()) {
      if (try_fused_1nn_tile<DataT, IdxT>(
            nearest_idx, nearest_dist, x, y, xn, yn, m, n, k, metric, sqrt, stream)) {
        return;
      }
    }
  }

  RAFT_EXPECTS(cutlass_kvp_scratch != nullptr, "CUTLASS fused 1-NN requires a scratch KVP buffer");

  if (initOutBuffer) {
    initFused1nnOutput(nearest_idx, nearest_dist, m, std::numeric_limits<DataT>::max(), stream);
  }

  MinAndDistanceReduceOpImpl<IdxT, DataT> cutlass_redOp;
  cutlass_redOp.out_kvp = cutlass_kvp_scratch;
  initialize<DataT, KVP, IdxT, decltype(cutlass_redOp)>(
    cutlass_kvp_scratch, m, maxVal, cutlass_redOp, stream);

  RAFT_CUDA_TRY(cudaMemsetAsync(workspace, 0, sizeof(int) * m, stream));

  switch (metric) {
    case cuvs::distance::DistanceType::CosineExpanded:
      fusedCosineNN<DataT, IdxT, P, decltype(cutlass_redOp), KVPReduceOpT>(nearest_idx,
                                                                           nearest_dist,
                                                                           x,
                                                                           y,
                                                                           xn,
                                                                           yn,
                                                                           m,
                                                                           n,
                                                                           k,
                                                                           workspace,
                                                                           cutlass_redOp,
                                                                           pairRedOp,
                                                                           sqrt,
                                                                           cutlass_kvp_scratch,
                                                                           stream);
      break;
    case cuvs::distance::DistanceType::L2SqrtExpanded:
    case cuvs::distance::DistanceType::L2Expanded:
      fusedL2NNImpl<DataT, IdxT, P, decltype(cutlass_redOp), KVPReduceOpT>(nearest_idx,
                                                                           nearest_dist,
                                                                           x,
                                                                           y,
                                                                           xn,
                                                                           yn,
                                                                           m,
                                                                           n,
                                                                           k,
                                                                           workspace,
                                                                           cutlass_redOp,
                                                                           pairRedOp,
                                                                           sqrt,
                                                                           false,
                                                                           cutlass_kvp_scratch,
                                                                           stream);
      break;
    case cuvs::distance::DistanceType::InnerProduct: break;
    default: assert("only cosine/l2 metric is supported with fusedDistanceNN\n"); break;
  }

  unpackFused1nnKvpToSoa(nearest_idx, nearest_dist, cutlass_kvp_scratch, m, stream);
}

}  // namespace detail
}  // namespace distance
}  // namespace cuvs
