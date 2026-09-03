/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <raft/core/kvp.hpp>        // raft::KeyValuePair
#include <raft/core/operators.hpp>  // raft::identity_op
#include <raft/core/resource/device_properties.hpp>
#include <raft/linalg/contractions.cuh>  // Policy
#include <raft/util/arch.cuh>            // raft::util::arch::SM_*
#include <raft/util/cuda_utils.cuh>      // raft::ceildiv, raft::shfl

#include <cstddef>  // size_t
#include <cstdint>
#include <limits>  // std::numeric_limits

namespace cuvs {
namespace distance {

namespace detail {

/** Explicit implementation selected for the fused 1-NN primitive. */
enum class Fused1nnBackend : std::uint8_t {
  Cutile,
  Cutlass,
  Unfused,
  Auto,
};

/** Tuning used only by the bounded-workspace unfused backend. */
struct UnfusedTop1nnTuning {
  std::size_t row_tile       = 8192;
  std::size_t candidate_tile = 8192;
};

struct Top1nnTuning {
  UnfusedTop1nnTuning unfused{};
};

/** Select the pre-cuTile implementation used by algorithm-level AUTO dispatch. */
template <typename IdxT>
constexpr Fused1nnBackend fused_1nn_legacy_backend(int cc_major,
                                                   IdxT m,
                                                   IdxT n,
                                                   cuvs::distance::DistanceType metric)
{
  const bool legacy_fused_metric = metric == cuvs::distance::DistanceType::L2Expanded ||
                                   metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                                   metric == cuvs::distance::DistanceType::CosineExpanded;
  if (!legacy_fused_metric) { return Fused1nnBackend::Unfused; }
  if (cc_major <= 8 || (cc_major == 9 && (m >= 4096 || n >= 4096))) {
    return Fused1nnBackend::Cutlass;
  }
  return Fused1nnBackend::Unfused;
}

/** Resolve AUTO before allocating backend-native output and workspace storage. */
template <typename DataT, typename IdxT>
Fused1nnBackend resolve_fused_1nn_backend(const raft::resources& handle,
                                          const DataT* x,
                                          const DataT* y,
                                          IdxT m,
                                          IdxT n,
                                          IdxT k,
                                          cuvs::distance::DistanceType metric)
{
#if CUDART_VERSION >= 13000
  if constexpr (is_fused_1nn_cutile_data_v<DataT>) {
    if (can_launch_fused_1nn_tile(x, y, m, n, k, metric)) { return Fused1nnBackend::Cutile; }
  }
#endif
  const auto prop = raft::resource::get_device_properties(handle);
  return fused_1nn_legacy_backend(prop.major, m, n, metric);
}

/**
 * Output-independent backend probe. Call this before allocating backend-native result storage.
 * cuTile delegates to its launcher/ABI probe; CUTLASS is available for the legacy L2/cosine
 * fused primitive only.
 */
template <typename DataT, typename IdxT>
bool can_launch_fused_1nn_backend(Fused1nnBackend backend,
                                  const DataT* x,
                                  const DataT* y,
                                  IdxT m,
                                  IdxT n,
                                  IdxT k,
                                  cuvs::distance::DistanceType metric)
{
  if (backend == Fused1nnBackend::Cutile) {
    if constexpr (is_fused_1nn_cutile_data_v<DataT>) {
      return can_launch_fused_1nn_tile(x, y, m, n, k, metric);
    }
    return false;
  }
  const bool supported_metric = metric == cuvs::distance::DistanceType::L2Expanded ||
                                metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                                metric == cuvs::distance::DistanceType::CosineExpanded;
  return (backend == Fused1nnBackend::Cutlass || backend == Fused1nnBackend::Unfused) &&
         supported_metric && x != nullptr && y != nullptr && m > 0 && n > 0 && k > 0;
}

template <typename DataT,
          typename OutT,
          typename IdxT,
          typename Policy,
          typename ReduceOpT,
          typename KVPReduceOpT>
void fusedDistanceNNImpl(OutT* min,
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
                         cudaStream_t stream)
{
  // The kernel policy is determined by fusedDistanceNN.
  typedef Policy P;

  dim3 blk(P::Nthreads);
  auto nblks            = raft::ceildiv<int>(m, P::Nthreads);
  constexpr auto maxVal = std::numeric_limits<DataT>::max();
  typedef raft::KeyValuePair<IdxT, DataT> KVPair;

  RAFT_CUDA_TRY(cudaMemsetAsync(workspace, 0, sizeof(int) * m, stream));
  if (initOutBuffer) {
    initKernel<DataT, OutT, IdxT, ReduceOpT>
      <<<nblks, P::Nthreads, 0, stream>>>(min, m, maxVal, redOp);
    RAFT_CUDA_TRY(cudaGetLastError());
  }

  switch (metric) {
    case cuvs::distance::DistanceType::CosineExpanded:
      fusedCosineNN<DataT, OutT, IdxT, P, ReduceOpT, KVPReduceOpT>(
        min, x, y, xn, yn, m, n, k, workspace, redOp, pairRedOp, sqrt, stream);
      break;
    case cuvs::distance::DistanceType::L2SqrtExpanded:
    case cuvs::distance::DistanceType::L2Expanded:
      // initOutBuffer is take care by fusedDistanceNNImpl() so we set it false to fusedL2NNImpl.
      fusedL2NNImpl<DataT, OutT, IdxT, P, ReduceOpT, KVPReduceOpT>(
        min, x, y, xn, yn, m, n, k, workspace, redOp, pairRedOp, sqrt, false, stream);
      break;
    default: assert("only cosine/l2 metric is supported with fusedDistanceNN\n"); break;
  }
}

}  // namespace detail
}  // namespace distance
}  // namespace cuvs
