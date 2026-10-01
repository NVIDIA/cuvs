/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "distance_ops/l2_exp.cuh"  // ops::l2_exp_distance_op
#include "fused_distance_nn/cutlass_base.cuh"
#include "fused_distance_nn/fused_bitwise_hamming_nn.cuh"
#include "fused_distance_nn/fused_cosine_nn.cuh"
#include "fused_distance_nn/fused_l2_nn.cuh"
#include "fused_distance_nn/helper_structs.cuh"
#include "fused_distance_nn/simt_kernel.cuh"
#include "pairwise_distance_base.cuh"  // PairwiseDistances
#include <cuvs/distance/distance.hpp>
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
  using AccT            = std::conditional_t<std::is_same_v<DataT, uint8_t>, uint32_t, DataT>;
  constexpr auto maxVal = std::numeric_limits<AccT>::max();

  RAFT_CUDA_TRY(cudaMemsetAsync(workspace, 0, sizeof(int) * m, stream));
  if (initOutBuffer) {
    initKernel<AccT, OutT, IdxT, ReduceOpT>
      <<<nblks, P::Nthreads, 0, stream>>>(min, m, maxVal, redOp);
    RAFT_CUDA_TRY(cudaGetLastError());
  }

  // An empty candidate set leaves the initialized (or supplied) result unchanged.
  if (n == 0) { return; }

  switch (metric) {
    case cuvs::distance::DistanceType::CosineExpanded:
      if constexpr (std::is_same_v<DataT, uint8_t> || std::is_same_v<DataT, int8_t>) {
        RAFT_FAIL("Cosine distance is not supported for uint8_t/int8_t data types");
      } else {
        fusedCosineNN<DataT, OutT, IdxT, P, ReduceOpT, KVPReduceOpT>(
          min, x, y, xn, yn, m, n, k, workspace, redOp, pairRedOp, sqrt, stream);
      }
      break;
    case cuvs::distance::DistanceType::L2SqrtExpanded:
    case cuvs::distance::DistanceType::L2Expanded:
      if constexpr (std::is_same_v<DataT, uint8_t> || std::is_same_v<DataT, int8_t>) {
        RAFT_FAIL("L2 distance is not supported for uint8_t/int8_t data types");
      } else {
        fusedL2NNImpl<DataT, OutT, IdxT, P, ReduceOpT, KVPReduceOpT>(
          min, x, y, xn, yn, m, n, k, workspace, redOp, pairRedOp, sqrt, false, stream);
      }
      break;
    case cuvs::distance::DistanceType::BitwiseHamming:
      if constexpr (std::is_same_v<DataT, uint8_t>) {
        fusedBitwiseHammingNN<DataT, OutT, IdxT, P, ReduceOpT, KVPReduceOpT>(
          min, x, y, xn, yn, m, n, k, workspace, redOp, pairRedOp, sqrt, stream);
      } else {
        RAFT_FAIL("BitwiseHamming distance only supports uint8_t data type");
      }
      break;
    default: RAFT_FAIL("only cosine/l2/bitwise hamming metric is supported with fusedDistanceNN");
  }
}

}  // namespace detail
}  // namespace distance
}  // namespace cuvs
