/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/operators.hpp>            // raft::abs, raft::max
#include <raft/util/cuda_dev_essentials.cuh>  // DI

namespace cuvs::distance::detail::ops {

/**
 * @brief the L_inf (Chebyshev) distance matrix calculation
 *
 * It computes the following equation:
 *
 *  c_ij = max_k | x_ik - y_kj |
 */
template <typename DataType, typename AccType, typename IdxType>
struct l_inf_distance_op {
  using DataT = DataType;
  using AccT  = AccType;
  using IdxT  = IdxType;

  // Load norms of input data
  static constexpr bool use_norms = false;
  // Whether the core function requires so many instructions that it makes sense
  // to reduce loop unrolling, etc. We do this to keep compile times in check.
  static constexpr bool expensive_inner_loop = false;

  // Size of shared memory. This is normally decided by the kernel policy, but
  // some ops such as correlation_distance_op use more.
  template <typename Policy>
  static constexpr size_t shared_mem_size()
  {
    return Policy::SmemSize;
  }

  DI void core(AccT& acc, DataT& x, DataT& y) const
  {
    // Widen to the accumulator type before doing any arithmetic. `__hsub` / `__habs` are FP16 ALU
    // instructions that need sm_53, and `raft::abs(__half)` static_asserts below that. The other
    // element-wise ops (l1, lp_unexp, jensen_shannon, kl_divergence) already do this; the
    // accumulation happens in `AccT` regardless, so the only change is that the subtraction is
    // now performed at accumulator precision.
    const auto diff = raft::abs(raft::to_float(x) - raft::to_float(y));
    acc             = raft::max(acc, static_cast<AccT>(diff));
  };

  template <typename Policy>
  DI void epilog(AccT acc[Policy::AccRowsPerTh][Policy::AccColsPerTh],
                 AccT* regxn,
                 AccT* regyn,
                 IdxT gridStrideX,
                 IdxT gridStrideY) const
  {
    return;
  }
};

}  // namespace cuvs::distance::detail::ops
