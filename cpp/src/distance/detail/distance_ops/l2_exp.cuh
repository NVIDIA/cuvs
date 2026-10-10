/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/math.hpp>
#include <raft/util/cuda_dev_essentials.cuh>  // DI

#include <cuda_fp16.h>

namespace cuvs::distance::detail::ops {

/**
 * Reserve 1 digit of precision from each floating-point type
 * for round-off error tolerance.
 * @tparam DataT
 */
template <typename DataT, typename AccT>
__device__ constexpr AccT get_clamp_precision()
{
  switch (sizeof(DataT)) {
    case 2: return AccT{1e-3};
    case 4: return AccT{1e-6};
    case 8: return AccT{1e-15};
    default: return AccT{0};
  }
}

/**
 * Whether an expanded L2 distance is round-off error of a self-distance and should be clamped to
 * zero. Self-neighboring points have (aNorm == bNorm) == accVal, but the dot product (accVal) can
 * have round-off errors, which will cause (aNorm == bNorm) ~ accVal instead.
 *
 * The round-off error scales with the norms, so the tolerance is relative to them for squared
 * norms below one. Otherwise, distinct points with equal norms and a small magnitude (for example,
 * a spectral embedding) would have their distances incorrectly clamped to zero.
 */
template <typename DataT, typename AccT>
DI bool is_self_distance_round_off(AccT dist, AccT aNorm, AccT bNorm)
{
  const AccT norm_product = aNorm * bNorm;
  const AccT norm_scale   = norm_product < AccT(1) ? norm_product : AccT(1);
  return (aNorm == bNorm) && (dist * dist < get_clamp_precision<DataT, AccT>() * norm_scale);
}

// Epilogue operator for CUTLASS based kernel
template <typename DataT, typename AccT>
struct l2_exp_cutlass_op {
  bool sqrt;

  __device__ l2_exp_cutlass_op() noexcept : sqrt(false) {}
  __device__ l2_exp_cutlass_op(bool isSqrt) noexcept : sqrt(isSqrt) {}
  inline __device__ AccT operator()(AccT aNorm, AccT bNorm, AccT accVal) const noexcept
  {
    AccT outVal = aNorm + bNorm - AccT(2.0) * accVal;
    outVal      = outVal * AccT(!is_self_distance_round_off<DataT, AccT>(outVal, aNorm, bNorm));
    return sqrt ? raft::sqrt(outVal * static_cast<AccT>(outVal > AccT(0))) : outVal;
  }

  __device__ AccT operator()(DataT aData) const noexcept
  {
    if constexpr (std::is_same_v<DataT, half> && std::is_same_v<AccT, float>) {
      return __half2float(aData);
    } else {
      return aData;
    }
  }
};

/**
 * @brief the expanded euclidean distance matrix calculation
 *
 * It computes the following equation:
 *
 * c_ij = - 2 sum_k x_ik * y_kj + ||x_i.||_2 + ||y_.j||_2
 *
 */
template <typename DataType, typename AccType, typename IdxType>
struct l2_exp_distance_op {
  using DataT = DataType;
  using AccT  = AccType;
  using IdxT  = IdxType;

  const bool sqrt;

  l2_exp_distance_op(bool sqrt_) noexcept : sqrt(sqrt_) {}

  // Load norms of input data
  static constexpr bool use_norms = true;
  // Whether the core function requires so many instructions that it makes sense
  // to reduce loop unrolling, etc. We do this to keep compile times in check.
  static constexpr bool expensive_inner_loop = false;

  // Size of shared memory. This is normally decided by the kernel policy, but
  // some ops such as correlation_distance_op use more.
  template <typename Policy>
  static constexpr size_t shared_mem_size()
  {
    return Policy::SmemSize + ((Policy::Mblk + Policy::Nblk) * sizeof(AccT));
  }

  DI void core(AccT& acc, DataT& x, DataT& y) const
  {
    if constexpr ((std::is_same_v<AccT, float> && std::is_same_v<DataT, half>)) {
      acc += __half2float(x) * __half2float(y);
    } else {
      acc += x * y;
    }
  };

  template <typename Policy>
  DI void epilog(AccT acc[Policy::AccRowsPerTh][Policy::AccColsPerTh],
                 AccT* regxn,
                 AccT* regyn,
                 IdxT gridStrideX,
                 IdxT gridStrideY) const
  {
#pragma unroll
    for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
#pragma unroll
      for (int j = 0; j < Policy::AccColsPerTh; ++j) {
        AccT accVal = acc[i][j];
        AccT val    = regxn[i] + regyn[j] - (AccT)2.0 * accVal;
        acc[i][j] =
          val * static_cast<AccT>((val > AccT(0))) *
          static_cast<AccT>(!is_self_distance_round_off<DataT, AccT>(val, regxn[i], regyn[j]));
      }
    }
    if (sqrt) {
#pragma unroll
      for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
#pragma unroll
        for (int j = 0; j < Policy::AccColsPerTh; ++j) {
          acc[i][j] = raft::sqrt(acc[i][j]);
        }
      }
    }
  }

  constexpr l2_exp_cutlass_op<DataT, AccT> get_cutlass_op() const
  {
    return l2_exp_cutlass_op<DataT, AccT>(sqrt);
  }
};

}  // namespace cuvs::distance::detail::ops
