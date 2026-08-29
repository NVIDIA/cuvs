/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file half_compat.cuh
 * @brief Half-precision arithmetic helpers that also work on pre-Pascal hardware.
 *
 * Native `__half` / `__half2` arithmetic intrinsics are only defined for
 * `__CUDA_ARCH__ >= 530`; on sm_50 (first-generation Maxwell, e.g. GM10x) the
 * FP16 storage format exists but there are no FP16 ALU instructions.
 *
 * The helpers below use the packed FP16 instructions where available and widen
 * to `float` otherwise. Note that on sm_53+ hardware the packed path is also
 * genuinely faster (two lanes per instruction), so this is not merely a
 * portability shim.
 *
 * Gating is done on `__CUDA_ARCH__` rather than on the project-wide
 * `CUVS_MIN_CUDA_ARCH` on purpose: these are pure device-side computations with
 * no host-visible layout, so the decision can be made independently for every
 * device compilation pass, and this header stays usable from NVRTC.
 */

#pragma once

#include <raft/core/detail/macros.hpp>  // RAFT_INLINE_FUNCTION
#include <raft/core/math.hpp>           // raft::sqrt

#include <cuda_fp16.h>

#include <cmath>
#include <type_traits>

namespace cuvs::util {

/** True when the current device compilation pass has native FP16 ALU support. */
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
#define CUVS_DEVICE_HAS_NATIVE_HALF_MATH 1
#else
#define CUVS_DEVICE_HAS_NATIVE_HALF_MATH 0
#endif

/** @brief Horizontal sum of the two halves of a `half2`, as `float`. */
__device__ __forceinline__ float half2_reduce_add(const __half2 a)
{
#if CUVS_DEVICE_HAS_NATIVE_HALF_MATH
  return static_cast<float>(__low2half(a) + __high2half(a));
#else
  return __low2float(a) + __high2float(a);
#endif
}

/**
 * @brief Squared L2 contribution of a two-element VPQ residual.
 *
 * Computes `sum((q - c - v)^2)` over the two packed lanes and returns it as a
 * `float`.
 *
 * @param q packed pair of query values
 * @param c packed pair of PQ codebook centroids
 * @param v packed pair of VQ codebook values
 */
__device__ __forceinline__ float half2_sq_residual(const __half2 q,
                                                   const __half2 c,
                                                   const __half2 v)
{
#if CUVS_DEVICE_HAS_NATIVE_HALF_MATH
  __half2 d = q - c - v;
  d         = d * d;
  return static_cast<float>(__low2half(d) + __high2half(d));
#else
  // No FP16 ALU on this target: widen to fp32. This is a strict accuracy
  // improvement, at the cost of operating on one lane at a time.
  const float dlo = __low2float(q) - __low2float(c) - __low2float(v);
  const float dhi = __high2float(q) - __high2float(c) - __high2float(v);
  return dlo * dlo + dhi * dhi;
#endif
}

/**
 * @brief Type-preserving square root that also works on pre-Pascal devices.
 *
 * A drop-in replacement for `raft::sqrt_op` for element-wise passes over possibly-fp16 data.
 * `raft::sqrt(__half)` lowers to `hsqrt`, an FP16 ALU instruction that requires sm_53 and whose
 * RAFT overload static_asserts below that. For `__half` inputs on such targets the value is
 * widened to `float`, which is also slightly more accurate than `hsqrt`.
 */
struct sqrt_op {
  template <typename T>
  RAFT_INLINE_FUNCTION auto operator()(const T& x) const -> T
  {
    if constexpr (std::is_same_v<T, __half>) {
#if CUVS_DEVICE_HAS_NATIVE_HALF_MATH
      return raft::sqrt(x);
#else
      return __float2half(::sqrtf(__half2float(x)));
#endif
    } else {
      return raft::sqrt(x);
    }
  }
};

}  // namespace cuvs::util
