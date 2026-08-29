/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file atomic_compat.cuh
 * @brief Pre-Pascal definitions for atomics that CUDA only declares from sm_60 onwards.
 *
 * `atomicAdd(double*, double)` and the block-scoped `atomicAdd_block` family are declared in
 * `sm_60_atomic_functions.h` under `!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600`. In a device
 * pass for an older architecture they therefore do not exist at all, and any header that calls
 * them unqualified fails to compile -- including a few RAFT headers that cuVS instantiates with
 * `double` (`raft/linalg/detail/strided_reduction.cuh`) or that come from the sparse linear
 * algebra primitives (`raft/sparse/linalg/detail/utils.cuh`).
 *
 * This header supplies the missing overloads for exactly those device passes:
 *
 *   - `atomicAdd(double*, double)` is emulated with the standard 64-bit compare-and-swap loop.
 *   - the `atomicAdd_block` overloads forward to their device-scoped counterparts, which is always
 *     correct (a wider scope is a stronger guarantee) and merely gives up the L1-local fast path.
 *
 * Because it must be visible before the RAFT header that uses the symbol is parsed -- the calls
 * are dependent, but the argument types are fundamental, so ADL contributes nothing and only the
 * declarations visible at template definition context are considered -- include this header
 * *first* in any translation unit that needs it.
 *
 * On sm_60 and newer, and in the host pass, this header expands to nothing.
 */

#pragma once

#include <cuda_runtime.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)

/**
 * @brief Software `atomicAdd` for `double` on pre-Pascal devices.
 *
 * Uses the documented CAS loop. The `__longlong_as_double(assumed) + val` comparison is written
 * with the raw bit pattern so that a NaN payload cannot make the loop spin forever.
 */
__device__ __forceinline__ double atomicAdd(double* address, double val)
{
  auto* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;
  do {
    assumed = old;
    old     = atomicCAS(
      address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    // Compare the bit patterns rather than the values: this terminates even when the stored
    // value is NaN, for which `old != assumed` would always hold.
  } while (assumed != old);
  return __longlong_as_double(old);
}

// Block-scoped atomics are an sm_60 feature. Falling back to the device-scoped operation is
// always safe; only the memory-ordering scope is wider than requested.
__device__ __forceinline__ int atomicAdd_block(int* address, int val)
{
  return atomicAdd(address, val);
}

__device__ __forceinline__ unsigned int atomicAdd_block(unsigned int* address, unsigned int val)
{
  return atomicAdd(address, val);
}

__device__ __forceinline__ unsigned long long int atomicAdd_block(unsigned long long int* address,
                                                                  unsigned long long int val)
{
  return atomicAdd(address, val);
}

__device__ __forceinline__ float atomicAdd_block(float* address, float val)
{
  return atomicAdd(address, val);
}

__device__ __forceinline__ double atomicAdd_block(double* address, double val)
{
  return atomicAdd(address, val);
}

#endif  // __CUDA_ARCH__ < 600
