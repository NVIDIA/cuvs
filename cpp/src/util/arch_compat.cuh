/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file arch_compat.cuh
 * @brief Portable replacements for device intrinsics that are not available on
 *        every compute capability cuVS can be built for.
 *
 * Each helper compiles down to the native intrinsic on architectures that
 * provide it, and to a functionally equivalent emulation elsewhere. They are
 * written so that the *emulated* control flow stays warp-uniform, which is a
 * hard requirement on pre-Volta hardware where a warp has a single program
 * counter and no independent thread scheduling.
 */

#pragma once

#include <cuvs/detail/arch_config.hpp>

#include <cuda_runtime.h>

namespace cuvs::util::arch_compat {

/** Lane index of the calling thread within its warp. */
__device__ __forceinline__ unsigned int lane_id()
{
  unsigned int lane;
  asm("mov.u32 %0, %%laneid;" : "=r"(lane));
  return lane;
}

/**
 * @brief `atomicAdd_block` where available, plain `atomicAdd` otherwise.
 *
 * Block-scoped atomics require sm_60. Falling back to the device-scoped atomic
 * is always correct -- it is a strictly stronger ordering guarantee -- it just
 * gives up the L1-local fast path.
 */
template <typename T>
__device__ __forceinline__ T atomic_add_block(T* address, T val)
{
#if CUVS_HAS_BLOCK_SCOPED_ATOMICS
  return atomicAdd_block(address, val);
#else
  return atomicAdd(address, val);
#endif
}

/**
 * @brief `__match_any_sync` where available, emulated otherwise.
 *
 * Returns the mask of lanes in @p mask whose @p value equals the caller's.
 *
 * `__match_any_sync` is an sm_70 instruction. The emulation repeatedly elects
 * the lowest not-yet-classified lane as a leader, broadcasts its value and
 * ballots for equality, which partitions @p mask into peer groups. The loop
 * trip count depends only on `unclaimed`, which every participating lane
 * computes identically, so control flow remains warp-uniform and the
 * `__shfl_sync` / `__ballot_sync` collectives always see the full @p mask.
 *
 * Cost is O(number of distinct values) shuffles instead of a single
 * instruction; for the segmented-reduction use case in the sparse SpMV kernels
 * the rows are sorted, so the number of distinct values per warp is small.
 */
template <typename T>
__device__ __forceinline__ unsigned int match_any_sync(unsigned int mask, T value)
{
#if CUVS_HAS_VOLTA_WARP_PRIMITIVES
  return __match_any_sync(mask, value);
#else
  unsigned int const self = 1u << lane_id();
  unsigned int peers      = 0;
  unsigned int unclaimed  = mask;

  while (unclaimed != 0u) {
    // `unclaimed` is identical in every participating lane, hence so is `leader`.
    int const leader     = __ffs(unclaimed) - 1;
    T const leader_value = __shfl_sync(mask, value, leader);
    unsigned int group   = __ballot_sync(mask, value == leader_value) & mask;

    if (group & self) { peers = group; }
    unclaimed &= ~group;
  }

  return peers;
#endif
}

}  // namespace cuvs::util::arch_compat
