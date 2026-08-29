/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file warp_intrinsic_compat.cuh
 * @brief Pre-Volta definition of `__match_any_sync` for dependency headers that call it
 *        unconditionally.
 *
 * `__match_any_sync` is an sm_70 instruction; CUDA only declares it for `__CUDA_ARCH__ >= 700`.
 * libcu++'s block-scoped barrier calls it from the body of `__arrive_sm70`
 * (`cuda/__barrier/barrier_block_scope.h`). That body is only *executed* under an
 * `NV_IF_TARGET(NV_PROVIDES_SM_70, ...)`, but it is still *parsed* for every device pass, so an
 * sm_50 or sm_60 compilation fails with "the global scope has no __match_any_sync" -- reached, in
 * cuVS, through `cuco::bloom_filter` -> `cuda/annotated_ptr` -> `cuda/__memcpy_async`.
 *
 * Following the same rule as `atomic_compat.cuh`, the fix lives in cuVS rather than in the
 * dependency: CCCL and cuco are CPM-fetched, so patching them in place would be invisible to a
 * reviewer, lost on a version bump and untestable in CI. cuVS instead makes the missing symbol
 * visible before the dependency header is parsed.
 *
 * The definition is the warp-uniform ballot emulation from `arch_compat.cuh`, so it is correct if
 * it is ever actually called -- although on a pre-Volta device it should not be, since libcu++
 * dispatches to a different implementation there.
 *
 * Expands to nothing on sm_70+ and in the host pass.
 *
 * > **Include-order constraint.** This header must be included **before** any header that reaches
 * > `cuda/__barrier/barrier_block_scope.h`. A clang-format/IWYU pass that sorts includes will
 * > silently break the pre-Volta build.
 */

#pragma once

#include <cuda_runtime.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)

namespace cuvs::util::detail {

/**
 * @brief Ballot-based emulation of `__match_any_sync`.
 *
 * Repeatedly elects the lowest not-yet-classified lane as leader, broadcasts its value and ballots
 * for equality, which partitions @p mask into peer groups. The loop trip count depends only on
 * `unclaimed`, which every participating lane computes identically, so control flow stays
 * warp-uniform -- mandatory pre-Volta, where a warp has a single program counter.
 */
template <typename T>
__device__ __forceinline__ unsigned int match_any_sync_emulated(unsigned int mask, T value)
{
  unsigned int const self = 1u << (threadIdx.x & 31u);
  unsigned int peers      = 0;
  unsigned int unclaimed  = mask;

  while (unclaimed != 0u) {
    int const leader     = __ffs(static_cast<int>(unclaimed)) - 1;
    T const leader_value = __shfl_sync(mask, value, leader);
    unsigned int group   = __ballot_sync(mask, value == leader_value) & mask;

    if (group & self) { peers = group; }
    unclaimed &= ~group;
  }

  return peers;
}

}  // namespace cuvs::util::detail

// The dependency headers call this unqualified, with fundamental argument types, so the
// declaration has to be at global scope and visible at their definition context.
#define CUVS_DEFINE_MATCH_ANY_SYNC(T)                                             \
  __device__ __forceinline__ unsigned int __match_any_sync(unsigned int mask, T value) \
  {                                                                               \
    return cuvs::util::detail::match_any_sync_emulated(mask, value);              \
  }

CUVS_DEFINE_MATCH_ANY_SYNC(int)
CUVS_DEFINE_MATCH_ANY_SYNC(unsigned int)
CUVS_DEFINE_MATCH_ANY_SYNC(long)
CUVS_DEFINE_MATCH_ANY_SYNC(unsigned long)
CUVS_DEFINE_MATCH_ANY_SYNC(long long)
CUVS_DEFINE_MATCH_ANY_SYNC(unsigned long long)

#undef CUVS_DEFINE_MATCH_ANY_SYNC

#endif  // __CUDA_ARCH__ < 700
