/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdint.h>
#include <sys/mman.h>

#include <cstdlib>
#include <cstring>
#include <new>

#include <raft/util/integer_utils.hpp>

namespace cuvs::neighbors::ivf_rabitq::detail {

#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))

namespace memory {
template <size_t alignment, class T, bool HUGE_PAGE = false>
inline T* align_mm(size_t nbytes)
{
  static_assert(alignment != 0 && (alignment & (alignment - 1)) == 0);
  static_assert(alignment % alignof(void*) == 0);

  size_t size = raft::round_up_safe<size_t>(nbytes, alignment);
  if (size == 0) { size = alignment; }
  void* p = std::aligned_alloc(alignment, size);
  if (p == nullptr) { throw std::bad_alloc{}; }
  if constexpr (HUGE_PAGE) { madvise(p, size, MADV_HUGEPAGE); }
  std::memset(p, 0, size);
  return static_cast<T*>(p);
}

}  // namespace memory
}  // namespace cuvs::neighbors::ivf_rabitq::detail
