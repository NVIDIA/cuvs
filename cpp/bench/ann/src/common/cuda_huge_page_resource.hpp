/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/error.hpp>
#include <raft/core/logger_macros.hpp>

#include <rmm/detail/aligned.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <sys/mman.h>

#include <cstddef>
#include <cstring>

namespace raft::mr {
/**
 * @brief Memory resource that uses mmap to allocate memory with huge pages.
 * It is assumed that the allocated memory is directly accessible on device. This currently only
 * works on GH systems.
 *
 * TODO(tfeher): consider improving or removing this helper once we made progress with
 * https://github.com/rapidsai/raft/issues/1819
 */
class cuda_huge_page_resource {
 public:
  cuda_huge_page_resource()                                                  = default;
  ~cuda_huge_page_resource()                                                 = default;
  cuda_huge_page_resource(cuda_huge_page_resource const&)                    = default;
  cuda_huge_page_resource(cuda_huge_page_resource&&)                         = default;
  auto operator=(cuda_huge_page_resource const&) -> cuda_huge_page_resource& = default;
  auto operator=(cuda_huge_page_resource&&) -> cuda_huge_page_resource&      = default;

  void* allocate(cuda::stream_ref,
                 std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    void* addr{nullptr};
    addr = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (addr == MAP_FAILED) { RAFT_FAIL("huge_page_resource::MAP FAILED"); }
    if (madvise(addr, bytes, MADV_HUGEPAGE) == -1) {
      munmap(addr, bytes);
      RAFT_FAIL("huge_page_resource::madvise MADV_HUGEPAGE");
    }
    memset(addr, 0, bytes);
    return addr;
  }

  void deallocate(cuda::stream_ref,
                  void* ptr,
                  std::size_t size,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    if (munmap(ptr, size) == -1) { RAFT_LOG_ERROR("huge_page_resource::munmap failed"); }
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return allocate(cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    deallocate(cuda::stream_ref{cudaStream_t{nullptr}}, ptr, bytes, alignment);
  }

  [[nodiscard]] bool operator==(cuda_huge_page_resource const&) const noexcept { return true; }
  [[nodiscard]] bool operator!=(cuda_huge_page_resource const&) const noexcept { return false; }

  constexpr friend void get_property(cuda_huge_page_resource const&,
                                     cuda::mr::device_accessible) noexcept
  {
  }
  constexpr friend void get_property(cuda_huge_page_resource const&,
                                     cuda::mr::host_accessible) noexcept
  {
  }
};
static_assert(cuda::mr::synchronous_resource<cuda_huge_page_resource>);
static_assert(cuda::mr::resource<cuda_huge_page_resource>);
static_assert(cuda::mr::resource_with<cuda_huge_page_resource, cuda::mr::device_accessible>);
static_assert(cuda::mr::resource_with<cuda_huge_page_resource, cuda::mr::host_accessible>);
}  // namespace raft::mr
