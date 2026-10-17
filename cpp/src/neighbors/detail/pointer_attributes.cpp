/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/common.hpp>

#include <raft/util/cuda_rt_essentials.hpp>

#include <cuda_runtime_api.h>

#include <mutex>
#include <unordered_map>

namespace cuvs::neighbors::detail {

namespace {

/**
 * Whether `device` reads pageable (unregistered) host memory through the host page tables (ATS,
 * e.g. Grace Hopper / Grace Blackwell). The result is cached per device.
 */
auto pageable_memory_uses_host_page_tables(int device) -> bool
{
  static std::mutex mutex;
  static std::unordered_map<int, bool> cache;
  std::lock_guard<std::mutex> lock(mutex);
  if (auto it = cache.find(device); it != cache.end()) { return it->second; }
  int uses_host_page_tables = 0;
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(
    &uses_host_page_tables, cudaDevAttrPageableMemoryAccessUsesHostPageTables, device));
  cache.emplace(device, uses_host_page_tables != 0);
  return uses_host_page_tables != 0;
}

}  // namespace

auto is_device_accessible(const cudaPointerAttributes& attr) -> bool
{
  switch (attr.type) {
    case cudaMemoryTypeDevice:
    case cudaMemoryTypeManaged: return true;
    case cudaMemoryTypeHost: return attr.devicePointer != nullptr;
    default: {
      if (attr.devicePointer == nullptr) { return false; }
      int device = 0;
      RAFT_CUDA_TRY(cudaGetDevice(&device));
      return pageable_memory_uses_host_page_tables(device);
    }
  }
}

}  // namespace cuvs::neighbors::detail
