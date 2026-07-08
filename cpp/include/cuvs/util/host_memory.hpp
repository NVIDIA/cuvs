/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/error.hpp>

#include <cuvs/core/export.hpp>

#include <cstddef>
#include <optional>

namespace CUVS_EXPORT cuvs {
namespace util {

/** @brief Snapshot of host and cgroup memory available to the current process. */
struct host_memory_info {
  /** Host-wide MemAvailable from /proc/meminfo. */
  size_t system_available;
  /** Effective memory available after applying any cgroup limit. */
  size_t available;
  /** Hard limit of the most constrained cgroup ancestor, when finite. */
  std::optional<size_t> cgroup_limit;
  /** Current usage charged to the most constrained cgroup ancestor. */
  std::optional<size_t> cgroup_current;
};

/**
 * @brief Get host memory available to the current process.
 *
 * Combines host-wide MemAvailable with cgroup v1 or v2 memory headroom. When a finite cgroup limit
 * is found, the effective value is `min(MemAvailable, limit - current_usage)`. Visible ancestor
 * cgroups are considered because a parent can impose a tighter limit than the process's leaf
 * cgroup.
 */
host_memory_info get_host_memory_info();

/**
 * @brief Get effective available host memory.
 *
 * Convenience wrapper around get_host_memory_info().
 *
 * @return Available memory in bytes
 */
size_t get_free_host_memory();

}  // namespace util
}  // namespace CUVS_EXPORT cuvs
