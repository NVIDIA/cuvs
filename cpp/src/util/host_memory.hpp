/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuvs/core/export.hpp>

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>

namespace cuvs::util::detail {

enum class cgroup_version { v1, v2 };

struct cgroup_memory_info {
  size_t limit;
  size_t current;
  size_t reclaimable_file;
  size_t working_set;
  size_t available;
  cgroup_version version;
  std::string path;
};

CUVS_EXPORT std::optional<cgroup_memory_info> get_cgroup_memory_info(
  const std::filesystem::path& proc_self_cgroup    = "/proc/self/cgroup",
  const std::filesystem::path& proc_self_mountinfo = "/proc/self/mountinfo");

}  // namespace cuvs::util::detail
