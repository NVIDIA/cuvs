/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>

#include <cuvs/detail/jit_lto/AlgorithmPlanner.hpp>

#include <raft/core/logger.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

std::shared_ptr<AlgorithmLauncher> AlgorithmPlanner::read_cache(std::string const& launch_key) const
{
  auto& launchers = jit_cache_.launchers;
  std::shared_lock<std::shared_mutex> read_lock(jit_cache_.mutex);
  if (auto it = launchers.find(launch_key); it != launchers.end()) { return it->second; }
  return nullptr;
}

std::shared_ptr<AlgorithmLauncher> AlgorithmPlanner::try_get_launcher()
{
  auto launch_key = this->get_planner_key();

  {
    std::shared_lock<std::shared_mutex> read_lock(jit_cache_.mutex);
    if (jit_cache_.build_failed.count(launch_key)) { return nullptr; }
    if (auto hit = read_cache(launch_key)) { return hit; }
  }

  std::unique_lock<std::shared_mutex> write_lock(jit_cache_.mutex);
  if (jit_cache_.build_failed.count(launch_key)) { return nullptr; }
  if (auto it = jit_cache_.launchers.find(launch_key); it != jit_cache_.launchers.end()) {
    return it->second;
  }

  RAFT_LOG_DEBUG("Building launcher for kernel entrypoint: %s", this->entrypoint.c_str());
  auto launcher = this->build();
  if (!launcher) {
    jit_cache_.build_failed.insert(launch_key);
    return nullptr;
  }
  jit_cache_.launchers[launch_key] = launcher;
  return launcher;
}

std::shared_ptr<AlgorithmLauncher> AlgorithmPlanner::get_launcher()
{
  auto launcher = try_get_launcher();
  if (!launcher) {
    RAFT_FAIL("Failed to build launcher for kernel entrypoint: %s", this->entrypoint.c_str());
  }
  return launcher;
}
