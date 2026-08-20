/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>

#include <cuvs/detail/jit_lto/TileAlgorithmPlanner.hpp>
#include <cuvs/detail/jit_lto/cutile_module.hpp>

#include <raft/core/logger.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

namespace cuvs::detail::jit_lto {

namespace {

template <typename FragmentT>
CutileTileConfig tile_config_from_fragment(const FragmentT* fragment, const std::string& entrypoint)
{
  if (fragment == nullptr) {
    RAFT_FAIL("cuTile planner '%s' has no registered fragments", entrypoint.c_str());
  }
  const int tile_m = fragment->get_tile_m();
  const int tile_n = fragment->get_tile_n();
  const int tile_k = fragment->get_tile_k();
  if (tile_m <= 0 || tile_n <= 0 || tile_k <= 0) {
    RAFT_FAIL(
      "cuTile planner '%s' is missing tile geometry in its static fragment (check "
      "register_cutile_fragment.cpp generation)",
      entrypoint.c_str());
  }
  return CutileTileConfig{tile_m, tile_n, tile_k};
}

}  // namespace

std::shared_ptr<rtcx::algorithm_launcher> TileAlgorithmPlanner::try_get_launcher()
{
  auto launch_key = this->get_planner_key();

  {
    std::shared_lock<std::shared_mutex> read_lock(launcher_cache_.mutex);
    if (launcher_cache_.build_failed.count(launch_key)) { return nullptr; }
    if (auto it = launcher_cache_.launchers.find(launch_key);
        it != launcher_cache_.launchers.end()) {
      return it->second;
    }
  }

  std::unique_lock<std::shared_mutex> write_lock(launcher_cache_.mutex);
  if (launcher_cache_.build_failed.count(launch_key)) { return nullptr; }
  if (auto it = launcher_cache_.launchers.find(launch_key); it != launcher_cache_.launchers.end()) {
    return it->second;
  }

  RAFT_LOG_DEBUG("Building launcher for kernel entrypoint: %s", entrypoint_.c_str());
  auto launcher = this->build();
  if (!launcher) {
    launcher_cache_.build_failed.insert(launch_key);
    return nullptr;
  }
  launcher_cache_.launchers[launch_key] = launcher;
  return launcher;
}

std::shared_ptr<rtcx::algorithm_launcher> TileAlgorithmPlanner::get_launcher()
{
  auto launcher = try_get_launcher();
  if (!launcher) {
    RAFT_FAIL("Failed to build launcher for kernel entrypoint: %s", entrypoint_.c_str());
  }
  return launcher;
}

std::string TileAlgorithmPlanner::get_planner_key() const
{
  std::string key = entrypoint_;
  for (const auto& fragment : cubin_fragments_) {
    key += fragment->get_key();
  }
  if (tileir_fragment_) { key += tileir_fragment_->get_key(); }
  return key;
}

CutileTileConfig TileAlgorithmPlanner::tile_config() const
{
  int cc_major = 0;
  int cc_minor = 0;
  if (cuvs::detail::jit_lto::get_device_compute_capability(cc_major, cc_minor)) {
    if (const auto* fragment = cuvs::detail::jit_lto::find_compatible_cubin_fragment(
          cc_major, cc_minor, cubin_fragments_)) {
      return tile_config_from_fragment(fragment, entrypoint_);
    }
  }

  if (tileir_fragment_) { return tile_config_from_fragment(tileir_fragment_.get(), entrypoint_); }

  if (!cubin_fragments_.empty()) {
    return tile_config_from_fragment(cubin_fragments_.front().get(), entrypoint_);
  }

  RAFT_FAIL("cuTile planner '%s' has no registered fragments", entrypoint_.c_str());
}

std::shared_ptr<rtcx::algorithm_launcher> TileAlgorithmPlanner::build()
{
  int cc_major = 0;
  int cc_minor = 0;
  if (!cuvs::detail::jit_lto::get_device_compute_capability(cc_major, cc_minor)) { return nullptr; }

  int driver_version = 0;
  if (cudaDriverGetVersion(&driver_version) != cudaSuccess) { return nullptr; }

  auto image = cuvs::detail::jit_lto::resolve_cutile_module_image(
    cc_major, cc_minor, driver_version, cubin_fragments_, tileir_fragment_.get());
  if (!image) { return nullptr; }

  return cuvs::detail::jit_lto::load_cutile_launcher(*image, entrypoint_);
}

}  // namespace cuvs::detail::jit_lto
