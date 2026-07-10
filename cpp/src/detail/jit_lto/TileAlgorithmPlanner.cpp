/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>
#include <string>

#include <cuvs/detail/jit_lto/AlgorithmPlanner.hpp>
#include <cuvs/detail/jit_lto/cutile_module.hpp>

#include <raft/core/logger.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

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

std::string TileAlgorithmPlanner::get_planner_key() const
{
  std::string key = this->entrypoint;
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
    for (const auto& fragment : cubin_fragments_) {
      if (fragment->get_cc_major() == cc_major && fragment->get_cc_minor() == cc_minor) {
        return tile_config_from_fragment(fragment.get(), entrypoint);
      }
    }
  }

  if (tileir_fragment_) { return tile_config_from_fragment(tileir_fragment_.get(), entrypoint); }

  if (!cubin_fragments_.empty()) {
    return tile_config_from_fragment(cubin_fragments_.front().get(), entrypoint);
  }

  RAFT_FAIL("cuTile planner '%s' has no registered fragments", entrypoint.c_str());
}

std::shared_ptr<AlgorithmLauncher> TileAlgorithmPlanner::build()
{
  int cc_major = 0;
  int cc_minor = 0;
  if (!cuvs::detail::jit_lto::get_device_compute_capability(cc_major, cc_minor)) { return nullptr; }

  int driver_version = 0;
  if (cudaDriverGetVersion(&driver_version) != cudaSuccess) { return nullptr; }

  auto image = cuvs::detail::jit_lto::resolve_cutile_module_image(
    cc_major, cc_minor, driver_version, cubin_fragments_, tileir_fragment_.get());
  if (!image) { return nullptr; }

  return cuvs::detail::jit_lto::load_cutile_launcher(*image, this->entrypoint);
}
