/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "CutileFragmentEntry.hpp"

#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <rtcx/algorithm_launcher.hpp>

namespace cuvs::detail::jit_lto {

struct CutileRuntimeCapabilities;

struct TileLauncherCache {
  std::shared_mutex mutex;
  std::unordered_map<std::string, std::shared_ptr<rtcx::algorithm_launcher>> launchers;
  // Cache expected compatibility misses so an unsupported module is not loaded on every call.
  // Unexpected CUDA errors are raised rather than inserted here.
  std::unordered_set<std::string> unavailable_launchers;
};

/** Loads prebuilt cubins or TileIR bytecode directly through the CUDA library API. */
struct TileAlgorithmPlanner {
  TileAlgorithmPlanner(std::string entrypoint, TileLauncherCache& launcher_cache)
    : entrypoint_(std::move(entrypoint)), launcher_cache_(launcher_cache)
  {
  }

  virtual ~TileAlgorithmPlanner() = default;

  std::shared_ptr<rtcx::algorithm_launcher> get_launcher();

  /** Returns nullptr when no module can be loaded for the current device (does not RAFT_FAIL). */
  std::shared_ptr<rtcx::algorithm_launcher> try_get_launcher();

  template <typename FragmentTag>
  void add_static_fragment()
  {
    cubin_fragments_.push_back(std::make_unique<StaticCubinFragmentEntry<FragmentTag>>());
  }

  template <typename FragmentTag>
  void add_static_tileir_fragment()
  {
    tileir_fragment_ = std::make_unique<StaticTileIrBytecodeFragmentEntry<FragmentTag>>();
  }

  /** Tile geometry from the cubin or TileIR fragment that would load on this device. */
  CutileTileConfig tile_config() const;

 protected:
  std::vector<std::unique_ptr<CubinFragmentEntry>> cubin_fragments_;
  std::unique_ptr<TileIrBytecodeFragmentEntry> tileir_fragment_;

 private:
  std::string get_planner_key(const CutileRuntimeCapabilities* capabilities) const;

  std::shared_ptr<rtcx::algorithm_launcher> build(const CutileRuntimeCapabilities* capabilities);

  std::string entrypoint_;
  TileLauncherCache& launcher_cache_;
};

}  // namespace cuvs::detail::jit_lto
