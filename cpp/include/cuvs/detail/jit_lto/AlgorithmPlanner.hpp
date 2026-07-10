/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <shared_mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "AlgorithmLauncher.hpp"
#include "FragmentEntry.hpp"

struct LauncherJitCache {
  std::shared_mutex mutex;
  std::unordered_map<std::string, std::shared_ptr<AlgorithmLauncher>> launchers;
  std::unordered_set<std::string> build_failed;
};

struct AlgorithmPlanner {
  AlgorithmPlanner(std::string entrypoint, LauncherJitCache& jit_cache)
    : entrypoint(std::move(entrypoint)), jit_cache_(jit_cache)
  {
  }

  virtual ~AlgorithmPlanner() = default;

  std::shared_ptr<AlgorithmLauncher> get_launcher();

  /** Returns nullptr when no module can be loaded for the current device (does not RAFT_FAIL). */
  std::shared_ptr<AlgorithmLauncher> try_get_launcher();

  std::string entrypoint;

 protected:
  virtual std::shared_ptr<AlgorithmLauncher> build() = 0;

  virtual std::string get_planner_key() const = 0;

  std::shared_ptr<AlgorithmLauncher> read_cache(std::string const& launch_key) const;

  LauncherJitCache& jit_cache_;
};

/** Links embedded LTO fatbin fragments at runtime via nvJitLink. */
struct LTOAlgorithmPlanner : AlgorithmPlanner {
  LTOAlgorithmPlanner(std::string entrypoint, LauncherJitCache& jit_cache)
    : AlgorithmPlanner(std::move(entrypoint), jit_cache)
  {
  }

  std::vector<std::unique_ptr<FragmentEntry>> fragments;

  template <typename T, typename = std::enable_if_t<std::is_convertible_v<T*, FragmentEntry*>>>
  void add_fragment(std::unique_ptr<T> fragment)
  {
    fragments.push_back(std::unique_ptr<FragmentEntry>(std::move(fragment)));
  }

  template <typename FragmentTag>
  void add_static_fragment()
  {
    add_fragment(std::make_unique<StaticFatbinFragmentEntry<FragmentTag>>());
  }

 protected:
  /** Extra link-time option strings passed to nvJitLink. */
  std::vector<std::string> linktime_extra_options;

  std::string get_planner_key() const override;

  std::shared_ptr<AlgorithmLauncher> build() override;
};

/** Loads prebuilt cubins or TileIR bytecode via cudaLibraryLoadData. */
struct TileAlgorithmPlanner : AlgorithmPlanner {
  TileAlgorithmPlanner(std::string entrypoint, LauncherJitCache& jit_cache)
    : AlgorithmPlanner(std::move(entrypoint), jit_cache)
  {
  }

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

  std::string get_planner_key() const override;

  std::shared_ptr<AlgorithmLauncher> build() override;
};
