/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/detail/jit_lto/AlgorithmPlanner.hpp>
#include <cuvs/detail/jit_lto/common_fragments.hpp>
#include <cuvs/detail/jit_lto/ivf_sq/scan_fragments.hpp>

#include <string>

namespace cuvs::neighbors::ivf_sq::detail {

struct IvfSqScanPlanner : AlgorithmPlanner {
  inline static LauncherJitCache launcher_jit_cache{};

  IvfSqScanPlanner() : AlgorithmPlanner("ivf_sq_scan", launcher_jit_cache) {}

  template <int Capacity>
  void add_entrypoint()
  {
    this->add_static_fragment<fragment_tag_ivf_sq_scan<Capacity>>();
  }

  template <typename FilterTag>
  void add_filter_device_function()
  {
    this->add_static_fragment<fragment_tag_ivf_sq_filter<FilterTag>>();
    this->add_static_fragment<
      cuvs::neighbors::detail::fragment_tag_sample_filter<cuvs::neighbors::detail::tag_bitset_u32,
                                                          cuvs::neighbors::detail::tag_index_i64,
                                                          FilterTag>>();
  }

  template <typename MetricTag>
  void add_setup_invariant_smem_function()
  {
    this->add_static_fragment<fragment_tag_setup_invariant_smem<MetricTag>>();
  }

  template <typename MetricTag>
  void add_setup_per_probe_smem_function()
  {
    this->add_static_fragment<fragment_tag_setup_per_probe_smem<MetricTag>>();
  }

  template <typename MetricTag>
  void add_accumulate_distance_function()
  {
    this->add_static_fragment<fragment_tag_accumulate_distance<MetricTag>>();
  }

  template <typename MetricTag>
  void add_finalize_distance_function()
  {
    this->add_static_fragment<fragment_tag_finalize_distance<MetricTag>>();
  }
};

}  // namespace cuvs::neighbors::ivf_sq::detail
