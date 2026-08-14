/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/detail/jit_lto/AlgorithmPlanner.hpp>
#include <cuvs/detail/jit_lto/FragmentEntry.hpp>
#include <cuvs/detail/jit_lto/common_fragments.hpp>
#include <cuvs/detail/jit_lto/ivf_pq/compute_similarity_fragments.hpp>

namespace cuvs::neighbors::ivf_pq::detail {

struct ComputeSimilarityPlanner : AlgorithmPlanner {
  inline static LauncherJitCache launcher_jit_cache{};

  ComputeSimilarityPlanner() : AlgorithmPlanner("compute_similarity", launcher_jit_cache) {}

  template <typename OutTag, typename LutTag>
  void add_entrypoint()
  {
    this->add_static_fragment<fragment_tag_compute_similarity<OutTag, LutTag>>();
  }

  template <typename LutTag, bool EnableSMemLut, uint32_t PqBits>
  void add_prepare_lut_function()
  {
    this->add_static_fragment<fragment_tag_prepare_lut<LutTag, EnableSMemLut, PqBits>>();
  }

  template <typename OutTag, bool kManageLocalTopK>
  void add_store_calculated_distances_function()
  {
    this->add_static_fragment<fragment_tag_store_calculated_distances<OutTag, kManageLocalTopK>>();
  }

  template <typename MetricTag>
  void add_precompute_base_diff_function()
  {
    this->add_static_fragment<fragment_tag_precompute_base_diff<MetricTag>>();
  }

  template <typename LutTag, typename MetricTag, bool PrecompBaseDiff, uint32_t PqBits>
  void add_create_lut_function()
  {
    this
      ->add_static_fragment<fragment_tag_create_lut<LutTag, MetricTag, PrecompBaseDiff, PqBits>>();
  }

  template <typename OutTag, typename LutTag, int Capacity>
  void add_compute_distances_function()
  {
    this->add_static_fragment<fragment_tag_compute_distances<OutTag, LutTag, Capacity>>();
  }

  template <typename OutTag, typename MetricTag>
  void add_get_early_stop_limit_function()
  {
    this->add_static_fragment<fragment_tag_get_early_stop_limit<OutTag, MetricTag>>();
  }

  template <typename FilterTag>
  void add_sample_filter_function()
  {
    this->add_static_fragment<fragment_tag_sample_filter<FilterTag>>();
    this->add_static_fragment<
      cuvs::neighbors::detail::fragment_tag_sample_filter<cuvs::neighbors::detail::tag_bitset_u32,
                                                          cuvs::neighbors::detail::tag_index_i64,
                                                          FilterTag>>();
  }

  template <uint32_t PqBits>
  void add_get_line_width_function()
  {
    this->add_static_fragment<fragment_tag_get_line_width<PqBits>>();
  }

  template <typename OutTag, typename LutTag, uint32_t PqBits>
  void add_compute_score_function()
  {
    this->add_static_fragment<fragment_tag_compute_score<OutTag, LutTag, PqBits>>();
  }

  template <typename OutTag, bool IncrementScore>
  void add_increment_score_function()
  {
    this->add_static_fragment<fragment_tag_increment_score<OutTag, IncrementScore>>();
  }
};

}  // namespace cuvs::neighbors::ivf_pq::detail
