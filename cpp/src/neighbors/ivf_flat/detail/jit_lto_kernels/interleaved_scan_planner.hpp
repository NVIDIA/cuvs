/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/detail/jit_lto/AlgorithmPlanner.hpp>
#include <cuvs/detail/jit_lto/FragmentEntry.hpp>
#include <cuvs/detail/jit_lto/common_fragments.hpp>
#include <cuvs/detail/jit_lto/ivf_flat/interleaved_scan_fragments.hpp>
#include <iostream>
#include <string>

namespace cuvs::neighbors::ivf_flat::detail {

struct InterleavedScanPlanner : AlgorithmPlanner {
  inline static LauncherJitCache launcher_jit_cache{};

  InterleavedScanPlanner() : AlgorithmPlanner("interleaved_scan", launcher_jit_cache) {}

  template <typename DataTag, typename AccTag, typename IdxTag, int Capacity, bool Ascending>
  void add_entrypoint()
  {
    this->add_static_fragment<
      fragment_tag_interleaved_scan<DataTag, AccTag, IdxTag, Capacity, Ascending>>();
  }

  template <typename DataTag, typename AccTag, bool ComputeNorm, int Veclen>
  void add_load_and_compute_dist_function()
  {
    this->add_static_fragment<
      fragment_tag_load_and_compute_dist<DataTag, AccTag, ComputeNorm, Veclen>>();
  }

  template <typename DataTag, typename AccTag, typename MetricTag, int Veclen>
  void add_metric_device_function()
  {
    this->add_static_fragment<fragment_tag_metric<DataTag, AccTag, MetricTag, Veclen>>();
  }

  void add_metric_udf_fragment(std::unique_ptr<UDFFatbinFragment> fragment)
  {
    this->add_fragment(std::move(fragment));
  }

  template <typename IndexTag, typename FilterTag>
  void add_filter_device_function()
  {
    this->add_static_fragment<fragment_tag_filter<IndexTag, FilterTag>>();
    this->add_static_fragment<
      cuvs::neighbors::detail::
        fragment_tag_sample_filter<cuvs::neighbors::detail::tag_bitset_u32, IndexTag, FilterTag>>();
  }

  template <typename PostLambdaTag>
  void add_post_lambda_device_function()
  {
    this->add_static_fragment<fragment_tag_post_lambda<PostLambdaTag>>();
  }
};

}  // namespace cuvs::neighbors::ivf_flat::detail
