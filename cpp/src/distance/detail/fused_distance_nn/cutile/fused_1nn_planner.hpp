/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <type_traits>

#include <cuvs/detail/jit_lto/AlgorithmPlanner.hpp>
#include <cuvs/detail/jit_lto/common_fragments.hpp>
#include <cuvs/detail/jit_lto/cutile_arch_tags.hpp>
#include <cuvs/detail/jit_lto/fused_distance_nn/fused_1nn_fragments.hpp>

#include "fused_1nn_cutile_tiles.hpp"

namespace cuvs::distance::detail {

/** Must match kernel_symbol() in fused_1nn_kernel.py (export uses with_symbol). */
template <typename DataTag, typename MetricTag, typename IndexTag>
inline const char* fused_1nn_kernel_entrypoint()
{
  constexpr bool is_i32 = std::is_same_v<IndexTag, cuvs::neighbors::detail::tag_index_i32>;
  constexpr bool is_i64 = std::is_same_v<IndexTag, cuvs::neighbors::detail::tag_index_i64>;
  static_assert(is_i32 || is_i64, "unsupported fused 1-NN cuTile index width");

  if constexpr (std::is_same_v<DataTag, cuvs::neighbors::detail::tag_f> &&
                std::is_same_v<MetricTag, metric_tag_ip>) {
    return is_i32 ? "fused_1nn_f_ip_i32" : "fused_1nn_f_ip_i64";
  } else if constexpr (std::is_same_v<DataTag, cuvs::neighbors::detail::tag_f> &&
                       std::is_same_v<MetricTag, metric_tag_l2>) {
    return is_i32 ? "fused_1nn_f_l2_i32" : "fused_1nn_f_l2_i64";
  } else if constexpr (std::is_same_v<DataTag, cuvs::neighbors::detail::tag_f> &&
                       std::is_same_v<MetricTag, metric_tag_cos>) {
    return is_i32 ? "fused_1nn_f_cos_i32" : "fused_1nn_f_cos_i64";
  } else if constexpr (std::is_same_v<DataTag, cuvs::neighbors::detail::tag_h> &&
                       std::is_same_v<MetricTag, metric_tag_ip>) {
    return is_i32 ? "fused_1nn_h_ip_i32" : "fused_1nn_h_ip_i64";
  } else if constexpr (std::is_same_v<DataTag, cuvs::neighbors::detail::tag_h> &&
                       std::is_same_v<MetricTag, metric_tag_l2>) {
    return is_i32 ? "fused_1nn_h_l2_i32" : "fused_1nn_h_l2_i64";
  } else if constexpr (std::is_same_v<DataTag, cuvs::neighbors::detail::tag_h> &&
                       std::is_same_v<MetricTag, metric_tag_cos>) {
    return is_i32 ? "fused_1nn_h_cos_i32" : "fused_1nn_h_cos_i64";
  } else {
    static_assert(sizeof(DataTag) == 0, "unsupported fused 1-NN cuTile data/metric combination");
    return "";
  }
}

template <typename DataT, cuvs::distance::DistanceType Metric, typename IdxT>
struct Fused1nnTilePlanner : TileAlgorithmPlanner {
  using DataTag   = fused_1nn_data_tag_t<DataT>;
  using MetricTag = fused_1nn_metric_tag_t<Metric>;
  using IndexTag  = fused_1nn_index_tag_t<IdxT>;

  inline static LauncherJitCache launcher_jit_cache{};

  Fused1nnTilePlanner()
    : TileAlgorithmPlanner(fused_1nn_kernel_entrypoint<DataTag, MetricTag, IndexTag>(),
                           launcher_jit_cache)
  {
  }

  /** Registers embedded cubin modules (one per SM); see register_cutile_fragment.cpp object files.
   */
  void add_entrypoint()
  {
    using cuvs::detail::jit_lto::cutile_arch_12_0;
    using cuvs::detail::jit_lto::cutile_arch_8_0;
    using cuvs::detail::jit_lto::cutile_arch_8_6;
    using cuvs::detail::jit_lto::cutile_arch_9_0;

    this->add_static_fragment<fragment_tag_fused_1nn_cubin<DataTag,
                                                           MetricTag,
                                                           IndexTag,
                                                           fused_1nn_matrix_tile_cutile_arch_8_0,
                                                           cutile_arch_8_0>>();
    this->add_static_fragment<fragment_tag_fused_1nn_cubin<DataTag,
                                                           MetricTag,
                                                           IndexTag,
                                                           fused_1nn_matrix_tile_cutile_arch_8_6,
                                                           cutile_arch_8_6>>();
    this->add_static_fragment<fragment_tag_fused_1nn_cubin<DataTag,
                                                           MetricTag,
                                                           IndexTag,
                                                           fused_1nn_matrix_tile_cutile_arch_9_0,
                                                           cutile_arch_9_0>>();
    this->add_static_fragment<fragment_tag_fused_1nn_cubin<DataTag,
                                                           MetricTag,
                                                           IndexTag,
                                                           fused_1nn_matrix_tile_cutile_arch_12_0,
                                                           cutile_arch_12_0>>();
  }

  void add_tileir_fallback()
  {
    this->add_static_tileir_fragment<
      fragment_tag_fused_1nn_tileir<DataTag, MetricTag, IndexTag, fused_1nn_matrix_tile>>();
  }
};

}  // namespace cuvs::distance::detail
