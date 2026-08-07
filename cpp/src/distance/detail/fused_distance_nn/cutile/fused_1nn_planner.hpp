/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
template <typename DataTag, typename AbiTag>
inline const char* fused_1nn_kernel_entrypoint()
{
  constexpr bool is_relaxed = std::is_same_v<AbiTag, cutile_abi_relaxed>;
  static_assert(is_relaxed || std::is_same_v<AbiTag, cutile_abi_strict>,
                "unsupported fused 1-NN cuTile ABI");

  if constexpr (std::is_same_v<DataTag, cuvs::neighbors::detail::tag_f>) {
    return is_relaxed ? "fused_1nn_f_i32_relaxed" : "fused_1nn_f_i32";
  } else if constexpr (std::is_same_v<DataTag, cuvs::neighbors::detail::tag_h>) {
    return is_relaxed ? "fused_1nn_h_i32_relaxed" : "fused_1nn_h_i32";
  } else {
    static_assert(sizeof(DataTag) == 0, "unsupported fused 1-NN cuTile data type");
    return "";
  }
}

template <typename DataT, typename AbiTag>
struct Fused1nnTilePlanner : TileAlgorithmPlanner {
  using DataTag  = fused_1nn_data_tag_t<DataT>;
  using IndexTag = cuvs::neighbors::detail::tag_index_i32;

  inline static LauncherJitCache launcher_jit_cache{};

  Fused1nnTilePlanner()
    : TileAlgorithmPlanner(fused_1nn_kernel_entrypoint<DataTag, AbiTag>(), launcher_jit_cache)
  {
  }

  /** Registers embedded cubin modules (one per SM); see register_cutile_fragment.cpp object files.
   */
  void add_entrypoint()
  {
    using cuvs::detail::jit_lto::cutile_arch_10_0;
    using cuvs::detail::jit_lto::cutile_arch_12_0;
    using cuvs::detail::jit_lto::cutile_arch_8_0;
    using cuvs::detail::jit_lto::cutile_arch_8_6;
    using cuvs::detail::jit_lto::cutile_arch_9_0;

    constexpr bool is_relaxed = std::is_same_v<AbiTag, cutile_abi_relaxed>;
    using Tile80              = std::conditional_t<is_relaxed,
                                                   fused_1nn_matrix_tile_cutile_arch_8_0_relaxed,
                                                   fused_1nn_matrix_tile_cutile_arch_8_0_strict>;
    using Tile86              = std::conditional_t<is_relaxed,
                                                   fused_1nn_matrix_tile_cutile_arch_8_6_relaxed,
                                                   fused_1nn_matrix_tile_cutile_arch_8_6_strict>;
    using Tile90              = std::conditional_t<is_relaxed,
                                                   fused_1nn_matrix_tile_cutile_arch_9_0_relaxed,
                                                   fused_1nn_matrix_tile_cutile_arch_9_0_strict>;
    using Tile100             = std::conditional_t<is_relaxed,
                                                   fused_1nn_matrix_tile_cutile_arch_10_0_relaxed,
                                                   fused_1nn_matrix_tile_cutile_arch_10_0_strict>;
    using Tile120             = std::conditional_t<is_relaxed,
                                                   fused_1nn_matrix_tile_cutile_arch_12_0_relaxed,
                                                   fused_1nn_matrix_tile_cutile_arch_12_0_strict>;

    this->add_static_fragment<
      fragment_tag_fused_1nn_cubin<DataTag, IndexTag, Tile80, AbiTag, cutile_arch_8_0>>();
    this->add_static_fragment<
      fragment_tag_fused_1nn_cubin<DataTag, IndexTag, Tile86, AbiTag, cutile_arch_8_6>>();
    this->add_static_fragment<
      fragment_tag_fused_1nn_cubin<DataTag, IndexTag, Tile90, AbiTag, cutile_arch_9_0>>();
    this->add_static_fragment<
      fragment_tag_fused_1nn_cubin<DataTag, IndexTag, Tile100, AbiTag, cutile_arch_10_0>>();
    this->add_static_fragment<
      fragment_tag_fused_1nn_cubin<DataTag, IndexTag, Tile120, AbiTag, cutile_arch_12_0>>();
  }

  void add_tileir_fallback()
  {
    constexpr bool is_relaxed = std::is_same_v<AbiTag, cutile_abi_relaxed>;
    using TileIr              = std::conditional_t<is_relaxed,
                                                   fused_1nn_matrix_tile_tileir_relaxed,
                                                   fused_1nn_matrix_tile_tileir_strict>;
    this->add_static_tileir_fragment<
      fragment_tag_fused_1nn_tileir<DataTag, IndexTag, TileIr, AbiTag>>();
  }
};

}  // namespace cuvs::distance::detail
