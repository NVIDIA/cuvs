/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <type_traits>

#include <cuvs/detail/jit_lto/TileAlgorithmPlanner.hpp>
#include <cuvs/detail/jit_lto/common_fragments.hpp>
#include <cuvs/detail/jit_lto/cutile_arch_tags.hpp>
#include <cuvs/detail/jit_lto/fused_distance_nn/fused_1nn_fragments.hpp>

#include "fused_1nn_cutile_tiles.hpp"

namespace cuvs::distance::detail {

/** Must match kernel_symbol() in fused_1nn_kernel.py (export uses with_symbol). */
template <typename DataTag, typename AbiTag>
inline const char* fused_1nn_kernel_entrypoint()
{
  constexpr bool is_relaxed  = std::is_same_v<AbiTag, cutile_abi_relaxed>;
  constexpr bool is_high_dim = std::is_same_v<AbiTag, cutile_abi_high_dim>;
  static_assert(is_relaxed || is_high_dim || std::is_same_v<AbiTag, cutile_abi_strict>,
                "unsupported fused 1-NN cuTile ABI");

  if constexpr (std::is_same_v<DataTag, cuvs::neighbors::detail::tag_f>) {
    if constexpr (is_high_dim) return "fused_1nn_f_i32_hd";
    return is_relaxed ? "fused_1nn_f_i32_relaxed" : "fused_1nn_f_i32";
  } else if constexpr (std::is_same_v<DataTag, cuvs::neighbors::detail::tag_h>) {
    if constexpr (is_high_dim) return "fused_1nn_h_i32_hd";
    return is_relaxed ? "fused_1nn_h_i32_relaxed" : "fused_1nn_h_i32";
  } else {
    static_assert(sizeof(DataTag) == 0, "unsupported fused 1-NN cuTile data type");
    return "";
  }
}

template <typename DataT, typename AbiTag>
struct Fused1nnTilePlanner : cuvs::detail::jit_lto::TileAlgorithmPlanner {
  using DataTag  = std::conditional_t<std::is_same_v<DataT, float>,
                                      cuvs::neighbors::detail::tag_f,
                                      cuvs::neighbors::detail::tag_h>;
  using IndexTag = cuvs::neighbors::detail::tag_index_i32;

  inline static cuvs::detail::jit_lto::TileLauncherCache launcher_cache{};

  Fused1nnTilePlanner()
    : TileAlgorithmPlanner(fused_1nn_kernel_entrypoint<DataTag, AbiTag>(), launcher_cache)
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
    constexpr bool is_float   = std::is_same_v<DataTag, cuvs::neighbors::detail::tag_f>;
    using Tile80 =
      std::conditional_t<is_float,
                         std::conditional_t<is_relaxed,
                                            fused_1nn_matrix_tile_f_cutile_arch_8_0_relaxed,
                                            fused_1nn_matrix_tile_f_cutile_arch_8_0_strict>,
                         std::conditional_t<is_relaxed,
                                            fused_1nn_matrix_tile_h_cutile_arch_8_0_relaxed,
                                            fused_1nn_matrix_tile_h_cutile_arch_8_0_strict>>;
    using Tile86 =
      std::conditional_t<is_float,
                         std::conditional_t<is_relaxed,
                                            fused_1nn_matrix_tile_f_cutile_arch_8_6_relaxed,
                                            fused_1nn_matrix_tile_f_cutile_arch_8_6_strict>,
                         std::conditional_t<is_relaxed,
                                            fused_1nn_matrix_tile_h_cutile_arch_8_6_relaxed,
                                            fused_1nn_matrix_tile_h_cutile_arch_8_6_strict>>;
    using Tile90 =
      std::conditional_t<is_float,
                         std::conditional_t<is_relaxed,
                                            fused_1nn_matrix_tile_f_cutile_arch_9_0_relaxed,
                                            fused_1nn_matrix_tile_f_cutile_arch_9_0_strict>,
                         std::conditional_t<is_relaxed,
                                            fused_1nn_matrix_tile_h_cutile_arch_9_0_relaxed,
                                            fused_1nn_matrix_tile_h_cutile_arch_9_0_strict>>;
    using Tile100 =
      std::conditional_t<is_float,
                         std::conditional_t<is_relaxed,
                                            fused_1nn_matrix_tile_f_cutile_arch_10_0_relaxed,
                                            fused_1nn_matrix_tile_f_cutile_arch_10_0_strict>,
                         std::conditional_t<is_relaxed,
                                            fused_1nn_matrix_tile_h_cutile_arch_10_0_relaxed,
                                            fused_1nn_matrix_tile_h_cutile_arch_10_0_strict>>;
    using Tile120 =
      std::conditional_t<is_float,
                         std::conditional_t<is_relaxed,
                                            fused_1nn_matrix_tile_f_cutile_arch_12_0_relaxed,
                                            fused_1nn_matrix_tile_f_cutile_arch_12_0_strict>,
                         std::conditional_t<is_relaxed,
                                            fused_1nn_matrix_tile_h_cutile_arch_12_0_relaxed,
                                            fused_1nn_matrix_tile_h_cutile_arch_12_0_strict>>;

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
};

/**
 * Planner for high-dimensional tiles. Only sm_90, sm_100, and sm_120 (FP32 only) have high-dim
 * tiles; other architectures yield try_get_launcher() == nullptr (no matching cubin).
 *
 * The tile aliases below are generated from the JSON entries with abi_abbrev="hd". sm_80 and sm_86
 * do not have high-dim tiles, so no fragment is registered for them.
 */
template <typename DataT>
struct Fused1nnTilePlanner<DataT, cutile_abi_high_dim>
  : cuvs::detail::jit_lto::TileAlgorithmPlanner {
  using DataTag  = std::conditional_t<std::is_same_v<DataT, float>,
                                      cuvs::neighbors::detail::tag_f,
                                      cuvs::neighbors::detail::tag_h>;
  using IndexTag = cuvs::neighbors::detail::tag_index_i32;

  inline static cuvs::detail::jit_lto::TileLauncherCache launcher_cache{};

  Fused1nnTilePlanner()
    : TileAlgorithmPlanner(fused_1nn_kernel_entrypoint<DataTag, cutile_abi_high_dim>(),
                           launcher_cache)
  {
  }

  void add_entrypoint()
  {
    using cuvs::detail::jit_lto::cutile_arch_10_0;
    using cuvs::detail::jit_lto::cutile_arch_12_0;
    using cuvs::detail::jit_lto::cutile_arch_9_0;

    constexpr bool is_float = std::is_same_v<DataTag, cuvs::neighbors::detail::tag_f>;

    // sm_90 (H100/H200): both FP32 and FP16 have high-dim tiles.
    if constexpr (is_float) {
      this->add_static_fragment<
        fragment_tag_fused_1nn_cubin<DataTag,
                                     IndexTag,
                                     fused_1nn_matrix_tile_f_cutile_arch_9_0_hd,
                                     cutile_abi_high_dim,
                                     cutile_arch_9_0>>();
    } else {
      this->add_static_fragment<
        fragment_tag_fused_1nn_cubin<DataTag,
                                     IndexTag,
                                     fused_1nn_matrix_tile_h_cutile_arch_9_0_hd,
                                     cutile_abi_high_dim,
                                     cutile_arch_9_0>>();
    }

    // sm_100 (B100/B200): both FP32 and FP16 have high-dim tiles.
    if constexpr (is_float) {
      this->add_static_fragment<
        fragment_tag_fused_1nn_cubin<DataTag,
                                     IndexTag,
                                     fused_1nn_matrix_tile_f_cutile_arch_10_0_hd,
                                     cutile_abi_high_dim,
                                     cutile_arch_10_0>>();
    } else {
      this->add_static_fragment<
        fragment_tag_fused_1nn_cubin<DataTag,
                                     IndexTag,
                                     fused_1nn_matrix_tile_h_cutile_arch_10_0_hd,
                                     cutile_abi_high_dim,
                                     cutile_arch_10_0>>();
    }

    // sm_120 (RTX Pro 6000): FP32 only; FP16 crossover is negligible.
    if constexpr (is_float) {
      this->add_static_fragment<
        fragment_tag_fused_1nn_cubin<DataTag,
                                     IndexTag,
                                     fused_1nn_matrix_tile_f_cutile_arch_12_0_hd,
                                     cutile_abi_high_dim,
                                     cutile_arch_12_0>>();
    }
  }
};

}  // namespace cuvs::distance::detail
