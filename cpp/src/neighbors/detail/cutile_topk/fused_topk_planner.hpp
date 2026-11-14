/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_fp16.h>
#include <type_traits>

#include <cuvs/detail/jit_lto/TileAlgorithmPlanner.hpp>
#include <cuvs/detail/jit_lto/common_fragments.hpp>
#include <cuvs/detail/jit_lto/cutile_arch_tags.hpp>
#include <cuvs/detail/jit_lto/neighbors/fused_topk_fragments.hpp>

#include "fused_topk_pk2_cutile_tiles.hpp"

namespace cuvs::neighbors::detail {

template <typename DataT, typename AbiTag>
inline const char* fused_topk_pk2_kernel_entrypoint()
{
  constexpr bool is_relaxed = std::is_same_v<AbiTag, cuvs::distance::detail::cutile_abi_relaxed>;
  static_assert(is_relaxed || std::is_same_v<AbiTag, cuvs::distance::detail::cutile_abi_strict>,
                "unsupported fused physical-K2 cuTile ABI");
  static_assert(std::is_same_v<DataT, float> || std::is_same_v<DataT, half>);
  if constexpr (std::is_same_v<DataT, half>) {
    return is_relaxed ? "fused_topk_pk2_h_i32_relaxed" : "fused_topk_pk2_h_i32";
  }
  return is_relaxed ? "fused_topk_pk2_f_i32_relaxed" : "fused_topk_pk2_f_i32";
}

template <typename DataT, typename AbiTag>
struct FusedTopkPk2Planner : cuvs::detail::jit_lto::TileAlgorithmPlanner {
  static_assert(std::is_same_v<DataT, float> || std::is_same_v<DataT, half>);
  using DataTag  = std::conditional_t<std::is_same_v<DataT, half>,
                                      cuvs::neighbors::detail::tag_h,
                                      cuvs::neighbors::detail::tag_f>;
  using IndexTag = cuvs::neighbors::detail::tag_index_i32;

  inline static cuvs::detail::jit_lto::TileLauncherCache launcher_cache{};

  FusedTopkPk2Planner()
    : TileAlgorithmPlanner(fused_topk_pk2_kernel_entrypoint<DataT, AbiTag>(), launcher_cache)
  {
  }

  void add_entrypoint()
  {
    using cuvs::detail::jit_lto::cutile_arch_10_0;
    using cuvs::detail::jit_lto::cutile_arch_12_0;
    using cuvs::detail::jit_lto::cutile_arch_8_0;
    using cuvs::detail::jit_lto::cutile_arch_8_6;
    using cuvs::detail::jit_lto::cutile_arch_9_0;
    constexpr bool is_relaxed = std::is_same_v<AbiTag, cuvs::distance::detail::cutile_abi_relaxed>;
    if constexpr (std::is_same_v<DataT, half>) {
      using Tile120 = std::conditional_t<is_relaxed,
                                         fused_topk_pk2_matrix_tile_h_cutile_arch_12_0_relaxed,
                                         fused_topk_pk2_matrix_tile_h_cutile_arch_12_0_strict>;
      this->add_static_fragment<
        fragment_tag_fused_topk_cubin<DataTag, IndexTag, 2, Tile120, AbiTag, cutile_arch_12_0>>();
    } else {
      using Tile80  = std::conditional_t<is_relaxed,
                                         fused_topk_pk2_matrix_tile_f_cutile_arch_8_0_relaxed,
                                         fused_topk_pk2_matrix_tile_f_cutile_arch_8_0_strict>;
      using Tile86  = std::conditional_t<is_relaxed,
                                         fused_topk_pk2_matrix_tile_f_cutile_arch_8_6_relaxed,
                                         fused_topk_pk2_matrix_tile_f_cutile_arch_8_6_strict>;
      using Tile90  = std::conditional_t<is_relaxed,
                                         fused_topk_pk2_matrix_tile_f_cutile_arch_9_0_relaxed,
                                         fused_topk_pk2_matrix_tile_f_cutile_arch_9_0_strict>;
      using Tile100 = std::conditional_t<is_relaxed,
                                         fused_topk_pk2_matrix_tile_f_cutile_arch_10_0_relaxed,
                                         fused_topk_pk2_matrix_tile_f_cutile_arch_10_0_strict>;
      using Tile120 = std::conditional_t<is_relaxed,
                                         fused_topk_pk2_matrix_tile_f_cutile_arch_12_0_relaxed,
                                         fused_topk_pk2_matrix_tile_f_cutile_arch_12_0_strict>;
      using Abi     = AbiTag;
      this->add_static_fragment<
        fragment_tag_fused_topk_cubin<DataTag, IndexTag, 2, Tile80, Abi, cutile_arch_8_0>>();
      this->add_static_fragment<
        fragment_tag_fused_topk_cubin<DataTag, IndexTag, 2, Tile86, Abi, cutile_arch_8_6>>();
      this->add_static_fragment<
        fragment_tag_fused_topk_cubin<DataTag, IndexTag, 2, Tile90, Abi, cutile_arch_9_0>>();
      this->add_static_fragment<
        fragment_tag_fused_topk_cubin<DataTag, IndexTag, 2, Tile100, Abi, cutile_arch_10_0>>();
      this->add_static_fragment<
        fragment_tag_fused_topk_cubin<DataTag, IndexTag, 2, Tile120, Abi, cutile_arch_12_0>>();
    }
  }
};

}  // namespace cuvs::neighbors::detail
