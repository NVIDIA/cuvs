/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "fused_topk_pk16_planner.hpp"
#include <string>

namespace cuvs::neighbors::detail {
template <typename DataT, typename AbiTag, int Stage>
struct FusedTopkPk16StagePlanner : cuvs::detail::jit_lto::TileAlgorithmPlanner {
  static_assert(Stage == 0 || Stage == 1 || Stage == 2 || Stage == 3);
  static_assert(Stage != 2 || ((std::is_same_v<DataT, half> || std::is_same_v<DataT, float>) &&
                               std::is_same_v<AbiTag, cuvs::distance::detail::cutile_abi_strict>));
  using DataTag = std::conditional_t<std::is_same_v<DataT, half>, tag_h, tag_f>;
  static_assert(Stage != 3 || (std::is_same_v<DataT, float> &&
                               std::is_same_v<AbiTag, cuvs::distance::detail::cutile_abi_strict>));
  static constexpr bool is_refinement = Stage == 1 || Stage == 3;
  // Keep this tag consistent with fused_topk_pk16_stages_matrix.json.
  // This is a dtype/layout configuration, not a workload-size dispatch rule.
  static constexpr int selector_tile_m =
    std::is_same_v<DataT, half> && std::is_same_v<AbiTag, cuvs::distance::detail::cutile_abi_strict>
      ? 64
      : 32;
  static constexpr int selector_tile_k =
    std::is_same_v<DataT, half> && std::is_same_v<AbiTag, cuvs::distance::detail::cutile_abi_strict>
      ? 64
      : 32;
  inline static cuvs::detail::jit_lto::TileLauncherCache launcher_cache{};
  FusedTopkPk16StagePlanner()
    : TileAlgorithmPlanner(std::string(fused_topk_pk16_kernel_entrypoint<DataT, AbiTag>()) +
                             (Stage == 0   ? "_select"
                              : Stage == 1 ? "_refine"
                              : Stage == 2 ? "_select_witness"
                                           : "_refine_sortnet"),
                           launcher_cache)
  {
  }
  void add_entrypoint()
  {
    this->add_static_fragment<fragment_tag_fused_topk_pk16_stage_cubin<
      DataTag,
      tag_index_i32,
      Stage,
      cutile_tile_config<is_refinement ? 4 : selector_tile_m,
                         128,
                         is_refinement ? 32 : selector_tile_k>,
      AbiTag,
      cuvs::detail::jit_lto::cutile_arch_12_0>>();
  }
};
}  // namespace cuvs::neighbors::detail
