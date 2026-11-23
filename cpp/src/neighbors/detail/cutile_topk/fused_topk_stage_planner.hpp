/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda_fp16.h>
#include <cuvs/detail/jit_lto/TileAlgorithmPlanner.hpp>
#include <cuvs/detail/jit_lto/common_fragments.hpp>
#include <cuvs/detail/jit_lto/cutile_arch_tags.hpp>
#include <cuvs/detail/jit_lto/neighbors/fused_topk_fragments.hpp>
#include <string>
#include <type_traits>

namespace cuvs::neighbors::detail {

/** Each stage has an independent symbol, fragment type and device-aware cache. */
template <int PhysicalK, typename DataT, fused_topk_stage Stage>
struct FusedTopkStagePlanner : cuvs::detail::jit_lto::TileAlgorithmPlanner {
  static_assert(PhysicalK == 2 || PhysicalK == 4 || PhysicalK == 8);
  static_assert(std::is_same_v<DataT, float> || std::is_same_v<DataT, half>);
  static_assert(Stage != fused_topk_stage::witness ||
                (PhysicalK == 4 && std::is_same_v<DataT, half>));
  static_assert(Stage != fused_topk_stage::select_l2 || PhysicalK == 8);
  static constexpr bool largeq      = Stage == fused_topk_stage::select_largeq;
  static constexpr bool largecorpus = Stage == fused_topk_stage::select_largecorpus;
  static_assert(!(largeq || largecorpus) || PhysicalK == 8);
  static constexpr int tile_m = largeq           ? 64
                                : PhysicalK == 8 ? (Stage == fused_topk_stage::refine ? 4 : 32)
                                                 : (Stage == fused_topk_stage::refine ? 32 : 64);
  static constexpr int tile_n = largecorpus ? 256 : 128;
  static constexpr int tile_k = 32;
  using TileTag               = cutile_tile_config<tile_m, tile_n, tile_k>;
  using DataTag               = std::conditional_t<std::is_same_v<DataT, half>, tag_h, tag_f>;
  inline static cuvs::detail::jit_lto::TileLauncherCache launcher_cache{};

  static std::string entrypoint()
  {
    const char* suffix = (largeq || largecorpus)                ? "longscan_select"
                         : Stage == fused_topk_stage::select    ? "select"
                         : Stage == fused_topk_stage::refine    ? "refine"
                         : Stage == fused_topk_stage::select_l2 ? "select_l2"
                                                                : "witness";
    return "fused_topk_pk" + std::to_string(PhysicalK) +
           (std::is_same_v<DataT, half> ? "_h_i32_" : "_f_i32_") + suffix;
  }

  FusedTopkStagePlanner() : TileAlgorithmPlanner(entrypoint(), launcher_cache)
  {
    this->add_static_fragment<
      fragment_tag_fused_topk_stage_cubin<DataTag,
                                          tag_index_i32,
                                          PhysicalK,
                                          TileTag,
                                          cuvs::distance::detail::cutile_abi_strict,
                                          cuvs::detail::jit_lto::cutile_arch_12_0,
                                          Stage>>();
  }
};

}  // namespace cuvs::neighbors::detail
