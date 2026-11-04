/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/detail/jit_lto/fused_distance_nn/fused_1nn_fragments.hpp>

namespace cuvs::neighbors::detail {

using cuvs::distance::detail::cutile_tile_config;

template <typename DataTag,
          typename IndexTag,
          int PhysicalK,
          typename TileTag,
          typename AbiTag,
          typename ArchTag>
struct fragment_tag_fused_topk_cubin {
  static constexpr int cc_major   = ArchTag::cc_major;
  static constexpr int cc_minor   = ArchTag::cc_minor;
  static constexpr int physical_k = PhysicalK;
};

enum class fused_topk_stage { select, refine, witness };

template <typename DataTag,
          typename IndexTag,
          int PhysicalK,
          typename TileTag,
          typename AbiTag,
          typename ArchTag,
          fused_topk_stage Stage>
struct fragment_tag_fused_topk_stage_cubin
  : fragment_tag_fused_topk_cubin<DataTag, IndexTag, PhysicalK, TileTag, AbiTag, ArchTag> {};
// Distinct registration type: never alias the ordinary PK8 image or launcher cache.
template <typename DataTag, typename IndexTag, typename AbiTag, typename ArchTag>
struct fragment_tag_fused_topk_pk8_longscan_cubin
  : fragment_tag_fused_topk_cubin<DataTag,
                                  IndexTag,
                                  8,
                                  cutile_tile_config<32, 128, 32>,
                                  AbiTag,
                                  ArchTag> {};

}  // namespace cuvs::neighbors::detail
