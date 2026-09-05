/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace cuvs::distance::detail {

struct cutile_abi_strict {};
struct cutile_abi_relaxed {};

template <int TileM, int TileN, int TileK>
struct cutile_tile_config {
  static constexpr int tile_m = TileM;
  static constexpr int tile_n = TileN;
  static constexpr int tile_k = TileK;
};

template <typename DataTag, typename IndexTag, typename TileTag, typename AbiTag, typename ArchTag>
struct fragment_tag_fused_1nn_cubin {
  static constexpr int cc_major = ArchTag::cc_major;
  static constexpr int cc_minor = ArchTag::cc_minor;
};

template <typename DataTag, typename IndexTag, typename TileTag, typename AbiTag>
struct fragment_tag_fused_1nn_tileir {};

}  // namespace cuvs::distance::detail
