/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "fused_topk_pk8_planner.hpp"

namespace cuvs::neighbors::detail {

template <typename DataT, typename AbiTag, typename OutputIdxT>
struct FusedTopkPk8LongscanPlanner : cuvs::detail::jit_lto::TileAlgorithmPlanner {
  using Base = FusedTopkPk8Planner<DataT, AbiTag, OutputIdxT>;
  inline static cuvs::detail::jit_lto::TileLauncherCache launcher_cache{};

  static const char* entrypoint()
  {
    constexpr bool relaxed = std::is_same_v<AbiTag, cuvs::distance::detail::cutile_abi_relaxed>;
    if constexpr (std::is_same_v<OutputIdxT, int64_t>) {
      if constexpr (std::is_same_v<DataT, half>) {
        return relaxed ? "fused_topk_pk8_h_i64_longscan_relaxed" : "fused_topk_pk8_h_i64_longscan";
      }
      return relaxed ? "fused_topk_pk8_f_i64_longscan_relaxed" : "fused_topk_pk8_f_i64_longscan";
    }
    if constexpr (std::is_same_v<DataT, half>) {
      return relaxed ? "fused_topk_pk8_h_i32_longscan_relaxed" : "fused_topk_pk8_h_i32_longscan";
    }
    return relaxed ? "fused_topk_pk8_f_i32_longscan_relaxed" : "fused_topk_pk8_f_i32_longscan";
  }

  FusedTopkPk8LongscanPlanner() : TileAlgorithmPlanner(entrypoint(), launcher_cache) {}

  void add_entrypoint()
  {
    this->add_static_fragment<
      fragment_tag_fused_topk_pk8_longscan_cubin<typename Base::DataTag,
                                                 typename Base::IndexTag,
                                                 AbiTag,
                                                 cuvs::detail::jit_lto::cutile_arch_12_0>>();
  }
};
}  // namespace cuvs::neighbors::detail
