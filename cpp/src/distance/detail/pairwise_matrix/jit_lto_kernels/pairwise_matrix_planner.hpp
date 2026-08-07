/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <AlgorithmPlanner.hpp>
#include <cuvs/detail/jit_lto/pairwise_matrix/pairwise_matrix_fragments.hpp>

namespace cuvs::distance::detail {

inline constexpr char kPairwiseMatrixJitEntrypoint[] = "pairwise_matrix_kernel";

template <typename DistanceTag_,
          typename DataTag_,
          typename AccTag_,
          typename OutTag_,
          typename IndexTag_,
          typename FinOpTag_,
          typename LayoutTag_,
          int Veclen_>
struct PairwiseMatrixPlanner : AlgorithmPlanner {
  using DistanceTag = DistanceTag_;
  using DataTag     = DataTag_;
  using AccTag      = AccTag_;
  using OutTag      = OutTag_;
  using IndexTag    = IndexTag_;
  using FinOpTag    = FinOpTag_;
  using LayoutTag   = LayoutTag_;

  static constexpr int Veclen = Veclen_;

  inline static LauncherJitCache launcher_jit_cache{};

  PairwiseMatrixPlanner() : AlgorithmPlanner(kPairwiseMatrixJitEntrypoint, launcher_jit_cache) {}

  void add_entrypoint()
  {
    this->add_static_fragment<fragment_tag_pairwise_matrix<DistanceTag,
                                                           DataTag,
                                                           AccTag,
                                                           OutTag,
                                                           IndexTag,
                                                           FinOpTag,
                                                           LayoutTag,
                                                           Veclen>>();
  }

  void add_compute_distance_function()
  {
    this->add_static_fragment<
      fragment_tag_compute_distance<DistanceTag, DataTag, AccTag, IndexTag>>();
  }

  void add_compute_distance_epilog_function()
  {
    this->add_static_fragment<fragment_tag_compute_distance_epilog<DistanceTag,
                                                                   DataTag,
                                                                   AccTag,
                                                                   IndexTag,
                                                                   LayoutTag,
                                                                   Veclen>>();
  }
};

}  // namespace cuvs::distance::detail
