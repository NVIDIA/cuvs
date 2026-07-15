/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cagra_planner_base.hpp"
#include <cuvs/detail/jit_lto/cagra/cagra_fragments.hpp>
#include <cuvs/distance/distance.hpp>

namespace cuvs::neighbors::cagra::detail::multi_cta_search {

template <typename DataTag,
          typename IndexTag,
          typename DistanceTag,
          typename SourceIndexTag,
          typename QueryTag,
          typename CodebookTag,
          typename SampleFilterJitTag>
struct CagraMultiCtaSearchPlanner
  : CagraPlannerBase<DataTag, IndexTag, DistanceTag, QueryTag, CodebookTag, SampleFilterJitTag> {
  static inline LauncherJitCache launcher_jit_cache{};

  CagraMultiCtaSearchPlanner(cuvs::distance::DistanceType /*metric*/,
                             uint32_t /*team_size*/,
                             uint32_t /*dataset_block_dim*/,
                             bool /*is_vpq*/,
                             uint32_t /*pq_bits*/,
                             uint32_t /*pq_len*/)
    : CagraPlannerBase<DataTag, IndexTag, DistanceTag, QueryTag, CodebookTag, SampleFilterJitTag>(
        "search_multi_cta", launcher_jit_cache)
  {
  }

  void add_search_multi_cta_kernel_fragment()
  {
    this->template add_static_fragment<
      fragment_tag_search_multi_cta<DataTag, SourceIndexTag, IndexTag, DistanceTag>>();
  }
};

}  // namespace cuvs::neighbors::cagra::detail::multi_cta_search
