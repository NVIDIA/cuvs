/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/detail/jit_lto/flowann/flowann_fragments.hpp>

#include <neighbors/detail/cagra/jit_lto_kernels/cagra_planner_base.hpp>

namespace cuvs::neighbors::cagra::experimental::flowann::detail {

template <typename DataTag,
          typename IndexTag,
          typename DistanceTag,
          typename SourceIndexTag,
          typename QueryTag,
          typename CodebookTag,
          typename SampleFilterJitTag>
struct single_cta_planner
  : cuvs::neighbors::cagra::detail::
      CagraPlannerBase<DataTag, IndexTag, DistanceTag, QueryTag, CodebookTag, SampleFilterJitTag> {
  using base_type = cuvs::neighbors::cagra::detail::
    CagraPlannerBase<DataTag, IndexTag, DistanceTag, QueryTag, CodebookTag, SampleFilterJitTag>;
  static inline rtcx::launcher_jit_cache launcher_cache{};

  single_cta_planner() : base_type("flowann_search_single_cta", launcher_cache) {}

  void add_search_kernel(bool topk_by_bitonic_sort, bool bitonic_sort_and_merge_multi_warps)
  {
    if (topk_by_bitonic_sort && bitonic_sort_and_merge_multi_warps) {
      this->template add_static_fragment<fragment_tag_search_single_cta<DataTag,
                                                                        SourceIndexTag,
                                                                        IndexTag,
                                                                        DistanceTag,
                                                                        true,
                                                                        true>>();
    } else if (topk_by_bitonic_sort) {
      this->template add_static_fragment<fragment_tag_search_single_cta<DataTag,
                                                                        SourceIndexTag,
                                                                        IndexTag,
                                                                        DistanceTag,
                                                                        true,
                                                                        false>>();
    } else if (bitonic_sort_and_merge_multi_warps) {
      this->template add_static_fragment<fragment_tag_search_single_cta<DataTag,
                                                                        SourceIndexTag,
                                                                        IndexTag,
                                                                        DistanceTag,
                                                                        false,
                                                                        true>>();
    } else {
      this->template add_static_fragment<fragment_tag_search_single_cta<DataTag,
                                                                        SourceIndexTag,
                                                                        IndexTag,
                                                                        DistanceTag,
                                                                        false,
                                                                        false>>();
    }
  }
};

template <typename DataTag,
          typename IndexTag,
          typename DistanceTag,
          typename SourceIndexTag,
          typename QueryTag,
          typename CodebookTag,
          typename SampleFilterJitTag>
struct multi_cta_planner
  : cuvs::neighbors::cagra::detail::
      CagraPlannerBase<DataTag, IndexTag, DistanceTag, QueryTag, CodebookTag, SampleFilterJitTag> {
  using base_type = cuvs::neighbors::cagra::detail::
    CagraPlannerBase<DataTag, IndexTag, DistanceTag, QueryTag, CodebookTag, SampleFilterJitTag>;
  static inline rtcx::launcher_jit_cache launcher_cache{};

  multi_cta_planner() : base_type("flowann_search_multi_cta", launcher_cache) {}

  void add_search_kernel()
  {
    this->template add_static_fragment<
      fragment_tag_search_multi_cta<DataTag, SourceIndexTag, IndexTag, DistanceTag>>();
  }
};

}  // namespace cuvs::neighbors::cagra::experimental::flowann::detail
