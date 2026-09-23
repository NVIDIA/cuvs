/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace cuvs::neighbors::cagra::experimental::flowann::detail {

template <typename DataTag,
          typename SourceIndexTag,
          typename IndexTag,
          typename DistanceTag,
          bool TopkByBitonicSort,
          bool BitonicSortAndMergeMultiWarps>
struct fragment_tag_search_single_cta {};

template <typename DataTag, typename SourceIndexTag, typename IndexTag, typename DistanceTag>
struct fragment_tag_search_multi_cta {};

}  // namespace cuvs::neighbors::cagra::experimental::flowann::detail
