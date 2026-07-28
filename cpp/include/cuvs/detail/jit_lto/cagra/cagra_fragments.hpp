/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace cuvs::neighbors::cagra::detail {

struct tag_dist_f {};
struct tag_metric_l2 {};
struct tag_metric_inner_product {};
struct tag_metric_cosine {};
struct tag_metric_hamming {};
struct tag_codebook_none {};
struct tag_codebook_half {};
struct tag_smem_f16 {};
struct tag_smem_e5m2 {};
struct tag_metric_l1 {};
struct tag_norm_noop {};
struct tag_norm_cosine {};

/// Multi-kernel planners that do not link `sample_filter` into the JIT link (e.g.
/// `random_pickup`). Real filters use `cuvs::neighbors::detail::tag_filter_*` on
/// `CagraPlannerBase`.
struct tag_cagra_jit_sample_filter_link_absent {};

template <typename DataTag,
          typename IndexTag,
          typename DistanceTag,
          typename QueryTag,
          typename CodebookTag,
          uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PqBits,
          uint32_t PqLen,
          typename SmemTag>
struct fragment_tag_setup_workspace {};

template <typename DataTag,
          typename IndexTag,
          typename DistanceTag,
          typename QueryTag,
          typename CodebookTag,
          uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PqBits,
          uint32_t PqLen,
          typename SmemTag>
struct fragment_tag_compute_distance {};

template <typename QueryTag, typename DistanceTag, typename MetricTag>
struct fragment_tag_dist_op {};

template <typename DataTag,
          typename IndexTag,
          typename DistanceTag,
          typename QueryTag,
          uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          typename NormTag>
struct fragment_tag_apply_normalization_standard {};

template <typename DataTag,
          typename SourceIndexTag,
          typename IndexTag,
          typename DistanceTag,
          bool TopkByBitonicSort,
          bool BitonicSortAndMergeMultiWarps>
struct fragment_tag_search_single_cta {};

template <typename DataTag,
          typename SourceIndexTag,
          typename IndexTag,
          typename DistanceTag,
          bool TopkByBitonicSort,
          bool BitonicSortAndMergeMultiWarps>
struct fragment_tag_search_single_cta_p {};

template <typename DataTag, typename SourceIndexTag, typename IndexTag, typename DistanceTag>
struct fragment_tag_search_multi_cta {};

template <typename DataTag, typename IndexTag, typename DistanceTag>
struct fragment_tag_random_pickup {};

template <typename DataTag, typename IndexTag, typename DistanceTag, typename SourceIndexTag>
struct fragment_tag_compute_distance_to_child_nodes {};

template <typename IndexTag, typename DistanceTag, typename SourceIndexTag>
struct fragment_tag_apply_filter_kernel {};

template <typename BitsetTag, typename SourceIndexTag, typename FilterTag>
struct fragment_tag_sample_filter {};

}  // namespace cuvs::neighbors::cagra::detail
