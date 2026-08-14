/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace cuvs::neighbors::ivf_flat::detail {

// Tag types for accumulator types
struct tag_acc_f {};
struct tag_acc_h {};
struct tag_acc_i32 {};
struct tag_acc_u32 {};

// Tag types for distance metrics
struct tag_metric_euclidean {};
struct tag_metric_inner_product {};
struct tag_metric_custom_udf {};

// Tag types for post-processing
struct tag_post_process_identity {};
struct tag_post_process_sqrt {};
struct tag_post_process_compose {};

template <typename DataTag, typename AccTag, typename IdxTag, int Capacity, bool Ascending>
struct fragment_tag_interleaved_scan {};

template <typename DataTag, typename AccTag, bool ComputeNorm, int Veclen>
struct fragment_tag_load_and_compute_dist {};

template <typename DataTag, typename AccTag, typename MetricTag, int Veclen>
struct fragment_tag_metric {};

template <typename IndexTag, typename FilterTag>
struct fragment_tag_filter {};

template <typename PostLambdaTag>
struct fragment_tag_post_lambda {};

}  // namespace cuvs::neighbors::ivf_flat::detail
