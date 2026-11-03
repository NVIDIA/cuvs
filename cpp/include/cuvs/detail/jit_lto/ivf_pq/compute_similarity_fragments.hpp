/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace cuvs::neighbors::ivf_pq::detail {

struct tag_out_f {};
struct tag_out_h {};

struct tag_lut_f {};
struct tag_lut_h {};
struct tag_lut_fp8_signed {};
struct tag_lut_fp8_unsigned {};

struct tag_metric_none {};
struct tag_metric_euclidean {};
struct tag_metric_inner_product {};

template <typename OutTag, typename LutTag>
struct fragment_tag_compute_similarity {};

template <typename LutTag, bool EnableSMemLut, uint32_t PqBits>
struct fragment_tag_prepare_lut {};

template <typename OutTag, bool kManageLocalTopK>
struct fragment_tag_store_calculated_distances {};

template <typename MetricTag>
struct fragment_tag_precompute_base_diff {};

template <typename LutTag, typename MetricTag, bool PrecompBaseDiff, uint32_t PqBits>
struct fragment_tag_create_lut {};

template <typename OutTag, typename LutTag, int Capacity>
struct fragment_tag_compute_distances {};

template <typename OutTag, typename MetricTag>
struct fragment_tag_get_early_stop_limit {};

template <typename FilterTag>
struct fragment_tag_sample_filter {};

template <uint32_t PqBits>
struct fragment_tag_get_line_width {};

template <typename OutTag, typename LutTag, uint32_t PqBits>
struct fragment_tag_compute_score {};

template <typename OutTag, bool Increment>
struct fragment_tag_increment_score {};

}  // namespace cuvs::neighbors::ivf_pq::detail
