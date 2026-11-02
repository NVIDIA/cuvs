/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace cuvs::neighbors::detail {

struct tag_f {};
struct tag_h {};
struct tag_d {};
struct tag_i8 {};
struct tag_u8 {};
struct tag_filter_none {};
struct tag_filter_bitset {};
struct tag_filter_bloom_filter {};
struct tag_filter_udf {};

struct tag_bitset_u32 {};

struct tag_index_i32 {};
struct tag_index_u32 {};
struct tag_index_i64 {};

template <typename BitsetTag, typename IndexTag, typename FilterTag>
struct fragment_tag_sample_filter {};

}  // namespace cuvs::neighbors::detail
