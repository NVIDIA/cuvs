/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuco/roaring_bitmap_ref.cuh>

#include <cstdint>

namespace cuvs::neighbors::detail {

/// Per-query cuco Roaring bitmap references for linked @c sample_filter in CAGRA JIT LTO.
template <typename Key = std::uint32_t>
struct roaring_filter_data_t {
  using ref_type = cuco::experimental::roaring_bitmap_ref<Key>;

  // Each entry points at a reference owned and initialized by one roaring_allowlist. The pointer
  // table is built by roaring_filter; CAGRA only follows it.
  ref_type const* const* refs{nullptr};
  std::uint8_t const* empty_rows{nullptr};
  std::uint32_t num_queries{};
  std::uint64_t dataset_rows{};
};

}  // namespace cuvs::neighbors::detail
