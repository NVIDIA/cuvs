/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuco/bloom_filter.cuh>

#include <cstddef>
#include <cstdint>

namespace cuvs::neighbors::detail {

/// Bitset (and length metadata) for linked @c sample_filter in JIT LTO; passed by value to
/// @c __global__ entry points. `bitset_ptr == nullptr` means no bitset (none filter, or not a
/// bitset at runtime on host). Also passed as @c void* to the unified JIT @c sample_filter (see
/// @c sample_filter.cuh / @c bitset_filter).
template <typename SourceIndexT>
struct bitset_filter_data_t {
  std::uint32_t* bitset_ptr{nullptr};
  SourceIndexT bitset_len{};
  SourceIndexT original_nbits{};
};

/// Global cuco bloom filter ref for linked @c sample_filter in CAGRA JIT LTO.
template <typename Key = std::uint32_t>
struct bloom_filter_data_t {
  using ref_type = typename cuco::bloom_filter<Key>::ref_type<>;

  explicit bloom_filter_data_t(ref_type filter) : filter(filter) {}

  ref_type filter;
};

}  // namespace cuvs::neighbors::detail
