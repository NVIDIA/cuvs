/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <neighbors/detail/cagra/compute_distance.hpp>

#include <cstdint>

namespace cuvs::neighbors::cagra::detail::single_cta_search {

template <typename DataT, typename IndexT, typename DistanceT>
struct alignas(16) multi_partition_desc_t {
  const dataset_descriptor_base_t<DataT, IndexT, DistanceT>* dataset_desc;
  const IndexT* graph;
  uint32_t graph_degree;
  uint32_t _pad;
  // This partition's own filter bitset. For now a VIEW into the caller's combined bitset (a device
  // word pointer at this partition's word-aligned slice start); the kernel tests it at
  // partition-local indices. Null ptr = no filter.
  std::uint32_t* bitset_ptr;
  std::int64_t bitset_len;  // number of bits in this partition's slice (metadata; test() ignores)
  std::int64_t original_nbits;  // original bit count; 0 = standard 32-bit packing
};

}  // namespace cuvs::neighbors::cagra::detail::single_cta_search

namespace cuvs::neighbors::cagra::detail::multi_cta_search {

template <typename DataT, typename IndexT, typename DistanceT>
struct alignas(16) multi_partition_desc_t {
  const dataset_descriptor_base_t<DataT, IndexT, DistanceT>* dataset_desc;
  const IndexT* graph;
  uint32_t graph_degree;
  uint32_t _pad;
  // This partition's own filter bitset. For now a VIEW into the caller's combined bitset (a device
  // word pointer at this partition's word-aligned slice start); the kernel tests it at
  // partition-local indices. Null ptr = no filter.
  std::uint32_t* bitset_ptr;
  std::int64_t bitset_len;  // number of bits in this partition's slice (metadata; test() ignores)
  std::int64_t original_nbits;  // original bit count; 0 = standard 32-bit packing
};

}  // namespace cuvs::neighbors::cagra::detail::multi_cta_search
