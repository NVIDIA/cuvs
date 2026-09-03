/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "extern_device_functions.cuh"

#include "../../roaring_filter_data.cuh"
#include "../../sample_filter_data.cuh"

#include <cuco/bloom_filter_ref.cuh>

#include <raft/core/bitset.cuh>

#include <cstdint>

namespace cuvs::neighbors::detail {

template <typename SourceIndexT>
__device__ bool sample_filter_none_impl(uint32_t /*query_id*/,
                                        SourceIndexT /*node_id*/,
                                        void* /*filter_data*/)
{
  return true;
}

template <typename SourceIndexT>
__device__ bool sample_filter_bitset_impl(uint32_t /*query_id*/,
                                          SourceIndexT node_id,
                                          void* filter_data)
{
  if (filter_data == nullptr) { return true; }

  auto* data = static_cast<bitset_filter_data_t<SourceIndexT>*>(filter_data);
  if (data->bitset_ptr == nullptr) { return true; }

  raft::core::bitset_view<uint32_t, SourceIndexT> const view{
    data->bitset_ptr, data->bitset_len, data->original_nbits};
  return view.test(node_id);
}

template <typename SourceIndexT, typename Key = std::uint32_t>
__device__ bool sample_filter_bloom_filter_impl(uint32_t /*query_id*/,
                                                SourceIndexT node_id,
                                                void* filter_data)
{
  if (filter_data == nullptr) { return true; }

  auto* data = static_cast<bloom_filter_data_t<Key>*>(filter_data);
  return data->filter.contains(static_cast<Key>(node_id));
}

template <typename SourceIndexT, typename Key = std::uint32_t>
__device__ bool sample_filter_roaring_impl(uint32_t query_id,
                                           SourceIndexT node_id,
                                           void* filter_data)
{
  if (filter_data == nullptr) { return false; }

  auto const* data = static_cast<roaring_filter_data_t<Key> const*>(filter_data);
  if (query_id >= data->num_queries || static_cast<std::uint64_t>(node_id) >= data->dataset_rows ||
      data->empty_rows[query_id] != 0) {
    return false;
  }
  return data->refs[query_id]->contains(static_cast<Key>(node_id));
}

}  // namespace cuvs::neighbors::detail
