/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "search_single_cta_jit.cuh"

#include <neighbors/detail/cagra/jit_lto_kernels/search_multi_cta_jit.cuh>

namespace cuvs::neighbors::cagra::experimental::flowann::detail {

template <typename DataT, typename IndexT, typename DistanceT, typename SourceIndexT>
__device__ void search_multi_cta_kernel_jit(
  IndexT* result_indices,
  DistanceT* result_distances,
  const cuvs::neighbors::cagra::detail::dataset_descriptor_base_t<DataT, IndexT, DistanceT>*
    dataset_desc,
  const DataT* queries,
  const std::uint8_t* inner_graph,
  std::uint32_t inner_row_length,
  std::uint32_t node_per_cacheline,
  std::uint32_t n_bits,
  const std::uint32_t* subgraph_offsets,
  const std::uint32_t* subgraph_ids,
  queue_view* queues,
  std::uint32_t num_queues,
  std::uint32_t max_elements,
  std::uint32_t graph_degree,
  std::uint32_t cross_graph_degree,
  const SourceIndexT* source_indices,
  unsigned num_distillation,
  std::uint64_t rand_xor_mask,
  const IndexT* seeds,
  std::uint32_t num_seeds,
  const std::uint32_t* seed_cross_graph,
  const IndexT* seed_lookup_keys,
  const std::uint32_t* seed_lookup_values,
  std::uint32_t seed_lookup_capacity,
  std::uint32_t visited_hash_bitlen,
  IndexT* traversed_hashmap,
  std::uint32_t traversed_hash_bitlen,
  std::uint32_t itopk_size,
  std::uint32_t min_iterations,
  std::uint32_t max_iterations,
  std::uint32_t* num_executed_iterations,
  IndexT graph_size,
  std::uint32_t query_id_offset,
  float sync_window_scale,
  std::uint32_t sync_drop_threshold,
  cuvs::neighbors::cagra::detail::cagra_sample_filter<SourceIndexT> filter_payload)
{
  __shared__ graph_state<IndexT> state;
  auto const query_id = blockIdx.y;
  auto const queue_id = query_id % num_queues;
  tiered_graph_provider<DataT, IndexT, DistanceT> provider{inner_graph,
                                                           inner_row_length,
                                                           node_per_cacheline,
                                                           n_bits,
                                                           subgraph_offsets,
                                                           subgraph_ids,
                                                           seeds,
                                                           seed_cross_graph,
                                                           seed_lookup_keys,
                                                           seed_lookup_values,
                                                           seed_lookup_capacity,
                                                           num_seeds,
                                                           queues[queue_id],
                                                           graph_degree,
                                                           cross_graph_degree,
                                                           graph_size,
                                                           sync_window_scale,
                                                           sync_drop_threshold,
                                                           &state};
  cuvs::neighbors::cagra::detail::multi_cta_search::
    search_core<DataT, IndexT, DistanceT, SourceIndexT>(
      result_indices,
      result_distances,
      dataset_desc,
      queries,
      provider,
      max_elements,
      graph_degree,
      (1 + additional_buffer_width) * graph_degree,
      source_indices,
      num_distillation,
      rand_xor_mask,
      seeds,
      num_seeds,
      visited_hash_bitlen,
      traversed_hashmap,
      traversed_hash_bitlen,
      itopk_size,
      min_iterations,
      max_iterations,
      num_executed_iterations,
      graph_size,
      query_id_offset,
      filter_payload);
}

}  // namespace cuvs::neighbors::cagra::experimental::flowann::detail
