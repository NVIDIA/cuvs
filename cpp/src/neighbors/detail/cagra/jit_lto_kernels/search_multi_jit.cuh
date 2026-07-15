/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../neighbors_device_intrinsics.cuh"
#include "../hashmap.hpp"
#include "../utils.hpp"
#include "cagra_filter_payload.cuh"

#include <cstdint>
#include <cuda_fp16.h>

#include "extern_device_functions.cuh"

namespace cuvs::neighbors::cagra::detail::multi_kernel_search {

template <typename DataT, typename IndexT, typename DistanceT>
__device__ void random_pickup_kernel_jit(
  const dataset_descriptor_base_t<DataT, IndexT, DistanceT>* dataset_desc,
  const DataT* const queries_ptr,  // [num_queries, dataset_dim]
  const std::size_t num_pickup,
  const unsigned num_distilation,
  const uint64_t rand_xor_mask,
  const IndexT* seed_ptr,  // [num_queries, num_seeds]
  const uint32_t num_seeds,
  IndexT* const result_indices_ptr,       // [num_queries, ldr]
  DistanceT* const result_distances_ptr,  // [num_queries, ldr]
  const std::uint32_t ldr,                // (*) ldr >= num_pickup
  IndexT* const visited_hashmap_ptr,      // [num_queries, 1 << bitlen]
  const std::uint32_t hash_bitlen,
  const IndexT graph_size)
{
  using DATA_T     = DataT;
  using INDEX_T    = IndexT;
  using DISTANCE_T = DistanceT;

  // Get team_size_bits directly from base descriptor
  uint32_t team_size_bits = dataset_desc->team_size_bitshift();

  const auto ldb               = hashmap::get_size(hash_bitlen);
  const auto global_team_index = (blockIdx.x * blockDim.x + threadIdx.x) >> team_size_bits;
  const uint32_t query_id      = blockIdx.y;
  if (global_team_index >= num_pickup) { return; }
  extern __shared__ uint8_t smem[];

  auto smem_desc =
    setup_workspace<DataT, IndexT, DistanceT>(dataset_desc, smem, queries_ptr, query_id);
  __syncthreads();

  IndexT dataset_size            = smem_desc->size;
  const INDEX_T seed_index_limit = graph_size > 0 ? graph_size : dataset_size;
  const auto args_load           = smem_desc->args.load();

  INDEX_T best_index_team_local;
  DISTANCE_T best_norm2_team_local = utils::get_max_value<DISTANCE_T>();
  for (unsigned i = 0; i < num_distilation; i++) {
    INDEX_T seed_index;
    if (seed_ptr && (global_team_index < num_seeds)) {
      seed_index = seed_ptr[global_team_index + (num_seeds * query_id)];
    } else {
      seed_index =
        device::xorshift64((global_team_index ^ rand_xor_mask) * (i + 1)) % seed_index_limit;
    }

    const auto norm2 =
      compute_distance<DataT, IndexT, DistanceT>(args_load, seed_index, true, team_size_bits);

    if (norm2 < best_norm2_team_local) {
      best_norm2_team_local = norm2;
      best_index_team_local = seed_index;
    }
  }

  const auto store_gmem_index = global_team_index + (ldr * query_id);
  if ((threadIdx.x & ((1u << team_size_bits) - 1u)) == 0) {
    if (hashmap::insert(
          visited_hashmap_ptr + (ldb * query_id), hash_bitlen, best_index_team_local)) {
      result_distances_ptr[store_gmem_index] = best_norm2_team_local;
      result_indices_ptr[store_gmem_index]   = best_index_team_local;
    } else {
      result_distances_ptr[store_gmem_index] = utils::get_max_value<DISTANCE_T>();
      result_indices_ptr[store_gmem_index]   = utils::get_max_value<INDEX_T>();
    }
  }
}

using cuvs::neighbors::detail::sample_filter;
template <typename DataT, typename IndexT, typename DistanceT, typename SourceIndexT>
__device__ void compute_distance_to_child_nodes_kernel_jit(
  const IndexT* const parent_node_list,  // [num_queries, search_width]
  IndexT* const parent_candidates_ptr,   // [num_queries, search_width]
  DistanceT* const parent_distance_ptr,  // [num_queries, search_width]
  const std::size_t lds,
  const std::uint32_t search_width,
  const dataset_descriptor_base_t<DataT, IndexT, DistanceT>* dataset_desc,
  const IndexT* const neighbor_graph_ptr,  // [dataset_size, graph_degree]
  const std::uint32_t graph_degree,
  const SourceIndexT* source_indices_ptr,
  const DataT* query_ptr,             // [num_queries, data_dim]
  IndexT* const visited_hashmap_ptr,  // [num_queries, 1 << hash_bitlen]
  const std::uint32_t hash_bitlen,
  IndexT* const result_indices_ptr,       // [num_queries, ldd]
  DistanceT* const result_distances_ptr,  // [num_queries, ldd]
  const std::uint32_t ldd,                // (*) ldd >= search_width * graph_degree
  cagra_sample_filter<SourceIndexT> filter_payload)
{
  using INDEX_T    = IndexT;
  using DISTANCE_T = DistanceT;

  // Get team_size_bits directly from base descriptor
  uint32_t team_size_bits = dataset_desc->team_size_bitshift();

  const auto team_size       = 1u << team_size_bits;
  const uint32_t ldb         = hashmap::get_size(hash_bitlen);
  const auto tid             = threadIdx.x + blockDim.x * blockIdx.x;
  const auto global_team_id  = tid >> team_size_bits;
  const auto local_query_id  = blockIdx.y;
  const auto filter_query_id = filter_payload.query_id_offset + local_query_id;

  extern __shared__ uint8_t smem[];
  auto smem_desc =
    setup_workspace<DataT, IndexT, DistanceT>(dataset_desc, smem, query_ptr, local_query_id);

  __syncthreads();
  if (global_team_id >= search_width * graph_degree) { return; }

  const std::size_t parent_list_index =
    parent_node_list[global_team_id / graph_degree + (search_width * blockIdx.y)];

  if (parent_list_index == utils::get_max_value<INDEX_T>()) { return; }

  constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;
  const auto raw_parent_index = parent_candidates_ptr[parent_list_index + (lds * local_query_id)];

  if (raw_parent_index == utils::get_max_value<INDEX_T>()) {
    result_distances_ptr[ldd * local_query_id + global_team_id] =
      utils::get_max_value<DISTANCE_T>();
    return;
  }
  const auto parent_index = raw_parent_index & ~index_msb_1_mask;

  const auto neighbor_list_head_ptr = neighbor_graph_ptr + (graph_degree * parent_index);

  const std::size_t child_id = neighbor_list_head_ptr[global_team_id % graph_degree];

  const auto compute_distance_flag = hashmap::insert<INDEX_T>(
    team_size, visited_hashmap_ptr + (ldb * blockIdx.y), hash_bitlen, child_id);

  const auto args  = smem_desc->args.load();
  DISTANCE_T norm2 = compute_distance<DataT, IndexT, DistanceT>(
    args, static_cast<INDEX_T>(child_id), compute_distance_flag, team_size_bits);

  if (compute_distance_flag) {
    if ((threadIdx.x & (team_size - 1)) == 0) {
      result_indices_ptr[ldd * local_query_id + global_team_id]   = child_id;
      result_distances_ptr[ldd * local_query_id + global_team_id] = norm2;
    }
  } else {
    if ((threadIdx.x & (team_size - 1)) == 0) {
      result_distances_ptr[ldd * local_query_id + global_team_id] =
        utils::get_max_value<DISTANCE_T>();
    }
  }

  const SourceIndexT node_id = source_indices_ptr == nullptr
                                 ? static_cast<SourceIndexT>(parent_index)
                                 : static_cast<SourceIndexT>(source_indices_ptr[parent_index]);
  if (!sample_filter<SourceIndexT>(filter_query_id, node_id, filter_payload.sample_filter_data())) {
    parent_candidates_ptr[parent_list_index + (lds * local_query_id)] =
      utils::get_max_value<INDEX_T>();
    parent_distance_ptr[parent_list_index + (lds * local_query_id)] =
      utils::get_max_value<DISTANCE_T>();
  }
}

template <typename IndexT,
          typename DistanceT,
          typename SourceIndexT>
__device__ void apply_filter_kernel_jit(
  const SourceIndexT* source_indices_ptr,  // [num_queries, search_width]
  IndexT* const result_indices_ptr,
  DistanceT* const result_distances_ptr,
  const std::size_t lds,
  const std::uint32_t result_buffer_size,
  const std::uint32_t num_queries,
  const std::uint32_t query_id_offset,
  cagra_sample_filter<SourceIndexT> filter_payload)
{
  constexpr IndexT index_msb_1_mask = utils::gen_index_msb_1_mask<IndexT>::value;
  const auto tid                    = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= result_buffer_size * num_queries) { return; }
  const auto i     = tid % result_buffer_size;
  const auto j     = tid / result_buffer_size;
  const auto index = i + j * lds;

  if (result_indices_ptr[index] != ~index_msb_1_mask) {
    // Use extern sample_filter function with 3 params: query_id, node_id, filter_data.
    // The payload maps built-in bitset filters and UDF context pointers to the linked ABI.
    SourceIndexT node_id = source_indices_ptr == nullptr
                             ? static_cast<SourceIndexT>(result_indices_ptr[index])
                             : source_indices_ptr[result_indices_ptr[index]];

    if (!sample_filter<SourceIndexT>(
          query_id_offset + j, node_id, filter_payload.sample_filter_data())) {
      result_indices_ptr[index]   = utils::get_max_value<IndexT>();
      result_distances_ptr[index] = utils::get_max_value<DistanceT>();
    }
  }
}

}  // namespace cuvs::neighbors::cagra::detail::multi_kernel_search
