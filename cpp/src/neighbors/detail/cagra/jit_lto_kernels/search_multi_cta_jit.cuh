/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../hashmap.hpp"
#include "../utils.hpp"

#include <neighbors/detail/cagra/search_multi_cta_helpers.cuh>

#include <raft/util/integer_utils.hpp>

#include <cstdint>
#include <cuda_fp16.h>
#include <type_traits>

#ifdef _CLK_BREAKDOWN
#include <cstdio>
#endif

#include "cagra_filter_payload.cuh"
#include "device_common_jit.cuh"
#include "extern_device_functions.cuh"

namespace cuvs::neighbors::cagra::detail::multi_cta_search {

using cuvs::neighbors::cagra::detail::device::compute_distance_to_child_nodes_jit;
using cuvs::neighbors::cagra::detail::device::compute_distance_to_random_nodes_jit;
using cuvs::neighbors::detail::sample_filter;
template <typename DataT, typename IndexT, typename DistanceT, typename SourceIndexT>
__device__ void search_kernel_jit(
  IndexT* const result_indices_ptr,       // [num_queries, num_cta_per_query, itopk_size]
  DistanceT* const result_distances_ptr,  // [num_queries, num_cta_per_query, itopk_size]
  const dataset_descriptor_base_t<DataT, IndexT, DistanceT>* dataset_desc,
  const DataT* const queries_ptr,  // [num_queries, dataset_dim]
  const IndexT* const knn_graph,   // [dataset_size, graph_degree]
  const uint32_t max_elements,
  const uint32_t graph_degree,
  const SourceIndexT* source_indices_ptr,  // [num_queries, search_width]
  const unsigned num_distilation,
  const uint64_t rand_xor_mask,
  const IndexT* seed_ptr,  // [num_queries, num_seeds]
  const uint32_t num_seeds,
  const uint32_t visited_hash_bitlen,
  IndexT* const traversed_hashmap_ptr,  // [num_queries, 1 << traversed_hash_bitlen]
  const uint32_t traversed_hash_bitlen,
  const uint32_t itopk_size,
  const uint32_t min_iteration,
  const uint32_t max_iteration,
  uint32_t* const num_executed_iterations, /* stats */
  const IndexT graph_size,
  const uint32_t query_id_offset,  // Offset to add to query_id when calling filter
  cagra_sample_filter<SourceIndexT> filter_payload)
{
  using DATA_T     = DataT;
  using INDEX_T    = IndexT;
  using DISTANCE_T = DistanceT;

  auto to_source_index = [source_indices_ptr](INDEX_T x) {
    return source_indices_ptr == nullptr ? static_cast<SourceIndexT>(x) : source_indices_ptr[x];
  };

  const auto num_queries       = gridDim.y;
  const auto query_id          = blockIdx.y;
  const auto num_cta_per_query = gridDim.x;
  const auto cta_id            = blockIdx.x;  // local CTA ID

#ifdef _CLK_BREAKDOWN
  uint64_t clk_init                 = 0;
  uint64_t clk_compute_1st_distance = 0;
  uint64_t clk_topk                 = 0;
  uint64_t clk_pickup_parents       = 0;
  uint64_t clk_compute_distance     = 0;
  uint64_t clk_start;
#define _CLK_START() clk_start = clock64()
#define _CLK_REC(V)  V += clock64() - clk_start;
#else
#define _CLK_START()
#define _CLK_REC(V)
#endif
  _CLK_START();

  extern __shared__ uint8_t smem[];

  // Layout of result_buffer
  // +----------------+---------+---------------------------+
  // | internal_top_k | padding | neighbors of parent nodes |
  // | <itopk_size>   | upto 32 | <graph_degree>            |
  // +----------------+---------+---------------------------+
  // |<---        result_buffer_size_32                 --->|
  const auto result_buffer_size    = itopk_size + graph_degree;
  const auto result_buffer_size_32 = raft::round_up_safe<uint32_t>(result_buffer_size, 32);
  assert(result_buffer_size_32 <= max_elements);

  // Get dim and smem_ws_size_in_bytes directly from base descriptor
  uint32_t dim                   = dataset_desc->args.dim;
  uint32_t smem_ws_size_in_bytes = dataset_desc->smem_ws_size_in_bytes();

  auto smem_desc =
    setup_workspace<DataT, IndexT, DistanceT>(dataset_desc, smem, queries_ptr, query_id);

  auto* __restrict__ result_indices_buffer =
    reinterpret_cast<INDEX_T*>(smem + smem_ws_size_in_bytes);
  auto* __restrict__ result_distances_buffer =
    reinterpret_cast<DISTANCE_T*>(result_indices_buffer + result_buffer_size_32);
  auto* __restrict__ local_visited_hashmap_ptr =
    reinterpret_cast<INDEX_T*>(result_distances_buffer + result_buffer_size_32);
  auto* __restrict__ parent_indices_buffer =
    reinterpret_cast<INDEX_T*>(local_visited_hashmap_ptr + hashmap::get_size(visited_hash_bitlen));
  auto* __restrict__ result_position = reinterpret_cast<int*>(parent_indices_buffer + 1);

  INDEX_T* const local_traversed_hashmap_ptr =
    traversed_hashmap_ptr + (hashmap::get_size(traversed_hash_bitlen) * query_id);

  constexpr INDEX_T invalid_index    = ~static_cast<INDEX_T>(0);
  constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;

  for (unsigned i = threadIdx.x; i < result_buffer_size_32; i += blockDim.x) {
    result_indices_buffer[i]   = invalid_index;
    result_distances_buffer[i] = utils::get_max_value<DISTANCE_T>();
  }
  hashmap::init<INDEX_T>(local_visited_hashmap_ptr, visited_hash_bitlen);
  __syncthreads();
  _CLK_REC(clk_init);

  // compute distance to randomly selecting nodes using JIT version
  _CLK_START();
  const INDEX_T* const local_seed_ptr = seed_ptr ? seed_ptr + (num_seeds * query_id) : nullptr;
  uint32_t block_id                   = cta_id + (num_cta_per_query * query_id);
  uint32_t num_blocks                 = num_cta_per_query * num_queries;

  compute_distance_to_random_nodes_jit<IndexT, DistanceT, DataT>(result_indices_buffer,
                                                                 result_distances_buffer,
                                                                 smem_desc,
                                                                 graph_degree,
                                                                 num_distilation,
                                                                 rand_xor_mask,
                                                                 local_seed_ptr,
                                                                 num_seeds,
                                                                 local_visited_hashmap_ptr,
                                                                 visited_hash_bitlen,
                                                                 local_traversed_hashmap_ptr,
                                                                 traversed_hash_bitlen,
                                                                 block_id,
                                                                 num_blocks,
                                                                 graph_size);
  __syncthreads();
  _CLK_REC(clk_compute_1st_distance);

  uint32_t iter = 0;
  while (1) {
    _CLK_START();
    if (threadIdx.x < 32) {
      // [1st warp] Topk with bitonic sort
      if constexpr (std::is_same_v<INDEX_T, uint32_t>) {
        // use a non-template wrapper function to avoid pre-inlining the topk_by_bitonic_sort
        // function (vs post-inlining, this impacts register pressure)
        if (max_elements <= 64) {
          topk_by_bitonic_sort_wrapper_64(
            result_distances_buffer, result_indices_buffer, result_buffer_size_32);
        } else if (max_elements <= 128) {
          topk_by_bitonic_sort_wrapper_128(
            result_distances_buffer, result_indices_buffer, result_buffer_size_32);
        } else {
          assert(max_elements <= 256);
          topk_by_bitonic_sort_wrapper_256(
            result_distances_buffer, result_indices_buffer, result_buffer_size_32);
        }
      } else {
        if (max_elements <= 64) {
          topk_by_bitonic_sort<64, INDEX_T>(
            result_distances_buffer, result_indices_buffer, result_buffer_size_32);
        } else if (max_elements <= 128) {
          topk_by_bitonic_sort<128, INDEX_T>(
            result_distances_buffer, result_indices_buffer, result_buffer_size_32);
        } else {
          assert(max_elements <= 256);
          topk_by_bitonic_sort<256, INDEX_T>(
            result_distances_buffer, result_indices_buffer, result_buffer_size_32);
        }
      }
    }
    __syncthreads();
    _CLK_REC(clk_topk);

    if (iter + 1 >= max_iteration) { break; }

    _CLK_START();
    if (threadIdx.x < 32) {
      // [1st warp] Pick up a next parent
      pickup_next_parent<INDEX_T, DISTANCE_T>(parent_indices_buffer,
                                              result_indices_buffer,
                                              result_distances_buffer,
                                              local_traversed_hashmap_ptr,
                                              traversed_hash_bitlen);
    } else {
      // [Other warps] Reset visited hashmap
      hashmap::init<INDEX_T>(local_visited_hashmap_ptr, visited_hash_bitlen, 32);
    }
    __syncthreads();
    _CLK_REC(clk_pickup_parents);

    if ((parent_indices_buffer[0] == invalid_index) && (iter >= min_iteration)) { break; }

    _CLK_START();
    for (unsigned i = threadIdx.x; i < result_buffer_size_32; i += blockDim.x) {
      INDEX_T index = result_indices_buffer[i];
      if (index == invalid_index) { continue; }
      if ((i >= itopk_size) && (index & index_msb_1_mask)) {
        // Remove nodes kicked out of the itopk list from the traversed hash table.
        hashmap::remove<INDEX_T>(
          local_traversed_hashmap_ptr, traversed_hash_bitlen, index & ~index_msb_1_mask);
        result_indices_buffer[i]   = invalid_index;
        result_distances_buffer[i] = utils::get_max_value<DISTANCE_T>();
      } else {
        // Restore visited hashmap by putting nodes on result buffer in it.
        index &= ~index_msb_1_mask;
        hashmap::insert(local_visited_hashmap_ptr, visited_hash_bitlen, index);
      }
    }
    // Initialize buffer for compute_distance_to_child_nodes.
    if (threadIdx.x == blockDim.x - 1) { result_position[0] = result_buffer_size_32; }
    __syncthreads();

    // Compute the norms between child nodes and query node using JIT version
    compute_distance_to_child_nodes_jit<IndexT, DistanceT, DataT, 0>(result_indices_buffer,
                                                                     result_distances_buffer,
                                                                     smem_desc,
                                                                     knn_graph,
                                                                     graph_degree,
                                                                     local_visited_hashmap_ptr,
                                                                     visited_hash_bitlen,
                                                                     local_traversed_hashmap_ptr,
                                                                     traversed_hash_bitlen,
                                                                     parent_indices_buffer,
                                                                     result_indices_buffer,
                                                                     1,
                                                                     result_position,
                                                                     result_buffer_size_32);
    __syncthreads();

    // Check the state of the nodes in the result buffer which were not updated
    // by the compute_distance_to_child_nodes above, and if it cannot be used as
    // a parent node, it is deactivated.
    for (uint32_t i = threadIdx.x; i < result_position[0]; i += blockDim.x) {
      INDEX_T index = result_indices_buffer[i];
      if (index == invalid_index || index & index_msb_1_mask) { continue; }
      if (hashmap::search<INDEX_T, 1>(local_traversed_hashmap_ptr, traversed_hash_bitlen, index)) {
        result_indices_buffer[i]   = invalid_index;
        result_distances_buffer[i] = utils::get_max_value<DISTANCE_T>();
      }
    }
    __syncthreads();
    _CLK_REC(clk_compute_distance);

    // Filtering - use extern sample_filter function (linked via JIT LTO)
    for (unsigned p = threadIdx.x; p < 1; p += blockDim.x) {
      if (parent_indices_buffer[p] != invalid_index) {
        const auto parent_id = result_indices_buffer[parent_indices_buffer[p]] & ~index_msb_1_mask;
        if (!sample_filter<SourceIndexT>(query_id + query_id_offset,
                                         to_source_index(parent_id),
                                         filter_payload.sample_filter_data())) {
          // If the parent must not be in the resulting top-k list, remove from the parent list
          result_distances_buffer[parent_indices_buffer[p]] = utils::get_max_value<DISTANCE_T>();
          result_indices_buffer[parent_indices_buffer[p]]   = invalid_index;
        }
      }
    }
    __syncthreads();

    iter++;
  }

  // Filtering - use extern sample_filter function (linked via JIT LTO)
  for (uint32_t i = threadIdx.x; i < result_buffer_size_32; i += blockDim.x) {
    INDEX_T index = result_indices_buffer[i];
    if (index == invalid_index) { continue; }
    index &= ~index_msb_1_mask;
    if (!sample_filter<SourceIndexT>(query_id + query_id_offset,
                                     to_source_index(index),
                                     filter_payload.sample_filter_data())) {
      result_indices_buffer[i]   = invalid_index;
      result_distances_buffer[i] = utils::get_max_value<DISTANCE_T>();
    }
  }
  __syncthreads();

  // Output search results (1st warp only).
  if (threadIdx.x < 32) {
    uint32_t offset = 0;
    for (uint32_t i = threadIdx.x; i < result_buffer_size_32; i += 32) {
      INDEX_T index = result_indices_buffer[i];
      bool is_valid = false;
      if (index != invalid_index) {
        if (index & index_msb_1_mask) {
          is_valid = true;
          index &= ~index_msb_1_mask;
        } else if ((offset < itopk_size) &&
                   hashmap::insert<INDEX_T, 1>(
                     local_traversed_hashmap_ptr, traversed_hash_bitlen, index)) {
          // If a node that is not used as a parent can be inserted into
          // the traversed hash table, it is considered a valid result.
          is_valid = true;
        }
      }
      const auto mask = __ballot_sync(0xffffffff, is_valid);
      if (is_valid) {
        const auto j = offset + __popc(mask & ((1 << threadIdx.x) - 1));
        if (j < itopk_size) {
          uint32_t k            = j + (itopk_size * (cta_id + (num_cta_per_query * query_id)));
          result_indices_ptr[k] = index & ~index_msb_1_mask;
          if (result_distances_ptr != nullptr) {
            DISTANCE_T dist         = result_distances_buffer[i];
            result_distances_ptr[k] = dist;
          }
        } else {
          // If it is valid and registered in the traversed hash table but is
          // not output as a result, it is removed from the hash table.
          hashmap::remove<INDEX_T>(local_traversed_hashmap_ptr, traversed_hash_bitlen, index);
        }
      }
      offset += __popc(mask);
    }
    // If the number of outputs is insufficient, fill in with invalid results.
    for (uint32_t i = offset + threadIdx.x; i < itopk_size; i += 32) {
      uint32_t k            = i + (itopk_size * (cta_id + (num_cta_per_query * query_id)));
      result_indices_ptr[k] = invalid_index;
      if (result_distances_ptr != nullptr) {
        result_distances_ptr[k] = utils::get_max_value<DISTANCE_T>();
      }
    }
  }

  if (threadIdx.x == 0 && cta_id == 0 && num_executed_iterations != nullptr) {
    num_executed_iterations[query_id] = iter + 1;
  }

#ifdef _CLK_BREAKDOWN
  if ((threadIdx.x == 0 || threadIdx.x == blockDim.x - 1) && (blockIdx.x == 0) &&
      ((query_id * 3) % gridDim.y < 3)) {
    printf(
      "%s:%d "
      "query, %d, thread, %d"
      ", init, %lu"
      ", 1st_distance, %lu"
      ", topk, %lu"
      ", pickup_parents, %lu"
      ", distance, %lu"
      "\n",
      __FILE__,
      __LINE__,
      query_id,
      threadIdx.x,
      clk_init,
      clk_compute_1st_distance,
      clk_topk,
      clk_pickup_parents,
      clk_compute_distance);
  }
#endif
}

}  // namespace cuvs::neighbors::cagra::detail::multi_cta_search
