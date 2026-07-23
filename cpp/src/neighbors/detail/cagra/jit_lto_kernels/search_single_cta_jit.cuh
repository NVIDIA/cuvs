/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Device-only helpers - split out to avoid pulling host launcher code into JIT translation units
#include "search_single_cta_device_helpers.cuh"

// neighbors_device_intrinsics / memory_ops come via search_single_cta_device_helpers.cuh
#include "../hashmap.hpp"
#include "../topk_by_radix.cuh"
#include "../utils.hpp"

#include <raft/core/operators.hpp>      // For raft::shfl_xor
#include <raft/util/integer_utils.hpp>  // For raft::round_up_safe
#include <raft/util/pow2_utils.cuh>

#include <cub/warp/warp_scan.cuh>

#include <cuda/atomic>
#include <cuda/std/atomic>

#include <cassert>  // For assert()

#ifdef _CLK_BREAKDOWN
#include <cstdio>  // For printf() in debug mode
#endif

// Include extern function declarations before namespace so they're available to kernel definitions
#include "cagra_filter_payload.cuh"
#include "extern_device_functions.cuh"
// Include shared JIT device functions
#include "device_common_jit.cuh"

namespace cuvs::neighbors::cagra::detail::single_cta_search {

// Sample filter extern function
// sample_filter is declared in extern_device_functions.cuh
using cuvs::neighbors::detail::sample_filter;

// JIT versions of compute_distance_to_random_nodes and compute_distance_to_child_nodes
// are now shared in device_common_jit.cuh - use fully qualified names
using cuvs::neighbors::cagra::detail::device::compute_distance_to_child_nodes_jit;
using cuvs::neighbors::cagra::detail::device::compute_distance_to_random_nodes_jit;

// JIT search_core - setup_workspace/compute_distance via function pointers
template <bool TOPK_BY_BITONIC_SORT,
          bool BITONIC_SORT_AND_MERGE_MULTI_WARPS,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT>
RAFT_DEVICE_INLINE_FUNCTION void search_core(
  uintptr_t result_indices_ptr,
  DistanceT* const result_distances_ptr,
  const std::uint32_t top_k,
  const DataT* const queries_ptr,
  const IndexT* const knn_graph,
  const std::uint32_t graph_degree,
  const SourceIndexT* source_indices_ptr,
  const unsigned num_distilation,
  const uint64_t rand_xor_mask,
  const IndexT* seed_ptr,
  const uint32_t num_seeds,
  IndexT* const visited_hashmap_ptr,
  const std::uint32_t max_candidates,
  const std::uint32_t max_itopk,
  const std::uint32_t internal_topk,
  const std::uint32_t search_width,
  const std::uint32_t min_iteration,
  const std::uint32_t max_iteration,
  std::uint32_t* const num_executed_iterations,
  const std::uint32_t hash_bitlen,
  const std::uint32_t small_hash_bitlen,
  const std::uint32_t small_hash_reset_interval,
  const std::uint32_t query_id,
  const std::uint32_t query_id_offset,  // Offset to add to query_id when calling filter
  const dataset_descriptor_base_t<DataT, IndexT, DistanceT>* dataset_desc,
  cagra_sample_filter<SourceIndexT> filter_payload,
  const IndexT graph_size = 0)  // Original number of bits
{
  using LOAD_T = device::LOAD_128BIT_T;

  auto to_source_index = [source_indices_ptr](IndexT x) {
    return source_indices_ptr == nullptr ? static_cast<SourceIndexT>(x) : source_indices_ptr[x];
  };

#ifdef _CLK_BREAKDOWN
  std::uint64_t clk_init                 = 0;
  std::uint64_t clk_compute_1st_distance = 0;
  std::uint64_t clk_topk                 = 0;
  std::uint64_t clk_reset_hash           = 0;
  std::uint64_t clk_pickup_parents       = 0;
  std::uint64_t clk_restore_hash         = 0;
  std::uint64_t clk_compute_distance     = 0;
  std::uint64_t clk_start;
#define _CLK_START() clk_start = clock64()
#define _CLK_REC(V)  V += clock64() - clk_start;
#else
#define _CLK_START()
#define _CLK_REC(V)
#endif
  _CLK_START();

  extern __shared__ uint8_t smem[];

  // Layout of result_buffer
  const auto result_buffer_size    = internal_topk + (search_width * graph_degree);
  const auto result_buffer_size_32 = raft::round_up_safe<uint32_t>(result_buffer_size, 32);
  const auto small_hash_size       = hashmap::get_size(small_hash_bitlen);

  // Get dim and smem_ws_size directly from base descriptor
  uint32_t dim                   = dataset_desc->args.dim;
  uint32_t smem_ws_size_in_bytes = dataset_desc->smem_ws_size_in_bytes();

  auto smem_desc =
    setup_workspace<DataT, IndexT, DistanceT>(dataset_desc, smem, queries_ptr, query_id);

  auto* __restrict__ result_indices_buffer =
    reinterpret_cast<IndexT*>(smem + smem_ws_size_in_bytes);
  auto* __restrict__ result_distances_buffer =
    reinterpret_cast<DistanceT*>(result_indices_buffer + result_buffer_size_32);
  auto* __restrict__ visited_hash_buffer =
    reinterpret_cast<IndexT*>(result_distances_buffer + result_buffer_size_32);
  auto* __restrict__ parent_list_buffer =
    reinterpret_cast<IndexT*>(visited_hash_buffer + small_hash_size);
  auto* __restrict__ topk_ws = reinterpret_cast<std::uint32_t*>(parent_list_buffer + search_width);
  auto* terminate_flag       = reinterpret_cast<std::uint32_t*>(topk_ws + 3);
  auto* __restrict__ smem_work_ptr = reinterpret_cast<std::uint32_t*>(terminate_flag + 1);

  // A flag for filtering.
  auto filter_flag = terminate_flag;

  if (threadIdx.x == 0) {
    terminate_flag[0] = 0;
    topk_ws[0]        = ~0u;
  }

  // Init hashmap
  IndexT* local_visited_hashmap_ptr;
  if (small_hash_bitlen) {
    local_visited_hashmap_ptr = visited_hash_buffer;
  } else {
    local_visited_hashmap_ptr = visited_hashmap_ptr + (hashmap::get_size(hash_bitlen) * blockIdx.y);
  }
  hashmap::init(local_visited_hashmap_ptr, hash_bitlen, 0);
  __syncthreads();
  _CLK_REC(clk_init);

  // compute distance to randomly selecting nodes using JIT version
  _CLK_START();
  const IndexT* const local_seed_ptr = seed_ptr ? seed_ptr + (num_seeds * query_id) : nullptr;
  // Get dataset_size directly from base descriptor
  IndexT dataset_size = smem_desc->size;
  compute_distance_to_random_nodes_jit<IndexT, DistanceT, DataT>(result_indices_buffer,
                                                                 result_distances_buffer,
                                                                 smem_desc,
                                                                 result_buffer_size,
                                                                 num_distilation,
                                                                 rand_xor_mask,
                                                                 local_seed_ptr,
                                                                 num_seeds,
                                                                 local_visited_hashmap_ptr,
                                                                 hash_bitlen,
                                                                 (IndexT*)nullptr,
                                                                 0,
                                                                 0,
                                                                 1,
                                                                 graph_size);
  __syncthreads();
  _CLK_REC(clk_compute_1st_distance);

  std::uint32_t iter = 0;
  while (1) {
    // sort
    if constexpr (TOPK_BY_BITONIC_SORT) {
      assert(blockDim.x >= 64);
      const bool bitonic_sort_and_full_multi_warps = (max_candidates > 128) ? true : false;

      // reset small-hash table.
      if ((iter + 1) % small_hash_reset_interval == 0) {
        _CLK_START();
        unsigned hash_start_tid;
        if (blockDim.x == 32) {
          hash_start_tid = 0;
        } else if (blockDim.x == 64) {
          if (bitonic_sort_and_full_multi_warps || BITONIC_SORT_AND_MERGE_MULTI_WARPS) {
            hash_start_tid = 0;
          } else {
            hash_start_tid = 32;
          }
        } else {
          if (bitonic_sort_and_full_multi_warps || BITONIC_SORT_AND_MERGE_MULTI_WARPS) {
            hash_start_tid = 64;
          } else {
            hash_start_tid = 32;
          }
        }
        hashmap::init(local_visited_hashmap_ptr, hash_bitlen, hash_start_tid);
        _CLK_REC(clk_reset_hash);
      }

      // topk with bitonic sort
      _CLK_START();
      // For JIT version, we always check filter_flag at runtime since sample_filter is extern
      if (*filter_flag != 0) {
        // Move the filtered out index to the end of the itopk list
        for (unsigned i = 0; i < search_width; i++) {
          move_invalid_to_end_of_list(
            result_indices_buffer, result_distances_buffer, internal_topk);
        }
        if (threadIdx.x == 0) { *terminate_flag = 0; }
      }
      topk_by_bitonic_sort_and_merge<BITONIC_SORT_AND_MERGE_MULTI_WARPS>(
        result_distances_buffer,
        result_indices_buffer,
        max_itopk,
        internal_topk,
        result_distances_buffer + internal_topk,
        result_indices_buffer + internal_topk,
        max_candidates,
        search_width * graph_degree,
        topk_ws,
        (iter == 0));
      __syncthreads();
      _CLK_REC(clk_topk);
    } else {
      _CLK_START();
      // topk with radix block sort
      topk_by_radix_sort<IndexT>{}(max_itopk,
                                   internal_topk,
                                   result_buffer_size,
                                   reinterpret_cast<std::uint32_t*>(result_distances_buffer),
                                   result_indices_buffer,
                                   reinterpret_cast<std::uint32_t*>(result_distances_buffer),
                                   result_indices_buffer,
                                   nullptr,
                                   topk_ws,
                                   true,
                                   smem_work_ptr);
      _CLK_REC(clk_topk);

      // reset small-hash table
      if ((iter + 1) % small_hash_reset_interval == 0) {
        _CLK_START();
        hashmap::init(local_visited_hashmap_ptr, hash_bitlen);
        _CLK_REC(clk_reset_hash);
      }
    }
    __syncthreads();

    if (iter + 1 == max_iteration) { break; }

    // pick up next parents
    if (threadIdx.x < 32) {
      _CLK_START();
      pickup_next_parents<TOPK_BY_BITONIC_SORT, IndexT>(
        terminate_flag, parent_list_buffer, result_indices_buffer, internal_topk, search_width);
      _CLK_REC(clk_pickup_parents);
    }

    // restore small-hash table by putting internal-topk indices in it
    if ((iter + 1) % small_hash_reset_interval == 0) {
      const unsigned first_tid = ((blockDim.x <= 32) ? 0 : 32);
      _CLK_START();
      hashmap_restore(
        local_visited_hashmap_ptr, hash_bitlen, result_indices_buffer, internal_topk, first_tid);
      _CLK_REC(clk_restore_hash);
    }
    __syncthreads();

    if (*terminate_flag && iter >= min_iteration) { break; }

    __syncthreads();
    // compute the norms between child nodes and query node using JIT version
    _CLK_START();
    compute_distance_to_child_nodes_jit<IndexT, DistanceT, DataT>(
      result_indices_buffer + internal_topk,
      result_distances_buffer + internal_topk,
      smem_desc,
      knn_graph,
      graph_degree,
      local_visited_hashmap_ptr,
      hash_bitlen,
      (IndexT*)nullptr,
      0u,
      parent_list_buffer,
      result_indices_buffer,
      search_width);
    // Critical: __syncthreads() must be reached by ALL threads
    // If any thread is stuck in compute_distance_to_child_nodes_jit, this will hang
    __syncthreads();
    _CLK_REC(clk_compute_distance);

    // Filtering - use extern sample_filter function
    if (threadIdx.x == 0) { *filter_flag = 0; }
    __syncthreads();

    constexpr IndexT index_msb_1_mask = utils::gen_index_msb_1_mask<IndexT>::value;
    const IndexT invalid_index        = utils::get_max_value<IndexT>();

    for (unsigned p = threadIdx.x; p < search_width; p += blockDim.x) {
      if (parent_list_buffer[p] != invalid_index) {
        const auto parent_id = result_indices_buffer[parent_list_buffer[p]] & ~index_msb_1_mask;
        if (!sample_filter<SourceIndexT>(query_id + query_id_offset,
                                         to_source_index(parent_id),
                                         filter_payload.sample_filter_data())) {
          result_distances_buffer[parent_list_buffer[p]] = utils::get_max_value<DistanceT>();
          result_indices_buffer[parent_list_buffer[p]]   = invalid_index;
          *filter_flag                                   = 1;
        }
      }
    }
    __syncthreads();

    iter++;
  }

  // Post process for filtering - use extern sample_filter function
  constexpr IndexT index_msb_1_mask = utils::gen_index_msb_1_mask<IndexT>::value;
  const IndexT invalid_index        = utils::get_max_value<IndexT>();

  for (unsigned i = threadIdx.x; i < internal_topk + search_width * graph_degree; i += blockDim.x) {
    const auto node_id = result_indices_buffer[i] & ~index_msb_1_mask;
    if (node_id != (invalid_index & ~index_msb_1_mask) &&
        !sample_filter<SourceIndexT>(query_id + query_id_offset,
                                     to_source_index(node_id),
                                     filter_payload.sample_filter_data())) {
      result_distances_buffer[i] = utils::get_max_value<DistanceT>();
      result_indices_buffer[i]   = invalid_index;
    }
  }

  __syncthreads();
  // Move invalid index items to the end of the buffer without sorting the entire buffer
  using scan_op_t    = cub::WarpScan<unsigned>;
  auto& temp_storage = *reinterpret_cast<typename scan_op_t::TempStorage*>(smem_work_ptr);

  constexpr std::uint32_t warp_size = 32;
  if (threadIdx.x < warp_size) {
    std::uint32_t num_found_valid = 0;
    for (std::uint32_t buffer_offset = 0; buffer_offset < internal_topk;
         buffer_offset += warp_size) {
      const auto src_position = buffer_offset + threadIdx.x;
      const std::uint32_t is_valid_index =
        (result_indices_buffer[src_position] & (~index_msb_1_mask)) == invalid_index ? 0 : 1;
      std::uint32_t new_position;
      scan_op_t(temp_storage).InclusiveSum(is_valid_index, new_position);
      if (is_valid_index) {
        const auto dst_position               = num_found_valid + (new_position - 1);
        result_indices_buffer[dst_position]   = result_indices_buffer[src_position];
        result_distances_buffer[dst_position] = result_distances_buffer[src_position];
      }

      num_found_valid += new_position;
      for (std::uint32_t offset = (warp_size >> 1); offset > 0; offset >>= 1) {
        const auto v = raft::shfl_xor(num_found_valid, offset);
        if ((threadIdx.x & offset) == 0) { num_found_valid = v; }
      }

      if (num_found_valid >= top_k) { break; }
    }

    if (num_found_valid < top_k) {
      for (std::uint32_t i = num_found_valid + threadIdx.x; i < internal_topk; i += warp_size) {
        result_indices_buffer[i]   = invalid_index;
        result_distances_buffer[i] = utils::get_max_value<DistanceT>();
      }
    }
  }

  // If the sufficient number of valid indexes are not in the internal topk, pick up from the
  // candidate list.
  if (top_k > internal_topk || result_indices_buffer[top_k - 1] == invalid_index) {
    __syncthreads();
    topk_by_bitonic_sort_and_merge<BITONIC_SORT_AND_MERGE_MULTI_WARPS>(
      result_distances_buffer,
      result_indices_buffer,
      max_itopk,
      internal_topk,
      result_distances_buffer + internal_topk,
      result_indices_buffer + internal_topk,
      max_candidates,
      search_width * graph_degree,
      topk_ws,
      (iter == 0));
  }
  __syncthreads();

  // NB: The indices pointer is tagged with its element size.
  const uint32_t index_element_tag = result_indices_ptr & 0x3;
  result_indices_ptr ^= index_element_tag;
  auto write_indices =
    index_element_tag == 3
      ? [](uintptr_t ptr,
           uint32_t i,
           SourceIndexT x) { reinterpret_cast<uint64_t*>(ptr)[i] = static_cast<uint64_t>(x); }
    : index_element_tag == 2
      ? [](uintptr_t ptr,
           uint32_t i,
           SourceIndexT x) { reinterpret_cast<uint32_t*>(ptr)[i] = static_cast<uint32_t>(x); }
    : index_element_tag == 1
      ? [](uintptr_t ptr,
           uint32_t i,
           SourceIndexT x) { reinterpret_cast<uint16_t*>(ptr)[i] = static_cast<uint16_t>(x); }
      : [](uintptr_t ptr, uint32_t i, SourceIndexT x) {
          reinterpret_cast<uint8_t*>(ptr)[i] = static_cast<uint8_t>(x);
        };
  for (std::uint32_t i = threadIdx.x; i < top_k; i += blockDim.x) {
    unsigned j  = i + (top_k * query_id);
    unsigned ii = i;
    if constexpr (TOPK_BY_BITONIC_SORT) { ii = device::swizzling(i); }
    if (result_distances_ptr != nullptr) { result_distances_ptr[j] = result_distances_buffer[ii]; }

    auto internal_index =
      result_indices_buffer[ii] & ~index_msb_1_mask;  // clear most significant bit
    auto source_index = to_source_index(internal_index);
    write_indices(result_indices_ptr, j, source_index);
  }
  if (threadIdx.x == 0 && num_executed_iterations != nullptr) {
    num_executed_iterations[query_id] = iter + 1;
  }
#ifdef _CLK_BREAKDOWN
  if ((threadIdx.x == 0 || threadIdx.x == blockDim.x - 1) && ((query_id * 3) % gridDim.y < 3)) {
    printf(
      "%s:%d "
      "query, %d, thread, %d"
      ", init, %lu"
      ", 1st_distance, %lu"
      ", topk, %lu"
      ", reset_hash, %lu"
      ", pickup_parents, %lu"
      ", restore_hash, %lu"
      ", distance, %lu"
      "\n",
      __FILE__,
      __LINE__,
      query_id,
      threadIdx.x,
      clk_init,
      clk_compute_1st_distance,
      clk_topk,
      clk_reset_hash,
      clk_pickup_parents,
      clk_restore_hash,
      clk_compute_distance);
  }
#endif
}

// JIT device implementation - called from extern "C" __global__ entry in generated .cu
template <bool TOPK_BY_BITONIC_SORT,
          bool BITONIC_SORT_AND_MERGE_MULTI_WARPS,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT>
__device__ void search_kernel_jit(
  uintptr_t result_indices_ptr,
  DistanceT* const result_distances_ptr,
  const std::uint32_t top_k,
  const DataT* const queries_ptr,
  const IndexT* const knn_graph,
  const std::uint32_t graph_degree,
  const SourceIndexT* source_indices_ptr,
  const unsigned num_distilation,
  const uint64_t rand_xor_mask,
  const IndexT* seed_ptr,
  const uint32_t num_seeds,
  IndexT* const visited_hashmap_ptr,
  const std::uint32_t max_candidates,
  const std::uint32_t max_itopk,
  const std::uint32_t internal_topk,
  const std::uint32_t search_width,
  const std::uint32_t min_iteration,
  const std::uint32_t max_iteration,
  std::uint32_t* const num_executed_iterations,
  const std::uint32_t hash_bitlen,
  const std::uint32_t small_hash_bitlen,
  const std::uint32_t small_hash_reset_interval,
  const std::uint32_t query_id_offset,  // Offset to add to query_id when calling filter
  const dataset_descriptor_base_t<DataT, IndexT, DistanceT>* dataset_desc,
  const IndexT graph_size,
  cagra_sample_filter<SourceIndexT> filter_payload)
{
  const auto query_id = blockIdx.y;
  search_core<TOPK_BY_BITONIC_SORT,
              BITONIC_SORT_AND_MERGE_MULTI_WARPS,
              DataT,
              IndexT,
              DistanceT,
              SourceIndexT>(result_indices_ptr,
                            result_distances_ptr,
                            top_k,
                            queries_ptr,
                            knn_graph,
                            graph_degree,
                            source_indices_ptr,
                            num_distilation,
                            rand_xor_mask,
                            seed_ptr,
                            num_seeds,
                            visited_hashmap_ptr,
                            max_candidates,
                            max_itopk,
                            internal_topk,
                            search_width,
                            min_iteration,
                            max_iteration,
                            num_executed_iterations,
                            hash_bitlen,
                            small_hash_bitlen,
                            small_hash_reset_interval,
                            query_id,
                            query_id_offset,
                            dataset_desc,
                            filter_payload,
                            graph_size);
}

// JIT persistent device implementation - called from extern "C" __global__ entry in generated .cu
template <bool TOPK_BY_BITONIC_SORT,
          bool BITONIC_SORT_AND_MERGE_MULTI_WARPS,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT>
__device__ void search_single_cta_p_impl(
  worker_handle_t* worker_handles,
  job_desc_t<job_desc_traits<DataT, IndexT, DistanceT>>* job_descriptors,
  uint32_t* completion_counters,
  const IndexT* const knn_graph,  // [dataset_size, graph_degree]
  const std::uint32_t graph_degree,
  const SourceIndexT* source_indices_ptr,
  const unsigned num_distilation,
  const uint64_t rand_xor_mask,
  const IndexT* seed_ptr,  // [num_queries, num_seeds]
  const uint32_t num_seeds,
  IndexT* const visited_hashmap_ptr,  // [num_queries, 1 << hash_bitlen]
  const std::uint32_t max_candidates,
  const std::uint32_t max_itopk,
  const std::uint32_t internal_topk,
  const std::uint32_t search_width,
  const std::uint32_t min_iteration,
  const std::uint32_t max_iteration,
  std::uint32_t* const num_executed_iterations,  // [num_queries]
  const std::uint32_t hash_bitlen,
  const std::uint32_t small_hash_bitlen,
  const std::uint32_t small_hash_reset_interval,
  const std::uint32_t query_id_offset,  // Offset to add to query_id when calling filter
  const dataset_descriptor_base_t<DataT, IndexT, DistanceT>* dataset_desc,
  const IndexT graph_size,
  cagra_sample_filter<SourceIndexT> filter_payload)
{
  using job_desc_type = job_desc_t<job_desc_traits<DataT, IndexT, DistanceT>>;
  __shared__ typename job_desc_type::input_t job_descriptor;
  __shared__ worker_handle_t::data_t worker_data;

  auto& worker_handle = worker_handles[blockIdx.y].data;
  uint32_t job_ix;

  while (true) {
    // wait the writing phase
    if (threadIdx.x == 0) {
      worker_handle_t::data_t worker_data_local;
      do {
        worker_data_local = worker_handle.load(cuda::memory_order_relaxed);
      } while (worker_data_local.handle == kWaitForWork);
      if (worker_data_local.handle != kNoMoreWork) {
        worker_handle.store({kWaitForWork}, cuda::memory_order_relaxed);
      }
      job_ix = worker_data_local.value.desc_id;
      cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);
      worker_data = worker_data_local;
    }
    if (threadIdx.x < raft::WarpSize) {
      // Sync one warp and copy descriptor data
      static_assert(job_desc_type::kBlobSize <= raft::WarpSize);
      constexpr uint32_t kMaxJobsNum = 8192;
      job_ix                         = raft::shfl(job_ix, 0);
      if (threadIdx.x < job_desc_type::kBlobSize && job_ix < kMaxJobsNum) {
        job_descriptor.blob[threadIdx.x] = job_descriptors[job_ix].input.blob[threadIdx.x];
      }
    }
    __syncthreads();
    if (worker_data.handle == kNoMoreWork) { break; }

    // reading phase
    auto result_indices_ptr    = job_descriptor.value.result_indices_ptr;
    auto* result_distances_ptr = job_descriptor.value.result_distances_ptr;
    auto* queries_ptr          = job_descriptor.value.queries_ptr;
    auto top_k                 = job_descriptor.value.top_k;
    auto n_queries             = job_descriptor.value.n_queries;
    auto query_id              = worker_data.value.query_id;

    // work phase - use JIT search_core
    search_core<TOPK_BY_BITONIC_SORT,
                BITONIC_SORT_AND_MERGE_MULTI_WARPS,
                DataT,
                IndexT,
                DistanceT,
                SourceIndexT>(result_indices_ptr,
                              result_distances_ptr,
                              top_k,
                              queries_ptr,
                              knn_graph,
                              graph_degree,
                              source_indices_ptr,
                              num_distilation,
                              rand_xor_mask,
                              seed_ptr,
                              num_seeds,
                              visited_hashmap_ptr,
                              max_candidates,
                              max_itopk,
                              internal_topk,
                              search_width,
                              min_iteration,
                              max_iteration,
                              num_executed_iterations,
                              hash_bitlen,
                              small_hash_bitlen,
                              small_hash_reset_interval,
                              query_id,
                              query_id_offset,
                              dataset_desc,
                              filter_payload,
                              graph_size);

    // make sure all writes are visible even for the host
    //     (e.g. when result buffers are in pinned memory)
    cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);

    // arrive to mark the end of the work phase
    __syncthreads();
    if (threadIdx.x == 0) {
      auto completed_count = atomicInc(completion_counters + job_ix, n_queries - 1) + 1;
      if (completed_count >= n_queries) {
        job_descriptors[job_ix].completion_flag.store(true, cuda::memory_order_relaxed);
      }
    }
  }
}

}  // namespace cuvs::neighbors::cagra::detail::single_cta_search
