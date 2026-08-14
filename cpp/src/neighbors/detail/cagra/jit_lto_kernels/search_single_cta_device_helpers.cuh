/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Device-only includes - no host-side headers
#include "../../neighbors_device_intrinsics.cuh"
#include "../bitonic.hpp"
#include "../device_memory_ops.hpp"
#include "../hashmap.hpp"
#include "../utils.hpp"

#include <raft/core/detail/macros.hpp>
#include <raft/util/warp_primitives.cuh>

#include <raft/util/integer_utils.hpp>

#include <cassert>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <cuda/atomic>
#include <cuda_runtime.h>  // For uint4

namespace cuvs::neighbors::cagra::detail::single_cta_search {

// Type bundle for `job_desc_t` (DATA_T / INDEX_T / DISTANCE_T for persistent single-CTA kernels).
template <typename DataT, typename IndexT, typename DistanceT>
struct job_desc_traits {
  using DATA_T     = DataT;
  using INDEX_T    = IndexT;
  using DISTANCE_T = DistanceT;
};

// Constants for persistent kernels (shared by JIT device code and host launcher)
constexpr size_t kCacheLineBytes            = 64;
constexpr uint32_t kMaxJobsNum              = 8192;
constexpr uint32_t kMaxWorkersNum           = 4096;
constexpr uint32_t kMaxWorkersPerThread     = 256;
constexpr uint32_t kSoftMaxWorkersPerThread = 16;

// Worker handle for persistent kernels
struct alignas(kCacheLineBytes) worker_handle_t {
  using handle_t = uint64_t;
  struct value_t {
    uint32_t desc_id;
    uint32_t query_id;
  };
  union data_t {
    handle_t handle;
    value_t value;
  };
  cuda::atomic<data_t, cuda::thread_scope_system> data;
};
static_assert(sizeof(worker_handle_t::value_t) == sizeof(worker_handle_t::handle_t));
static_assert(
  cuda::atomic<worker_handle_t::data_t, cuda::thread_scope_system>::is_always_lock_free);

constexpr worker_handle_t::handle_t kWaitForWork = std::numeric_limits<uint64_t>::max();
constexpr worker_handle_t::handle_t kNoMoreWork  = kWaitForWork - 1;

constexpr auto is_worker_busy(worker_handle_t::handle_t h) -> bool
{
  return (h != kWaitForWork) && (h != kNoMoreWork);
}

// Job descriptor for persistent kernels
template <typename DATASET_DESCRIPTOR_T>
struct alignas(kCacheLineBytes) job_desc_t {
  using index_type    = typename DATASET_DESCRIPTOR_T::INDEX_T;
  using distance_type = typename DATASET_DESCRIPTOR_T::DISTANCE_T;
  using data_type     = typename DATASET_DESCRIPTOR_T::DATA_T;
  // The algorithm input parameters
  struct value_t {
    uintptr_t result_indices_ptr;         // [num_queries, top_k]
    distance_type* result_distances_ptr;  // [num_queries, top_k]
    const data_type* queries_ptr;         // [num_queries, dataset_dim]
    uint32_t top_k;
    uint32_t n_queries;
  };
  using blob_elem_type = uint4;
  constexpr static inline size_t kBlobSize =
    raft::div_rounding_up_safe(sizeof(value_t), sizeof(blob_elem_type));
  // Union facilitates loading the input by a warp in a single request
  union input_t {
    blob_elem_type blob[kBlobSize];  // NOLINT
    value_t value;
  } input;
  // Last thread triggers this flag.
  cuda::atomic<bool, cuda::thread_scope_system> completion_flag;
};

// Pick up next parent nodes from the internal topk list
template <bool TOPK_BY_BITONIC_SORT, class INDEX_T>
RAFT_DEVICE_INLINE_FUNCTION void pickup_next_parents(std::uint32_t* const terminate_flag,
                                                     INDEX_T* const next_parent_indices,
                                                     INDEX_T* const internal_topk_indices,
                                                     const std::size_t internal_topk_size,
                                                     const std::uint32_t search_width)
{
  constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;

  for (std::uint32_t i = threadIdx.x; i < search_width; i += 32) {
    next_parent_indices[i] = utils::get_max_value<INDEX_T>();
  }
  std::uint32_t itopk_max = internal_topk_size;
  if (itopk_max % 32) { itopk_max += 32 - (itopk_max % 32); }
  std::uint32_t num_new_parents = 0;
  for (std::uint32_t j = threadIdx.x; j < itopk_max; j += 32) {
    std::uint32_t jj = j;
    if (TOPK_BY_BITONIC_SORT) { jj = device::swizzling(j); }
    INDEX_T index;
    int new_parent = 0;
    if (j < internal_topk_size) {
      index = internal_topk_indices[jj];
      if ((index & index_msb_1_mask) == 0) {  // check if most significant bit is set
        new_parent = 1;
      }
    }
    const std::uint32_t ballot_mask = __ballot_sync(0xffffffff, new_parent);
    if (new_parent) {
      const auto i = __popc(ballot_mask & ((1 << threadIdx.x) - 1)) + num_new_parents;
      if (i < search_width) {
        next_parent_indices[i] = jj;
        // set most significant bit as used node
        internal_topk_indices[jj] |= index_msb_1_mask;
      }
    }
    num_new_parents += __popc(ballot_mask);
    if (num_new_parents >= search_width) { break; }
  }
  if (threadIdx.x == 0 && (num_new_parents == 0)) { *terminate_flag = 1; }
}

// Helper function for bitonic sort and full
template <unsigned MAX_CANDIDATES, bool MULTI_WARPS, class IdxT = void>
RAFT_DEVICE_INLINE_FUNCTION void topk_by_bitonic_sort_and_full(
  float* candidate_distances,  // [num_candidates]
  IdxT* candidate_indices,     // [num_candidates]
  const std::uint32_t num_candidates,
  const std::uint32_t num_itopk)
{
  const unsigned lane_id = threadIdx.x % raft::warp_size();
  const unsigned warp_id = threadIdx.x / raft::warp_size();
  static_assert(MAX_CANDIDATES <= 256);
  if constexpr (!MULTI_WARPS) {
    if (warp_id > 0) { return; }
    constexpr unsigned N = (MAX_CANDIDATES + (raft::warp_size() - 1)) / raft::warp_size();
    float key[N];
    IdxT val[N];
    /* Candidates -> Reg */
    for (unsigned i = 0; i < N; i++) {
      unsigned j = lane_id + (raft::warp_size() * i);
      if (j < num_candidates) {
        key[i] = candidate_distances[j];
        val[i] = candidate_indices[j];
      } else {
        key[i] = utils::get_max_value<float>();
        val[i] = utils::get_max_value<IdxT>();
      }
    }
    /* Sort */
    bitonic::warp_sort<float, IdxT, N>(key, val);
    /* Reg -> Temp_itopk */
    for (unsigned i = 0; i < N; i++) {
      unsigned j = (N * lane_id) + i;
      if (j < num_candidates && j < num_itopk) {
        candidate_distances[device::swizzling(j)] = key[i];
        candidate_indices[device::swizzling(j)]   = val[i];
      }
    }
  } else {
    assert(blockDim.x >= 64);
    // Use two warps (64 threads)
    constexpr unsigned max_candidates_per_warp = (MAX_CANDIDATES + 1) / 2;
    static_assert(max_candidates_per_warp <= 128);
    constexpr unsigned N = (max_candidates_per_warp + (raft::warp_size() - 1)) / raft::warp_size();
    float key[N];
    IdxT val[N];
    if (warp_id < 2) {
      /* Candidates -> Reg */
      for (unsigned i = 0; i < N; i++) {
        unsigned jl = lane_id + (raft::warp_size() * i);
        unsigned j  = jl + (max_candidates_per_warp * warp_id);
        if (j < num_candidates) {
          key[i] = candidate_distances[j];
          val[i] = candidate_indices[j];
        } else {
          key[i] = utils::get_max_value<float>();
          val[i] = utils::get_max_value<IdxT>();
        }
      }
      /* Sort */
      bitonic::warp_sort<float, IdxT, N>(key, val);
      /* Reg -> Temp_candidates */
      for (unsigned i = 0; i < N; i++) {
        unsigned jl = (N * lane_id) + i;
        unsigned j  = jl + (max_candidates_per_warp * warp_id);
        if (j < num_candidates && jl < num_itopk) {
          candidate_distances[device::swizzling(j)] = key[i];
          candidate_indices[device::swizzling(j)]   = val[i];
        }
      }
    }
    __syncthreads();

    unsigned num_warps_used = (num_itopk + max_candidates_per_warp - 1) / max_candidates_per_warp;
    if (warp_id < num_warps_used) {
      /* Temp_candidates -> Reg */
      for (unsigned i = 0; i < N; i++) {
        unsigned jl = (N * lane_id) + i;
        unsigned kl = max_candidates_per_warp - 1 - jl;
        unsigned j  = jl + (max_candidates_per_warp * warp_id);
        unsigned k  = MAX_CANDIDATES - 1 - j;
        if (j >= num_candidates || k >= num_candidates || kl >= num_itopk) continue;
        float temp_key = candidate_distances[device::swizzling(k)];
        if (key[i] == temp_key) continue;
        if ((warp_id == 0) == (key[i] > temp_key)) {
          key[i] = temp_key;
          val[i] = candidate_indices[device::swizzling(k)];
        }
      }
    }
    if (num_warps_used > 1) { __syncthreads(); }
    if (warp_id < num_warps_used) {
      /* Merge */
      bitonic::warp_merge<float, IdxT, N>(key, val, raft::warp_size());
      /* Reg -> Temp_itopk */
      for (unsigned i = 0; i < N; i++) {
        unsigned jl = (N * lane_id) + i;
        unsigned j  = jl + (max_candidates_per_warp * warp_id);
        if (j < num_candidates && j < num_itopk) {
          candidate_distances[device::swizzling(j)] = key[i];
          candidate_indices[device::swizzling(j)]   = val[i];
        }
      }
    }
    if (num_warps_used > 1) { __syncthreads(); }
  }
}

// Wrapper functions to avoid pre-inlining (impacts register pressure)
RAFT_DEVICE_INLINE_FUNCTION void topk_by_bitonic_sort_and_full_wrapper_64_false(
  float* candidate_distances,        // [num_candidates]
  std::uint32_t* candidate_indices,  // [num_candidates]
  const std::uint32_t num_candidates,
  const std::uint32_t num_itopk)
{
  topk_by_bitonic_sort_and_full<64, false, uint32_t>(
    candidate_distances, candidate_indices, num_candidates, num_itopk);
}

RAFT_DEVICE_INLINE_FUNCTION void topk_by_bitonic_sort_and_full_wrapper_128_false(
  float* candidate_distances,        // [num_candidates]
  std::uint32_t* candidate_indices,  // [num_candidates]
  const std::uint32_t num_candidates,
  const std::uint32_t num_itopk)
{
  topk_by_bitonic_sort_and_full<128, false, uint32_t>(
    candidate_distances, candidate_indices, num_candidates, num_itopk);
}

RAFT_DEVICE_INLINE_FUNCTION void topk_by_bitonic_sort_and_full_wrapper_256_false(
  float* candidate_distances,        // [num_candidates]
  std::uint32_t* candidate_indices,  // [num_candidates]
  const std::uint32_t num_candidates,
  const std::uint32_t num_itopk)
{
  topk_by_bitonic_sort_and_full<256, false, uint32_t>(
    candidate_distances, candidate_indices, num_candidates, num_itopk);
}

// TopK by bitonic sort and merge (template version with MAX_ITOPK)
template <unsigned MAX_ITOPK, bool MULTI_WARPS, class IdxT = void>
RAFT_DEVICE_INLINE_FUNCTION void topk_by_bitonic_sort_and_merge(
  float* itopk_distances,  // [num_itopk]
  IdxT* itopk_indices,     // [num_itopk]
  const std::uint32_t num_itopk,
  float* candidate_distances,  // [num_candidates]
  IdxT* candidate_indices,     // [num_candidates]
  const std::uint32_t num_candidates,
  std::uint32_t* work_buf,
  const bool first)
{
  const unsigned lane_id = threadIdx.x % raft::warp_size();
  const unsigned warp_id = threadIdx.x / raft::warp_size();

  static_assert(MAX_ITOPK <= 512);
  if constexpr (!MULTI_WARPS) {
    static_assert(MAX_ITOPK <= 256);
    if (warp_id > 0) { return; }
    constexpr unsigned N = (MAX_ITOPK + (raft::warp_size() - 1)) / raft::warp_size();
    float key[N];
    IdxT val[N];
    if (first) {
      /* Load itopk results */
      for (unsigned i = 0; i < N; i++) {
        unsigned j = lane_id + (raft::warp_size() * i);
        if (j < num_itopk) {
          key[i] = itopk_distances[j];
          val[i] = itopk_indices[j];
        } else {
          key[i] = utils::get_max_value<float>();
          val[i] = utils::get_max_value<IdxT>();
        }
      }
      /* Warp Sort */
      bitonic::warp_sort<float, IdxT, N>(key, val);
    } else {
      /* Load itopk results */
      for (unsigned i = 0; i < N; i++) {
        unsigned j = (N * lane_id) + i;
        if (j < num_itopk) {
          key[i] = itopk_distances[device::swizzling(j)];
          val[i] = itopk_indices[device::swizzling(j)];
        } else {
          key[i] = utils::get_max_value<float>();
          val[i] = utils::get_max_value<IdxT>();
        }
      }
    }
    /* Merge candidates */
    for (unsigned i = 0; i < N; i++) {
      unsigned j = (N * lane_id) + i;  // [0:max_itopk-1]
      unsigned k = MAX_ITOPK - 1 - j;
      if (k >= num_itopk || k >= num_candidates) continue;
      float candidate_key = candidate_distances[device::swizzling(k)];
      if (key[i] > candidate_key) {
        key[i] = candidate_key;
        val[i] = candidate_indices[device::swizzling(k)];
      }
    }
    /* Warp Merge */
    bitonic::warp_merge<float, IdxT, N>(key, val, raft::warp_size());
    /* Store new itopk results */
    for (unsigned i = 0; i < N; i++) {
      unsigned j = (N * lane_id) + i;
      if (j < num_itopk) {
        itopk_distances[device::swizzling(j)] = key[i];
        itopk_indices[device::swizzling(j)]   = val[i];
      }
    }
  } else {
    static_assert(MAX_ITOPK == 512);
    assert(blockDim.x >= 64);
    // Use two warps (64 threads) or more
    constexpr unsigned max_itopk_per_warp = (MAX_ITOPK + 1) / 2;
    constexpr unsigned N = (max_itopk_per_warp + (raft::warp_size() - 1)) / raft::warp_size();
    float key[N];
    IdxT val[N];
    if (first) {
      /* Load itop results (not sorted) */
      if (warp_id < 2) {
        for (unsigned i = 0; i < N; i++) {
          unsigned j = lane_id + (raft::warp_size() * i) + (max_itopk_per_warp * warp_id);
          if (j < num_itopk) {
            key[i] = itopk_distances[j];
            val[i] = itopk_indices[j];
          } else {
            key[i] = utils::get_max_value<float>();
            val[i] = utils::get_max_value<IdxT>();
          }
        }
        /* Warp Sort */
        bitonic::warp_sort<float, IdxT, N>(key, val);
        /* Store intermedidate results */
        for (unsigned i = 0; i < N; i++) {
          unsigned j = (N * threadIdx.x) + i;
          if (j >= num_itopk) continue;
          itopk_distances[device::swizzling(j)] = key[i];
          itopk_indices[device::swizzling(j)]   = val[i];
        }
      }
      __syncthreads();
      if (warp_id < 2) {
        /* Load intermedidate results */
        for (unsigned i = 0; i < N; i++) {
          unsigned j = (N * threadIdx.x) + i;
          unsigned k = MAX_ITOPK - 1 - j;
          if (k >= num_itopk) continue;
          float temp_key = itopk_distances[device::swizzling(k)];
          if (key[i] == temp_key) continue;
          if ((warp_id == 0) == (key[i] > temp_key)) {
            key[i] = temp_key;
            val[i] = itopk_indices[device::swizzling(k)];
          }
        }
        /* Warp Merge */
        bitonic::warp_merge<float, IdxT, N>(key, val, raft::warp_size());
      }
      __syncthreads();
      /* Store itopk results (sorted) */
      if (warp_id < 2) {
        for (unsigned i = 0; i < N; i++) {
          unsigned j = (N * threadIdx.x) + i;
          if (j >= num_itopk) continue;
          itopk_distances[device::swizzling(j)] = key[i];
          itopk_indices[device::swizzling(j)]   = val[i];
        }
      }
    }
    const uint32_t num_itopk_div2 = num_itopk / 2;
    if (threadIdx.x < 3) {
      // work_buf is used to obtain turning points in 1st and 2nd half of itopk afer merge.
      work_buf[threadIdx.x] = num_itopk_div2;
    }
    __syncthreads();

    // Merge candidates (using whole threads)
    for (unsigned k = threadIdx.x; k < (num_candidates < num_itopk ? num_candidates : num_itopk);
         k += blockDim.x) {
      const unsigned j          = num_itopk - 1 - k;
      const float itopk_key     = itopk_distances[device::swizzling(j)];
      const float candidate_key = candidate_distances[device::swizzling(k)];
      if (itopk_key > candidate_key) {
        itopk_distances[device::swizzling(j)] = candidate_key;
        itopk_indices[device::swizzling(j)]   = candidate_indices[device::swizzling(k)];
        if (j < num_itopk_div2) {
          atomicMin(work_buf + 2, j);
        } else {
          atomicMin(work_buf + 1, j - num_itopk_div2);
        }
      }
    }
    __syncthreads();

    // Merge 1st and 2nd half of itopk (using whole threads)
    for (unsigned j = threadIdx.x; j < num_itopk_div2; j += blockDim.x) {
      const unsigned k = j + num_itopk_div2;
      float key_0      = itopk_distances[device::swizzling(j)];
      float key_1      = itopk_distances[device::swizzling(k)];
      if (key_0 > key_1) {
        itopk_distances[device::swizzling(j)] = key_1;
        itopk_distances[device::swizzling(k)] = key_0;
        IdxT val_0                            = itopk_indices[device::swizzling(j)];
        IdxT val_1                            = itopk_indices[device::swizzling(k)];
        itopk_indices[device::swizzling(j)]   = val_1;
        itopk_indices[device::swizzling(k)]   = val_0;
        atomicMin(work_buf + 0, j);
      }
    }
    if (threadIdx.x == blockDim.x - 1) {
      if (work_buf[2] < num_itopk_div2) { work_buf[1] = work_buf[2]; }
    }
    __syncthreads();
    // Warp-0 merges 1st half of itopk, warp-1 does 2nd half.
    if (warp_id < 2) {
      // Load intermedidate itopk results
      const uint32_t turning_point = work_buf[warp_id];  // turning_point <= num_itopk_div2
      for (unsigned i = 0; i < N; i++) {
        unsigned k = num_itopk;
        unsigned j = (N * lane_id) + i;
        if (j < turning_point) {
          k = j + (num_itopk_div2 * warp_id);
        } else if (j >= (MAX_ITOPK / 2 - num_itopk_div2)) {
          j -= (MAX_ITOPK / 2 - num_itopk_div2);
          if ((turning_point <= j) && (j < num_itopk_div2)) { k = j + (num_itopk_div2 * warp_id); }
        }
        if (k < num_itopk) {
          key[i] = itopk_distances[device::swizzling(k)];
          val[i] = itopk_indices[device::swizzling(k)];
        } else {
          key[i] = utils::get_max_value<float>();
          val[i] = utils::get_max_value<IdxT>();
        }
      }
      /* Warp Merge */
      bitonic::warp_merge<float, IdxT, N>(key, val, raft::warp_size());
      /* Store new itopk results */
      for (unsigned i = 0; i < N; i++) {
        const unsigned j = (N * lane_id) + i;
        if (j < num_itopk_div2) {
          unsigned k                            = j + (num_itopk_div2 * warp_id);
          itopk_distances[device::swizzling(k)] = key[i];
          itopk_indices[device::swizzling(k)]   = val[i];
        }
      }
    }
  }
}

// Wrapper functions to avoid pre-inlining
RAFT_DEVICE_INLINE_FUNCTION void topk_by_bitonic_sort_and_merge_wrapper_64_false(
  float* itopk_distances,   // [num_itopk]
  uint32_t* itopk_indices,  // [num_itopk]
  const std::uint32_t num_itopk,
  float* candidate_distances,   // [num_candidates]
  uint32_t* candidate_indices,  // [num_candidates]
  const std::uint32_t num_candidates,
  std::uint32_t* work_buf,
  const bool first)
{
  topk_by_bitonic_sort_and_merge<64, false, uint32_t>(itopk_distances,
                                                      itopk_indices,
                                                      num_itopk,
                                                      candidate_distances,
                                                      candidate_indices,
                                                      num_candidates,
                                                      work_buf,
                                                      first);
}

RAFT_DEVICE_INLINE_FUNCTION void topk_by_bitonic_sort_and_merge_wrapper_128_false(
  float* itopk_distances,   // [num_itopk]
  uint32_t* itopk_indices,  // [num_itopk]
  const std::uint32_t num_itopk,
  float* candidate_distances,   // [num_candidates]
  uint32_t* candidate_indices,  // [num_candidates]
  const std::uint32_t num_candidates,
  std::uint32_t* work_buf,
  const bool first)
{
  topk_by_bitonic_sort_and_merge<128, false, uint32_t>(itopk_distances,
                                                       itopk_indices,
                                                       num_itopk,
                                                       candidate_distances,
                                                       candidate_indices,
                                                       num_candidates,
                                                       work_buf,
                                                       first);
}

RAFT_DEVICE_INLINE_FUNCTION void topk_by_bitonic_sort_and_merge_wrapper_256_false(
  float* itopk_distances,   // [num_itopk]
  uint32_t* itopk_indices,  // [num_itopk]
  const std::uint32_t num_itopk,
  float* candidate_distances,   // [num_candidates]
  uint32_t* candidate_indices,  // [num_candidates]
  const std::uint32_t num_candidates,
  std::uint32_t* work_buf,
  const bool first)
{
  topk_by_bitonic_sort_and_merge<256, false, uint32_t>(itopk_distances,
                                                       itopk_indices,
                                                       num_itopk,
                                                       candidate_distances,
                                                       candidate_indices,
                                                       num_candidates,
                                                       work_buf,
                                                       first);
}

// TopK by bitonic sort and merge (runtime version)
template <bool MULTI_WARPS, class IdxT>
RAFT_DEVICE_INLINE_FUNCTION void topk_by_bitonic_sort_and_merge(
  float* itopk_distances,  // [num_itopk]
  IdxT* itopk_indices,     // [num_itopk]
  const std::uint32_t max_itopk,
  const std::uint32_t num_itopk,
  float* candidate_distances,  // [num_candidates]
  IdxT* candidate_indices,     // [num_candidates]
  const std::uint32_t max_candidates,
  const std::uint32_t num_candidates,
  std::uint32_t* work_buf,
  const bool first)
{
  static_assert(std::is_same_v<IdxT, uint32_t>);
  assert(max_itopk <= 512);
  assert(max_candidates <= 256);
  assert(!MULTI_WARPS || blockDim.x >= 64);

  // use a non-template wrapper function to avoid pre-inlining the topk_by_bitonic_sort_and_full
  // function (vs post-inlining, this impacts register pressure)
  if (max_candidates <= 64) {
    topk_by_bitonic_sort_and_full_wrapper_64_false(
      candidate_distances, candidate_indices, num_candidates, num_itopk);
  } else if (max_candidates <= 128) {
    topk_by_bitonic_sort_and_full_wrapper_128_false(
      candidate_distances, candidate_indices, num_candidates, num_itopk);
  } else {
    topk_by_bitonic_sort_and_full_wrapper_256_false(
      candidate_distances, candidate_indices, num_candidates, num_itopk);
  }

  if constexpr (!MULTI_WARPS) {
    assert(max_itopk <= 256);
    // use a non-template wrapper function to avoid pre-inlining the topk_by_bitonic_sort_and_merge
    // function (vs post-inlining, this impacts register pressure)
    if (max_itopk <= 64) {
      topk_by_bitonic_sort_and_merge_wrapper_64_false(itopk_distances,
                                                      itopk_indices,
                                                      num_itopk,
                                                      candidate_distances,
                                                      candidate_indices,
                                                      num_candidates,
                                                      work_buf,
                                                      first);
    } else if (max_itopk <= 128) {
      topk_by_bitonic_sort_and_merge_wrapper_128_false(itopk_distances,
                                                       itopk_indices,
                                                       num_itopk,
                                                       candidate_distances,
                                                       candidate_indices,
                                                       num_candidates,
                                                       work_buf,
                                                       first);
    } else {
      topk_by_bitonic_sort_and_merge_wrapper_256_false(itopk_distances,
                                                       itopk_indices,
                                                       num_itopk,
                                                       candidate_distances,
                                                       candidate_indices,
                                                       num_candidates,
                                                       work_buf,
                                                       first);
    }
  } else {
    assert(max_itopk > 256);
    topk_by_bitonic_sort_and_merge<512, MULTI_WARPS, uint32_t>(itopk_distances,
                                                               itopk_indices,
                                                               num_itopk,
                                                               candidate_distances,
                                                               candidate_indices,
                                                               num_candidates,
                                                               work_buf,
                                                               first);
  }
}

// This function move the invalid index element to the end of the itopk list.
// Require : array_length % 32 == 0 && The invalid entry is only one.
template <class IdxT>
RAFT_DEVICE_INLINE_FUNCTION void move_invalid_to_end_of_list(IdxT* const index_array,
                                                             float* const distance_array,
                                                             const std::uint32_t array_length)
{
  constexpr std::uint32_t warp_size     = 32;
  constexpr std::uint32_t invalid_index = utils::get_max_value<IdxT>();
  const std::uint32_t lane_id           = threadIdx.x % warp_size;

  if (threadIdx.x >= warp_size) { return; }

  bool found_invalid = false;
  if (array_length % warp_size == 0) {
    for (std::uint32_t i = lane_id; i < array_length; i += warp_size) {
      const auto index    = index_array[i];
      const auto distance = distance_array[i];

      if (found_invalid) {
        index_array[i - 1]    = index;
        distance_array[i - 1] = distance;
      } else {
        // Check if the index is invalid
        const auto I_found_invalid = (index == invalid_index);
        const auto who_has_invalid = raft::ballot(I_found_invalid);
        // if a value that is loaded by a smaller lane id thread, shift the array
        if (who_has_invalid << (warp_size - lane_id)) {
          index_array[i - 1]    = index;
          distance_array[i - 1] = distance;
        }

        found_invalid = who_has_invalid;
      }
    }
  }
  if (lane_id == 0) {
    index_array[array_length - 1]    = invalid_index;
    distance_array[array_length - 1] = utils::get_max_value<float>();
  }
}

template <class INDEX_T>
RAFT_DEVICE_INLINE_FUNCTION void hashmap_restore(INDEX_T* const hashmap_ptr,
                                                 const size_t hashmap_bitlen,
                                                 const INDEX_T* itopk_indices,
                                                 const uint32_t itopk_size,
                                                 const uint32_t first_tid = 0)
{
  constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;
  if (threadIdx.x < first_tid) return;
  for (unsigned i = threadIdx.x - first_tid; i < itopk_size; i += blockDim.x - first_tid) {
    auto key = itopk_indices[i] & ~index_msb_1_mask;  // clear most significant bit
    hashmap::insert(hashmap_ptr, hashmap_bitlen, key);
  }
}

}  // namespace cuvs::neighbors::cagra::detail::single_cta_search
