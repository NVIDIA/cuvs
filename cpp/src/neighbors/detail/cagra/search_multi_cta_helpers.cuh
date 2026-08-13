/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <neighbors/detail/cagra/bitonic.hpp>
#include <neighbors/detail/cagra/hashmap.hpp>
#include <neighbors/detail/cagra/utils.hpp>
#include <raft/core/operators.hpp>

namespace cuvs::neighbors::cagra::detail::multi_cta_search {

template <class INDEX_T, class DISTANCE_T>
RAFT_DEVICE_INLINE_FUNCTION void pickup_next_parent(
  INDEX_T* const next_parent_indices,
  INDEX_T* const itopk_indices,       // [itopk_size * 2]
  DISTANCE_T* const itopk_distances,  // [itopk_size * 2]
  INDEX_T* const hash_ptr,
  const uint32_t hash_bitlen)
{
  constexpr uint32_t itopk_size      = 32;
  constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;
  constexpr INDEX_T invalid_index    = ~static_cast<INDEX_T>(0);

  const unsigned warp_id = threadIdx.x / 32;
  if (warp_id > 0) { return; }
  if (threadIdx.x == 0) { next_parent_indices[0] = invalid_index; }
  __syncwarp();

  int j = -1;
  for (unsigned i = threadIdx.x; i < itopk_size * 2; i += 32) {
    INDEX_T index    = itopk_indices[i];
    int is_invalid   = 0;
    int is_candidate = 0;
    if (index == invalid_index) {
      is_invalid = 1;
    } else if (index & index_msb_1_mask) {
    } else {
      is_candidate = 1;
    }

    const auto ballot_mask  = __ballot_sync(0xffffffff, is_candidate);
    const auto candidate_id = __popc(ballot_mask & ((1 << threadIdx.x) - 1));
    for (int k = 0; k < __popc(ballot_mask); k++) {
      int flag_done = 0;
      if (is_candidate && candidate_id == k) {
        is_candidate = 0;
        if (hashmap::insert<INDEX_T, 1>(hash_ptr, hash_bitlen, index)) {
          // Use this candidate as next parent
          index |= index_msb_1_mask;  // set most significant bit as used node
          if (i < itopk_size) {
            next_parent_indices[0] = i;
            itopk_indices[i]       = index;
          } else {
            next_parent_indices[0] = j;
            // Move the next parent node from i-th position to j-th position
            itopk_indices[j]   = index;
            itopk_distances[j] = itopk_distances[i];
            itopk_indices[i]   = invalid_index;
            itopk_distances[i] = utils::get_max_value<DISTANCE_T>();
          }
          flag_done = 1;
        } else {
          // Deactivate the node since it has been used by other CTA.
          itopk_indices[i]   = invalid_index;
          itopk_distances[i] = utils::get_max_value<DISTANCE_T>();
          is_invalid         = 1;
        }
      }
      if (__any_sync(0xffffffff, (flag_done > 0))) { return; }
    }
    if (i < itopk_size) {
      j = 31 - __clz(__ballot_sync(0xffffffff, is_invalid));
      if (j < 0) { return; }
    }
  }
}

template <unsigned MAX_ELEMENTS, class INDEX_T>
RAFT_DEVICE_INLINE_FUNCTION void topk_by_bitonic_sort(float* distances,  // [num_elements]
                                                      INDEX_T* indices,  // [num_elements]
                                                      const uint32_t num_elements)
{
  const unsigned warp_id = threadIdx.x / raft::warp_size();
  if (warp_id > 0) { return; }
  const unsigned lane_id = threadIdx.x % raft::warp_size();
  constexpr unsigned N   = (MAX_ELEMENTS + (raft::warp_size() - 1)) / raft::warp_size();
  float key[N];
  INDEX_T val[N];
  for (unsigned i = 0; i < N; i++) {
    unsigned j = lane_id + (raft::warp_size() * i);
    if (j < num_elements) {
      key[i] = distances[j];
      val[i] = indices[j];
    } else {
      key[i] = utils::get_max_value<float>();
      val[i] = ~static_cast<INDEX_T>(0);
    }
  }
  /* Warp Sort */
  bitonic::warp_sort<float, INDEX_T, N>(key, val);
  /* Store sorted results */
  for (unsigned i = 0; i < N; i++) {
    unsigned j = (N * lane_id) + i;
    if (j < num_elements) {
      distances[j] = key[i];
      indices[j]   = val[i];
    }
  }
}

RAFT_DEVICE_INLINE_FUNCTION void topk_by_bitonic_sort_wrapper_64(
  float* distances,   // [num_elements]
  uint32_t* indices,  // [num_elements]
  const uint32_t num_elements)
{
  topk_by_bitonic_sort<64, uint32_t>(distances, indices, num_elements);
}

RAFT_DEVICE_INLINE_FUNCTION void topk_by_bitonic_sort_wrapper_128(
  float* distances,   // [num_elements]
  uint32_t* indices,  // [num_elements]
  const uint32_t num_elements)
{
  topk_by_bitonic_sort<128, uint32_t>(distances, indices, num_elements);
}

RAFT_DEVICE_INLINE_FUNCTION void topk_by_bitonic_sort_wrapper_256(
  float* distances,   // [num_elements]
  uint32_t* indices,  // [num_elements]
  const uint32_t num_elements)
{
  topk_by_bitonic_sort<256, uint32_t>(distances, indices, num_elements);
}

}  // namespace cuvs::neighbors::cagra::detail::multi_cta_search
