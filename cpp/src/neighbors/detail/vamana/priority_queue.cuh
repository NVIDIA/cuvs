/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "vamana_structs.cuh"
#include <raft/util/warp_primitives.cuh>
#include <stdio.h>

namespace cuvs::neighbors::vamana::detail {

/***************************************************************************************
***************************************************************************************/
/**
 * @defgroup vamana_priority_queue Vamana Priority queue structure
 * @{
 */

/**
 * @brief Priority Queue structure used by Vamana GreedySearch during
 * graph construction.
 *
 * The structure keeps the nearest visited neighbors seen thus far during
 * search and lets us efficiently find the next node to visit during the search.
 * Stores a total of KVAL pairs, where currently KVAL must be 2i-1 for some integer
 * i since the heap must be complete.
 * This size is determined during vamana build with the "queue_size" parameter (default 127)
 *
 * The queue and all methods are device-side, with a work group size or 32 (one warp).
 * During search, each warp creates their own queue to search a single query at a time.
 * The device memory pointed to by `vals` is assigned during the call to `initialize`.
 * The Vamana GreedySearch call uses shared memory, but any device-accessible memory is applicable.
 *
 *
 * @tparam IdxT type of the vector indices (represent dataset.extent(0))
 * @tparam accT type of distances between vectors (accumuator type)
 *
 */
template <typename IdxT, typename accT>
class PriorityQueue {
 public:
  int KVAL;
  int insert_pointer;
  DistPair<IdxT, accT>* vals;
  DistPair<IdxT, accT> temp;

  int* q_size;
  // Enforce max-heap property on the entries
  __forceinline__ __device__ void heapify()
  {
    int i        = 0;
    int swapDest = 0;

    while (2 * i + 2 < KVAL) {
      swapDest = 2 * i;
      swapDest +=
        (vals[i].dist > vals[2 * i + 1].dist && vals[2 * i + 2].dist >= vals[2 * i + 1].dist);
      swapDest +=
        2 * (vals[i].dist > vals[2 * i + 2].dist && vals[2 * i + 1].dist > vals[2 * i + 2].dist);

      if (swapDest == 2 * i) return;

      swap(&vals[i], &vals[swapDest]);

      i = swapDest;
    }
  }

  // Starts the heapify process starting at a particular index
  __forceinline__ __device__ void heapifyAt(int idx)
  {
    int i        = idx;
    int swapDest = 0;

    while (2 * i + 2 < KVAL) {
      swapDest = 2 * i;
      swapDest +=
        (vals[i].dist > vals[2 * i + 1].dist && vals[2 * i + 2].dist <= vals[2 * i + 1].dist);
      swapDest +=
        2 * (vals[i].dist > vals[2 * i + 2].dist && vals[2 * i + 1].dist < vals[2 * i + 2].dist);

      if (swapDest == 2 * i) return;

      swap(&vals[i], &vals[swapDest]);
      i = swapDest;
    }
  }

  // Heapify from the bottom up, used with insert_back
  __forceinline__ __device__ void heapifyReverseAt(int idx)
  {
    int i        = idx;
    int swapDest = 0;
    while (i > 0) {
      swapDest = ((i - 1) / 2);
      if (vals[swapDest].dist <= vals[i].dist) return;

      swap(&vals[i], &vals[swapDest]);
      i = swapDest;
    }
  }

  __device__ void reset()
  {
    *q_size = 0;
    for (int i = 0; i < KVAL; i++) {
      vals[i].dist = raft::upper_bound<accT>();
      vals[i].idx  = raft::upper_bound<IdxT>();
    }
  }

  __device__ void initialize(DistPair<IdxT, accT>* v, int _kval, int* _q_size)
  {
    vals           = v;
    KVAL           = _kval;
    insert_pointer = _kval / 2;
    q_size         = _q_size;
    reset();
  }

  // Initialize all nodes of the heap to +infinity
  __device__ void initialize()
  {
    for (int i = 0; i < KVAL; i++) {
      vals[i].idx  = raft::upper_bound<IdxT>();
      vals[i].dist = raft::upper_bound<accT>();
    }
  }

  __device__ void write_to_gmem(int* gmem)
  {
    for (int i = 0; i < KVAL; i++) {
      gmem[i] = vals[i].idx;
    }
  }

  // Replace the root of the heap with new pair
  __device__ void insert(accT newDist, IdxT newIdx)
  {
    vals[0].dist = newDist;
    vals[0].idx  = newIdx;

    heapify();
  }

  // Replace a specific element in the heap (and maintain heap properties)
  __device__ void insertAt(accT newDist, IdxT newIdx, int idx)
  {
    vals[idx].dist = newDist;
    vals[idx].idx  = newIdx;

    heapifyAt(idx);
  }

  // Return value of the root of the heap (largest value)
  __device__ accT top() { return vals[0].dist; }

  __device__ IdxT top_node() { return vals[0].idx; }

  __device__ void insert_back(accT newDist, IdxT newIdx)
  {
    if (newDist < vals[insert_pointer].dist) {
      if (vals[insert_pointer].idx == raft::upper_bound<IdxT>()) *q_size += 1;
      vals[insert_pointer].dist = newDist;
      vals[insert_pointer].idx  = newIdx;
      heapifyReverseAt(insert_pointer);
    }
    insert_pointer++;

    if (insert_pointer == KVAL) insert_pointer = KVAL / 2;
  }

  // Pop root node off and heapify
  __device__ DistPair<IdxT, accT> pop()
  {
    DistPair<IdxT, accT> result;
    result.dist  = vals[0].dist;
    result.idx   = vals[0].idx;
    vals[0].dist = raft::upper_bound<accT>();
    vals[0].idx  = raft::upper_bound<IdxT>();
    heapify();
    *q_size -= 1;
    return result;
  }
};

/***************************************************************************************
 * Node structure used for simplified lists during GreedySearch.
 * Used for other operations like checking for duplicates, etc.
 ****************************************************************************************/
template <typename SUMTYPE>
class __align__(16) Node {
 public:
  SUMTYPE distance;
  int nodeid;
};

// Less-than operator between two Nodes.
template <typename SUMTYPE>
__host__ __device__ bool operator<(const Node<SUMTYPE>& first, const Node<SUMTYPE>& other)
{
  return first.distance < other.distance;
}

// Less-than operator between two Nodes.
template <typename SUMTYPE>
__host__ __device__ bool operator>(const Node<SUMTYPE>& first, const Node<SUMTYPE>& other)
{
  return first.distance > other.distance;
}

// each warp scans its own pq with laneId and stride 32 to find duplicates
template <typename accT>
__device__ bool check_duplicate_warp(const Node<accT>* pq,
                                     const int size,
                                     Node<accT> new_node,
                                     int laneId)
{
  bool found = false;
  for (int i = laneId; i < size; i += 32) {
    if (pq[i].nodeid == new_node.nodeid) {
      found = true;
      break;
    }
  }
  unsigned mask = raft::ballot(found);
  return (mask != 0);
}

// Warp-level version: no __syncthreads, uses laneId for single-thread ops and warp shuffle
template <typename SUMTYPE>
__inline__ __device__ void parallel_pq_max_enqueue_warp(Node<SUMTYPE>* pq,
                                                        int* size,
                                                        const int pq_size,
                                                        Node<SUMTYPE> input_data,
                                                        SUMTYPE* cur_max_val,
                                                        int* max_idx,
                                                        int laneId)
{
  if (*size < pq_size) {
    if (laneId == 0) {
      pq[*size].distance = input_data.distance;
      pq[*size].nodeid   = input_data.nodeid;
      *size              = *size + 1;
      if (input_data.distance > (*cur_max_val)) {
        *cur_max_val = input_data.distance;
        *max_idx     = *size - 1;
      }
    }
    return;
  } else {
    if (input_data.distance >= (*cur_max_val)) { return; }
    if (laneId == 0) {
      pq[*max_idx].distance = input_data.distance;
      pq[*max_idx].nodeid   = input_data.nodeid;
    }
    int idx         = 0;
    SUMTYPE max_val = pq[0].distance;

    for (int i = laneId; i < pq_size; i += 32) {
      if (pq[i].distance > max_val) {
        max_val = pq[i].distance;
        idx     = i;
      }
    }

    for (int offset = 16; offset > 0; offset /= 2) {
      SUMTYPE new_max_val = raft::shfl_up(max_val, offset);
      int new_idx         = raft::shfl_up(idx, offset);
      if (new_max_val > max_val) {
        max_val = new_max_val;
        idx     = new_idx;
      }
    }

    if (laneId == 31) {
      *max_idx     = idx;
      *cur_max_val = max_val;
    }
  }
}

// Warp-level version: lane 0 does insert_back, no __syncthreads
template <typename QueryT, typename DataT, typename accT, typename IdxT>
__forceinline__ __device__ void enqueue_all_neighbors_warp(int num_neighbors,
                                                           Point<QueryT, accT>* query_vec,
                                                           const DataT* vec_ptr,
                                                           IdxT* neighbor_array,
                                                           PriorityQueue<IdxT, accT>& heap_queue,
                                                           int dim,
                                                           cuvs::distance::DistanceType metric,
                                                           int laneId)
{
  for (int i = 0; i < num_neighbors; i++) {
    const DataT* neighbor_vec = &vec_ptr[(size_t)(neighbor_array[i]) * (size_t)(dim)];
    accT dist_out;
    if constexpr (std::is_same_v<QueryT, __half>) {
      dist_out =
        dist_warp_half_query<accT, DataT>(query_vec->coords, neighbor_vec, dim, metric, laneId);
    } else if constexpr (std::is_same_v<QueryT, float> && is_cuda_fp16_v<DataT>) {
      dist_out = dist_warp<accT>(query_vec->coords, neighbor_vec, dim, metric, laneId);
    } else {
      static_assert(std::is_same_v<QueryT, DataT>);
      dist_out = dist_warp<QueryT, accT>(query_vec->coords, neighbor_vec, dim, metric, laneId);
    }
    if (laneId == 0) { heap_queue.insert_back(dist_out, neighbor_array[i]); }
  }
}

// Half-precision version with two code paths based on query being fp16 or fp32
template <typename T, typename accT, typename IdxT>
__forceinline__ __device__ void enqueue_all_neighbors_warp(
  int num_neighbors,
  bool fp16_query_smem,
  __half* s_coords_half,
  typename greedy_search_query_coord<T>::type* s_coords,
  const T* vec_ptr,
  IdxT* neighbor_array,
  PriorityQueue<IdxT, accT>& heap_queue,
  int dim,
  cuvs::distance::DistanceType metric,
  int laneId)
{
  if (fp16_query_smem) {
    Point<__half, accT> query_vec;
    query_vec.coords = s_coords_half;
    query_vec.Dim    = dim;
    enqueue_all_neighbors_warp<__half, T, accT, IdxT>(
      num_neighbors, &query_vec, vec_ptr, neighbor_array, heap_queue, dim, metric, laneId);
  } else if constexpr (is_cuda_fp16_v<T>) {
    Point<float, accT> query_vec;
    query_vec.coords = reinterpret_cast<float*>(s_coords);
    query_vec.Dim    = dim;
    enqueue_all_neighbors_warp<float, T, accT, IdxT>(
      num_neighbors, &query_vec, vec_ptr, neighbor_array, heap_queue, dim, metric, laneId);
  } else {
    Point<T, accT> query_vec;
    query_vec.coords = reinterpret_cast<T*>(s_coords);
    query_vec.Dim    = dim;
    enqueue_all_neighbors_warp<T, T, accT, IdxT>(
      num_neighbors, &query_vec, vec_ptr, neighbor_array, heap_queue, dim, metric, laneId);
  }
}

}  // namespace cuvs::neighbors::vamana::detail
