/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cub/cub.cuh>

#include "macros.cuh"
#include "priority_queue.cuh"
#include "vamana_structs.cuh"
#include <cuvs/neighbors/vamana.hpp>

#include <cuvs/distance/distance.hpp>
#include <raft/util/warp_primitives.cuh>
#include <rmm/resource_ref.hpp>

#include <chrono>
#include <cstdio>
#include <type_traits>
#include <vector>

namespace cuvs::neighbors::vamana::detail {

/* @defgroup greedy_search_detail greedy search
 * @{
 */

/* Combines edge and candidate lists, removes duplicates, and sorts by distance
 * Uses CUB primitives, so needs to be templated. Called with Macros for supported sizes above */
template <typename accT, typename IdxT, int CANDS>
__forceinline__ __device__ void sort_visited(
  QueryCandidates<IdxT, accT>* query,
  typename cub::BlockMergeSort<DistPair<IdxT, accT>, 32, (CANDS / 32)>::TempStorage* sort_mem)
{
  const int ELTS   = CANDS / 32;
  using BlockSortT = cub::BlockMergeSort<DistPair<IdxT, accT>, 32, ELTS>;
  DistPair<IdxT, accT> tmp[ELTS];
  for (int i = 0; i < ELTS; i++) {
    tmp[i].idx  = query->ids[ELTS * threadIdx.x + i];
    tmp[i].dist = query->dists[ELTS * threadIdx.x + i];
  }

  __syncthreads();
  BlockSortT(*sort_mem).Sort(tmp, CmpDist<IdxT, accT>());
  __syncthreads();

  for (int i = 0; i < ELTS; i++) {
    query->ids[ELTS * threadIdx.x + i]   = tmp[i].idx;
    query->dists[ELTS * threadIdx.x + i] = tmp[i].dist;
  }
  __syncthreads();
}

namespace {

template <typename T, typename accT, typename IdxT = uint32_t>
__global__ void SortPairsKernel(void* query_list_ptr, int num_queries, int topk)
{
  union ShmemLayout {
    typename cub::BlockMergeSort<DistPair<IdxT, accT>, 32, 1>::TempStorage sort_mem;
  };
  extern __shared__ __align__(alignof(ShmemLayout)) char smem[];

  QueryCandidates<IdxT, accT>* query_list =
    static_cast<QueryCandidates<IdxT, accT>*>(query_list_ptr);

  for (int i = blockIdx.x; i < num_queries; i += gridDim.x) {
    __syncthreads();
    SEARCH_SELECT_SORT(topk);
  }
}

/********************************************************************************************
  GPU kernel to perform a batched GreedySearch on a graph. Since this is used for
  Vamana construction, the entire visited list is kept and stored within the query_list.
  Uses 128 threads per block (4 warps), each warp processes one query independently
  with per-warp scratch space to avoid block synchronization overhead.

  Input - graph with edge lists, dataset vectors, query_list_ptr with the ids of dataset
          vectors to be searched. All inputs, including dataset,  must be device accessible.

  Output - the id and dist lists in query_list_ptr will be updated with the nodes visited
           during the GreedySearch.
**********************************************************************************************/
template <typename T,
          typename accT,
          typename IdxT = uint32_t,
          typename Accessor =
            raft::host_device_accessor<cuda::std::default_accessor<T>, raft::memory_type::host>>
__global__ __launch_bounds__(128, 12) void GreedySearchKernel(
  raft::device_matrix_view<IdxT, int64_t> graph,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
  void* query_list_ptr,
  int num_queries,
  int medoid_id,
  int topk,
  cuvs::distance::DistanceType metric,
  int max_queue_size,
  Node<accT>* topk_pq_mem)
{
  const int warpIdx = threadIdx.x / 32;
  const int laneId  = threadIdx.x % 32;

  int dim    = dataset.extent(1);
  int degree = graph.extent(1);

  QueryCandidates<IdxT, accT>* query_list =
    static_cast<QueryCandidates<IdxT, accT>*>(query_list_ptr);

  using QueryCoordT = typename greedy_search_query_coord<T>::type;

  int align_padding = raft::alignTo(dim, 16) - dim;

  // Only use fp16 coords in shared memory if type is fp16 and dim >= 512
  const bool fp16_query_smem = greedy_search_use_fp16_query_smem<T>(dim);

  extern __shared__ __align__(16) char smem[];

  // Per-warp shared memory layout: coords, neighbor_array, candidate_queue
  const int coords_size      = (dim + align_padding) * greedy_search_query_smem_elem_size<T>(dim);
  const int neighbor_size    = degree * sizeof(IdxT);
  const int queue_size_bytes = max_queue_size * sizeof(DistPair<IdxT, accT>);
  const int per_warp_size    = (coords_size + neighbor_size + queue_size_bytes + 15) & ~15;

  char* warp_smem       = &smem[warpIdx * per_warp_size];
  __half* s_coords_half = reinterpret_cast<__half*>(warp_smem);
  QueryCoordT* s_coords = reinterpret_cast<QueryCoordT*>(warp_smem);
  IdxT* neighbor_array  = reinterpret_cast<IdxT*>(warp_smem + coords_size);
  DistPair<IdxT, accT>* candidate_queue_smem =
    reinterpret_cast<DistPair<IdxT, accT>*>(warp_smem + coords_size + neighbor_size);

  // 4 warps per block
  static __shared__ int topk_q_size[4];
  static __shared__ int cand_q_size[4];
  static __shared__ accT cur_k_max[4];
  static __shared__ int k_max_idx[4];
  static __shared__ int num_neighbors[4];

  // Different code path for fp16 since it is gated by dim and datatype
  Point<__half, accT> s_query_half;
  Point<QueryCoordT, accT> s_query;
  if (fp16_query_smem) {
    s_query_half.Dim    = dim;
    s_query_half.coords = s_coords_half;
  } else {
    s_query.Dim    = dim;
    s_query.coords = s_coords;
  }

  PriorityQueue<IdxT, accT> heap_queue;
  if (laneId == 0) {
    heap_queue.initialize(candidate_queue_smem, max_queue_size, &cand_q_size[warpIdx]);
  }

  Node<accT>* topk_pq = &topk_pq_mem[(blockIdx.x * 4 + warpIdx) * topk];
  const T* vec_ptr    = &dataset(0, 0);

  for (int i = blockIdx.x * 4 + warpIdx; i < num_queries; i += gridDim.x * 4) {
    query_list[i].reset_warp(laneId);

    int cur_query_id = query_list[i].queryId;
    if (fp16_query_smem) {
      if constexpr (is_cuda_fp16_v<T>) {
        update_shared_point_warp_fp16_query_smem<accT>(
          &s_query_half, vec_ptr, cur_query_id, dim, laneId);
      } else if constexpr (std::is_same_v<T, float>) {
        update_shared_point_warp_fp16_query_smem<accT>(
          &s_query_half, vec_ptr, cur_query_id, dim, laneId);
      } else {
        update_shared_point_warp_fp16_query_smem<T, accT>(
          &s_query_half, vec_ptr, cur_query_id, dim, laneId);
      }
    } else if constexpr (is_cuda_fp16_v<T>) {
      update_shared_point_warp_half_to_float<accT>(&s_query, vec_ptr, cur_query_id, dim, laneId);
    } else {
      update_shared_point_warp<T, accT>(&s_query, vec_ptr, cur_query_id, dim, laneId);
    }

    if (laneId == 0) {
      topk_q_size[warpIdx] = 0;
      cand_q_size[warpIdx] = 0;
      if (fp16_query_smem) {
        s_query_half.id = cur_query_id;
      } else {
        s_query.id = cur_query_id;
      }
      cur_k_max[warpIdx] = 0;
      k_max_idx[warpIdx] = 0;
      heap_queue.reset();
    }

    accT medoid_dist;
    if (fp16_query_smem) {
      medoid_dist = dist_warp_half_query<accT, T>(
        s_coords_half, &vec_ptr[(size_t)medoid_id * (size_t)dim], dim, metric, laneId);
    } else if constexpr (is_cuda_fp16_v<T>) {
      medoid_dist =
        dist_warp<accT>(s_coords, &vec_ptr[(size_t)medoid_id * (size_t)dim], dim, metric, laneId);
    } else {
      medoid_dist = dist_warp<T, accT>(
        s_coords, &vec_ptr[(size_t)medoid_id * (size_t)dim], dim, metric, laneId);
    }

    if (laneId == 0) { heap_queue.insert_back(medoid_dist, medoid_id); }

    while (cand_q_size[warpIdx] != 0) {
      int cand_num;
      accT cur_distance;
      if (laneId == 0) {
        DistPair<IdxT, accT> test_cand_out = heap_queue.pop();
        cand_num                           = test_cand_out.idx;
        cur_distance                       = test_cand_out.dist;
      }
      cand_num     = raft::shfl(cand_num, 0);
      cur_distance = raft::shfl(cur_distance, 0);

      if (query_list[i].check_visited_warp(cand_num, cur_distance, laneId)) { continue; }

      bool done      = false;
      bool pass_flag = false;

      if (topk_q_size[warpIdx] == topk) {
        if (laneId == 0) {
          if (cur_k_max[warpIdx] <= cur_distance) { done = true; }
        }
        done = raft::shfl(done, 0);
        if (done) {
          if (query_list[i].size < topk) {
            pass_flag = true;
          } else if (query_list[i].size >= topk) {
            break;
          }
        }
      }

      Node<accT> new_cand;
      new_cand.distance = cur_distance;
      new_cand.nodeid   = cand_num;

      if (check_duplicate_warp(topk_pq, topk_q_size[warpIdx], new_cand, laneId) == false) {
        if (!pass_flag) {
          parallel_pq_max_enqueue_warp<accT>(topk_pq,
                                             &topk_q_size[warpIdx],
                                             topk,
                                             new_cand,
                                             &cur_k_max[warpIdx],
                                             &k_max_idx[warpIdx],
                                             laneId);
        }
      } else {
        continue;
      }

      num_neighbors[warpIdx] = degree;

      for (size_t j = laneId; j < degree; j += 32) {
        neighbor_array[j] = graph(cand_num, j);
        if (neighbor_array[j] == raft::upper_bound<IdxT>())
          atomicMin(&num_neighbors[warpIdx],
                    (int)j);  // warp-wide min to find the number of neighbors
      }

      enqueue_all_neighbors_warp(num_neighbors[warpIdx],
                                 fp16_query_smem,
                                 s_coords_half,
                                 s_coords,
                                 vec_ptr,
                                 neighbor_array,
                                 heap_queue,
                                 dim,
                                 metric,
                                 laneId);
    }

    bool self_found = false;
    for (int j = laneId; j < query_list[i].size; j += 32) {
      if (query_list[i].ids[j] == cur_query_id) {
        query_list[i].dists[j] = raft::upper_bound<accT>();
        query_list[i].ids[j]   = raft::upper_bound<IdxT>();
        self_found             = true;  // Flat to reduce size by 1
      }
    }
    self_found = (raft::ballot(self_found) != 0);

    for (int j = query_list[i].size + laneId; j < query_list[i].maxSize; j += 32) {
      query_list[i].ids[j]   = raft::upper_bound<IdxT>();
      query_list[i].dists[j] = raft::upper_bound<accT>();
    }

    if (self_found && laneId == 0) { query_list[i].size--; }
  }

  return;
}

}  // namespace

/**
 * @}
 */

}  // namespace cuvs::neighbors::vamana::detail
