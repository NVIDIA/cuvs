/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_fp16.h>
#include <type_traits>

namespace cuvs::neighbors::vamana::detail {

// RobustPrune wide-dim optimizations: smem candidate cache and GreedySearch distance reuse.
static constexpr int kRobustPruneCandCacheMinDim = 128;

// Minimum dim for 8 warps on degree-64 occlusion sweep (half). Below this, barrier tax wins.
static constexpr int kRobustPruneMultiWarpMinDimHalf = 960;

// Occlusion sweep is multi-warp (occId += num_warps). Extra warps hide wide-dim djk latency,
// but narrow dim, low degree, and byte-wide (int8) / half distances have cheap djk -- barrier
// overhead wins unless dim is very wide.
template <typename T>
__host__ __device__ inline int robust_prune_block_dim(int dim, int degree)
{
  if (degree < 64 || dim < kRobustPruneCandCacheMinDim) { return 128; }
  if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) { return 128; }
  if constexpr (std::is_same_v<T, half>) {
    if (dim < kRobustPruneMultiWarpMinDimHalf) { return 128; }
    return 256;
  }
  return 256;
}

/* Macros to compute the shared memory requirements for CUB primitives used by search and prune */
#define COMPUTE_SMEM_SIZE(degree, visited_size, DEG, CANDS)                                     \
  if (degree == DEG && visited_size <= CANDS && visited_size > CANDS / 2) {                     \
    sort_smem_size = static_cast<int>(                                                          \
      sizeof(typename cub::BlockMergeSort<DistPair<IdxT, accT>, 32, CANDS / 32>::TempStorage)); \
  }

// Current supported sizes for degree and visited_size. Note that visited_size must be > degree
#define SELECT_SORT_SMEM_SIZE(degree, visited_size)  \
  COMPUTE_SMEM_SIZE(degree, visited_size, 32, 64);   \
  COMPUTE_SMEM_SIZE(degree, visited_size, 32, 128);  \
  COMPUTE_SMEM_SIZE(degree, visited_size, 32, 256);  \
  COMPUTE_SMEM_SIZE(degree, visited_size, 32, 512);  \
  COMPUTE_SMEM_SIZE(degree, visited_size, 64, 128);  \
  COMPUTE_SMEM_SIZE(degree, visited_size, 64, 256);  \
  COMPUTE_SMEM_SIZE(degree, visited_size, 64, 512);  \
  COMPUTE_SMEM_SIZE(degree, visited_size, 128, 256); \
  COMPUTE_SMEM_SIZE(degree, visited_size, 128, 512); \
  COMPUTE_SMEM_SIZE(degree, visited_size, 256, 512); \
  COMPUTE_SMEM_SIZE(degree, visited_size, 256, 1024);

/* Macros to call the CUB BlockSort primitives for supported sizes for GREEDY SEARCH */
#define SEARCH_CALL_SORT(topk, CANDS)                                             \
  if (topk <= CANDS && topk > CANDS / 2) {                                        \
    using BlockSortT = cub::BlockMergeSort<DistPair<IdxT, accT>, 32, CANDS / 32>; \
    auto& sort_mem   = reinterpret_cast<typename BlockSortT::TempStorage&>(smem); \
    sort_visited<accT, IdxT, CANDS>(&query_list[i], &sort_mem);                   \
  }

// SEARCH only relies on visited_size (not degree) for shared memory.
#define SEARCH_SELECT_SORT(topk) \
  SEARCH_CALL_SORT(topk, 64);    \
  SEARCH_CALL_SORT(topk, 128);   \
  SEARCH_CALL_SORT(topk, 256);   \
  SEARCH_CALL_SORT(topk, 512);   \
  SEARCH_CALL_SORT(topk, 1024);

}  // namespace cuvs::neighbors::vamana::detail
