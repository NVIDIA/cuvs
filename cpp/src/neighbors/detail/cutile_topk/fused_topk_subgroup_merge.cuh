/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <limits>

namespace cuvs::neighbors::detail {

// A power-of-two subgroup owns one query and one sorted list per lane.
// Valid IDs are globally unique within a query. Inputs contain at most
// GroupSize splits. Every lane in a participating group follows the same loop;
// separate queries communicate only through their own subgroup mask.
template <int PhysicalK, int GroupSize, typename IdxT>
__global__ void merge_fused_topk_subgroups(IdxT* neighbors,
                                           float* distances,
                                           const int* partial_indices,
                                           const float* partial_distances,
                                           int query_rows,
                                           int logical_k,
                                           int num_splits,
                                           bool inner_product)
{
  static_assert(GroupSize == 2 || GroupSize == 4 || GroupSize == 8 || GroupSize == 16);
  constexpr int invalid = std::numeric_limits<int>::max();
  constexpr float inf   = std::numeric_limits<float>::infinity();
  const int lane        = threadIdx.x % GroupSize;
  const int group_base  = (threadIdx.x % 32) - lane;
  const unsigned mask   = ((1u << GroupSize) - 1u) << group_base;
  const int64_t first   = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / GroupSize;
  const int64_t stride  = int64_t(gridDim.x) * (blockDim.x / GroupSize);
  const int64_t width   = int64_t(num_splits) * PhysicalK;
  for (int64_t row = first; row < query_rows; row += stride) {
    int position       = 0;
    const int64_t base = row * width + int64_t(lane) * PhysicalK;
    int id             = lane < num_splits ? partial_indices[base] : -1;
    int index          = id < 0 ? invalid : id;
    float value        = id < 0 ? inf : partial_distances[base];
    float score        = id < 0 ? inf : (inner_product ? -value : value);
    for (int out = 0; out < logical_k; ++out) {
      float best_score = score;
      int best_index   = index;
      int winner       = lane;
#pragma unroll
      for (int delta = GroupSize / 2; delta > 0; delta /= 2) {
        const float other_score = __shfl_down_sync(mask, best_score, delta, GroupSize);
        const int other_index   = __shfl_down_sync(mask, best_index, delta, GroupSize);
        const int other_winner  = __shfl_down_sync(mask, winner, delta, GroupSize);
        if (lane + delta < GroupSize &&
            (other_score < best_score || (other_score == best_score && other_index < best_index))) {
          best_score = other_score;
          best_index = other_index;
          winner     = other_winner;
        }
      }
      winner               = __shfl_sync(mask, winner, 0, GroupSize);
      best_index           = __shfl_sync(mask, best_index, 0, GroupSize);
      const float selected = __shfl_sync(mask, value, winner, GroupSize);
      if (lane == 0) {
        neighbors[row * logical_k + out] = best_index == invalid ? IdxT(-1) : IdxT(best_index);
        distances[row * logical_k + out] = best_index == invalid ? inf : selected;
      }
      if (out + 1 < logical_k && lane == winner && best_index != invalid) {
        ++position;
        id    = position < PhysicalK ? partial_indices[base + position] : -1;
        index = id < 0 ? invalid : id;
        value = id < 0 ? inf : partial_distances[base + position];
        score = id < 0 ? inf : (inner_product ? -value : value);
      }
    }
  }
}

}  // namespace cuvs::neighbors::detail
