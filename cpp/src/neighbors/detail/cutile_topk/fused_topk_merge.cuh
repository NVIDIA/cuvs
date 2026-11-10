/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <limits>

namespace cuvs::neighbors::detail {

// The producer emits sorted lists with globally unique valid corpus IDs.
// One warp owns a query; each lane owns up to four sorted split lists (at most 128 splits).
// The original value is carried separately from the ordering key, preserving
// signed-zero bits. This is not a general duplicate-ID deduplicating merge.
template <int PhysicalK, typename IdxT>
__global__ void merge_fused_topk_sorted_splits(IdxT* neighbors,
                                               float* distances,
                                               const int* partial_indices,
                                               const float* partial_distances,
                                               int query_rows,
                                               int logical_k,
                                               int num_splits,
                                               bool inner_product)
{
  constexpr unsigned mask = 0xffffffffu;
  constexpr int invalid   = std::numeric_limits<int>::max();
  constexpr float inf     = std::numeric_limits<float>::infinity();
  const int lane          = threadIdx.x;
  const int64_t width     = static_cast<int64_t>(num_splits) * PhysicalK;
  for (int64_t row = blockIdx.x; row < query_rows; row += gridDim.x) {
    int position[4] = {0, 0, 0, 0};
    int index[4];
    float value[4];
    float score[4];
#pragma unroll
    for (int list = 0; list < 4; ++list) {
      const int split      = lane + 32 * list;
      const int64_t offset = row * width + static_cast<int64_t>(split) * PhysicalK;
      const int id         = split < num_splits ? partial_indices[offset] : -1;
      index[list]          = id < 0 ? invalid : id;
      value[list]          = id < 0 ? inf : partial_distances[offset];
      score[list]          = id < 0 ? inf : (inner_product ? -value[list] : value[list]);
    }
    for (int out = 0; out < logical_k; ++out) {
      int list = 0;
#pragma unroll
      for (int head = 1; head < 4; ++head) {
        if (score[head] < score[list] ||
            (score[head] == score[list] && index[head] < index[list])) {
          list = head;
        }
      }
      float best_score = score[list];
      int best_index   = index[list];
      int winner       = lane;
#pragma unroll
      for (int delta = 16; delta > 0; delta /= 2) {
        const float other_score = __shfl_down_sync(mask, best_score, delta);
        const int other_index   = __shfl_down_sync(mask, best_index, delta);
        const int other_winner  = __shfl_down_sync(mask, winner, delta);
        if (lane + delta < 32 &&
            (other_score < best_score || (other_score == best_score && other_index < best_index))) {
          best_score = other_score;
          best_index = other_index;
          winner     = other_winner;
        }
      }
      winner               = __shfl_sync(mask, winner, 0);
      best_index           = __shfl_sync(mask, best_index, 0);
      const float selected = __shfl_sync(mask, value[list], winner);
      if (lane == 0) {
        const int64_t offset = row * logical_k + out;
        neighbors[offset] =
          best_index == invalid ? static_cast<IdxT>(-1) : static_cast<IdxT>(best_index);
        distances[offset] = best_index == invalid ? inf : selected;
      }
      // All-invalid heads must not advance or access an exhausted list.
      if (lane == winner && best_index != invalid) {
        const int next       = ++position[list];
        const int split      = lane + 32 * list;
        const int64_t offset = row * width + static_cast<int64_t>(split) * PhysicalK + next;
        const int id         = next < PhysicalK ? partial_indices[offset] : -1;
        index[list]          = id < 0 ? invalid : id;
        value[list]          = id < 0 ? inf : partial_distances[offset];
        score[list]          = id < 0 ? inf : (inner_product ? -value[list] : value[list]);
      }
    }
  }
}

}  // namespace cuvs::neighbors::detail
