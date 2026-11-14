/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <limits>
namespace cuvs::neighbors::detail {
// Merge up to two/four sorted lists per query. Statically indexed arrays and
// unrolled loops let the compiler retain heads in registers, without shuffles.
template <int PhysicalK, int Heads, typename IdxT>
__global__ void merge_fused_topk_register_heads(IdxT* neighbors,
                                                float* distances,
                                                const int* partial_indices,
                                                const float* partial_distances,
                                                int query_rows,
                                                int logical_k,
                                                int num_splits,
                                                bool inner_product)
{
  static_assert(Heads == 2 || Heads == 4);
  constexpr int invalid = std::numeric_limits<int>::max();
  constexpr float inf   = std::numeric_limits<float>::infinity();
  for (int64_t row = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; row < query_rows;
       row += int64_t(gridDim.x) * blockDim.x) {
    int index[Heads], position[Heads];
    float value[Heads], score[Heads];
    const int64_t base = row * (num_splits * PhysicalK);
#pragma unroll
    for (int h = 0; h < Heads; ++h) {
      position[h] = 0;
      int id      = h < num_splits ? partial_indices[base + h * PhysicalK] : -1;
      value[h]    = id < 0 ? inf : partial_distances[base + h * PhysicalK];
      score[h]    = id < 0 ? inf : (inner_product ? -value[h] : value[h]);
      index[h]    = id < 0 ? invalid : id;
    }
#pragma unroll
    for (int out = 0; out < PhysicalK; ++out) {
      if (out < logical_k) {
        int best_index = index[0], winner = 0;
        float best_score = score[0], best_value = value[0];
#pragma unroll
        for (int h = 1; h < Heads; ++h) {
          if (score[h] < best_score || (score[h] == best_score && index[h] < best_index)) {
            best_index = index[h];
            best_score = score[h];
            best_value = value[h];
            winner     = h;
          }
        }
        neighbors[row * logical_k + out] = best_index == invalid ? IdxT(-1) : IdxT(best_index);
        distances[row * logical_k + out] = best_index == invalid ? inf : best_value;
        if (out + 1 < logical_k && best_index != invalid) {
#pragma unroll
          for (int h = 0; h < Heads; ++h) {
            if (winner == h) {
              ++position[h];
              int id =
                position[h] < PhysicalK ? partial_indices[base + h * PhysicalK + position[h]] : -1;
              value[h] = id < 0 ? inf : partial_distances[base + h * PhysicalK + position[h]];
              score[h] = id < 0 ? inf : (inner_product ? -value[h] : value[h]);
              index[h] = id < 0 ? invalid : id;
            }
          }
        }
      }
    }
  }
}
}  // namespace cuvs::neighbors::detail
