/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#include <cuvs/detail/jit_lto/ivf_pq/compute_similarity_fragments.hpp>
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>

namespace cuvs::neighbors::ivf_pq::detail {

template <typename LutT, typename MetricTag, bool PrecompBaseDiff, uint32_t PqBits>
__device__ void create_lut_impl(uint32_t pq_dim,
                                uint32_t pq_len,
                                uint32_t label,
                                const float* pq_centers,
                                codebook_gen codebook_kind,
                                LutT* lut_scores,
                                uint8_t* lut_end,
                                const float* query,
                                const float* cluster_center)
{
  constexpr uint32_t PqShift = 1u << PqBits;
  constexpr uint32_t PqMask  = PqShift - 1u;

  const uint32_t lut_size = pq_dim * PqShift;

  const float* pq_center;
  if (codebook_kind == codebook_gen::PER_SUBSPACE) {
    pq_center = pq_centers;
  } else {
    pq_center = pq_centers + (pq_len << PqBits) * label;
  }

  // Create a lookup table
  // For each subspace, the lookup table stores the distance between the actual query vector
  // (projected into the subspace) and all possible pq vectors in that subspace.
  for (uint32_t i = threadIdx.x; i < lut_size; i += blockDim.x) {
    const uint32_t i_pq  = i >> PqBits;
    uint32_t j           = i_pq * pq_len;
    const uint32_t j_end = pq_len + j;
    auto cur_pq_center =
      pq_center + (i & PqMask) + (codebook_kind == codebook_gen::PER_SUBSPACE ? j * PqShift : 0u);
    float score = 0.0;
    do {
      float pq_c = *cur_pq_center;
      cur_pq_center += PqShift;
      if constexpr (std::is_same_v<MetricTag, tag_metric_euclidean>) {
        float diff;
        if constexpr (PrecompBaseDiff) {
          diff = reinterpret_cast<float*>(lut_end)[j];
        } else {
          diff = query[j] - cluster_center[j];
        }
        diff -= pq_c;
        score += diff * diff;
      } else if constexpr (std::is_same_v<MetricTag, tag_metric_inner_product>) {
        // NB: we negate the scores as we hardcoded select-topk to always compute the minimum
        float q;
        if constexpr (PrecompBaseDiff) {
          float2 pvals = reinterpret_cast<float2*>(lut_end)[j];
          q            = pvals.x;
          score -= pvals.y;
        } else {
          q = query[j];
          score -= q * cluster_center[j];
        }
        score -= q * pq_c;
      } else {
        static_assert(sizeof(MetricTag*) == 0, "Invalid MetricTag");
      }
    } while (++j < j_end);
    lut_scores[i] = LutT(score);
  }
}

}  // namespace cuvs::neighbors::ivf_pq::detail
