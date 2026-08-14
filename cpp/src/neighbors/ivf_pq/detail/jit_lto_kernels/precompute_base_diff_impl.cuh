/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#include <cuvs/distance/distance.hpp>

namespace cuvs::neighbors::ivf_pq::detail {

__device__ void precompute_base_diff_euclidean_impl(uint32_t dim,
                                                    uint8_t* lut_end,
                                                    const float* query,
                                                    const float* cluster_center)
{
  // Reduce number of memory reads later by pre-computing parts of the score
  for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    reinterpret_cast<float*>(lut_end)[i] = query[i] - cluster_center[i];
  }
  __syncthreads();
}

__device__ void precompute_base_diff_inner_product_impl(uint32_t dim,
                                                        uint8_t* lut_end,
                                                        const float* query,
                                                        const float* cluster_center)
{
  // Reduce number of memory reads later by pre-computing parts of the score
  float2 pvals;
  for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    pvals.x                               = query[i];
    pvals.y                               = cluster_center[i] * pvals.x;
    reinterpret_cast<float2*>(lut_end)[i] = pvals;
  }
  __syncthreads();
}

__device__ void precompute_base_diff_none_impl(uint32_t dim,
                                               uint8_t* lut_end,
                                               const float* query,
                                               const float* cluster_center)
{
}

}  // namespace cuvs::neighbors::ivf_pq::detail
