/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace cuvs::neighbors::ivf_sq::detail {

// Phase 1 invariant smem setup. Called once per query.
//
// Loads the metric-invariant smem array. The metric-variant array is left
// untouched and will be filled per-probe by setup_per_probe_smem.

__device__ void setup_invariant_smem_l2_impl(uint32_t dim,
                                             const float* __restrict__ query,
                                             const float* __restrict__ sq_vmin,
                                             float* __restrict__ s_aux,
                                             float* __restrict__ /* s_query_term */)
{
  for (uint32_t d = threadIdx.x; d < dim; d += blockDim.x) {
    s_aux[d] = query[d] - sq_vmin[d];
  }
}

__device__ void setup_invariant_smem_ip_impl(uint32_t dim,
                                             const float* __restrict__ query,
                                             const float* __restrict__ /* sq_vmin */,
                                             float* __restrict__ /* s_aux */,
                                             float* __restrict__ s_query_term)
{
  for (uint32_t d = threadIdx.x; d < dim; d += blockDim.x) {
    s_query_term[d] = query[d];
  }
}

__device__ void setup_invariant_smem_cosine_impl(uint32_t dim,
                                                 const float* __restrict__ query,
                                                 const float* __restrict__ /* sq_vmin */,
                                                 float* __restrict__ /* s_aux */,
                                                 float* __restrict__ s_query_term)
{
  for (uint32_t d = threadIdx.x; d < dim; d += blockDim.x) {
    s_query_term[d] = query[d];
  }
}

}  // namespace cuvs::neighbors::ivf_sq::detail
