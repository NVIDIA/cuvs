/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace cuvs::neighbors::ivf_sq::detail {

// Phase 2 per-probe smem setup. Called once per (query, probe).
//
// Loads the per-probe smem array using the invariant array filled in Phase 1.
//
// L2:        s_query_term[d] = s_aux[d]   - centroid[d]   (where s_aux holds query - sq_vmin)
// IP/Cosine: s_aux[d]        = centroid[d] + sq_vmin[d]
//
// The "in" and "out" smem arrays do not alias — they are distinct regions of
// the kernel's smem layout — so __restrict__ is safe.

__device__ void setup_per_probe_smem_l2_impl(uint32_t dim,
                                             const float* __restrict__ centroid,
                                             const float* __restrict__ /* sq_vmin */,
                                             float* __restrict__ s_aux,
                                             float* __restrict__ s_query_term)
{
  for (uint32_t d = threadIdx.x; d < dim; d += blockDim.x) {
    s_query_term[d] = s_aux[d] - centroid[d];
  }
}

__device__ void setup_per_probe_smem_ip_impl(uint32_t dim,
                                             const float* __restrict__ centroid,
                                             const float* __restrict__ sq_vmin,
                                             float* __restrict__ s_aux,
                                             float* __restrict__ /* s_query_term */)
{
  for (uint32_t d = threadIdx.x; d < dim; d += blockDim.x) {
    s_aux[d] = centroid[d] + sq_vmin[d];
  }
}

__device__ void setup_per_probe_smem_cosine_impl(uint32_t dim,
                                                 const float* __restrict__ centroid,
                                                 const float* __restrict__ sq_vmin,
                                                 float* __restrict__ s_aux,
                                                 float* __restrict__ /* s_query_term */)
{
  for (uint32_t d = threadIdx.x; d < dim; d += blockDim.x) {
    s_aux[d] = centroid[d] + sq_vmin[d];
  }
}

}  // namespace cuvs::neighbors::ivf_sq::detail
