/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace cuvs::neighbors::ivf_sq::detail {

// Per-row distance finalize. Called once per scanned row.
//
// All metrics return a min-close value so the warpsort epilogue and the
// host-side select_k can be hardcoded to ascending / select-min:
//   L2:     return dist                              (squared L2 is min-close)
//   IP:     return -dist                             (negate so min-close;
//                                                     postprocess_distances
//                                                     undoes the sign)
//   Cosine: denom = query_norm * sqrtf(v_norm_sq);
//           return denom > 0 ? 1 - dist/denom : 0    (cosine distance is min-close)

__device__ float finalize_distance_l2_impl(float dist,
                                           float /* v_norm_sq */,
                                           float /* query_norm */)
{
  return dist;
}

__device__ float finalize_distance_ip_impl(float dist,
                                           float /* v_norm_sq */,
                                           float /* query_norm */)
{
  return -dist;
}

__device__ float finalize_distance_cosine_impl(float dist, float v_norm_sq, float query_norm)
{
  float denom = query_norm * sqrtf(v_norm_sq);
  return (denom > 0.0f) ? 1.0f - dist / denom : 0.0f;
}

}  // namespace cuvs::neighbors::ivf_sq::detail
