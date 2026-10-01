/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace cuvs::neighbors::ivf_sq::detail {

// Per-element distance accumulator. Called inside the unrolled inner loop of
// ivf_sq_scan_impl. After JIT-LTO inlining, these bodies fold directly into
// the unrolled loop and the v_norm_sq plumbing is dead-code-eliminated for
// non-cosine metrics.
//
// L2:        diff = qt - code*scale; dist += diff*diff
// IP:        v    = aux + code*scale; dist += qt*v
// Cosine:    as IP, plus v_norm_sq += v*v

__device__ void accumulate_distance_l2_impl(
  float qt, float /* aux */, float scale, uint8_t code, float& dist, float& /* v_norm_sq */)
{
  float recon = float(code) * scale;
  float diff  = qt - recon;
  dist += diff * diff;
}

__device__ void accumulate_distance_ip_impl(
  float qt, float aux, float scale, uint8_t code, float& dist, float& /* v_norm_sq */)
{
  float recon = float(code) * scale;
  float v_d   = aux + recon;
  dist += qt * v_d;
}

__device__ void accumulate_distance_cosine_impl(
  float qt, float aux, float scale, uint8_t code, float& dist, float& v_norm_sq)
{
  float recon = float(code) * scale;
  float v_d   = aux + recon;
  dist += qt * v_d;
  v_norm_sq += v_d * v_d;
}

}  // namespace cuvs::neighbors::ivf_sq::detail
