/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>

namespace cuvs::neighbors::ivf_rabitq::detail {

inline float normalize_query16_scalar(float* unit_q,  // out: normalised vector (length D)
                                      const float* __restrict__ q,
                                      const float* __restrict__ c,
                                      float norm,
                                      size_t D)
{
  constexpr float eps = 1e-5f;

  if (norm > eps) {
    const float inv_norm = 1.0f / norm;
    float sum            = 0.0f;

    for (size_t i = 0; i < D; ++i) {
      float u   = (q[i] - c[i]) * inv_norm;  // (q - c) / norm
      unit_q[i] = u;
      sum += u;  // running sum
    }
    return sum;  // same as _mm512_reduce_add_ps
  } else         // q‑c is (almost) the zero vector
  {
    float value = 1.0f / std::sqrt(static_cast<float>(D));  // 1 / √D
    std::fill(unit_q, unit_q + D, value);
    return static_cast<float>(D) * value;  // == √D
  }
}

inline float L2SqrCPU_STL(const float* h_x, const float* h_y, size_t N)
{
  // Using STL algorithms
  return std::inner_product(
    h_x,
    h_x + N,                // First range
    h_y,                    // Second range begin
    0.0f,                   // Initial value
    std::plus<float>(),     // Sum operation
    [](float a, float b) {  // Product operation (replaced with squared difference)
      float diff = a - b;
      return diff * diff;
    });
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
