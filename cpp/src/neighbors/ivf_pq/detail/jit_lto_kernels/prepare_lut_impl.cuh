/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace cuvs::neighbors::ivf_pq::detail {

template <typename LutT, bool EnableSMemLut, uint32_t PqBits>
__device__ void prepare_lut_impl(uint8_t* smem_buf,
                                 uint32_t pq_dim,
                                 LutT*& lut_scores,
                                 uint8_t*& lut_end)
{
  constexpr uint32_t PqShift = 1u << PqBits;  // NOLINT

  const uint32_t lut_size = pq_dim * PqShift;

  if constexpr (EnableSMemLut) {
    lut_scores = reinterpret_cast<LutT*>(smem_buf);
    lut_end    = reinterpret_cast<uint8_t*>(lut_scores + lut_size);
  } else {
    lut_scores += lut_size * blockIdx.x;
    lut_end = smem_buf;
  }
}

}  // namespace cuvs::neighbors::ivf_pq::detail
