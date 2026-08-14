/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#include <raft/util/cudart_utils.hpp>

#include <cuvs/neighbors/ivf_pq.hpp>

namespace cuvs::neighbors::ivf_pq::detail {

template <uint32_t PqBits>
__device__ uint32_t get_line_width_impl(uint32_t pq_dim)
{
  constexpr uint32_t kChunkSize = (kIndexGroupVecLen * 8u) / PqBits;
  return raft::div_rounding_up_unsafe(pq_dim, kChunkSize) * kIndexGroupVecLen;
}

}  // namespace cuvs::neighbors::ivf_pq::detail
