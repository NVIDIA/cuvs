/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/util/cuda_utils.cuh>

namespace cuvs::neighbors::ivf_flat::detail {

template <typename T, typename AccT, int Veclen>
__device__ void compute_dist_euclidean_impl(AccT& acc, AccT x, AccT y)
{
  if constexpr (std::is_same_v<T, uint8_t> && std::is_same_v<AccT, uint32_t>) {
    if constexpr (Veclen > 1) {
      const auto diff = __vabsdiffu4(x, y);
      acc             = raft::dp4a(diff, diff, acc);
    } else {
      const auto diff = __usad(x, y, 0u);
      acc += diff * diff;
    }
  } else if constexpr (std::is_same_v<T, int8_t> && std::is_same_v<AccT, int32_t>) {
    if constexpr (Veclen > 1) {
      // Note that we enforce here that the unsigned version of dp4a is used, because the difference
      // between two int8 numbers can be greater than 127 and therefore represented as a negative
      // number in int8. Casting from int8 to int32 would yield incorrect results, while casting
      // from uint8 to uint32 is correct.
      const auto diff = __vabsdiffs4(x, y);
      acc             = raft::dp4a(diff, diff, static_cast<uint32_t>(acc));
    } else {
      const auto diff = x - y;
      acc += diff * diff;
    }
  } else {
    const auto diff = x - y;
    acc += diff * diff;
  }
}

template <typename T, typename AccT, int Veclen>
__device__ void compute_dist_inner_product_impl(AccT& acc, AccT x, AccT y)
{
  if constexpr (Veclen > 1 && (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>)) {
    acc = raft::dp4a(x, y, acc);
  } else {
    acc += x * y;
  }
}

}  // namespace cuvs::neighbors::ivf_flat::detail
