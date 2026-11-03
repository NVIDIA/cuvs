/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Bitwise Hamming over byte-packed binary vectors (uint8_t data, uint32_t acc).
// `x` and `y` are already promoted to uint32_t by the load fragment: for Veclen >= 4 they pack
// 4 source bytes per accumulator word, for Veclen == 2 the upper 16 bits are zero from the
// uint16->uint32 zero-extension, and for Veclen == 1 only the low 8 bits carry a byte.
template <typename T, typename AccT, int Veclen>
__device__ void compute_dist_bitwise_hamming_impl(AccT& acc, AccT x, AccT y)
{
  static_assert(std::is_same_v<T, uint8_t> && std::is_same_v<AccT, uint32_t>,
                "compute_dist_bitwise_hamming_impl is only valid for uint8_t/uint32_t");
  if constexpr (Veclen > 1) {
    acc += __popc(x ^ y);
  } else {
    acc += __popc(static_cast<uint32_t>(x ^ y) & 0xffu);
  }
}

}  // namespace cuvs::neighbors::ivf_flat::detail
