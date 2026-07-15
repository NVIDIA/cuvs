/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/warp_primitives.cuh>

#include <cuda_fp16.h>

#include <cfloat>
#include <cstdint>

namespace cuvs::neighbors::detail::device {

// warpSize for compile time calculation
constexpr unsigned warp_size = 32;

// using LOAD_256BIT_T = ulonglong4;
using LOAD_128BIT_T = uint4;
using LOAD_64BIT_T  = uint64_t;

template <class LOAD_T, class DATA_T>
RAFT_DEVICE_INLINE_FUNCTION constexpr unsigned get_vlen()
{
  static_assert(sizeof(DATA_T) > 0, "get_vlen: DATA_T must have positive size");
  return static_cast<unsigned>(sizeof(LOAD_T) / sizeof(DATA_T));
}

/** Xorshift rondem number generator.
 *
 * See https://en.wikipedia.org/wiki/Xorshift#xorshift for reference.
 */
_RAFT_HOST_DEVICE inline uint64_t xorshift64(uint64_t u)
{
  u ^= u >> 12;
  u ^= u << 25;
  u ^= u >> 27;
  return u * 0x2545F4914F6CDD1DULL;
}

template <uint32_t TeamSize, typename T>
RAFT_DEVICE_INLINE_FUNCTION auto team_sum(T x) -> T
{
#pragma unroll
  for (uint32_t stride = TeamSize >> 1; stride > 0; stride >>= 1) {
    x += raft::shfl_xor(x, stride, TeamSize);
  }
  return x;
}

template <typename T>
RAFT_DEVICE_INLINE_FUNCTION auto team_sum(T x, uint32_t team_size_bitshift) -> T
{
  switch (team_size_bitshift) {
    case 5: x += raft::shfl_xor(x, 16); [[fallthrough]];
    case 4: x += raft::shfl_xor(x, 8); [[fallthrough]];
    case 3: x += raft::shfl_xor(x, 4); [[fallthrough]];
    case 2: x += raft::shfl_xor(x, 2); [[fallthrough]];
    case 1: x += raft::shfl_xor(x, 1); [[fallthrough]];
    default: return x;
  }
}

template <uint32_t Dim = 1024, uint32_t Stride = 128, typename T>
RAFT_DEVICE_INLINE_FUNCTION constexpr auto swizzling(T x) -> T
{
  // Address swizzling reduces bank conflicts in shared memory, but increases
  // the amount of operation instead.
  if constexpr (Stride <= 32) {
    return x;
  } else if constexpr (Dim <= 1024) {
    return x ^ (x >> 5);
  } else {
    return x ^ ((x >> 5) & 0x1f);
  }
}

}  // namespace cuvs::neighbors::detail::device

// CAGRA JIT kernels extend `cuvs::neighbors::cagra::detail::device` in other headers; re-export
// the shared helpers there under the historical nested name.
namespace cuvs::neighbors::cagra::detail::device {
using namespace cuvs::neighbors::detail::device;
}  // namespace cuvs::neighbors::cagra::detail::device
