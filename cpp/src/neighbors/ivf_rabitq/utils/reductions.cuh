/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/util/cuda_dev_essentials.cuh>

#include <cuda_runtime.h>

namespace cuvs::neighbors::ivf_rabitq::detail {

// Warp-level sum reduction using shuffle instructions.
template <typename T>
__inline__ __device__ T warpReduceSum(T val)
{
#pragma unroll
  for (int offset = raft::WarpSize / 2; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// N floats summed component-wise. Passing this to cub::BlockReduce (with its Sum()) performs N
// independent block-wide sums in one collective call: one warp-shuffle tree and one set of barriers
// instead of N of each. Used with cub::BlockReduce<FloatSumN<N>, BlockSize>.
template <int N>
struct FloatSumN {
  float v[N];

  __device__ __forceinline__ FloatSumN operator+(const FloatSumN& o) const
  {
    FloatSumN r;
#pragma unroll
    for (int k = 0; k < N; ++k) {
      r.v[k] = v[k] + o.v[k];
    }
    return r;
  }
};

}  // namespace cuvs::neighbors::ivf_rabitq::detail
