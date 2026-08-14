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

// Block-level sum reduction via per-warp shfl + shared-memory exchange.
// Returns the full block sum in lane 0 of warp 0; other lanes hold partial values.
template <typename T>
__inline__ __device__ T blockReduceSum(T val)
{
  __shared__ T shared[32];  // up to 1024 threads -> 32 warps
  int lane = threadIdx.x & 31;
  int wid  = threadIdx.x >> 5;

  val = warpReduceSum(val);
  // Wait until every warp has finished reading shared[] in the previous call
  // before this call overwrites it. Doing this after the warp-local reduction
  // overlaps the extra synchronization with useful work.
  __syncthreads();
  if (lane == 0) shared[wid] = val;
  __syncthreads();

  T out = (threadIdx.x < blockDim.x / 32) ? shared[lane] : T(0);
  if (wid == 0) out = warpReduceSum(out);
  return out;
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
