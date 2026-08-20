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

// Block-level sum reduction of N independent values in a SINGLE pass over a
// private (per-instantiation) shared scratch. Each value gets its own column of
// shared[N][32], so all N are reduced within one write -> __syncthreads() -> read
// window. Because a group of sums is reduced in one call, the scratch is never
// reused by a back-to-back call, so no inter-call barrier is required: the
// shared-memory reuse race is removed by construction rather than patched with a
// guard. On return vals[k] holds the full block sum of the input vals[k] in lane 0
// of warp 0 (other lanes/warps hold partial values), matching blockReduceSum.
//
// Assumes blockDim.x is a multiple of the warp size and uses at most 32 warps.
template <int N, typename T>
__inline__ __device__ void blockReduceSumN(T (&vals)[N])
{
  __shared__ T shared[N][32];  // one column per value; up to 1024 threads -> 32 warps
  int lane = threadIdx.x & 31;
  int wid  = threadIdx.x >> 5;

#pragma unroll
  for (int k = 0; k < N; ++k) {
    vals[k] = warpReduceSum(vals[k]);
  }
  if (lane == 0) {
#pragma unroll
    for (int k = 0; k < N; ++k) {
      shared[k][wid] = vals[k];
    }
  }
  __syncthreads();

  bool active = threadIdx.x < blockDim.x / 32;
#pragma unroll
  for (int k = 0; k < N; ++k) {
    T v = active ? shared[k][lane] : T(0);
    if (wid == 0) v = warpReduceSum(v);
    vals[k] = v;
  }
}

// Single-value convenience that routes through the fused family (N = 1). Standalone
// reductions (never called back-to-back over the same scratch) use this and stay
// race-free by construction; N = 1 gives each such site its own shared column,
// distinct from any blockReduceSumN<N != 1> used elsewhere in the same kernel.
template <typename T>
__inline__ __device__ T blockReduceSum(T val)
{
  T v[1] = {val};
  blockReduceSumN<1>(v);
  return v[0];
}

// Block-level max all-reduce: butterfly shuffles + one shared-memory exchange,
// a single __syncthreads(), and EVERY thread receives the full block maximum
// (no separate broadcast barrier needed). Because fmaxf is exactly associative
// and commutative, the result is bit-identical to a tree reduction over the
// same inputs. Uses its own private scratch; the same one-call-per-kernel
// contract as blockReduceSumN applies. Assumes blockDim.x is a multiple of the
// warp size and uses at most 32 warps.
__inline__ __device__ float blockAllReduceMax(float val)
{
  __shared__ float shared[32];
  int lane            = threadIdx.x & 31;
  int wid             = threadIdx.x >> 5;
  int nwarps          = blockDim.x >> 5;
  const float kNegInf = __int_as_float(0xff800000);  // -inf

#pragma unroll
  for (int offset = raft::WarpSize / 2; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
  }
  if (lane == 0) { shared[wid] = val; }
  __syncthreads();

  float v = (lane < nwarps) ? shared[lane] : kNegInf;
#pragma unroll
  for (int offset = raft::WarpSize / 2; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));
  }
  return v;
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
