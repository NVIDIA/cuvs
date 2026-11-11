/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Block reductions used by the RaBitQ quantizer kernels, built on cub::BlockReduce.
// warpReduceSum and FloatSumN come from ivf_rabitq.
//
// Requirements / contract:
//   - blockDim.x == kBlockSize.
//   - blockReduceSum leaves the result in thread 0 only, like cub::BlockReduce. Separate repeated
//     calls of one instantiation with a __syncthreads() (cub's rule for reusing temp storage).
//   - The AllReduce variants publish the result through shared memory behind a __syncthreads(),
//     so every thread receives it, they may be called repeatedly, and the barrier also orders all
//     shared/global writes issued before the call ahead of anything after it.
//

#ifndef RABITQ_GPU_BLOCK_REDUCE_CUH
#define RABITQ_GPU_BLOCK_REDUCE_CUH

#include "../../../../neighbors/ivf_rabitq/utils/reductions.cuh"

#include <cub/block/block_reduce.cuh>
#include <cuda/functional>

namespace cuvs::preprocessing::quantize::rabitq::detail {

namespace blockred {

using cuvs::neighbors::ivf_rabitq::detail::FloatSumN;
using cuvs::neighbors::ivf_rabitq::detail::warpReduceSum;

/// Sum-reduce N values across the block in one pass; on return vals[k] holds
/// the full block sum in THREAD 0 only (other threads hold partials).
template <int kBlockSize, int N>
__device__ __forceinline__ void blockReduceSum(float (&vals)[N])
{
  using BlockReduce = cub::BlockReduce<FloatSumN<N>, kBlockSize>;
  __shared__ typename BlockReduce::TempStorage tmp;

  FloatSumN<N> s;
#pragma unroll
  for (int k = 0; k < N; ++k)
    s.v[k] = vals[k];
  s = BlockReduce(tmp).Sum(s);
#pragma unroll
  for (int k = 0; k < N; ++k)
    vals[k] = s.v[k];
}

/// Single-value sum; the result is valid in thread 0 only.
template <int kBlockSize>
__device__ __forceinline__ float blockReduceSum(float val)
{
  using BlockReduce = cub::BlockReduce<float, kBlockSize>;
  __shared__ typename BlockReduce::TempStorage tmp;
  return BlockReduce(tmp).Sum(val);
}

/// Sum-reduce one value across the block; every thread receives the result.
template <int kBlockSize>
__device__ __forceinline__ float blockAllReduceSum(float val)
{
  using BlockReduce = cub::BlockReduce<float, kBlockSize>;
  __shared__ typename BlockReduce::TempStorage tmp;
  __shared__ float result;

  float r = BlockReduce(tmp).Sum(val);
  if (threadIdx.x == 0) result = r;
  __syncthreads();
  return result;
}

/// Max-reduce one value across the block; every thread receives the result.
template <int kBlockSize>
__device__ __forceinline__ float blockAllReduceMax(float val)
{
  using BlockReduce = cub::BlockReduce<float, kBlockSize>;
  __shared__ typename BlockReduce::TempStorage tmp;
  __shared__ float result;

  float r = BlockReduce(tmp).Reduce(val, cuda::maximum<>{});
  if (threadIdx.x == 0) result = r;
  __syncthreads();
  return result;
}

}  // namespace blockred

}  // namespace cuvs::preprocessing::quantize::rabitq::detail

#endif  // RABITQ_GPU_BLOCK_REDUCE_CUH
