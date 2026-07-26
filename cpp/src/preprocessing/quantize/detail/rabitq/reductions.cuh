/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Imported from the standalone RaBitQ GPU quantizer. Behaviour unchanged;
 * only include paths, error macros and the enclosing namespace differ.
 */

//
// Warp/block reductions: warp shuffles + one shared-memory exchange and a
// single __syncthreads() per block-level call. Naming follows the
// cuVS/RAFT device-utility convention (camelCase; blockReduceSum leaves the
// result in thread 0 like raft::blockReduce; the AllReduce variants deliver
// it to every thread via butterfly shuffles, with no extra barrier).
//
// Requirements / contract:
//   - blockDim.x is a multiple of 32, at most 1024 threads.
//   - Each block-level template instantiation owns one static shared scratch
//     per kernel. Call an instantiation at most once per kernel launch (or
//     separate repeated calls with a __syncthreads()): a back-to-back second
//     call would overwrite the scratch while stragglers of the first still
//     read it. Distinct N (and Sum vs Max, thread-0 vs AllReduce) use
//     distinct scratches.
//   - The internal __syncthreads() also orders all shared/global writes
//     issued before the call ahead of anything after it.
//

#ifndef RABITQ_GPU_BLOCK_REDUCE_CUH
#define RABITQ_GPU_BLOCK_REDUCE_CUH

#include <cuda_runtime.h>

namespace cuvs::preprocessing::quantize::rabitq::detail {

namespace blockred {

/// Warp-level sum; the full warp sum lands in lane 0 (other lanes hold
/// partials), matching raft::warpReduce.
__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffff, val, off);
    return val;
}

/// Sum-reduce N values across the block in one pass; on return vals[k] holds
/// the full block sum in THREAD 0 only (other threads hold partials) — the
/// same result contract as raft::blockReduce / cuVS blockReduceSum.
template <int N>
__device__ __forceinline__ void blockReduceSum(float (&vals)[N]) {
    __shared__ float sh[N][32];
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    const int nw   = blockDim.x >> 5;

    #pragma unroll
    for (int k = 0; k < N; ++k) {
        float v = warpReduceSum(vals[k]);
        if (lane == 0) sh[k][wid] = v;
    }
    __syncthreads();
    if (wid == 0) {
        #pragma unroll
        for (int k = 0; k < N; ++k) {
            float v = (lane < nw) ? sh[k][lane] : 0.0f;
            vals[k] = warpReduceSum(v);
        }
    }
}

/// Single-value convenience routed through the N = 1 instantiation.
__device__ __forceinline__ float blockReduceSum(float val) {
    float v[1] = {val};
    blockReduceSum(v);
    return v[0];
}

/// Sum-reduce N values across the block; on return vals[k] holds the full
/// block sum in EVERY thread (butterfly all-reduce, no broadcast barrier).
template <int N>
__device__ __forceinline__ void blockAllReduceSum(float (&vals)[N]) {
    __shared__ float sh[N][32];
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    const int nw   = blockDim.x >> 5;

    #pragma unroll
    for (int k = 0; k < N; ++k) {
        float v = vals[k];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_xor_sync(0xffffffff, v, off);
        if (lane == 0) sh[k][wid] = v;
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < N; ++k) {
        float v = (lane < nw) ? sh[k][lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_xor_sync(0xffffffff, v, off);
        vals[k] = v;   // every lane of every warp
    }
}

/// Max-reduce one value across the block; every thread receives the result.
__device__ __forceinline__ float blockAllReduceMax(float val) {
    __shared__ float sh[32];
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    const int nw   = blockDim.x >> 5;
    const float kNegInf = __int_as_float(0xff800000);

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, off));
    if (lane == 0) sh[wid] = val;
    __syncthreads();
    float v = (lane < nw) ? sh[lane] : kNegInf;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
    return v;
}

}  // namespace blockred

}  // namespace cuvs::preprocessing::quantize::rabitq::detail

#endif  // RABITQ_GPU_BLOCK_REDUCE_CUH
