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

namespace {
__constant__ float d_kTightStart_opt[9] = {
  0.0f,
  0.15f,
  0.20f,
  0.52f,
  0.59f,
  0.71f,
  0.75f,
  0.77f,
  0.81f,
};
}  // namespace

__inline__ __device__ float compute_best_rescale_parallel(
  float* s_xp_norm, int D, int EX_BITS, float* reuse_space, int BlockSize)
{
  int tid              = threadIdx.x;
  constexpr float kEps = 1e-5f;
  constexpr int kNEnum = 10;

  //=========================================================================
  // Step 1: Find maximum value using parallel reduction
  //=========================================================================
  float local_max = 0.0f;
  for (int i = tid; i < D; i += BlockSize) {
    local_max = fmaxf(local_max, fabsf(s_xp_norm[i]));
  }

  // Block-level reduction for max
  float* s_reduce;
  s_reduce      = reuse_space;
  s_reduce[tid] = local_max;
  __syncthreads();

  for (int stride = BlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) { s_reduce[tid] = fmaxf(s_reduce[tid], s_reduce[tid + stride]); }
    __syncthreads();
  }

  float max_o = s_reduce[0];  // full max, already visible to all threads via the reduction's sync
  if (max_o < kEps) return 1.0f;

  //=========================================================================
  // Step 2: Each thread evaluates multiple critical points
  //=========================================================================
  float t_end   = static_cast<float>((1 << EX_BITS) - 1 + kNEnum) / max_o;
  float t_start = t_end * d_kTightStart_opt[EX_BITS];

  //=========================================================================
  // Phase 1: Coarse grid search
  //=========================================================================
  const int COARSE_SAMPLES = 64;  // Though not fully utilize each thread, but very fast
  float best_coarse_ip     = 0.0f;
  float best_coarse_t      = t_start;

  for (int i = tid; i < COARSE_SAMPLES; i += BlockSize) {
    float t = t_start + (t_end - t_start) * i / (COARSE_SAMPLES - 1);

    // Quick approximate evaluation (sample subset of dimensions)
    float numerator       = 0.0f;
    float sqr_denominator = static_cast<float>(D) * 0.25f;

    // Full evaluation for coarse grid
    for (int j = 0; j < D; j++) {
      float val     = fabsf(s_xp_norm[j]);
      int quantized = min(static_cast<int>((t * val) + kEps), (1 << EX_BITS) - 1);
      numerator += (quantized + 0.5f) * val;
      sqr_denominator += quantized * quantized + quantized;
    }

    float ip = numerator / sqrtf(sqr_denominator);
    if (ip > best_coarse_ip) {
      best_coarse_ip = ip;
      best_coarse_t  = t;
    }
  }

  // Parallel reduction to find best coarse point
  float* s_coarse_ip;
  s_coarse_ip = reuse_space + BlockSize;
  float* s_coarse_t;
  s_coarse_t       = s_coarse_ip + BlockSize;
  s_coarse_ip[tid] = best_coarse_ip;
  s_coarse_t[tid]  = best_coarse_t;
  __syncthreads();

  for (int stride = BlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      if (s_coarse_ip[tid + stride] > s_coarse_ip[tid]) {
        s_coarse_ip[tid] = s_coarse_ip[tid + stride];
        s_coarse_t[tid]  = s_coarse_t[tid + stride];
      }
    }
    __syncthreads();
  }

  //=========================================================================
  // Phase 2: Fine search around best coarse point
  //=========================================================================
  float center_t   = s_coarse_t[0];
  float range      = (t_end - t_start) / COARSE_SAMPLES;
  float fine_start = fmaxf(t_start, center_t - range);
  float fine_end   = fminf(t_end, center_t + range);

  const int FINE_SAMPLES = 32;
  float best_fine_ip     = 0.0f;
  float best_fine_t      = center_t;

  for (int i = tid; i < FINE_SAMPLES; i += BlockSize) {
    float t = fine_start + (fine_end - fine_start) * i / (FINE_SAMPLES - 1);

    // Full evaluation
    float numerator       = 0.0f;
    float sqr_denominator = static_cast<float>(D) * 0.25f;

    for (int j = 0; j < D; j++) {
      float val     = fabsf(s_xp_norm[j]);
      int quantized = min(static_cast<int>((t * val) + kEps), (1 << EX_BITS) - 1);
      numerator += (quantized + 0.5f) * val;
      sqr_denominator += quantized * quantized + quantized;
    }

    float ip = numerator / sqrtf(sqr_denominator);
    if (ip > best_fine_ip) {
      best_fine_ip = ip;
      best_fine_t  = t;
    }
  }

  // Final reduction
  float* s_fine_ip;
  s_fine_ip = s_coarse_ip;
  float* s_fine_t;
  s_fine_t       = s_coarse_t;
  s_fine_ip[tid] = best_fine_ip;
  s_fine_t[tid]  = best_fine_t;
  __syncthreads();

  for (int stride = BlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      if (s_fine_ip[tid + stride] > s_fine_ip[tid]) {
        s_fine_ip[tid] = s_fine_ip[tid + stride];
        s_fine_t[tid]  = s_fine_t[tid + stride];
      }
    }
    __syncthreads();
  }

  return s_fine_t[0];
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
