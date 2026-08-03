/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Imported from the standalone RaBitQ GPU quantizer. Behaviour unchanged;
 * only include paths, error macros and the enclosing namespace differ.
 */

//
// Implementation of the rescale-factor search helpers extracted from
// IVF-RaBitQ-GPU-main/inc/gpu_index/quantizer_gpu_fast.cu.
//

#include "rescale_search.cuh"
#include "tight_start_constants.cuh"
#include "reductions.cuh"

#include <cub/cub.cuh>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resource/device_properties.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

#include <rmm/device_uvector.hpp>

namespace cuvs::preprocessing::quantize::rabitq::detail {

// ---------------------------------------------------------------------------
// Warp-cooperative single-sample evaluator for the rescale search.
// ---------------------------------------------------------------------------
static __device__ __forceinline__ float evaluate_rescale_sample_warp(
        const float* __restrict__ s_xp_norm, int D, int EX_BITS, float t, int lane_id)
{
    constexpr float kEps = 1e-5f;
    int max_code = (1 << EX_BITS) - 1;
    float numerator = 0.0f;
    float sqr_denom = (lane_id == 0) ? static_cast<float>(D) * 0.25f : 0.0f;

    for (int j = lane_id; j < D; j += 32) {
        float val = fabsf(s_xp_norm[j]);
        int quantized = min(__float2int_rd(t * val + kEps), max_code);
        numerator += (quantized + 0.5f) * val;
        sqr_denom += quantized * quantized + quantized;
    }

    numerator = blockred::warpReduceSum(numerator);
    sqr_denom = blockred::warpReduceSum(sqr_denom);
    return numerator / sqrtf(sqr_denom);
}

// ---------------------------------------------------------------------------
// Warp-cooperative rescale search. See header for contract.
// ---------------------------------------------------------------------------
__device__ float compute_best_rescale_parallel(
        float* s_xp_norm,
        int D,
        int EX_BITS,
        float* reuse_space,
        int BlockSize,
        int n_coarse_samples,
        int n_fine_samples)
{
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int nWarps = BlockSize / 32;

    constexpr float kEps = 1e-5f;
    constexpr int kNEnum = 10;
    int coarse_samples = n_coarse_samples > 1 ? n_coarse_samples : 1;
    int fine_samples = n_fine_samples > 1 ? n_fine_samples : 1;
    int coarse_denom = coarse_samples > 1 ? coarse_samples - 1 : 1;
    int fine_denom = fine_samples > 1 ? fine_samples - 1 : 1;

    // block-wide max of |s_xp_norm|
    float local_max = 0.0f;
    for (int i = tid; i < D; i += BlockSize) {
        local_max = fmaxf(local_max, fabsf(s_xp_norm[i]));
    }

    float* s_reduce = reuse_space;
    s_reduce[tid] = local_max;
    __syncthreads();
    for (int stride = BlockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_reduce[tid] = fmaxf(s_reduce[tid], s_reduce[tid + stride]);
        __syncthreads();
    }
    __shared__ float max_o_shared;
    if (tid == 0) max_o_shared = s_reduce[0];
    __syncthreads();
    float max_o = max_o_shared;

    if (max_o < kEps) return 1.0f;

    float t_end = static_cast<float>((1 << EX_BITS) - 1 + kNEnum) / max_o;
    float t_start = t_end * d_kTightStart_opt[EX_BITS];

    float* s_warp_ip = reuse_space + BlockSize;
    float* s_warp_t  = s_warp_ip + nWarps;

    // Tournament winners are broadcast through a dedicated shared slot
    // (never aliased by the workspace and never reused by the fine phase),
    // so the pre-existing post-tournament barriers are the only
    // synchronization needed — no publish/overwrite race by construction.
    __shared__ float s_best_t;

    // coarse grid search
    float best_coarse_ip = 0.0f;
    float best_coarse_t  = t_start;
    for (int base = 0; base < coarse_samples; base += nWarps) {
        int si = base + warp_id;
        float tc = (si < coarse_samples)
            ? t_start + (t_end - t_start) * si / coarse_denom
            : t_start;
        float ip = (si < coarse_samples)
            ? evaluate_rescale_sample_warp(s_xp_norm, D, EX_BITS, tc, lane_id)
            : 0.0f;
        if (lane_id == 0 && ip > best_coarse_ip) {
            best_coarse_ip = ip;
            best_coarse_t  = tc;
        }
    }

    if (lane_id == 0) {
        s_warp_ip[warp_id] = best_coarse_ip;
        s_warp_t[warp_id]  = best_coarse_t;
    }
    __syncthreads();

    if (warp_id == 0) {
        float ip = (lane_id < nWarps) ? s_warp_ip[lane_id] : -1.0f;
        float tc = (lane_id < nWarps) ? s_warp_t[lane_id]  : 0.0f;
        for (int s = 16; s > 0; s >>= 1) {
            float oi = __shfl_down_sync(0xffffffff, ip, s);
            float ot = __shfl_down_sync(0xffffffff, tc, s);
            if (oi > ip) { ip = oi; tc = ot; }
        }
        if (lane_id == 0) s_best_t = tc;
    }
    __syncthreads();

    float center_t = s_best_t;
    float range = (t_end - t_start) / coarse_samples;
    float fine_start = fmaxf(t_start, center_t - range);
    float fine_end   = fminf(t_end,   center_t + range);

    // fine grid search
    float best_fine_ip = 0.0f;
    float best_fine_t  = center_t;
    for (int base = 0; base < fine_samples; base += nWarps) {
        int si = base + warp_id;
        float tf = (si < fine_samples)
            ? fine_start + (fine_end - fine_start) * si / fine_denom
            : center_t;
        float ip = (si < fine_samples)
            ? evaluate_rescale_sample_warp(s_xp_norm, D, EX_BITS, tf, lane_id)
            : 0.0f;
        if (lane_id == 0 && ip > best_fine_ip) {
            best_fine_ip = ip;
            best_fine_t  = tf;
        }
    }

    if (lane_id == 0) {
        s_warp_ip[warp_id] = best_fine_ip;
        s_warp_t[warp_id]  = best_fine_t;
    }
    __syncthreads();

    if (warp_id == 0) {
        float ip = (lane_id < nWarps) ? s_warp_ip[lane_id] : -1.0f;
        float tf = (lane_id < nWarps) ? s_warp_t[lane_id]  : 0.0f;
        for (int s = 16; s > 0; s >>= 1) {
            float oi = __shfl_down_sync(0xffffffff, ip, s);
            float ot = __shfl_down_sync(0xffffffff, tf, s);
            if (oi > ip) { ip = oi; tf = ot; }
        }
        // Safe overwrite of s_best_t: the fine-publish barrier above ordered
        // every thread's read of the coarse center before it. Returning the
        // static slot also keeps the result immune to any workspace reuse
        // in the caller.
        if (lane_id == 0) s_best_t = tf;
    }
    __syncthreads();

    return s_best_t;
}

// ---------------------------------------------------------------------------
// Fully-fused kernel: generate a random Gaussian row, normalize, search for
// the optimal rescale factor. One block = one sample row.
// ---------------------------------------------------------------------------
__global__ void rabitq_rescale_sample_kernel(
        float* __restrict__ output_factors,
        int rows,
        int cols,
        int ex_bits,
        unsigned long long seed,
        int coarse_samples,
        int fine_samples)
{
    const int row_id = blockIdx.x;
    if (row_id >= rows) return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    extern __shared__ float shared_mem[];
    float* row_data    = shared_mem;
    float* reuse_space = &row_data[cols];

    curandState rng_state;
    curand_init(seed, row_id * block_size + tid, 0, &rng_state);

    for (int i = tid; i < cols; i += block_size) {
        row_data[i] = curand_normal(&rng_state);
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += block_size) {
        float val = row_data[i];
        local_sum += val * val;
    }

    float norm_squared = blockred::blockReduceSum(local_sum);

    __shared__ float inv_norm;
    if (tid == 0) inv_norm = rsqrtf(norm_squared);
    __syncthreads();

    for (int i = tid; i < cols; i += block_size) {
        row_data[i] = fabsf(row_data[i] * inv_norm);
    }
    __syncthreads();

    float rescale_factor = compute_best_rescale_parallel(
        row_data, cols, ex_bits, reuse_space, block_size,
        coarse_samples, fine_samples);

    if (tid == 0) output_factors[row_id] = rescale_factor;
}

// ---------------------------------------------------------------------------
// Host entry point — average the per-row factors on device.
// ---------------------------------------------------------------------------
float get_const_scaling_factor(raft::resources const& res,
                               size_t dim, size_t ex_bits,
                               uint64_t seed,
                               int coarse_samples,
                               int fine_samples) {
    constexpr long kConstNum = 100;

    auto stream    = raft::resource::get_cuda_stream(res);
    auto workspace = raft::resource::get_workspace_resource_ref(res);

    rmm::device_uvector<float> d_factors(static_cast<size_t>(kConstNum), stream, workspace);
    rmm::device_uvector<float> d_sum(1, stream, workspace);

    int block_size = 256;
    if (dim <= 512) block_size = 128;
    if (dim >= 1536) block_size = 512;

    size_t shared_mem_size = (dim + 3 * block_size) * sizeof(float);

    // Device properties of the *resource's* device (cached on `res`), not device 0.
    cudaDeviceProp const& prop = raft::resource::get_device_properties(res);
    if (shared_mem_size > prop.sharedMemPerBlock) {
        block_size = 128;
        shared_mem_size = (dim + 3 * block_size) * sizeof(float);
    }

    rabitq_rescale_sample_kernel<<<kConstNum, block_size, shared_mem_size, stream>>>(
        d_factors.data(), kConstNum, static_cast<int>(dim), static_cast<int>(ex_bits), seed,
        coarse_samples, fine_samples);
    RAFT_CUDA_TRY(cudaGetLastError());

    size_t temp_storage_bytes = 0;
    RAFT_CUDA_TRY(cub::DeviceReduce::Sum(
        nullptr, temp_storage_bytes, d_factors.data(), d_sum.data(), kConstNum, stream));

    rmm::device_uvector<char> d_temp_storage(temp_storage_bytes, stream, workspace);
    RAFT_CUDA_TRY(cub::DeviceReduce::Sum(
        d_temp_storage.data(), temp_storage_bytes, d_factors.data(), d_sum.data(), kConstNum,
        stream));

    float sum;
    RAFT_CUDA_TRY(
        cudaMemcpyAsync(&sum, d_sum.data(), sizeof(float), cudaMemcpyDeviceToHost, stream));
    raft::resource::sync_stream(res);

    return sum / kConstNum;
}

}  // namespace cuvs::preprocessing::quantize::rabitq::detail
