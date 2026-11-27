/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Packed RaBitQ quantization and per-vector factors on rotated residuals.

#include "../../../../neighbors/detail/smem_utils.cuh"
#include "quantize.cuh"
#include "reductions.cuh"
#include "rescale_search.cuh"
#include "tight_start_constants.cuh"

#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <raft/core/error.hpp>
#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/reduction.cuh>

namespace cuvs::preprocessing::quantize::rabitq::detail {

static __device__ __forceinline__ float evaluate_rescale_sample(
  const float* __restrict__ s_abs_norm, int D, int ex_bits, float t, int lane_id)
{
  constexpr float kEps = 1e-5f;
  int max_code         = (1 << ex_bits) - 1;
  float numerator      = 0.0f;
  float sqr_denom      = (lane_id == 0) ? static_cast<float>(D) * 0.25f : 0.0f;

  for (int j = lane_id; j < D; j += 32) {
    float val     = s_abs_norm[j];
    int quantized = min(__float2int_rd(t * val + kEps), max_code);
    numerator += (quantized + 0.5f) * val;
    sqr_denom += quantized * quantized + quantized;
  }

  numerator = raft::warpReduce(numerator);
  sqr_denom = raft::warpReduce(sqr_denom);

  return numerator / sqrtf(sqr_denom);
}

// ---------------------------------------------------------------------------
// Fused rescale search + quantize + factor kernel (fast and non-fast paths).
// Accumulate per-vector factors while the quantized values are in registers;
// packing then reads only the low-bit codes in shared memory. kFullFactors
// selects the factor set:
//   false: (delta, vl) via d_delta / d_vl / delta_mode
//   true : (f_add, f_rescale, f_error) via d_centroid / d_factors
// ---------------------------------------------------------------------------
template <int kBlockSize, bool kFullFactors>
__global__ void quantize_fused_kernel(const float* __restrict__ d_residual,
                                      packed_codes_output codes,
                                      int n_rows,
                                      int padded_dim,
                                      int ex_bits,
                                      float const_scaling_factor,
                                      bool use_fast,
                                      int n_coarse_samples,
                                      int n_fine_samples,
                                      float* __restrict__ d_delta,
                                      float* __restrict__ d_vl,
                                      int delta_mode,
                                      const float* __restrict__ d_centroid,
                                      float* __restrict__ d_factors)
{
  constexpr int kNWarps = kBlockSize / 32;
  constexpr float kEps  = 1e-5f;
  constexpr int kNEnum  = 10;
  int coarse_samples    = n_coarse_samples > 1 ? n_coarse_samples : 1;
  int fine_samples      = n_fine_samples > 1 ? n_fine_samples : 1;
  int coarse_denom      = coarse_samples > 1 ? coarse_samples - 1 : 1;
  int fine_denom        = fine_samples > 1 ? fine_samples - 1 : 1;

  extern __shared__ char smem[];
  float* s_abs_norm = reinterpret_cast<float*>(smem);
  float* s_warp_ip  = s_abs_norm + padded_dim;
  float* s_warp_t   = s_warp_ip + kNWarps;

  int vec_id = blockIdx.x;
  if (vec_id >= n_rows) return;

  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;

  const float* res = d_residual + (size_t)vec_id * padded_dim;
  // Scale search is finished before quantization starts, so its scratch can
  // also hold the low-bit codes awaiting packing.
  uint8_t* code = reinterpret_cast<uint8_t*>(smem);

  // norm (fused block reduction; result in every thread)
  float norm_sum[1] = {0.0f};
  for (int i = threadIdx.x; i < padded_dim; i += kBlockSize)
    norm_sum[0] += res[i] * res[i];
  norm_sum[0]    = blockred::blockAllReduceSum<kBlockSize>(norm_sum[0]);
  float inv_norm = rsqrtf(norm_sum[0] + 1e-30f);

  float t;

  if (use_fast || ex_bits == 0) {
    t = use_fast ? const_scaling_factor : 1.0f;
  } else {
    // cache abs(res)*inv_norm in smem + find block max. The barrier
    // inside block_max_all also publishes the s_abs_norm writes to the
    // whole block before the sampling loops below read them.
    float local_max = 0.0f;
    for (int i = threadIdx.x; i < padded_dim; i += kBlockSize) {
      float val     = fabsf(res[i]) * inv_norm;
      s_abs_norm[i] = val;
      local_max     = fmaxf(local_max, val);
    }
    float max_o = blockred::blockAllReduceMax<kBlockSize>(local_max);

    if (max_o < kEps) {
      t = 1.0f;
    } else {
      float t_end   = static_cast<float>((1 << ex_bits) - 1 + kNEnum) / max_o;
      float t_start = t_end * d_kTightStart_opt[ex_bits];

      // coarse grid
      float best_coarse_ip = 0.0f, best_coarse_t = t_start;
      for (int base = 0; base < coarse_samples; base += kNWarps) {
        int si = base + warp_id;
        float tc =
          (si < coarse_samples) ? t_start + (t_end - t_start) * si / coarse_denom : t_start;
        float ip = (si < coarse_samples)
                     ? evaluate_rescale_sample(s_abs_norm, padded_dim, ex_bits, tc, lane_id)
                     : 0.0f;
        if (lane_id == 0 && ip > best_coarse_ip) {
          best_coarse_ip = ip;
          best_coarse_t  = tc;
        }
      }

      // Tournament winners are broadcast through a dedicated shared
      // slot (never reused by the fine phase), so the pre-existing
      // post-tournament barrier is the only synchronization needed —
      // no publish/overwrite race by construction.
      __shared__ float s_best_t;

      if (lane_id == 0) {
        s_warp_ip[warp_id] = best_coarse_ip;
        s_warp_t[warp_id]  = best_coarse_t;
      }
      __syncthreads();
      if (warp_id == 0) {
        float ip = (lane_id < kNWarps) ? s_warp_ip[lane_id] : -1.0f;
        float tc = (lane_id < kNWarps) ? s_warp_t[lane_id] : 0.0f;
        for (int s = kNWarps / 2; s > 0; s >>= 1) {
          float oi = __shfl_down_sync(0xffffffff, ip, s);
          float ot = __shfl_down_sync(0xffffffff, tc, s);
          if (oi > ip) {
            ip = oi;
            tc = ot;
          }
        }
        if (lane_id == 0) s_best_t = tc;
      }
      __syncthreads();

      float center_t   = s_best_t;
      float range      = (t_end - t_start) / coarse_samples;
      float fine_start = fmaxf(t_start, center_t - range);
      float fine_end   = fminf(t_end, center_t + range);

      // fine grid
      float best_fine_ip = 0.0f, best_fine_t = center_t;
      for (int base = 0; base < fine_samples; base += kNWarps) {
        int si = base + warp_id;
        float tf =
          (si < fine_samples) ? fine_start + (fine_end - fine_start) * si / fine_denom : center_t;
        float ip = (si < fine_samples)
                     ? evaluate_rescale_sample(s_abs_norm, padded_dim, ex_bits, tf, lane_id)
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
        float ip = (lane_id < kNWarps) ? s_warp_ip[lane_id] : -1.0f;
        float tf = (lane_id < kNWarps) ? s_warp_t[lane_id] : 0.0f;
        for (int s = kNWarps / 2; s > 0; s >>= 1) {
          float oi = __shfl_down_sync(0xffffffff, ip, s);
          float ot = __shfl_down_sync(0xffffffff, tf, s);
          if (oi > ip) {
            ip = oi;
            tf = ot;
          }
        }
        // Safe overwrite of s_best_t: the fine-publish barrier above
        // ordered every thread's read of the coarse center before it.
        if (lane_id == 0) s_best_t = tf;
      }
      __syncthreads();
      t = s_best_t;
    }
  }

  // quantize (abs + branch) + accumulate the factor sums in the same pass
  int mask   = (1 << ex_bits) - 1;
  int offset = 1 << ex_bits;
  float cb   = -((float)(1 << ex_bits) - 0.5f);

  // Reconstruction uses { |u|^2, res . u }; distance factors use
  // { res . u, centroid . u, |u|^2 }. Quantization and output stores are shared.
  constexpr int kNumSums = kFullFactors ? 3 : 2;
  float sums[kNumSums]   = {};
  for (int i = threadIdx.x; i < padded_dim; i += kBlockSize) {
    float r       = res[i];
    float abs_val = fabsf(r) * inv_norm;
    int k         = __float2int_rd(t * abs_val + kEps);
    if (k > mask) k = mask;
    int total = (r >= 0.0f) ? (k + offset) : (mask - k);
    if (ex_bits == 8) {
      codes.extra_bits[static_cast<size_t>(vec_id) * padded_dim + i] = total & mask;
    } else if (ex_bits > 0) {
      code[i] = static_cast<uint8_t>(total & mask);
    }
    uint32_t signs = __brev(__ballot_sync(0xffffffff, r >= 0.0f));
    if (lane_id == 0) { codes.signs[(i / 32) * codes.sign_stride + vec_id] = signs; }
    float u = (float)total + cb;
    if constexpr (kFullFactors) {
      sums[0] += r * u;
      sums[1] += d_centroid[i] * u;
      sums[2] += u * u;
    } else {
      sums[0] += u * u;
      sums[1] += r * u;
    }
  }
  blockred::blockReduceSum<kBlockSize>(sums);
  if (threadIdx.x == 0) {
    if constexpr (kFullFactors) {
      constexpr float kEpsilon = 1.9f;
      float l2_sq              = norm_sum[0];
      float ip_resi_xucb       = sums[0];
      float ip_cent_xucb       = sums[1];
      float xu_sq              = sums[2];

      float l2_norm = sqrtf(l2_sq);
      float denom   = ip_resi_xucb + 1e-30f;

      float f_add     = l2_sq + 2.0f * l2_sq * (ip_cent_xucb / denom);
      float f_rescale = -2.0f * l2_sq / denom;

      float ratio   = (l2_sq * xu_sq) / (denom * denom);
      float inner   = fmaxf(0.0f, (ratio - 1.0f) / ((float)padded_dim - 1.0f));
      float f_error = 2.0f * l2_norm * kEpsilon * sqrtf(inner);

      float* factors = d_factors + (size_t)vec_id * 3;
      factors[0]     = f_add;
      factors[1]     = f_rescale;
      factors[2]     = f_error;
    } else {
      float norm_res = sqrtf(norm_sum[0]);
      float norm_ucb = sqrtf(sums[0]);
      float cos_sim  = sums[1] / (norm_res * norm_ucb + 1e-30f);

      float ratio = norm_res / (norm_ucb + 1e-30f);
      float delta;
      if (delta_mode == 1)
        delta = ratio / (cos_sim + 1e-30f);
      else if (delta_mode == 2)
        delta = ratio;
      else
        delta = ratio * cos_sim;
      d_delta[vec_id] = delta;
      d_vl[vec_id]    = delta * cb;
    }
  }

  // For 1–7 extra bits, quantization leaves one code byte per dimension in
  // shared memory. Each thread packs a complete destination byte so stores
  // cannot race.
  if (ex_bits > 0 && ex_bits < 8) {
    __syncthreads();
    const int bytes_per_row = padded_dim * ex_bits / 8;
    uint8_t* out            = codes.extra_bits + static_cast<size_t>(vec_id) * bytes_per_row;
    for (int byte = threadIdx.x; byte < bytes_per_row; byte += kBlockSize) {
      uint8_t packed = 0;
      for (int bit = 0; bit < 8; ++bit) {
        int position = byte * 8 + bit;
        int dim      = position / ex_bits;
        int shift    = ex_bits - 1 - position % ex_bits;
        packed |= ((code[dim] >> shift) & 1u) << (7 - bit);
      }
      out[byte] = packed;
    }
  }
}

// ---------------------------------------------------------------------------
// Launcher + free-function entry points.
// ---------------------------------------------------------------------------
// Keys the shared-memory state per kernel. Both output modes have the same signature but not the
// same static shared memory, so one must not be raised to the other's limit.
template <bool kFullFactors>
struct quantize_smem_key {};

template <bool kFullFactors>
static void launch_quantize(cudaStream_t stream,
                            const float* d_residual,
                            size_t n_rows,
                            uint32_t padded_dim,
                            size_t ex_bits,
                            float const_scaling_factor,
                            bool use_fast,
                            int delta_mode,
                            int coarse_samples,
                            int fine_samples,
                            packed_codes_output codes,
                            float* d_delta,
                            float* d_vl,
                            const float* d_centroid,
                            float* d_factors)
{
  constexpr int block  = 256;
  int rows             = static_cast<int>(n_rows);
  int iD               = static_cast<int>(padded_dim);
  int iB               = static_cast<int>(ex_bits);
  constexpr int nwarps = block / 32;

  // Reuse scale-search scratch for packing. Zero and eight extra bits write
  // directly to their final buffers and need no packing scratch.
  const size_t search_smem =
    (use_fast || ex_bits == 0) ? 0 : (padded_dim + 2 * nwarps) * sizeof(float);
  const size_t packing_smem = ex_bits > 0 && ex_bits < 8 ? padded_dim : 0;
  const size_t q_smem       = std::max(search_smem, packing_smem);
  auto kernel               = &quantize_fused_kernel<block, kFullFactors>;
  cudaKernel_t cuda_kernel;
  RAFT_CUDA_TRY(cudaGetKernel(&cuda_kernel, reinterpret_cast<const void*>(kernel)));
  cuvs::neighbors::detail::safely_launch_kernel_with_smem_size<quantize_smem_key<kFullFactors>>(
    static_cast<std::uint32_t>(q_smem),
    [&]() {
      kernel<<<rows, block, q_smem, stream>>>(d_residual,
                                              codes,
                                              rows,
                                              iD,
                                              iB,
                                              const_scaling_factor,
                                              use_fast,
                                              coarse_samples,
                                              fine_samples,
                                              d_delta,
                                              d_vl,
                                              delta_mode,
                                              d_centroid,
                                              d_factors);
    },
    cuda_kernel);
  RAFT_CUDA_TRY(cudaGetLastError());
}

void quantize_fused_on_residuals(
  cudaStream_t stream,
  raft::device_matrix_view<const float, int64_t, raft::row_major> residuals,
  size_t ex_bits,
  float const_scaling_factor,
  bool use_fast,
  packed_codes_output codes,
  raft::device_vector_view<float, int64_t> delta,
  raft::device_vector_view<float, int64_t> vl,
  int delta_mode,
  int coarse_samples,
  int fine_samples)
{
  launch_quantize<false>(stream,
                         residuals.data_handle(),
                         static_cast<size_t>(residuals.extent(0)),
                         static_cast<uint32_t>(residuals.extent(1)),
                         ex_bits,
                         const_scaling_factor,
                         use_fast,
                         delta_mode,
                         coarse_samples,
                         fine_samples,
                         codes,
                         delta.data_handle(),
                         vl.data_handle(),
                         nullptr,
                         nullptr);
}

void quantize_full_on_residuals(
  cudaStream_t stream,
  raft::device_matrix_view<const float, int64_t, raft::row_major> residuals,
  raft::device_vector_view<const float, int64_t> centroid,
  size_t ex_bits,
  float const_scaling_factor,
  bool use_fast,
  packed_codes_output codes,
  raft::device_matrix_view<float, int64_t, raft::row_major> factors,
  int coarse_samples,
  int fine_samples)
{
  launch_quantize<true>(stream,
                        residuals.data_handle(),
                        static_cast<size_t>(residuals.extent(0)),
                        static_cast<uint32_t>(residuals.extent(1)),
                        ex_bits,
                        const_scaling_factor,
                        use_fast,
                        0,
                        coarse_samples,
                        fine_samples,
                        codes,
                        nullptr,
                        nullptr,
                        centroid.data_handle(),
                        factors.data_handle());
}

}  // namespace cuvs::preprocessing::quantize::rabitq::detail
