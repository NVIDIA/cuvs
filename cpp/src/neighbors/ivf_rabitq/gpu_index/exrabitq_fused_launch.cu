/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../utils/reductions.cuh"
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>

namespace cuvs::neighbors::ivf_rabitq::detail {

namespace {

template <unsigned int BlockSize>
__global__ void exrabitq_fused_kernel_batch_ori(
  // Inputs
  const int* __restrict__ d_bin_XP,
  const float* __restrict__ d_XP_norm,
  const float* __restrict__ d_XP,
  const float* __restrict__ d_centroid,
  size_t num_points,
  size_t D,
  size_t EX_BITS,
  float kConstEpsilon,
  // Outputs
  uint8_t* d_long_code,
  float* d_ex_factor)
{
  //=========================================================================
  // Setup: One block per row
  //=========================================================================
  int row = blockIdx.x;
  if (row >= num_points) return;

  // Dynamically allocated shared memory for one row's data.
  extern __shared__ float s_mem[];
  float* s_xp_norm    = s_mem;
  uint8_t* s_tmp_code = (uint8_t*)(s_xp_norm + D);

  int tid = threadIdx.x;

  //=========================================================================
  // Step 1: Coalesced load of all necessary data into shared memory
  //=========================================================================
  // Load data
  for (int j = tid; j < D; j += BlockSize) {
    s_xp_norm[j] = d_XP_norm[row * D + j];
    // Note: s_bin_xp, s_xp not loaded yet - we'll reuse as workspace
  }
  // Compute scaling factor (reusing unused buffers)
  float const_scaling_factor =
    compute_best_rescale_parallel(s_xp_norm,
                                  D,
                                  EX_BITS,
                                  (s_xp_norm + D),  // Reused as workspace
                                  BlockSize);

  //=========================================================================
  // Part A: ExRaBitQ Code Generation
  //=========================================================================
  const int mask          = (1 << EX_BITS) - 1;
  float thread_ipnorm_sum = 0.0f;

  // Parallel quantization and start of ip_norm reduction
  for (int j = tid; j < D; j += BlockSize) {
    float val    = fabsf(s_xp_norm[j]);
    int code_val = static_cast<int>((const_scaling_factor * val) + 1e-5f);
    if (code_val >= (1 << EX_BITS)) code_val = (1 << EX_BITS) - 1;
    s_tmp_code[j] = static_cast<uint8_t>(code_val);
    thread_ipnorm_sum += (code_val + 0.5f) * val;
  }

  // Parallel bit-flipping
  for (int j = tid; j < D; j += BlockSize) {
    if (d_bin_XP[row * D + j] == 0) { s_tmp_code[j] = (~s_tmp_code[j]) & mask; }
  }

  // Finish ip_norm reduction
  float total_ipnorm = blockReduceSum(thread_ipnorm_sum);
  float ip_norm_inv  = 1.0f;
  if (tid == 0) {
    float inv   = 1.0f / total_ipnorm;
    ip_norm_inv = isfinite(inv) ? inv : 1.0f;
  }
  // Broadcast ip_norm_inv to all threads in the block
  ip_norm_inv = __shfl_sync(0xffffffff, ip_norm_inv, 0);

  //=========================================================================
  // Part B: Factor Computation
  //=========================================================================
  float l2_sqr = 0.f, ip_resi_xucb = 0.f, ip_cent_xucb = 0.f, xu_sq = 0.f;

  for (size_t j = tid; j < D; j += BlockSize) {
    float res  = d_XP[row * D + j];
    int xu_pre = s_tmp_code[j];
    xu_pre += static_cast<int>(res >= 0) << EX_BITS;
    float xu = float(xu_pre) - (static_cast<float>(1 << EX_BITS) - 0.5f);
    float c  = d_centroid[j];
    l2_sqr += res * res;
    ip_resi_xucb += res * xu;
    ip_cent_xucb += c * xu;
    xu_sq += xu * xu;
  }

  // Perform parallel reductions for all factor components
  l2_sqr       = blockReduceSum(l2_sqr);
  ip_resi_xucb = blockReduceSum(ip_resi_xucb);
  ip_cent_xucb = blockReduceSum(ip_cent_xucb);
  xu_sq        = blockReduceSum(xu_sq);

  // Thread 0 computes and writes the final factors
  if (tid == 0) {
    float denom = ip_resi_xucb;
    if (denom == 0.0f) denom = INFINITY;
    float l2_norm = sqrtf(fmaxf(l2_sqr, 0.f));

    float fadd     = l2_sqr + 2.f * l2_sqr * ip_cent_xucb / denom;
    float frescale = -2.f * l2_norm * ip_norm_inv;

    float ratio     = (l2_sqr * xu_sq) / (ip_resi_xucb * ip_resi_xucb);
    float inner     = (ratio - 1.f) / fmaxf(float(D - 1), 1.f);
    float tmp_error = l2_norm * kConstEpsilon * sqrtf(fmaxf(inner, 0.f));

    size_t base           = 2 * row;
    d_ex_factor[base + 0] = fadd;
    d_ex_factor[base + 1] = frescale;
  }

  //=========================================================================
  // Part C: Pack and Write Long Code (MINIMAL READS, PARALLEL, COALESCED)
  //=========================================================================
  int long_code_length = (D * EX_BITS + 7) / 8;
  uint8_t* out_ptr     = d_long_code + row * long_code_length;

  if (EX_BITS == 8) {
    for (int j = tid; j < D; j += BlockSize)
      out_ptr[j] = s_tmp_code[j];
  } else {
    const int num_codes_to_read = (8 + EX_BITS - 2) / EX_BITS + 1;
    for (int out_byte_idx = tid; out_byte_idx < long_code_length; out_byte_idx += BlockSize) {
      int start_bit       = out_byte_idx * 8;
      int start_code_idx  = start_bit / EX_BITS;
      int bit_offset      = start_bit % EX_BITS;
      uint64_t bit_buffer = 0;

      switch (num_codes_to_read) {
        case 2:
          if (start_code_idx < D) bit_buffer |= (uint64_t)s_tmp_code[start_code_idx];
          bit_buffer <<= EX_BITS;
          if (start_code_idx + 1 < D) bit_buffer |= (uint64_t)s_tmp_code[start_code_idx + 1];
          break;
        case 3:
          if (start_code_idx < D) bit_buffer |= (uint64_t)s_tmp_code[start_code_idx];
          bit_buffer <<= EX_BITS;
          if (start_code_idx + 1 < D) bit_buffer |= (uint64_t)s_tmp_code[start_code_idx + 1];
          bit_buffer <<= EX_BITS;
          if (start_code_idx + 2 < D) bit_buffer |= (uint64_t)s_tmp_code[start_code_idx + 2];
          break;
        case 4:
          if (start_code_idx < D) bit_buffer |= (uint64_t)s_tmp_code[start_code_idx];
          bit_buffer <<= EX_BITS;
          if (start_code_idx + 1 < D) bit_buffer |= (uint64_t)s_tmp_code[start_code_idx + 1];
          bit_buffer <<= EX_BITS;
          if (start_code_idx + 2 < D) bit_buffer |= (uint64_t)s_tmp_code[start_code_idx + 2];
          bit_buffer <<= EX_BITS;
          if (start_code_idx + 3 < D) bit_buffer |= (uint64_t)s_tmp_code[start_code_idx + 3];
          break;
        default:
          for (int k = 0; k < num_codes_to_read; ++k) {
            bit_buffer <<= EX_BITS;
            if (start_code_idx + k < D) bit_buffer |= (uint64_t)s_tmp_code[start_code_idx + k];
          }
          break;
      }
      int total_bits_in_buffer = num_codes_to_read * EX_BITS;
      int shift                = total_bits_in_buffer - bit_offset - 8;
      uint8_t out_byte         = (shift >= 0) ? (uint8_t)(bit_buffer >> shift) : 0;
      out_ptr[out_byte_idx]    = out_byte;
    }
  }
}

}  // namespace

void exrabitq_codes_and_factors_fused_ori(const int* d_bin_XP,
                                          const float* d_XP_norm,
                                          float* d_XP,
                                          uint8_t* d_long_code,
                                          float* d_ex_factor,
                                          const float* d_centroid,
                                          size_t num_points,
                                          size_t D,
                                          size_t EX_BITS,
                                          raft::resources const& handle)
{
  auto stream = raft::resource::get_cuda_stream(handle);

  const unsigned int BlockSize = 256;  // A good default, can be tuned
  dim3 gridDim(num_points);
  dim3 blockDim(BlockSize);

  // Calculate required shared memory size
  size_t shared_mem_size = D * sizeof(float) +         // s_xp_norm
                           D * sizeof(uint8_t) +       // s_tmp_code
                           BlockSize * sizeof(float);  // s_partials for reduction

  if (shared_mem_size <= D * sizeof(float) + 3 * BlockSize * sizeof(float)) {
    shared_mem_size =
      D * sizeof(float) +
      3 * BlockSize *
        sizeof(float);  // Make sure there are space to reuse for finding scaling factors
  }

  exrabitq_fused_kernel_batch_ori<BlockSize>
    <<<gridDim, blockDim, shared_mem_size, stream.get()>>>(d_bin_XP,
                                                           d_XP_norm,
                                                           d_XP,
                                                           d_centroid,
                                                           num_points,
                                                           D,
                                                           EX_BITS,
                                                           1.9f,  // kConstEpsilon
                                                           d_long_code,
                                                           d_ex_factor);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  raft::resource::sync_stream(handle);
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
