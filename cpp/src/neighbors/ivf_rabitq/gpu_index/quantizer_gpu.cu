/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 3/10/25.
//

#include "../../../core/map_launchers.hpp"
#include "../../../core/random_launchers.hpp"
#include "../../detail/smem_utils.cuh"
#include "../utils/reductions.cuh"
#include "quantizer_gpu.cuh"

#include "scaling_factor_launch.hpp"

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/map.cuh>
#include <raft/linalg/norm_types.hpp>
#include <raft/linalg/normalize.cuh>

#include <cub/device/device_reduce.cuh>

#include <atomic>
#include <queue>
#include <thread>

namespace cuvs::neighbors::ivf_rabitq::detail {

namespace {

// Forward declarations for free functions
void data_transformation_batch_opt(const float* d_data,
                                   const float* d_centroid,
                                   const PID* d_IDs,
                                   size_t num_points,
                                   const RotatorGPU& rotator,
                                   float* d_rotated_c,
                                   float* d_XP_norm,
                                   int* d_bin_XP,
                                   float* d_XP_output,
                                   float* d_X_and_C_pad,
                                   size_t DIM,
                                   size_t D,
                                   raft::resources const& handle);

void data_transformation_batch_opt_contiguous(const float* d_contiguous_data,
                                              const float* d_centroid,
                                              size_t num_points,
                                              const RotatorGPU& rotator,
                                              float* d_rotated_c,
                                              float* d_XP_norm,
                                              int* d_bin_XP,
                                              float* d_XP_output,
                                              float* d_X_and_C_pad,
                                              size_t DIM,
                                              size_t D,
                                              raft::resources const& handle);

void rabitq_codes_and_factors_fused(const float* d_rotated_c,
                                    const int* d_bin_XP,
                                    const float* d_XP,
                                    uint32_t* d_short_data,
                                    float* d_short_data_factors,
                                    size_t num_points,
                                    size_t D,
                                    raft::resources const& handle);

void exrabitq_codes_and_factors_fused(const int* d_bin_XP,
                                      const float* d_XP_norm,
                                      float* d_XP,
                                      uint8_t* d_long_code,
                                      float* d_ex_factor,
                                      const float* d_centroid,
                                      size_t num_points,
                                      size_t D,
                                      size_t EX_BITS,
                                      float const_scaling_factor,
                                      raft::resources const& handle);

//---------------------------------------------------------------------------
// Kernel: subtract_normalize_binarize_Kernel (Fused)
// This single kernel performs three sequential operations:
// 1. Subtracts the rotated centroid (d_CP) from each row of rotated data (d_XP).
// 2. Normalizes each resulting row (residual).
// 3. Binarizes the normalized row.
// It uses a block-per-row strategy with shared memory for the reduction.
//
// Template parameters:
// - BlockSize: The number of threads per block, should be a power of 2.
template <unsigned int BlockSize>
__global__ void subtract_normalize_binarize_Kernel(
  const float* __restrict__ d_XP,
  const float* __restrict__ d_CP,
  float* d_XP_residuals,  // Output for final residuals
  float* d_XP_norm,       // Output for normalized data
  int* d_bin_XP,          // Output for binarized data
  size_t num_points,
  size_t D)
{
  // Use one block to process one row.
  int row = blockIdx.x;
  if (row >= num_points) { return; }

  // Shared memory for the reduction and for broadcasting the final norm.
  extern __shared__ float s_mem[];
  float* s_partials = s_mem;  // Used for the reduction sum

  int tid_in_block    = threadIdx.x;
  float thread_sum_sq = 0.0f;

  // Each thread processes multiple elements if D > BlockSize
  for (int j = tid_in_block; j < D; j += BlockSize) {
    // Step 1: Calculate the residual (XP - CP)
    float residual = d_XP[row * D + j] - d_CP[j];

    // Step 2: Store the residual for later use and accumulate sum of squares
    thread_sum_sq += residual * residual;

    // Step 3 (part 1): Write the residual to its final output location
    d_XP_residuals[row * D + j] = residual;
  }

  s_partials[tid_in_block] = thread_sum_sq;
  __syncthreads();

  // Step 4: Parallel reduction within the block to find the total sum of squares
  for (unsigned int s = BlockSize / 2; s > 0; s >>= 1) {
    if (tid_in_block < s) { s_partials[tid_in_block] += s_partials[tid_in_block + s]; }
    __syncthreads();
  }

  // The final L2 norm is calculated by thread 0 and broadcast via shared memory
  if (tid_in_block == 0) { s_partials[0] = sqrtf(s_partials[0]); }
  __syncthreads();

  float norm = s_partials[0];

  // Step 5: Normalize and Binarize
  // Each thread calculates and writes the final normalized and binary values
  if (norm > 0 /*1e-6f*/) {  // Avoid division by zero
    for (int j = tid_in_block; j < D; j += BlockSize) {
      float residual         = d_XP_residuals[row * D + j];  // Read the just-written residual
      float normalized_val   = residual / norm;
      d_XP_norm[row * D + j] = normalized_val;
      d_bin_XP[row * D + j]  = (normalized_val > 0.0f) ? 1 : 0;
    }
  } else {  // Handle zero-norm case
    for (int j = tid_in_block; j < D; j += BlockSize) {
      d_XP_norm[row * D + j] = 0.0f;
      d_bin_XP[row * D + j]  = 0;
    }
  }
}

//---------------------------------------------------------------------------
// Kernel: gatherAndPadKernel (Combined)
// Populates a (num_points + 1) x D matrix, d_X_and_C_pad.
// - The first num_points rows are filled with data points padded to length D.
//   If d_IDs is non-null, rows are gathered from d_data via d_IDs;
//   if d_IDs is null, d_data is assumed contiguous and copied directly.
// - The last row (index num_points) is filled with the padded centroid.
__global__ void gatherAndPadKernel(const float* __restrict__ d_data,
                                   const PID* __restrict__ d_IDs,
                                   const float* __restrict__ d_centroid,
                                   float* d_X_and_C_pad,
                                   size_t num_points,
                                   size_t DIM,
                                   size_t D)
{
  // Calculate the global thread ID for a 1D grid over all elements.
  int global_idx        = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_elements = (num_points + 1) * D;

  if (global_idx < total_elements) {
    // Determine which row and column this thread is responsible for.
    int row_idx = global_idx / D;
    int col_idx = global_idx % D;

    if (row_idx < num_points) {
      // This thread is processing a data point.
      if (col_idx < DIM) {
        // Gather via d_IDs when provided; otherwise treat d_data as contiguous.
        size_t src_row = (d_IDs != nullptr) ? static_cast<size_t>(d_IDs[row_idx]) : row_idx;
        d_X_and_C_pad[global_idx] = d_data[src_row * DIM + col_idx];
      } else {
        // Pad with zero.
        d_X_and_C_pad[global_idx] = 0.0f;
      }
    } else {
      // This thread is processing the centroid (row_idx == num_points).
      if (col_idx < DIM) {
        // Copy from the centroid vector.
        d_X_and_C_pad[global_idx] = d_centroid[col_idx];
      } else {
        // Pad with zero.
        d_X_and_C_pad[global_idx] = 0.0f;
      }
    }
  }
}

//---------------------------------------------------------------------------
// Kernel: pack_and_compute_factors_kernel (Fused)
//
// This single kernel performs two independent operations for each data point (row):
// 1. Computes the RaBitQ factors (logic from RowwisePackedKernel).
// 2. Packs the binary representation into uint32_t codes (logic from pack_binary_kernel).
//
// It is launched with one block per data point (row).
__global__ void pack_and_compute_factors_kernel(
  // Inputs for factor computation
  const float* __restrict__ d_centroid,  // [D]
  const int* __restrict__ d_bin_XP,      // [N*D], values in {0,1}
  const float* __restrict__ d_XP,        // [N*D], residuals
  float kConstEpsilon,                   // e.g. 1.9f
  // Inputs for packing (d_bin_XP is shared)
  // Common parameters
  size_t N,
  size_t D,
  // Outputs
  float* __restrict__ d_factors,        // [3*N], packed factors
  uint32_t* __restrict__ d_packed_code  // [(D/32) * N], packed codes, transposed: all rows of a
                                        // word are contiguous (the layout the search kernels read)
)
{
  // Each block is responsible for one data point (row).
  size_t row = blockIdx.x;
  if (row >= N) return;

  //=========================================================================
  // Part 1: Compute RaBitQ factors (from RowwisePackedKernel)
  //=========================================================================
  float l2_sqr       = 0.f;
  float ip_resi_xucb = 0.f;
  float ip_cent_xucb = 0.f;
  float xu_sq        = 0.f;

  // Each thread in the block processes a subset of the columns
  for (size_t d = threadIdx.x; d < D; d += blockDim.x) {
    float res = d_XP[row * D + d];
    float xu  = float(d_bin_XP[row * D + d]) - 0.5f;
    float c   = d_centroid[d];

    l2_sqr += res * res;
    ip_resi_xucb += res * xu;
    ip_cent_xucb += c * xu;
    xu_sq += xu * xu;
  }

  // Perform parallel reduction within the block
  l2_sqr       = blockReduceSum(l2_sqr);
  ip_resi_xucb = blockReduceSum(ip_resi_xucb);
  ip_cent_xucb = blockReduceSum(ip_cent_xucb);
  xu_sq        = blockReduceSum(xu_sq);

  // Thread 0 performs the final calculations and writes the factors
  if (threadIdx.x == 0) {
    if (ip_resi_xucb == 0.0f) ip_resi_xucb = INFINITY;

    float l2_norm = sqrtf(fmaxf(l2_sqr, 0.f));
    float denom   = ip_resi_xucb;

    float fadd     = l2_sqr + 2.f * l2_sqr * (ip_cent_xucb / denom);
    float frescale = -2.f * l2_sqr / denom;

    float ratio     = (l2_sqr * xu_sq) / (denom * denom);
    float inner     = (ratio - 1.f) / fmaxf(float(D - 1), 1.f);
    inner           = fmaxf(inner, 0.f);
    float tmp_error = l2_norm * kConstEpsilon * sqrtf(inner);
    float ferr      = 2.f * tmp_error;

    size_t base         = 3 * row;
    d_factors[base + 0] = fadd;
    d_factors[base + 1] = frescale;
    d_factors[base + 2] = ferr;
  }

  //=========================================================================
  // Part 2: Pack binary codes (from pack_binary_kernel)
  //=========================================================================
  // Ensure all threads have finished with d_bin_XP for factor calculation
  // before potentially reusing registers for packing. A sync is good practice here.
  __syncthreads();

  // Each thread processes one or more blocks of 32 bits for the current row.
  size_t blocks_per_point = D / 32;
  for (size_t block_id = threadIdx.x; block_id < blocks_per_point; block_id += blockDim.x) {
    uint32_t cur = 0;
    // Process 32 bits for this block_id
    for (int i = 0; i < 32; i++) {
      int bit = d_bin_XP[row * D + block_id * 32 + i];
      cur |= ((uint32_t)bit << (31 - i));
    }
    d_packed_code[block_id * N + row] = cur;
  }
}

//---------------------------------------------------------------------------
// Kernel: exrabitq_fused_kernel_batch (Definitive Optimized Version)
//
// Fuses ExRaBitQ code generation and factor computation into a single kernel.
// Uses a "one block per row" strategy with shared memory, coalesced access,
// efficient reductions, and a fully parallel, minimal-read packing algorithm.
//
template <unsigned int BlockSize>
__global__ void exrabitq_fused_kernel_batch(
  // Inputs
  const int* __restrict__ d_bin_XP,
  const float* __restrict__ d_XP_norm,
  const float* __restrict__ d_XP,
  const float* __restrict__ d_centroid,
  size_t num_points,
  size_t D,
  size_t EX_BITS,
  float const_scaling_factor,
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
  for (int j = tid; j < D; j += BlockSize) {
    s_xp_norm[j] = d_XP_norm[row * D + j];
  }

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

void data_transformation_batch_opt(const float* d_data,
                                   const float* d_centroid,
                                   const PID* d_IDs,
                                   size_t num_points,
                                   const RotatorGPU& rotator,
                                   float* d_rotated_c,
                                   float* d_XP_norm,
                                   int* d_bin_XP,
                                   float* d_XP_output,
                                   float* d_X_and_C_pad,
                                   size_t DIM,
                                   size_t D,
                                   raft::resources const& handle)
{
  auto stream = raft::resource::get_cuda_stream(handle);

  // 1. Allocate a single temporary buffer for both padded data and the padded centroid.

  // 2. Launch a single kernel to gather and pad both data and centroid.
  int blockSize           = D < 256 ? 128 : 256;
  size_t totalPadElements = (num_points + 1) * D;
  int gridPadSize         = (totalPadElements + blockSize - 1) / blockSize;
  gatherAndPadKernel<<<gridPadSize, blockSize, 0, stream.get()>>>(
    d_data, d_IDs, d_centroid, d_X_and_C_pad, num_points, DIM, D);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // 3. Allocate a single output buffer for both rotated data (XP) and rotated centroid (CP).
  float* d_XP_and_CP = d_XP_output;

  // 4. Perform a single, combined rotation.
  // The input is d_X_and_C_pad, output is d_XP_and_CP. The number of "points" is num_points + 1.
  rotator.rotate(handle, d_X_and_C_pad, d_XP_and_CP, num_points + 1);

  // Create pointers to the specific results within the combined buffer.
  float* d_XP = d_XP_and_CP;
  float* d_CP = d_XP_and_CP + num_points * D;

  // 5. Save the rotated centroid: copy CP into d_rotated_c.
  raft::copy(d_rotated_c, d_CP, D, stream);

  // 6. Launch the single FUSED kernel for subtract, normalize, and binarize.
  const unsigned int FusedBlockSize = 256;  // A good default, can be tuned.
  dim3 gridDim(num_points);
  dim3 blockDim(FusedBlockSize);
  size_t sharedMemSize = FusedBlockSize * sizeof(float);

  subtract_normalize_binarize_Kernel<FusedBlockSize>
    <<<gridDim, blockDim, sharedMemSize, stream.get()>>>(
      d_XP,         // Input: Rotated data
      d_CP,         // Input: Rotated centroid
      d_XP_output,  // Output 1: Final residuals
      d_XP_norm,    // Output 2: Normalized residuals
      d_bin_XP,     // Output 3: Binarized data
      num_points,
      D);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

// Fused function to compute RaBitQ codes and factors in a single pass.
void rabitq_codes_and_factors_fused(const float* d_rotated_c,
                                    const int* d_bin_XP,
                                    const float* d_XP,
                                    uint32_t* d_short_data,
                                    float* d_short_data_factors,
                                    size_t num_points,
                                    size_t D,
                                    raft::resources const& handle)
{
  auto stream = raft::resource::get_cuda_stream(handle);

  int threads_per_block = 256;  // A good default, can be tuned
  dim3 grid(num_points);
  dim3 block(threads_per_block);

  pack_and_compute_factors_kernel<<<grid, block, 0, stream.get()>>>(
    d_rotated_c,
    d_bin_XP,
    d_XP,
    1.9f,  // kConstEpsilon, hardcoded from original call
    num_points,
    D,
    d_short_data_factors,
    d_short_data);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  raft::resource::sync_stream(handle);
}

//---------------------------------------------------------------------------
// Host function: exrabitq_codes_batch (Optimized)
// Launches a single fused kernel to compute ExRaBitQ codes and factors.
void exrabitq_codes_and_factors_fused(const int* d_bin_XP,
                                      const float* d_XP_norm,
                                      float* d_XP,
                                      uint8_t* d_long_code,
                                      float* d_ex_factor,
                                      const float* d_centroid,
                                      size_t num_points,
                                      size_t D,
                                      size_t EX_BITS,
                                      float const_scaling_factor,
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

  exrabitq_fused_kernel_batch<BlockSize>
    <<<gridDim, blockDim, shared_mem_size, stream.get()>>>(d_bin_XP,
                                                           d_XP_norm,
                                                           d_XP,
                                                           d_centroid,
                                                           num_points,
                                                           D,
                                                           EX_BITS,
                                                           const_scaling_factor,
                                                           1.9f,  // kConstEpsilon
                                                           d_long_code,
                                                           d_ex_factor);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  raft::resource::sync_stream(handle);
}

}  // anonymous namespace

void DataQuantizerGPU::quantize_batch_opt(const float* d_data,
                                          const float* d_centroid,
                                          const PID* d_IDs,
                                          size_t num_points,
                                          const RotatorGPU& rotator,
                                          uint32_t* d_short_data,
                                          float* d_short_data_factors,
                                          uint8_t* d_long_code,
                                          float* d_ex_factor,
                                          float* d_rotated_c)
{
  // 1. Data Transformation:
  data_transformation_batch_opt(d_data,
                                d_centroid,
                                d_IDs,
                                num_points,
                                rotator,
                                d_rotated_c,
                                d_XP_norm.data_handle(),
                                d_bin_XP.data_handle(),
                                d_XP.data_handle(),
                                d_X_and_C_pad.data_handle(),
                                DIM,
                                D,
                                handle_);

  rabitq_codes_and_factors_fused(d_rotated_c,
                                 d_bin_XP.data_handle(),
                                 d_XP.data_handle(),
                                 d_short_data,
                                 d_short_data_factors,
                                 num_points,
                                 D,
                                 handle_);

  // 5. Compute ExRaBitQ quantization codes.
  if (fast_quantize_flag) {
    exrabitq_codes_and_factors_fused(d_bin_XP.data_handle(),
                                     d_XP_norm.data_handle(),
                                     d_XP.data_handle(),
                                     d_long_code,
                                     d_ex_factor,
                                     d_rotated_c,
                                     num_points,
                                     D,
                                     EX_BITS,
                                     const_scaling_factor,
                                     handle_);
  } else {
    exrabitq_codes_and_factors_fused_ori(d_bin_XP.data_handle(),
                                         d_XP_norm.data_handle(),
                                         d_XP.data_handle(),
                                         d_long_code,
                                         d_ex_factor,
                                         d_rotated_c,
                                         num_points,
                                         D,
                                         EX_BITS,
                                         handle_);
  }
}

namespace {

void data_transformation_batch_opt_contiguous(const float* d_contiguous_data,
                                              const float* d_centroid,
                                              size_t num_points,
                                              const RotatorGPU& rotator,
                                              float* d_rotated_c,
                                              float* d_XP_norm,
                                              int* d_bin_XP,
                                              float* d_XP_output,
                                              float* d_X_and_C_pad,
                                              size_t DIM,
                                              size_t D,
                                              raft::resources const& handle)
{
  auto stream = raft::resource::get_cuda_stream(handle);

  // 1. Allocate a single temporary buffer for both padded data and the padded centroid.

  // 2. Launch a single kernel to copy, pad, and add centroid.
  int blockSize           = D < 256 ? 128 : 256;
  size_t totalPadElements = (num_points + 1) * D;
  int gridPadSize         = (totalPadElements + blockSize - 1) / blockSize;
  gatherAndPadKernel<<<gridPadSize, blockSize, 0, stream.get()>>>(
    d_contiguous_data, nullptr, d_centroid, d_X_and_C_pad, num_points, DIM, D);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // 3. Allocate a single output buffer for both rotated data (XP) and rotated centroid (CP).
  float* d_XP_and_CP = d_XP_output;

  // 4. Perform a single, combined rotation.
  // The input is d_X_and_C_pad, output is d_XP_and_CP. The number of "points" is num_points + 1.
  rotator.rotate(handle, d_X_and_C_pad, d_XP_and_CP, num_points + 1);

  // Create pointers to the specific results within the combined buffer.
  float* d_XP = d_XP_and_CP;
  float* d_CP = d_XP_and_CP + num_points * D;

  // 5. Save the rotated centroid: copy CP into d_rotated_c.
  raft::copy(d_rotated_c, d_CP, D, stream);

  // 6. Launch the single FUSED kernel for subtract, normalize, and binarize.
  const unsigned int FusedBlockSize = 256;  // A good default, can be tuned.
  dim3 gridDim(num_points);
  dim3 blockDim(FusedBlockSize);
  size_t sharedMemSize = FusedBlockSize * sizeof(float);

  subtract_normalize_binarize_Kernel<FusedBlockSize>
    <<<gridDim, blockDim, sharedMemSize, stream.get()>>>(
      d_XP,         // Input: Rotated data
      d_CP,         // Input: Rotated centroid
      d_XP_output,  // Output 1: Final residuals
      d_XP_norm,    // Output 2: Normalized residuals
      d_bin_XP,     // Output 3: Binarized data
      num_points,
      D);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // anonymous namespace

void DataQuantizerGPU::quantize_batch_opt_contiguous(const float* d_contiguous_data,
                                                     const float* d_centroid,
                                                     size_t num_points,
                                                     const RotatorGPU& rotator,
                                                     uint32_t* d_short_data,
                                                     float* d_short_data_factors,
                                                     uint8_t* d_long_code,
                                                     float* d_ex_factor,
                                                     float* d_rotated_c)
{
  // 1. Data Transformation:
  data_transformation_batch_opt_contiguous(d_contiguous_data,
                                           d_centroid,
                                           num_points,
                                           rotator,
                                           d_rotated_c,
                                           d_XP_norm.data_handle(),
                                           d_bin_XP.data_handle(),
                                           d_XP.data_handle(),
                                           d_X_and_C_pad.data_handle(),
                                           DIM,
                                           D,
                                           handle_);

  rabitq_codes_and_factors_fused(d_rotated_c,
                                 d_bin_XP.data_handle(),
                                 d_XP.data_handle(),
                                 d_short_data,
                                 d_short_data_factors,
                                 num_points,
                                 D,
                                 handle_);

  // 5. Compute ExRaBitQ quantization codes.
  if (fast_quantize_flag) {
    exrabitq_codes_and_factors_fused(d_bin_XP.data_handle(),
                                     d_XP_norm.data_handle(),
                                     d_XP.data_handle(),
                                     d_long_code,
                                     d_ex_factor,
                                     d_rotated_c,
                                     num_points,
                                     D,
                                     EX_BITS,
                                     const_scaling_factor,
                                     handle_);
  } else {
    exrabitq_codes_and_factors_fused_ori(d_bin_XP.data_handle(),
                                         d_XP_norm.data_handle(),
                                         d_XP.data_handle(),
                                         d_long_code,
                                         d_ex_factor,
                                         d_rotated_c,
                                         num_points,
                                         D,
                                         EX_BITS,
                                         handle_);
  }
}

namespace {

constexpr std::array<float, 9> kTightStart = {
  0,
  0.15,
  0.20,
  0.52,
  0.59,
  0.71,
  0.75,
  0.77,
  0.81,
};

template <typename T>
inline double best_rescale_factor(const T* o_abs, size_t dim, size_t ex_bits)
{
  constexpr double kEps = 1e-5;
  constexpr int kNEnum  = 10;
  double max_o          = *std::max_element(o_abs, o_abs + dim);

  double t_end   = static_cast<double>(((1 << ex_bits) - 1) + kNEnum) / max_o;
  double t_start = t_end * kTightStart[ex_bits];

  std::vector<int> cur_o_bar(dim);
  double sqr_denominator = static_cast<double>(dim) * 0.25;
  double numerator       = 0;

  for (size_t i = 0; i < dim; ++i) {
    int cur      = static_cast<int>((t_start * o_abs[i]) + kEps);
    cur_o_bar[i] = cur;
    sqr_denominator += cur * cur + cur;
    numerator += (cur + 0.5) * o_abs[i];
  }

  std::priority_queue<std::pair<double, size_t>,
                      std::vector<std::pair<double, size_t>>,
                      std::greater<>>
    next_t;

  for (size_t i = 0; i < dim; ++i) {
    next_t.emplace(static_cast<double>(cur_o_bar[i] + 1) / o_abs[i], i);
  }

  double max_ip = 0;
  double t      = 0;

  while (!next_t.empty()) {
    double cur_t     = next_t.top().first;
    size_t update_id = next_t.top().second;
    next_t.pop();

    cur_o_bar[update_id]++;
    int update_o_bar = cur_o_bar[update_id];
    sqr_denominator += 2 * update_o_bar;
    numerator += o_abs[update_id];

    double cur_ip = numerator / std::sqrt(sqr_denominator);
    if (cur_ip > max_ip) {
      max_ip = cur_ip;
      t      = cur_t;
    }

    if (update_o_bar < (1 << ex_bits) - 1) {
      double t_next = static_cast<double>(update_o_bar + 1) / o_abs[update_id];
      if (t_next < t_end) { next_t.emplace(t_next, update_id); }
    }
  }

  return t;
}

}  // anonymous namespace

float DataQuantizerGPU::get_const_scaling_factors(raft::resources const& handle,
                                                  size_t dim,
                                                  size_t ex_bits)
{
  constexpr long kConstNum = 100;

  // random matrix with normal distribution
  auto rand = raft::make_device_matrix<double, size_t>(handle, kConstNum, dim);
  raft::random::RngState rng(7ULL);
  cuvs::detail::random::normal(handle, rng, rand.data_handle(), kConstNum * dim, 0., 1.);
  // row-wise normalization
  auto row_normalized = raft::make_device_matrix<double, size_t>(handle, kConstNum, dim);
  raft::linalg::row_normalize<raft::linalg::L2Norm, double, size_t>(
    handle, rand.view(), row_normalized.view());
  // take abs values (reusing memory allocation for `rand`)
  cuvs::detail::map(handle,
                    rand.view(),
                    raft::abs_op{},
                    raft::make_device_vector_view<const double, size_t>(
                      row_normalized.data_handle(), kConstNum * dim));
  auto h_rand_row_normalized_abs = raft::make_host_matrix<double, size_t>(kConstNum, dim);
  raft::copy(h_rand_row_normalized_abs.data_handle(),
             rand.data_handle(),
             kConstNum * dim,
             raft::resource::get_cuda_stream(handle));
  raft::resource::sync_stream(handle);

  double sum = 0;
  for (long j = 0; j < kConstNum; ++j) {
    sum += best_rescale_factor(&h_rand_row_normalized_abs(j, 0), dim, ex_bits);
  }

  double t_const = sum / kConstNum;

  return (float)t_const;
}

float DataQuantizerGPU::get_const_scaling_factors_fully_gpu(size_t dim, size_t ex_bits)
{
  constexpr long kConstNum = 100;

  rmm::device_uvector<float> d_factors(kConstNum, stream_);
  rmm::device_uvector<float> d_sum(1, stream_);

  // Calculate block size (must be power of 2 for reductions)
  int block_size = 256;
  if (dim <= 512) block_size = 128;
  if (dim >= 1536) block_size = 512;

  // Calculate shared memory size
  size_t shared_mem_size = (dim +           // row_data
                            3 * block_size  // reuse_space for best_rescale
                            ) *
                           sizeof(float);

  unsigned long long seed = time(nullptr);
  launch_random_scaling_factors(
    d_factors.data(), kConstNum, dim, ex_bits, seed, block_size, shared_mem_size, stream_.get());
  RAFT_CUDA_TRY(cudaGetLastError());

  // Use CUB for reduction - handles any size optimally
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Sum(
    nullptr, temp_storage_bytes, d_factors.data(), d_sum.data(), kConstNum, stream_.get());

  {
    rmm::device_buffer d_temp_storage(temp_storage_bytes, stream_);
    cub::DeviceReduce::Sum(d_temp_storage.data(),
                           temp_storage_bytes,
                           d_factors.data(),
                           d_sum.data(),
                           kConstNum,
                           stream_.get());
  }
  RAFT_CUDA_TRY(cudaGetLastError());

  // Copy single value back and sync before reading host variable
  float sum;
  raft::copy(&sum, d_sum.data(), 1, stream_);
  raft::resource::sync_stream(handle_);

  return sum / kConstNum;
}

void DataQuantizerGPU::alloc_buffers(size_t num_points)
{
  const int64_t size_norm = static_cast<int64_t>(num_points) * D;
  const int64_t size_bin  = static_cast<int64_t>(num_points) * D;
  const int64_t size_xp   = static_cast<int64_t>(num_points + 1) * D;

  // Overwrite RAFT device vectors with new allocations
  d_XP_norm     = raft::make_device_vector<float, int64_t>(handle_, size_norm);
  d_bin_XP      = raft::make_device_vector<int, int64_t>(handle_, size_bin);
  d_XP          = raft::make_device_vector<float, int64_t>(handle_, size_xp);
  d_X_and_C_pad = raft::make_device_vector<float, int64_t>(handle_, size_xp);
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
