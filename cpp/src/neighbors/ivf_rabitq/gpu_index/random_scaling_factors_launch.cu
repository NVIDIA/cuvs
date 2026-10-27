/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../utils/reductions.cuh"
#include "scaling_factor_launch.hpp"
#include <curand_kernel.h>
#include <neighbors/detail/smem_utils.cuh>
#include <raft/util/cudart_utils.hpp>

namespace cuvs::neighbors::ivf_rabitq::detail {

namespace {

__global__ void fully_fused_kernel(float* __restrict__ output_factors,
                                   const int rows,
                                   const int cols,
                                   const int ex_bits,
                                   unsigned long long seed)
{
  const int row_id = blockIdx.x;
  if (row_id >= rows) return;

  const int tid        = threadIdx.x;
  const int block_size = blockDim.x;

  // Calculate shared memory layout
  // row_data: cols floats
  // reuse_space: 3 * block_size floats (for best_rescale computation)
  extern __shared__ float shared_mem[];
  float* row_data    = shared_mem;
  float* reuse_space = &row_data[cols];  // No reduction_buffer needed!

  // Initialize RNG state per thread
  curandState rng_state;
  curand_init(seed, row_id * block_size + tid, 0, &rng_state);

  // Generate random Gaussian values and calculate L2 norm
  float local_sum = 0.0f;
  for (int i = tid; i < cols; i += block_size) {
    float val   = curand_normal(&rng_state);
    row_data[i] = val;
    local_sum += val * val;
  }

  float norm_squared = blockReduceSum(local_sum);

  // full block sum only available on lane 0 of warp 0 (see definition of blockReduceSum); need to
  // make inv_norm a shared variable
  __shared__ float inv_norm;
  if (tid == 0) { inv_norm = rsqrtf(norm_squared); }
  __syncthreads();

  for (int i = tid; i < cols; i += block_size) {
    row_data[i] = fabsf(row_data[i] * inv_norm);
  }

  float rescale_factor =
    compute_best_rescale_parallel(row_data, cols, ex_bits, reuse_space, block_size);

  if (tid == 0) { output_factors[row_id] = rescale_factor; }
}

}  // namespace

void launch_random_scaling_factors(float* output,
                                   int rows,
                                   int cols,
                                   int ex_bits,
                                   unsigned long long seed,
                                   int block_size,
                                   size_t shared_mem_size,
                                   cudaStream_t stream)
{
  auto kernel_fn      = fully_fused_kernel;
  auto const launcher = [&]() {
    kernel_fn<<<rows, block_size, shared_mem_size, stream>>>(output, rows, cols, ex_bits, seed);
  };
  cudaKernel_t cuda_kernel;
  RAFT_CUDA_TRY(cudaGetKernel(&cuda_kernel, reinterpret_cast<const void*>(kernel_fn)));
  cuvs::neighbors::detail::safely_launch_kernel_with_smem_size<std::decay_t<decltype(kernel_fn)>>(
    static_cast<uint32_t>(shared_mem_size), launcher, cuda_kernel);
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
