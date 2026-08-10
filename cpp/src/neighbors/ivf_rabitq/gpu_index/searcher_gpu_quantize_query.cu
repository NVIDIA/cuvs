/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 4/14/25.
//

// This file implements `SearcherGPU::SearchClusterQueryPairsQuantizeQuery`.
#include "../../detail/smem_utils.cuh"
#include "../jit_lto_kernels/kernel_def.hpp"
#include "../jit_lto_kernels/launcher_factory.hpp"
#include "../utils/reductions.cuh"
#include "../utils/searcher_gpu_utils.hpp"
#include "searcher_gpu.cuh"
#include "searcher_gpu_common.cuh"

#include <cuvs/selection/select_k.hpp>
#include <raft/matrix/detail/select_warpsort.cuh>

#include <cub/block/block_reduce.cuh>

#include <thrust/fill.h>

#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <limits>

namespace cuvs::neighbors::ivf_rabitq::detail {

//---------------------------------------------------------------------------
// Kernel: exrabitq_quantize_query
//
// Quantize queries using exrabitq implementation, the output are always int8_t array
//
template <unsigned int BlockSize>
__global__ void exrabitq_quantize_query(
  // Inputs
  const float* __restrict__ d_XP,
  size_t num_points,
  size_t D,
  size_t EX_BITS,
  float const_scaling_factor,
  float kConstEpsilon,
  // Outputs
  int8_t* d_long_code,
  float* d_delta)
{
  //=========================================================================
  // Setup: One block per row
  //=========================================================================
  int row = blockIdx.x;
  if (row >= num_points) return;

  // Dynamically allocated shared memory for one row's data.
  extern __shared__ float s_mem[];
  float* s_xp     = s_mem;
  float* s_reduce = s_xp + D;  // For reduction

  int tid = threadIdx.x;

  //=========================================================================
  // Step 0: Load XP and compute L2 nrom && normalize
  //=========================================================================
  float thread_sum_sq = 0.0f;

  // local L2 norm
  for (int j = tid; j < D; j += BlockSize) {
    float xp_val = d_XP[row * D + j];
    s_xp[j]      = xp_val;
    thread_sum_sq += xp_val * xp_val;  // Direct L2 norm of XP
  }

  s_reduce[tid] = thread_sum_sq;
  __syncthreads();

  // global reduction
  for (unsigned int stride = BlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) { s_reduce[tid] += s_reduce[tid + stride]; }
    __syncthreads();
  }

  float norm     = sqrtf(s_reduce[0]);
  float norm_inv = (norm > 0) ? (1.0f / norm) : 0.0f;

  //=========================================================================
  // Part A+B+C: Quantize, accumulate factors, and write codes
  //=========================================================================
  // Each thread fully owns the j values it visits, so code_val stays in a
  // register: there is no need to materialize an int8_t scratch in shared
  // memory or to read it back in separate passes.
  const int max_code = (1 << (EX_BITS - 1)) - 1;
  const int min_code = -(1 << (EX_BITS - 1));
  int8_t* out_ptr    = d_long_code + row * D;  // long_code_length == D

  float ip_resi_xucb = 0.f, xu_sq = 0.f;
  for (int j = tid; j < D; j += BlockSize) {
    float val    = s_xp[j] * norm_inv;
    int code_val = __float2int_rn(const_scaling_factor * val);  // round-to-nearest-even
    code_val     = std::clamp(code_val, min_code, max_code);

    float xu = float(code_val);
    // just ignore the 0.5 since we are not going to store extra shift
    ip_resi_xucb += s_xp[j] * xu;  // for cos_similarity
    xu_sq += xu * xu;              // norm_quan^2
    out_ptr[j] = static_cast<int8_t>(code_val);
  }

  // only thread 0 in the block needs the results, so simply use blockReduceSum
  ip_resi_xucb = blockReduceSum(ip_resi_xucb);
  xu_sq        = blockReduceSum(xu_sq);

  // Thread 0 computes and writes the final factors
  if (tid == 0) {
    float norm_quan = sqrtf(fmaxf(xu_sq, 0.f));
    float delta;
    if (norm > 1e-6f && norm_quan > 1e-6f) {
      float cos_similarity = ip_resi_xucb / (norm * norm_quan);
      delta                = norm / norm_quan * cos_similarity;
    } else {
      delta = 0.f;
    }

    size_t base   = row;
    d_delta[base] = delta;
  }
}

__global__ void findQueryRanges(const float* __restrict__ queries,
                                float* __restrict__ query_ranges,
                                int num_queries,
                                int num_dimensions)
{
  const int query_idx = blockIdx.x;
  if (query_idx >= num_queries) return;

  const float* query = queries + query_idx * num_dimensions;

  using BlockReduceFloat = cub::BlockReduce<float, 256>;
  __shared__ typename BlockReduceFloat::TempStorage temp_storage_min;
  __shared__ typename BlockReduceFloat::TempStorage temp_storage_max;

  float local_min = FLT_MAX;
  float local_max = -FLT_MAX;

  for (int i = threadIdx.x; i < num_dimensions; i += blockDim.x) {
    float val = query[i];
    local_min = fminf(local_min, val);
    local_max = fmaxf(local_max, val);
  }

  float block_min = BlockReduceFloat(temp_storage_min).Reduce(local_min, cuda::minimum<>{});
  float block_max = BlockReduceFloat(temp_storage_max).Reduce(local_max, cuda::maximum<>{});

  if (threadIdx.x == 0) {
    query_ranges[query_idx * 2]     = block_min;
    query_ranges[query_idx * 2 + 1] = block_max;
  }
}

__global__ void quantizeQueriesToInt8(const float* __restrict__ queries,
                                      const float* __restrict__ query_ranges,
                                      int8_t* __restrict__ quantized_queries,
                                      float* __restrict__ widths,
                                      int num_queries,
                                      int num_dimensions)
{
  const int BQ            = 8;                    // Use full 8-bit range
  const float max_int_val = (1 << (BQ - 1)) - 1;  // 127

  int idx            = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = num_queries * num_dimensions;

  for (int i = idx; i < total_elements; i += gridDim.x * blockDim.x) {
    int query_idx = i / num_dimensions;
    int dim_idx   = i % num_dimensions;

    float vmin     = query_ranges[query_idx * 2];
    float vmax     = query_ranges[query_idx * 2 + 1];
    float vmax_abs = fmaxf(fabsf(vmin), fabsf(vmax));

    float width          = vmax_abs / max_int_val;
    float one_over_width = (width > 0) ? 1.0f / width : 0.0f;

    if (dim_idx == 0) { widths[query_idx] = width; }

    float val        = queries[query_idx * num_dimensions + dim_idx];
    float scaled     = val * one_over_width;
    scaled           = fmaxf(-128.0f, fminf(127.0f, scaled));
    int8_t quantized = (int8_t)__float2int_rn(scaled);

    quantized_queries[query_idx * num_dimensions + dim_idx] = quantized;
  }
}

__global__ void packInt8QueryBitPlanes(const int8_t* __restrict__ queries,
                                       uint32_t* __restrict__ packed_queries,
                                       int num_queries,
                                       int num_dimensions)
{
  const int dims_per_word = 32;
  const int num_words     = (num_dimensions + dims_per_word - 1) / dims_per_word;
  const int num_bits      = 8;

  int idx            = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = num_queries * num_bits * num_words;

  for (int i = idx; i < total_elements; i += gridDim.x * blockDim.x) {
    int query_idx = i / (num_bits * num_words);
    int remainder = i % (num_bits * num_words);
    int bit_idx   = remainder / num_words;
    int word_idx  = remainder % num_words;

    uint32_t packed_word = 0;

#pragma unroll 8
    for (int d = 0; d < dims_per_word; ++d) {
      int dim_idx = word_idx * dims_per_word + d;
      if (dim_idx < num_dimensions) {
        uint8_t val      = (uint8_t)queries[query_idx * num_dimensions + dim_idx];
        uint32_t bit_val = (val >> bit_idx) & 1;

        // FIXED: Match data bit ordering - dim 0 goes to bit 31, dim 31 to bit 0
        int bit_position = 31 - d;  // Reverse bit ordering!
        packed_word |= (bit_val << bit_position);
      }
    }

    packed_queries[i] = packed_word;
  }
}

__global__ void quantizeQueriesToInt4(
  const float* __restrict__ queries,
  const float* __restrict__ query_ranges,
  int8_t* __restrict__ quantized_queries,  // Still use int8_t for storage
  float* __restrict__ widths,
  int num_queries,
  int num_dimensions)
{
  const int BQ            = 4;                    // Use 4-bit range
  const float max_int_val = (1 << (BQ - 1)) - 1;  // 2^3 - 1 = 7

  int idx            = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = num_queries * num_dimensions;

  for (int i = idx; i < total_elements; i += gridDim.x * blockDim.x) {
    int query_idx = i / num_dimensions;
    int dim_idx   = i % num_dimensions;

    float vmin     = query_ranges[query_idx * 2];
    float vmax     = query_ranges[query_idx * 2 + 1];
    float vmax_abs = fmaxf(fabsf(vmin), fabsf(vmax));

    float width          = vmax_abs / max_int_val;
    float one_over_width = (width > 0) ? 1.0f / width : 0.0f;

    if (dim_idx == 0) { widths[query_idx] = width; }

    float val    = queries[query_idx * num_dimensions + dim_idx];
    float scaled = val * one_over_width;

    // Clamp to 4-bit range [-8, 7]
    scaled           = fmaxf(-8.0f, fminf(7.0f, scaled));
    int8_t quantized = (int8_t)__float2int_rn(scaled);

    quantized_queries[query_idx * num_dimensions + dim_idx] = quantized;
  }
}

__global__ void packInt4QueryBitPlanes(const int8_t* __restrict__ queries,
                                       uint32_t* __restrict__ packed_queries,
                                       int num_queries,
                                       int num_dimensions)
{
  const int dims_per_word = 32;
  const int num_words     = (num_dimensions + dims_per_word - 1) / dims_per_word;
  const int num_bits      = 4;  // Only 4 bit planes!

  int idx            = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = num_queries * num_bits * num_words;

  for (int i = idx; i < total_elements; i += gridDim.x * blockDim.x) {
    int query_idx = i / (num_bits * num_words);
    int remainder = i % (num_bits * num_words);
    int bit_idx   = remainder / num_words;
    int word_idx  = remainder % num_words;

    uint32_t packed_word = 0;

#pragma unroll 8
    for (int d = 0; d < dims_per_word; ++d) {
      int dim_idx = word_idx * dims_per_word + d;
      if (dim_idx < num_dimensions) {
        // For 4-bit values, we only care about the lower 4 bits
        // But need to handle sign extension properly
        uint8_t val      = (uint8_t)(queries[query_idx * num_dimensions + dim_idx] & 0xF);
        uint32_t bit_val = (val >> bit_idx) & 1;

        // Match data bit ordering - dim 0 goes to bit 31
        int bit_position = 31 - d;
        packed_word |= (bit_val << bit_position);
      }
    }

    packed_queries[i] = packed_word;
  }
}

// Search with qunatized query vectors
void SearcherGPU::SearchClusterQueryPairsQuantizeQuery(
  const IVFGPU& cur_ivf,
  IVFGPU::GPUClusterMeta* d_cluster_meta,
  ClusterQueryPair* d_sorted_pairs,
  size_t num_queries,
  raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
  const float* d_G_k1xSumq,
  const float* d_G_kbxSumq,
  size_t nprobe,
  size_t topk,
  raft::device_matrix_view<float, int64_t, raft::row_major> d_final_dists,
  raft::device_matrix_view<uint32_t, int64_t, raft::row_major> d_final_pids,
  bool use_4bit  // Add parameter to choose 4-bit or 8-bit
)
{
  // check if the inner products kernel should use block sort to keep a top-k priority queue vs.
  // outputting distances from all vectors in probed clusters
  const bool use_block_sort{topk <= kMaxTopKBlockSort};

  // query quantize
  const int num_bits  = use_4bit ? 4 : 8;  // Choose bit width
  const int num_words = (cur_ivf.get_num_padded_dim() + 31) / 32;

  // Allocate memory for quantization
  auto d_query_write_counters = raft::make_device_vector<int, int64_t>(handle_, num_queries);
  auto d_query_ranges         = raft::make_device_vector<float, int64_t>(handle_, 0);
  auto d_widths               = raft::make_device_vector<float, int64_t>(handle_, 0);
  auto d_quantized_queries    = raft::make_device_vector<int8_t, int64_t>(handle_, 0);
  auto d_packed_queries       = raft::make_device_vector<uint32_t, int64_t>(handle_, 0);
  auto d_topk_threshold_batch = raft::make_device_vector<float, int64_t>(handle_, 0);
  if (use_block_sort) {
    d_query_ranges      = raft::make_device_vector<float, int64_t>(handle_, num_queries * 2);
    d_widths            = raft::make_device_vector<float, int64_t>(handle_, num_queries);
    d_quantized_queries = raft::make_device_vector<int8_t, int64_t>(
      handle_, num_queries * cur_ivf.get_num_padded_dim());
    d_packed_queries =
      raft::make_device_vector<uint32_t, int64_t>(handle_, num_queries * num_bits * num_words);
    d_topk_threshold_batch = raft::make_device_vector<float, int64_t>(handle_, num_queries);
  }

  if (use_block_sort) {
    if (rabitq_quantize_flag_) {
      const int block_size = 256;
      const int grid_size  = num_queries;
      size_t shared_mem    = D * sizeof(float) + block_size * sizeof(float);
      exrabitq_quantize_query<block_size>
        <<<grid_size, block_size, shared_mem, stream_>>>(queries.data_handle(),
                                                         num_queries,
                                                         D,
                                                         num_bits,
                                                         best_rescaling_factor,
                                                         1.9f,
                                                         d_quantized_queries.data_handle(),
                                                         d_widths.data_handle());
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    } else {  // scalar quantize
      // Step 1: Find min/max for each query
      const int block_size = 256;
      const int grid_size  = num_queries;
      findQueryRanges<<<grid_size, block_size, 0, stream_>>>(queries.data_handle(),
                                                             d_query_ranges.data_handle(),
                                                             num_queries,
                                                             cur_ivf.get_num_padded_dim());
      RAFT_CUDA_TRY(cudaPeekAtLastError());

      // Step 2: Quantize queries to int8_t with BQ=8
      if (use_4bit) {
        quantizeQueriesToInt4<<<grid_size, block_size, 0, stream_>>>(
          queries.data_handle(),
          d_query_ranges.data_handle(),
          d_quantized_queries.data_handle(),
          d_widths.data_handle(),
          num_queries,
          cur_ivf.get_num_padded_dim());
        RAFT_CUDA_TRY(cudaPeekAtLastError());
      } else {
        quantizeQueriesToInt8<<<grid_size, block_size, 0, stream_>>>(
          queries.data_handle(),
          d_query_ranges.data_handle(),
          d_quantized_queries.data_handle(),
          d_widths.data_handle(),
          num_queries,
          cur_ivf.get_num_padded_dim());
        RAFT_CUDA_TRY(cudaPeekAtLastError());
      }
    }
  }

  // Step 3: Pack quantized queries into bit planes
  if (use_block_sort) {
    const int block_size = 256;
    const int grid_size  = (num_queries * num_bits * num_words + block_size - 1) / block_size;

    if (use_4bit) {
      packInt4QueryBitPlanes<<<grid_size, block_size, 0, stream_>>>(
        d_quantized_queries.data_handle(),
        d_packed_queries.data_handle(),
        num_queries,
        cur_ivf.get_num_padded_dim());
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    } else {
      packInt8QueryBitPlanes<<<grid_size, block_size, 0, stream_>>>(
        d_quantized_queries.data_handle(),
        d_packed_queries.data_handle(),
        num_queries,
        cur_ivf.get_num_padded_dim());
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }
  }

  // We minimize max_cluster_size to reduce shared memory usage when the probe clusters do not
  // include the largest cluster. This optimization is expected to be more effective when
  // num_queries and/or nprobe are low.
  uint32_t max_cluster_size;

  // For the intermediate distances (and associated IDs), we want to minimize the allocation both to
  // reduce memory footprint and to avoid unnecessary passes in the final RAFT select_k call. For
  // `use_block_sort = true`, the required allocation is simply num_queries * nprobe * topk. For
  // `use_block_sort = false`, the strategy here is to compute the sum of cluster sizes over all
  // probed clusters for each query, and use the maximum of these sums as the allocation size needed
  // per query. This avoids negative performance impact from any abnormally large cluster when using
  // the global maximum cluster size as the allocation size per query per probe.
  std::optional<size_t> max_probed_vectors_count =
    use_block_sort ? std::nullopt : std::optional<size_t>{0};

  // call utility function to evaluate max_cluster_size and max_probed_vectors_count
  get_max_probed_cluster_size_and_vectors_count(handle_,
                                                d_sorted_pairs,
                                                num_queries * nprobe,
                                                cur_ivf.get_cluster_meta().data_handle(),
                                                num_queries,
                                                max_cluster_size,
                                                max_probed_vectors_count);

  // allocate memory for intermediate output
  size_t n_cols     = use_block_sort ? nprobe * topk : max_probed_vectors_count.value();
  auto d_topk_dists = raft::make_device_matrix<float, int64_t>(handle_, num_queries, n_cols);
  auto d_topk_pids  = raft::make_device_matrix<uint32_t, int64_t>(handle_, num_queries, n_cols);

  // initialize distances
  thrust::fill(thrust::cuda::par.on(stream_),
               d_topk_dists.data_handle(),
               d_topk_dists.data_handle() + d_topk_dists.size(),
               std::numeric_limits<float>::infinity());

  thrust::fill(thrust::cuda::par.on(stream_),
               d_query_write_counters.data_handle(),
               d_query_write_counters.data_handle() + num_queries,
               0);

  if (use_block_sort) {
    thrust::fill(thrust::cuda::par.on(stream_),
                 d_topk_threshold_batch.data_handle(),
                 d_topk_threshold_batch.data_handle() + num_queries,
                 std::numeric_limits<float>::infinity());
  }

  // Launch modified kernel with packed queries instead of LUT
  size_t num_pairs = num_queries * nprobe;
  uint32_t gridDim{static_cast<uint32_t>(num_pairs)};
  uint32_t blockDim{256};

  // Recalculate shared memory for new approach
  size_t query_storage = D * sizeof(float);  // For shared query vector
  const int queue_buffer_smem_bytes =
    use_block_sort ? raft::matrix::detail::select::warpsort::calc_smem_size_for_block_wide<T, IdxT>(
                       blockDim / raft::WarpSize, kMaxTopKBlockSort)
                   : 0;

  // Now we need: packed query bits, candidate storage, and query vector
  // this part is also used to store ip2 results
  size_t packed_query_size = max((use_block_sort ? (num_bits * num_words * sizeof(uint32_t)) : 0),
                                 max_cluster_size * sizeof(float));
  size_t candidate_storage = use_block_sort ? max_cluster_size * (sizeof(float) + sizeof(int)) : 0;
  size_t shared_mem_size =
    max(packed_query_size + candidate_storage + query_storage, (size_t)queue_buffer_smem_bytes);

  ComputeInnerProductsKernelParams kernelParams;
  kernelParams.d_sorted_pairs          = d_sorted_pairs;
  kernelParams.d_query                 = queries.data_handle();
  kernelParams.d_short_data            = cur_ivf.get_short_data_device();
  kernelParams.d_cluster_meta          = d_cluster_meta;
  kernelParams.d_packed_queries        = d_packed_queries.data_handle();
  kernelParams.d_widths                = d_widths.data_handle();
  kernelParams.d_short_factors         = cur_ivf.get_short_factors_batch_device();
  kernelParams.d_G_k1xSumq             = d_G_k1xSumq;
  kernelParams.d_G_kbxSumq             = d_G_kbxSumq;
  kernelParams.d_centroid_distances    = get_centroid_distances();
  kernelParams.topk                    = topk;
  kernelParams.num_queries             = num_queries;
  kernelParams.nprobe                  = nprobe;
  kernelParams.num_pairs               = num_pairs;
  kernelParams.num_centroids           = cur_ivf.get_num_centroids();
  kernelParams.D                       = D;
  kernelParams.d_threshold             = d_topk_threshold_batch.data_handle();
  kernelParams.max_candidates_per_pair = max_cluster_size;
  kernelParams.max_candidates_per_query =
    use_block_sort ? 0 /* unused */ : max_probed_vectors_count.value();
  kernelParams.ex_bits      = cur_ivf.get_ex_bits();
  kernelParams.d_long_code  = cur_ivf.get_long_code_device();
  kernelParams.d_ex_factor  = reinterpret_cast<const float*>(cur_ivf.get_ex_factor_device());
  kernelParams.d_pids       = cur_ivf.get_ids_device();
  kernelParams.d_topk_dists = d_topk_dists.data_handle();
  kernelParams.d_topk_pids  = d_topk_pids.data_handle();
  kernelParams.d_query_write_counters = d_query_write_counters.data_handle();
  kernelParams.num_bits               = num_bits;
  kernelParams.num_words              = num_words;

  const int num_bits_for_dispatch = use_4bit ? 4 : 8;
  const bool with_ex              = (cur_ivf.get_ex_bits() != 0);
  auto jit_launcher = use_block_sort ? make_compute_inner_products_with_bitwise_block_sort_launcher(
                                         num_bits_for_dispatch, cur_ivf.get_ex_bits(), with_ex)
                                     : make_compute_inner_products_with_bitwise_launcher(
                                         cur_ivf.get_ex_bits(), with_ex);
  auto const& kernel_launcher = [&]() -> void {
    jit_launcher->dispatch<compute_inner_products_with_lut_func_t>(
      stream_, gridDim, blockDim, shared_mem_size, kernelParams);
  };
  cuvs::neighbors::detail::safely_launch_kernel_with_smem_size<
    compute_inner_products_with_lut_func_t>(
    static_cast<std::uint32_t>(shared_mem_size), kernel_launcher, jit_launcher->get_kernel());
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Merge results
  cuvs::selection::select_k(
    handle_,
    raft::make_device_matrix_view<const float, int64_t>(
      d_topk_dists.data_handle(), num_queries, n_cols),
    std::make_optional(raft::make_device_matrix_view<const uint32_t, int64_t>(
      d_topk_pids.data_handle(), num_queries, n_cols)),
    d_final_dists,
    d_final_pids,
    /*select_min=*/true,
    /*sorted=*/false);
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
