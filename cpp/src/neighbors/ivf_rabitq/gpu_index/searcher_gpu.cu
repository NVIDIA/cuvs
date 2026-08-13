/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 4/14/25.
//

// This file implements `SearcherGPU::SearchClusterQueryPairs`.
#include "../../detail/smem_utils.cuh"
#include "../jit_lto_kernels/kernel_def.hpp"
#include "../jit_lto_kernels/launcher_factory.hpp"
#include "../utils/memory.hpp"
#include "../utils/searcher_gpu_utils.hpp"
#include "searcher_gpu.cuh"
#include "searcher_gpu_common.cuh"

#include <cuvs/selection/select_k.hpp>
#include <raft/matrix/detail/select_warpsort.cuh>

#include <rmm/device_uvector.hpp>

#include <thrust/fill.h>

#include <cstdint>
#include <cuda_runtime.h>
#include <limits>
#include <optional>
#include <string>

namespace cuvs::neighbors::ivf_rabitq::detail {

SearcherGPU::SearcherGPU(raft::resources const& handle,
                         const float* q,
                         size_t d,
                         size_t ex_bits,
                         std::string mode,
                         DataQuantizerGPU::FastQuantizeFactors* fast_quantize_factors,
                         bool rabitq_quantize_flag)
  : D(d), query_(q), rabitq_quantize_flag_(rabitq_quantize_flag), mode_(mode), handle_(handle)
{
  RAFT_EXPECTS(D % 32 == 0, "D (%zu) must be divisible by 32", D);
  RAFT_EXPECTS(
    D % BITS_PER_CHUNK == 0, "D (%zu) must be divisible by BITS_PER_CHUNK (%d)", D, BITS_PER_CHUNK);
  set_unit_q(memory::align_mm<64, float>(D * sizeof(float)));
  set_quant_query(memory::align_mm<64, int16_t>(D * sizeof(int16_t)));
  if (mode_ == "quant4" || mode_ == "quant8") {
    RAFT_EXPECTS(fast_quantize_factors != nullptr,
                 "fast_quantize_factors must be set for quant4/quant8 mode");
    if (mode_ == "quant4") {
      best_rescaling_factor = fast_quantize_factors->const_scaling_factor_4bit;
    } else if (mode_ == "quant8") {
      best_rescaling_factor = fast_quantize_factors->const_scaling_factor_8bit;
    }
  }
}

void SearcherGPU::AllocateSearcherSpace(size_t num_centroids, size_t num_queries)
{
  centroid_distances_ =
    raft::make_device_vector<float, int64_t>(handle_, num_queries * num_centroids);
  q_norms_ = raft::make_device_vector<float, int64_t>(handle_, num_queries);
};

__global__ void precomputeAllLUTs(const float* d_query,      // Query vectors
                                  float* d_lut_for_queries,  // Output LUTs for all queries
                                  size_t num_queries,        // Number of queries
                                  size_t D                   // Dimension
)
{
  // Each block handles one query
  const int query_idx = blockIdx.x;
  if (query_idx >= num_queries) return;

  const int tid         = threadIdx.x;
  const int num_threads = blockDim.x;

  // Calculate LUT parameters
  const size_t num_chunks         = D / BITS_PER_CHUNK;
  const size_t lut_per_query_size = num_chunks * LUT_SIZE;

  // Pointer to this query's LUT in global memory
  float* query_lut = d_lut_for_queries + query_idx * lut_per_query_size;

  // Pointer to this query's vector
  const float* query_vec = d_query + query_idx * D;

  // Each thread computes part of the LUT
  for (size_t chunk_idx = tid; chunk_idx < num_chunks; chunk_idx += num_threads) {
    size_t dim_start = chunk_idx * BITS_PER_CHUNK;

    // Compute LUT entries for this chunk
    for (int lut_entry = 0; lut_entry < LUT_SIZE; lut_entry++) {
      float sum = 0.0f;

      // For each bit in the 4-bit pattern
      for (int bit_idx = 0; bit_idx < BITS_PER_CHUNK; bit_idx++) {
        size_t dim = dim_start + bit_idx;
        if (dim < D) {  // Check if within actual dimension
          // Check if bit is set in the pattern
          if (lut_entry & (1 << (BITS_PER_CHUNK - 1 - bit_idx))) { sum += query_vec[dim]; }
        }
      }

      // Store in global LUT
      size_t lut_offset     = chunk_idx * LUT_SIZE + lut_entry;
      query_lut[lut_offset] = sum;
    }
  }
}

// Launch function for precomputing LUTs
void launchPrecomputeLUTs(const float* d_query,
                          float* d_lut_for_queries,
                          size_t num_queries,
                          size_t D,
                          rmm::cuda_stream_view stream)
{
  // Launch precompute kernel
  dim3 gridDim(num_queries, 1, 1);
  dim3 blockDim(256, 1, 1);  // Can tune this

  precomputeAllLUTs<<<gridDim, blockDim, 0, stream>>>(d_query, d_lut_for_queries, num_queries, D);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

void SearcherGPU::SearchClusterQueryPairs(
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
  raft::device_matrix_view<uint32_t, int64_t, raft::row_major> d_final_pids)
{
  // First allocate space for LUT
  size_t lut_size = num_queries * (D / BITS_PER_CHUNK) * LUT_SIZE * sizeof(float);
  rmm::device_uvector<float> d_lut_for_queries(lut_size / sizeof(float), stream_);
  thrust::fill(thrust::cuda::par.on(stream_),
               d_lut_for_queries.data(),
               d_lut_for_queries.data() + d_lut_for_queries.size(),
               -std::numeric_limits<float>::infinity());
  // precompute LUTS
  launchPrecomputeLUTs(queries.data_handle(), d_lut_for_queries.data(), num_queries, D, stream_);

  // check if the inner products kernel should use block sort to keep a top-k priority queue vs.
  // outputting distances from all vectors in probed clusters
  const bool use_block_sort{topk <= kMaxTopKBlockSort};

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

  rmm::device_uvector<int> d_query_write_counters(num_queries, stream_);
  thrust::fill(thrust::cuda::par.on(stream_),
               d_query_write_counters.data(),
               d_query_write_counters.data() + num_queries,
               0);

  rmm::device_uvector<float> d_topk_threshold_batch(use_block_sort ? num_queries : 0, stream_);
  if (use_block_sort) {
    thrust::fill(thrust::cuda::par.on(stream_),
                 d_topk_threshold_batch.data(),
                 d_topk_threshold_batch.data() + num_queries,
                 std::numeric_limits<float>::infinity());
  }
  // Then launch kernel for computation
  size_t num_pairs = num_queries * nprobe;
  uint32_t gridDim{static_cast<uint32_t>(num_pairs)};
  uint32_t blockDim{256};
  size_t num_chunks = D / BITS_PER_CHUNK;
  size_t candidate_storage =
    max_cluster_size * (use_block_sort ? (2 * sizeof(float) /* ip, ip2 */ + sizeof(int) /* idx */)
                                       : sizeof(float) /* ip2 */);
  const int queue_buffer_smem_bytes =
    use_block_sort ? raft::matrix::detail::select::warpsort::calc_smem_size_for_block_wide<T, IdxT>(
                       blockDim / raft::WarpSize, kMaxTopKBlockSort)
                   : 0;

  ComputeInnerProductsKernelParams kernelParams;
  kernelParams.d_sorted_pairs          = d_sorted_pairs;
  kernelParams.d_query                 = queries.data_handle();
  kernelParams.d_short_data            = cur_ivf.get_short_data_device();
  kernelParams.d_cluster_meta          = d_cluster_meta;
  kernelParams.d_lut_for_queries_float = d_lut_for_queries.data();
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
  kernelParams.d_threshold             = d_topk_threshold_batch.data();
  kernelParams.max_candidates_per_pair = max_cluster_size;
  kernelParams.max_candidates_per_query =
    use_block_sort ? 0 /* unused */ : max_probed_vectors_count.value();
  kernelParams.ex_bits      = cur_ivf.get_ex_bits();
  kernelParams.d_long_code  = cur_ivf.get_long_code_device();
  kernelParams.d_ex_factor  = reinterpret_cast<const float*>(cur_ivf.get_ex_factor_device());
  kernelParams.d_pids       = cur_ivf.get_ids_device();
  kernelParams.d_topk_dists = d_topk_dists.data_handle();
  kernelParams.d_topk_pids  = d_topk_pids.data_handle();
  kernelParams.d_query_write_counters = d_query_write_counters.data();

  if (cur_ivf.get_ex_bits() != 0) {
    size_t shared_mem_size =
      num_chunks * LUT_SIZE * sizeof(float) + candidate_storage + queue_buffer_smem_bytes;
    auto jit_launcher = use_block_sort ? make_compute_inner_products_with_lut_block_sort_launcher(
                                           cur_ivf.get_ex_bits(), /*with_ex=*/true)
                                       : make_compute_inner_products_with_lut_launcher(
                                           cur_ivf.get_ex_bits(), /*with_ex=*/true);
    auto const& kernel_launcher = [&]() -> void {
      jit_launcher->dispatch<compute_inner_products_with_lut_func_t>(
        stream_, gridDim, blockDim, shared_mem_size, kernelParams);
    };
    cuvs::neighbors::detail::safely_launch_kernel_with_smem_size<
      compute_inner_products_with_lut_func_t>(
      static_cast<std::uint32_t>(shared_mem_size), kernel_launcher, jit_launcher->get_kernel());
  } else {
    size_t shared_mem_size =
      max(num_chunks * LUT_SIZE * sizeof(float) +
            (use_block_sort ? max_cluster_size * (sizeof(float) + sizeof(int)) : 0),
          (size_t)queue_buffer_smem_bytes);
    auto jit_launcher = use_block_sort ? make_compute_inner_products_with_lut_block_sort_launcher(
                                           /*ex_bits=*/0, /*with_ex=*/false)
                                       : make_compute_inner_products_with_lut_launcher(
                                           /*ex_bits=*/0, /*with_ex=*/false);
    auto const& kernel_launcher = [&]() -> void {
      jit_launcher->dispatch<compute_inner_products_with_lut_func_t>(
        stream_, gridDim, blockDim, shared_mem_size, kernelParams);
    };
    cuvs::neighbors::detail::safely_launch_kernel_with_smem_size<
      compute_inner_products_with_lut_func_t>(
      static_cast<std::uint32_t>(shared_mem_size), kernel_launcher, jit_launcher->get_kernel());
  }
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // merge results from different blocks
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
