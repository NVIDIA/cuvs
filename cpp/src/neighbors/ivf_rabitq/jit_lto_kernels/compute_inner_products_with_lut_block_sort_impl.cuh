/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../detail/smem_utils.cuh"
#include "../gpu_index/searcher_gpu_common.cuh"
#include "block_sort.cuh"
#include "device_functions.cuh"

#include <raft/util/cuda_dev_essentials.cuh>

#include <cstdint>

namespace cuvs::neighbors::ivf_rabitq::detail {

// Unified BlockSort LUT kernel. The WithEx-axis-specific body (LUT filter +
// IP2-refined block sort for WithEx=true, LUT filter + direct block sort for
// WithEx=false) lives in the lut_block_sort_emit_topk JIT-LTO fragment, so
// this kernel is emitted as a single fragment regardless of WithEx.
__device__ void compute_inner_products_with_lut_block_sort_impl(
  const ComputeInnerProductsKernelParams params)
{
  const int block_id = blockIdx.x;
  if (block_id >= params.num_pairs) return;

  ClusterQueryPair pair = params.d_sorted_pairs[block_id];
  int cluster_idx       = pair.cluster_idx;
  int query_idx         = pair.query_idx;

  if (cluster_idx >= params.num_centroids || query_idx >= params.num_queries) return;

  const size_t num_chunks         = params.D / BITS_PER_CHUNK;
  const size_t lut_per_query_size = num_chunks * LUT_SIZE;

  extern __shared__ __align__(256) char shared_mem_raw[];
  float* shared_lut = reinterpret_cast<float*>(shared_mem_raw);

  const int tid         = threadIdx.x;
  const int num_threads = blockDim.x;

  float* query_lut = params.d_lut_for_queries_float + query_idx * lut_per_query_size;
  for (size_t i = tid; i < lut_per_query_size; i += num_threads) {
    shared_lut[i] = query_lut[i];
  }
  __syncthreads();

  size_t num_vectors_in_cluster = params.d_cluster_meta[cluster_idx].num;
  size_t cluster_start_index    = params.d_cluster_meta[cluster_idx].start_index;
  float q_g_add   = params.d_centroid_distances[query_idx * params.num_centroids + cluster_idx];
  float q_k1xsumq = params.d_G_k1xSumq[query_idx];
  float threshold = params.d_threshold[query_idx];

  lut_block_sort_emit_topk(params,
                           query_idx,
                           cluster_idx,
                           num_vectors_in_cluster,
                           cluster_start_index,
                           q_g_add,
                           q_k1xsumq,
                           threshold);
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
