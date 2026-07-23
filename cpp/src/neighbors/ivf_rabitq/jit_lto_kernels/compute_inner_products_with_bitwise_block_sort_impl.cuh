/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Unified kernel using BlockSort. The WithEx-axis-specific tail (refine + sort
// for WithEx=true, direct sort for WithEx=false) lives in the
// bitwise_block_sort_emit_topk JIT-LTO fragment, so this kernel is emitted as
// a single fragment regardless of WithEx. The num_bits-dependent popc
// inner-product loop is dispatched at runtime through the
// compute_bitwise_quantized_ip_for_vec JIT-LTO fragment, so this kernel is
// also not templated on num_bits.
template <bool Signed>
__device__ void compute_inner_products_with_bitwise_block_sort_impl(
  const ComputeInnerProductsKernelParams params)
{
  const int block_id = blockIdx.x;
  if (block_id >= params.num_pairs) return;

  ClusterQueryPair pair = params.d_sorted_pairs[block_id];
  int cluster_idx       = pair.cluster_idx;
  int query_idx         = pair.query_idx;

  if (cluster_idx >= params.num_centroids || query_idx >= params.num_queries) return;

  size_t num_vectors_in_cluster = params.d_cluster_meta[cluster_idx].num;
  size_t cluster_start_index    = params.d_cluster_meta[cluster_idx].start_index;

  extern __shared__ __align__(256) char shared_mem_raw_2[];
  uint32_t* shared_packed_query = reinterpret_cast<uint32_t*>(shared_mem_raw_2);

  const int tid         = threadIdx.x;
  const int num_threads = blockDim.x;

  const uint32_t* query_packed_ptr =
    params.d_packed_queries + query_idx * params.num_bits * params.num_words;
  for (uint32_t i = tid; i < params.num_bits * params.num_words; i += num_threads) {
    shared_packed_query[i] = query_packed_ptr[i];
  }

  float query_width = params.d_widths[query_idx];

  __shared__ int num_candidates;
  float q_g_add   = params.d_centroid_distances[query_idx * params.num_centroids + cluster_idx];
  float q_k1xsumq = params.d_G_k1xSumq[query_idx];
  float q_g_error = sqrtf(q_g_add);
  float threshold = params.d_threshold[query_idx];
  // ‖q‖² for the InnerProduct pseudo-distance transform pd = (‖q−x‖² − ‖q‖² − ‖x‖²)/2 = −⟨q,x⟩.
  float q_sqr_norm = Signed ? params.d_q_sqr_norms[query_idx] : 0.0f;

  if (tid == 0) { num_candidates = 0; }
  __syncthreads();

  size_t packed_query_bytes   = max(params.num_bits * params.num_words * sizeof(uint32_t),
                                  params.max_candidates_per_pair * sizeof(float));
  float* shared_candidate_ips = reinterpret_cast<float*>(shared_mem_raw_2 + packed_query_bytes);
  int* shared_candidate_indices =
    reinterpret_cast<int*>(shared_candidate_ips + params.max_candidates_per_pair);
  float* shared_query = (float*)(shared_candidate_indices + params.max_candidates_per_pair);

  // Phase 1: Bitwise inner product filter
  for (size_t vec_base = 0; vec_base < num_vectors_in_cluster; vec_base += num_threads) {
    size_t vec_idx = vec_base + tid;

    bool is_candidate        = false;
    float local_ip_quantized = 0;

    if (vec_idx < num_vectors_in_cluster) {
      size_t factor_offset = cluster_start_index + vec_idx;
      float3 factors       = reinterpret_cast<const float3*>(params.d_short_factors)[factor_offset];
      float f_add          = factors.x;
      float f_rescale      = factors.y;
      float f_error        = factors.z;

      int32_t accumulator = compute_bitwise_quantized_ip_for_vec(params.d_short_data,
                                                                 shared_packed_query,
                                                                 cluster_start_index,
                                                                 num_vectors_in_cluster,
                                                                 vec_idx,
                                                                 params.num_words);

      float ip       = (float)accumulator * query_width;
      float est_dist = f_add + q_g_add + f_rescale * (ip + q_k1xsumq);
      float low_dist = est_dist - f_error * q_g_error;

      if constexpr (Signed) {
        // Convert the squared-L2 lower bound into the pseudo-distance lower bound. The map
        // est → (est − ‖q‖² − ‖x‖²)/2 is affine with positive slope, so a lower bound on the
        // L2 estimate is a valid lower bound on pd = −⟨q,x⟩; the `< threshold` prune stays correct.
        float x_sqr = params.d_vec_sqr_norms[cluster_start_index + vec_idx];
        low_dist    = (low_dist - q_sqr_norm - x_sqr) * 0.5f;
      }

      if (low_dist < threshold) {
        is_candidate       = true;
        local_ip_quantized = ip;
      }
    }

    __syncwarp();

    if (is_candidate) {
      int candidate_slot = atomicAdd(&num_candidates, 1);
      if (candidate_slot < params.max_candidates_per_pair) {
        shared_candidate_ips[candidate_slot]     = local_ip_quantized;
        shared_candidate_indices[candidate_slot] = vec_idx;
      }
    }
  }

  __syncthreads();

  if (num_candidates > 0) {
    for (size_t i = tid; i < params.D; i += num_threads) {
      shared_query[i] = params.d_query[query_idx * params.D + i];
    }
    __syncthreads();

    bitwise_block_sort_emit_topk(params,
                                 num_candidates,
                                 query_idx,
                                 cluster_idx,
                                 num_vectors_in_cluster,
                                 cluster_start_index,
                                 q_g_add,
                                 q_k1xsumq,
                                 threshold);
  }
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
