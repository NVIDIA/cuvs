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

template <bool WithEx>
__device__ void compute_inner_products_with_lut16_opt_block_sort_impl(
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

  const uint32_t num_chunks         = params.D / BITS_PER_CHUNK;
  const uint32_t lut_per_query_size = num_chunks * LUT_SIZE;

  extern __shared__ __align__(256) char shared_mem_raw[];
  lut_dtype* shared_lut_fp16 = reinterpret_cast<lut_dtype*>(shared_mem_raw);
  uint32_t lut_bytes         = num_chunks * LUT_SIZE * sizeof(lut_dtype);

  const int tid         = threadIdx.x;
  const int num_threads = blockDim.x;

  lut_dtype* query_lut = params.d_lut_for_queries_half + query_idx * lut_per_query_size;
  for (uint32_t i = tid; i < lut_per_query_size; i += num_threads) {
    shared_lut_fp16[i] = query_lut[i];
  }
  __syncthreads();

  __shared__ int num_candidates;
  float q_g_add   = params.d_centroid_distances[query_idx * params.num_centroids + cluster_idx];
  float q_k1xsumq = params.d_G_k1xSumq[query_idx];
  float threshold = params.d_threshold[query_idx];

  float q_g_error;
  if constexpr (WithEx) { q_g_error = sqrtf(q_g_add); }

  if (tid == 0) { num_candidates = 0; }
  __syncthreads();

  // WithEx: candidates placed at max(lut_bytes, max_candidates_per_pair * sizeof(float))
  // NoEX:   candidates placed right after LUT
  float* shared_candidate_ips;
  if constexpr (WithEx) {
    if (lut_bytes < params.max_candidates_per_pair * sizeof(float)) {
      shared_candidate_ips =
        reinterpret_cast<float*>(shared_mem_raw + params.max_candidates_per_pair * sizeof(float));
    } else {
      shared_candidate_ips = reinterpret_cast<float*>(shared_mem_raw + lut_bytes);
    }
  } else {
    shared_candidate_ips = reinterpret_cast<float*>(shared_mem_raw + lut_bytes);
  }
  // WithEx: stores local vec_idx; NoEX: stores PID
  int* shared_candidate_indices = (int*)(shared_candidate_ips + params.max_candidates_per_pair);

  const uint32_t short_code_length = params.D / 32;

  for (size_t vec_base = 0; vec_base < num_vectors_in_cluster; vec_base += num_threads) {
    size_t vec_idx = vec_base + tid;

    float local_ip    = 0.0f;
    bool is_candidate = false;

    if (vec_idx < num_vectors_in_cluster) {
      size_t factor_offset = cluster_start_index + vec_idx;
      float3 factors       = reinterpret_cast<const float3*>(params.d_short_factors)[factor_offset];
      float f_add          = factors.x;
      float f_rescale      = factors.y;

      float ip = compute_lut_ip_for_vec<__half>(params.d_short_data,
                                                shared_lut_fp16,
                                                cluster_start_index,
                                                num_vectors_in_cluster,
                                                vec_idx,
                                                short_code_length);

      float est_dist = f_add + q_g_add + f_rescale * (ip + q_k1xsumq);

      if constexpr (WithEx) {
        float f_error  = factors.z;
        float low_dist = est_dist - f_error * q_g_error;
        if (low_dist < threshold) {
          is_candidate = true;
          local_ip     = ip;
        }
      } else {
        if (est_dist < threshold) {
          is_candidate = true;
          local_ip     = est_dist;
        }
      }
    }

    __syncwarp();
    if (is_candidate) {
      int candidate_slot = atomicAdd(&num_candidates, 1);
      if (candidate_slot < params.max_candidates_per_pair) {
        shared_candidate_ips[candidate_slot] = local_ip;
        if constexpr (WithEx) {
          shared_candidate_indices[candidate_slot] = (int)vec_idx;
        } else {
          shared_candidate_indices[candidate_slot] =
            (int)params.d_pids[cluster_start_index + vec_idx];
        }
      }
    }
  }
  __syncthreads();
  if (num_candidates > 0) {
    __shared__ int probe_slot;
    uint32_t output_offset;
    {
      using block_sort_t = typename rabitq_block_sort<kMaxTopKBlockSort, T, IdxT>::type;
      block_sort_t queue(params.topk);

      if constexpr (WithEx) {
        float q_kbxsumq               = params.d_G_kbxSumq[query_idx];
        const uint32_t long_code_size = (params.D * params.ex_bits + 7) / 8;

        float* shared_query = (float*)(shared_candidate_indices + params.max_candidates_per_pair);
        for (uint32_t i = tid; i < params.D; i += num_threads) {
          shared_query[i] = params.d_query[query_idx * params.D + i];
        }
        __syncthreads();

        float* shared_ip2_results = reinterpret_cast<float*>(shared_lut_fp16);
        const int warp_id         = tid / raft::WarpSize;
        const int lane_id         = tid % raft::WarpSize;
        const int num_warps       = num_threads / raft::WarpSize;

        for (int cand_idx = warp_id; cand_idx < num_candidates; cand_idx += num_warps) {
          size_t global_vec_idx        = cluster_start_index + shared_candidate_indices[cand_idx];
          const uint8_t* vec_long_code = params.d_long_code + global_vec_idx * long_code_size;
          float ip2 =
            compute_ip2_from_long_codes_warp(vec_long_code, shared_query, params.D, lane_id);
          if (lane_id == 0) { shared_ip2_results[cand_idx] = ip2; }
        }
        __syncthreads();

        const int adds_per_thread = (num_candidates + num_threads - 1) / num_threads;
        for (int round = 0; round < adds_per_thread; round++) {
          int cand_idx = tid + round * num_threads;
          float ex_dist;
          uint32_t pid;
          if (cand_idx < num_candidates) {
            float ip              = shared_candidate_ips[cand_idx];
            float ip2             = shared_ip2_results[cand_idx];
            int local_vec_idx     = shared_candidate_indices[cand_idx];
            size_t global_vec_idx = cluster_start_index + local_vec_idx;
            float2 ex_factors = reinterpret_cast<const float2*>(params.d_ex_factor)[global_vec_idx];
            float f_ex_add    = ex_factors.x;
            float f_ex_rescale = ex_factors.y;
            ex_dist =
              f_ex_add + q_g_add +
              f_ex_rescale * (static_cast<float>(1 << params.ex_bits) * ip + ip2 + q_kbxsumq);
            pid = (uint32_t)params.d_pids[global_vec_idx];
          } else {
            ex_dist = INFINITY;
            pid     = 0;
          }
          queue.add(ex_dist, pid);
        }
      } else {
        const int candidates_per_thread = (num_candidates + num_threads - 1) / num_threads;
        for (int c = 0; c < candidates_per_thread; ++c) {
          int cand_idx = tid + c * num_threads;
          float dist;
          uint32_t pid;
          if (cand_idx < num_candidates) {
            dist = shared_candidate_ips[cand_idx];
            pid  = (uint32_t)shared_candidate_indices[cand_idx];
          } else {
            dist = INFINITY;
            pid  = 0;
          }
          queue.add(dist, pid);
        }
      }

      __syncthreads();

      queue.done((uint8_t*)shared_lut_fp16);

      if (tid == 0) { probe_slot = atomicAdd(&params.d_query_write_counters[query_idx], 1); }
      __syncthreads();

      if (probe_slot >= params.nprobe) { return; }

      output_offset = query_idx * (params.topk * params.nprobe) + probe_slot * params.topk;
      queue.store(params.d_topk_dists + output_offset,
                  (uint32_t*)(params.d_topk_pids + output_offset));
    }

    if (num_candidates >= params.topk) {
      update_threshold_atomicmin<false>(params.d_topk_dists,
                                        params.d_threshold,
                                        query_idx,
                                        params.topk,
                                        params.nprobe,
                                        probe_slot,
                                        threshold,
                                        tid);
    }
  }
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
