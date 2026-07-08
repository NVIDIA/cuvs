/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../gpu_index/searcher_gpu_common.cuh"
#include "device_functions.cuh"

#include <raft/util/cuda_dev_essentials.cuh>

#include <cstdint>

namespace cuvs::neighbors::ivf_rabitq::detail {

// Unified non-BlockSort half-precision LUT kernel. The WithEx-axis-specific
// body (IP2 precompute + per-vector distance write for WithEx=true, FP16-LUT
// distance write for WithEx=false) lives in the lut16_opt_emit_distances
// JIT-LTO fragment, so this kernel is emitted as a single fragment regardless
// of WithEx.
__device__ void compute_inner_products_with_lut16_opt_impl(
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
  float q_g_add = params.d_centroid_distances[query_idx * params.num_centroids + cluster_idx];

  lut16_opt_emit_distances(
    params, query_idx, cluster_idx, num_vectors_in_cluster, cluster_start_index, q_g_add);
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
