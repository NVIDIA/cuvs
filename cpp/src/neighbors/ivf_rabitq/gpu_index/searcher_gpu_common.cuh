/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

//
// Created by Stardust on 4/14/25.
//

#include "searcher_gpu.cuh"

#include <raft/util/cuda_dev_essentials.cuh>

#include <cstdint>
#include <cuda_runtime.h>

namespace cuvs::neighbors::ivf_rabitq::detail {

static constexpr int BITS_PER_CHUNK = 4;
static constexpr int LUT_SIZE       = (1 << BITS_PER_CHUNK);  // 16

// --- Tunables ---
using T    = float;
using IdxT = uint32_t;

using lut_dtype = __half;  // FP16

// POD struct consolidating parameters for all computeInnerProducts* kernels
struct ComputeInnerProductsKernelParams {
  const ClusterQueryPair* d_sorted_pairs       = nullptr;
  const float* d_query                         = nullptr;
  const uint32_t* d_short_data                 = nullptr;
  const IVFGPU::GPUClusterMeta* d_cluster_meta = nullptr;
  float* d_lut_for_queries_float               = nullptr;
  lut_dtype* d_lut_for_queries_half            = nullptr;
  const uint32_t* d_packed_queries             = nullptr;  // Packed query bit planes
  const float* d_widths                        = nullptr;  // Query scaling factors
  const float* d_short_factors                 = nullptr;
  const float* d_G_k1xSumq                     = nullptr;
  const float* d_G_kbxSumq                     = nullptr;
  const float* d_centroid_distances            = nullptr;
  uint32_t topk                                = 0;
  uint32_t num_queries                         = 0;
  uint32_t nprobe                              = 0;
  uint32_t num_pairs                           = 0;
  uint32_t num_centroids                       = 0;
  uint32_t D                                   = 0;
  const float* d_threshold                     = nullptr;  // threshold for each query
  uint32_t max_candidates_per_pair             = 0;        // max storage per pair, 1000 suggested
  uint32_t max_candidates_per_query =
    0;  // max number of vectors in probed clusters for any particular query
  uint32_t ex_bits            = 0;        // bits per dimension in ex codes
  const uint8_t* d_long_code  = nullptr;  // long codes for all vectors
  const float* d_ex_factor    = nullptr;  // ex factors for distance computation
  const PID* d_pids           = nullptr;  // PIDs for all vectors
  float* d_topk_dists         = nullptr;  // output top-k distances
  PID* d_topk_pids            = nullptr;  // output top-k PIDs
  int* d_query_write_counters = nullptr;
  uint32_t num_bits           = 0;  // number of bits (8 for int8)
  uint32_t num_words          = 0;  // approx. D/32
};

}  // namespace cuvs::neighbors::ivf_rabitq::detail
