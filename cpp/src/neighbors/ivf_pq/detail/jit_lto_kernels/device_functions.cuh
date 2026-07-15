/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <neighbors/sample_filter.cuh>

namespace cuvs::neighbors::ivf_pq::detail {

using vec_t = raft::TxN_t<uint32_t, kIndexGroupVecLen / sizeof(uint32_t)>;

template <typename LutT>
__device__ void prepare_lut(uint8_t* smem_buf,
                            uint32_t pq_dim,
                            LutT*& lut_scores,
                            uint8_t*& lut_end);

template <typename OutT>
__device__ void store_calculated_distances(OutT* _out_scores,
                                           uint32_t* _out_indices,
                                           uint32_t probe_ix,
                                           uint32_t n_probes,
                                           uint32_t query_ix,
                                           uint32_t topk,
                                           uint32_t max_samples,
                                           OutT*& out_scores,
                                           uint32_t*& out_indices);

__device__ void precompute_base_diff(uint32_t dim,
                                     uint8_t* lut_end,
                                     const float* query,
                                     const float* cluster_center);

template <typename LutT>
__device__ void create_lut(uint32_t pq_dim,
                           uint32_t pq_len,
                           uint32_t label,
                           const float* pq_centers,
                           codebook_gen codebook_kind,
                           LutT* lut_scores,
                           uint8_t* lut_end,
                           const float* query,
                           const float* cluster_center);

template <typename OutT, typename LutT>
__device__ void compute_distances(const uint32_t* chunk_indices,
                                  float* query_kths,
                                  uint32_t n_probes,
                                  uint32_t probe_ix,
                                  uint32_t query_ix,
                                  uint32_t max_samples,
                                  uint32_t topk,
                                  uint8_t* lut_end,
                                  uint32_t pq_dim,
                                  const uint8_t* const* pq_dataset,
                                  uint32_t label,
                                  uint32_t queries_offset,
                                  OutT* out_scores,
                                  uint32_t* out_indices,
                                  LutT* lut_scores,
                                  uint8_t* smem_buf,
                                  const int64_t* const* inds_ptrs,
                                  uint32_t* bitset_ptr,
                                  int64_t bitset_len,
                                  int64_t original_nbits);

template <typename OutT>
__device__ OutT get_early_stop_limit(OutT query_kth);

__device__ bool sample_filter(const int64_t* const* const inds_ptrs,
                              const uint32_t query_ix,
                              const uint32_t cluster_ix,
                              const uint32_t sample_ix,
                              uint32_t* bitset_ptr,
                              int64_t bitset_len,
                              int64_t original_nbits);

__device__ uint32_t get_line_width(uint32_t pq_dim);

template <typename OutT, typename LutT>
__device__ auto compute_score(uint32_t pq_dim,
                              const vec_t::io_t* pq_head,
                              const LutT* lut_scores,
                              OutT early_stop_limit) -> OutT;

template <typename OutT>
__device__ OutT increment_score(OutT score);

}  // namespace cuvs::neighbors::ivf_pq::detail
