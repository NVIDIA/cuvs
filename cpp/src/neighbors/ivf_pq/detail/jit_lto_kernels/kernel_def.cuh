/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#include <cuvs/neighbors/ivf_pq.hpp>
#include <neighbors/sample_filter.cuh>

namespace cuvs::neighbors::ivf_pq::detail {

template <typename OutT, typename LutT>
using compute_similarity_func_t = void(uint32_t dim,
                                       uint32_t n_probes,
                                       uint32_t pq_dim,
                                       uint32_t n_queries,
                                       uint32_t queries_offset,
                                       cuvs::neighbors::ivf_pq::codebook_gen codebook_kind,
                                       uint32_t topk,
                                       uint32_t max_samples,
                                       const float* cluster_centers,
                                       const float* pq_centers,
                                       const uint8_t* const* pq_dataset,
                                       const uint32_t* cluster_labels,
                                       const uint32_t* _chunk_indices,
                                       const float* queries,
                                       const uint32_t* index_list,
                                       float* query_kths,
                                       const int64_t* const* inds_ptrs,
                                       uint32_t* bitset_ptr,
                                       int64_t bitset_len,
                                       int64_t original_nbits,
                                       LutT* lut_scores,
                                       OutT* _out_scores,
                                       uint32_t* _out_indices);

}  // namespace cuvs::neighbors::ivf_pq::detail
