/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../sample_filter.cuh"  // none_sample_filter
#include "ivf_pq_fp_8bit.cuh"    // cuvs::neighbors::ivf_pq::detail::fp_8bit

#include "ivf_pq_compute_similarity.hpp"  // cuvs::neighbors::ivf_pq::detail::selected
#include <cuvs/detail/jit_lto/ivf_pq/compute_similarity_fragments.hpp>
#include <cuvs/distance/distance.hpp>  // cuvs::distance::DistanceType
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>    // cuvs::neighbors::ivf_pq::codebook_gen
#include <raft/core/detail/macros.hpp>  // RAFT_WEAK_FUNCTION
#include <rmm/cuda_stream_view.hpp>     // rmm::cuda_stream_view

#include <cuda_fp16.h>  // __half

namespace cuvs::neighbors::ivf_pq::detail {

// is_local_topk_feasible is not inline here, because we would have to define it
// here as well. That would run the risk of the definitions here and in the
// -inl.cuh header diverging.
auto RAFT_WEAK_FUNCTION is_local_topk_feasible(uint32_t k, uint32_t n_probes, uint32_t n_queries)
  -> bool;

template <typename OutT, typename LutT>
void compute_similarity_run(selected<OutT, LutT> s,
                            rmm::cuda_stream_view stream,
                            uint32_t dim,
                            uint32_t n_probes,
                            uint32_t pq_dim,
                            uint32_t n_queries,
                            uint32_t queries_offset,
                            codebook_gen codebook_kind,
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

/**
 * Use heuristics to choose an optimal instance of the search kernel.
 * It selects among a few kernel variants (with/out using shared mem for
 * lookup tables / precomputed distances) and tries to choose the block size
 * to maximize kernel occupancy.
 *
 * @param manage_local_topk
 *    whether use the fused calculate+select or just calculate the distances for each
 *    query and probed cluster.
 *
 * @param locality_hint
 *    beyond this limit do not consider increasing the number of active blocks per SM
 *    would improve locality anymore.
 */
template <typename OutT, typename LutT, typename FilterT, typename MetricTag, bool IncrementScore>
auto compute_similarity_select(const cudaDeviceProp& dev_props,
                               bool manage_local_topk,
                               int locality_hint,
                               double preferred_shmem_carveout,
                               uint32_t pq_bits,
                               uint32_t pq_dim,
                               uint32_t precomp_data_count,
                               uint32_t n_queries,
                               uint32_t n_probes,
                               uint32_t topk) -> selected<OutT, LutT>;

}  // namespace cuvs::neighbors::ivf_pq::detail

#define ARGS(...) __VA_ARGS__

#define instantiate_cuvs_neighbors_ivf_pq_detail_compute_similarity_select(    \
  OutT, LutT, FilterT, MetricTag, IncrementScore)                              \
  extern template auto cuvs::neighbors::ivf_pq::detail::                       \
    compute_similarity_select<OutT, LutT, FilterT, MetricTag, IncrementScore>( \
      const cudaDeviceProp& dev_props,                                         \
      bool manage_local_topk,                                                  \
      int locality_hint,                                                       \
      double preferred_shmem_carveout,                                         \
      uint32_t pq_bits,                                                        \
      uint32_t pq_dim,                                                         \
      uint32_t precomp_data_count,                                             \
      uint32_t n_queries,                                                      \
      uint32_t n_probes,                                                       \
      uint32_t topk) -> cuvs::neighbors::ivf_pq::detail::selected<OutT, LutT>;

#define instantiate_cuvs_neighbors_ivf_pq_detail_compute_similarity(OutT, LutT)             \
  instantiate_cuvs_neighbors_ivf_pq_detail_compute_similarity_select(                       \
    ARGS(OutT),                                                                             \
    ARGS(LutT),                                                                             \
    cuvs::neighbors::filtering::none_sample_filter,                                         \
    cuvs::neighbors::ivf_pq::detail::tag_metric_euclidean,                                  \
    false);                                                                                 \
  instantiate_cuvs_neighbors_ivf_pq_detail_compute_similarity_select(                       \
    ARGS(OutT),                                                                             \
    ARGS(LutT),                                                                             \
    ARGS(cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>),                     \
    cuvs::neighbors::ivf_pq::detail::tag_metric_euclidean,                                  \
    false);                                                                                 \
  instantiate_cuvs_neighbors_ivf_pq_detail_compute_similarity_select(                       \
    ARGS(OutT),                                                                             \
    ARGS(LutT),                                                                             \
    cuvs::neighbors::filtering::none_sample_filter,                                         \
    cuvs::neighbors::ivf_pq::detail::tag_metric_inner_product,                              \
    false);                                                                                 \
  instantiate_cuvs_neighbors_ivf_pq_detail_compute_similarity_select(                       \
    ARGS(OutT),                                                                             \
    ARGS(LutT),                                                                             \
    ARGS(cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>),                     \
    cuvs::neighbors::ivf_pq::detail::tag_metric_inner_product,                              \
    false);                                                                                 \
  instantiate_cuvs_neighbors_ivf_pq_detail_compute_similarity_select(                       \
    ARGS(OutT),                                                                             \
    ARGS(LutT),                                                                             \
    cuvs::neighbors::filtering::none_sample_filter,                                         \
    cuvs::neighbors::ivf_pq::detail::tag_metric_inner_product,                              \
    true);                                                                                  \
  instantiate_cuvs_neighbors_ivf_pq_detail_compute_similarity_select(                       \
    ARGS(OutT),                                                                             \
    ARGS(LutT),                                                                             \
    ARGS(cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>),                     \
    cuvs::neighbors::ivf_pq::detail::tag_metric_inner_product,                              \
    true);                                                                                  \
                                                                                            \
  extern template void cuvs::neighbors::ivf_pq::detail::compute_similarity_run<OutT, LutT>( \
    cuvs::neighbors::ivf_pq::detail::selected<OutT, LutT> s,                                \
    rmm::cuda_stream_view stream,                                                           \
    uint32_t dim,                                                                           \
    uint32_t n_probes,                                                                      \
    uint32_t pq_dim,                                                                        \
    uint32_t n_queries,                                                                     \
    uint32_t queries_offset,                                                                \
    cuvs::neighbors::ivf_pq::codebook_gen codebook_kind,                                    \
    uint32_t topk,                                                                          \
    uint32_t max_samples,                                                                   \
    const float* cluster_centers,                                                           \
    const float* pq_centers,                                                                \
    const uint8_t* const* pq_dataset,                                                       \
    const uint32_t* cluster_labels,                                                         \
    const uint32_t* _chunk_indices,                                                         \
    const float* queries,                                                                   \
    const uint32_t* index_list,                                                             \
    float* query_kths,                                                                      \
    const int64_t* const* inds_ptrs,                                                        \
    uint32_t* bitset_ptr,                                                                   \
    int64_t bitset_len,                                                                     \
    int64_t original_nbits,                                                                 \
    LutT* lut_scores,                                                                       \
    OutT* _out_scores,                                                                      \
    uint32_t* _out_indices);

instantiate_cuvs_neighbors_ivf_pq_detail_compute_similarity(
  half, ARGS(cuvs::neighbors::ivf_pq::detail::fp_8bit<5u, false>));
instantiate_cuvs_neighbors_ivf_pq_detail_compute_similarity(
  half, ARGS(cuvs::neighbors::ivf_pq::detail::fp_8bit<5u, true>));
instantiate_cuvs_neighbors_ivf_pq_detail_compute_similarity(half, half);
instantiate_cuvs_neighbors_ivf_pq_detail_compute_similarity(float, half);
instantiate_cuvs_neighbors_ivf_pq_detail_compute_similarity(float, float);
instantiate_cuvs_neighbors_ivf_pq_detail_compute_similarity(
  float, ARGS(cuvs::neighbors::ivf_pq::detail::fp_8bit<5u, false>));
instantiate_cuvs_neighbors_ivf_pq_detail_compute_similarity(
  float, ARGS(cuvs::neighbors::ivf_pq::detail::fp_8bit<5u, true>));

#undef ARGS

#undef instantiate_cuvs_neighbors_ivf_pq_detail_compute_similarity_select
#undef instantiate_cuvs_neighbors_ivf_pq_detail_compute_similarity
