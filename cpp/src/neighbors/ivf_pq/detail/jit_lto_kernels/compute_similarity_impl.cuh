/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#include "device_functions.cuh"

#include <raft/matrix/detail/select_warpsort.cuh>

namespace cuvs::neighbors::ivf_pq::detail {

/**
 * The main kernel that computes similarity scores across multiple queries and probes.
 * When `Capacity > 0`, it also selects top K candidates for each query and probe
 * (which need to be merged across probes afterwards).
 *
 * Each block processes a (query, probe) pair: it calculates the distance between the single query
 * vector and all the dataset vector in the cluster that we are probing.
 *
 * @tparam OutT
 *   The output type - distances.
 * @tparam LutT
 *   The lookup table element type (lut_scores).
 *
 * @param dim the dimensionality of the data (NB: after rotation transform, i.e. `index.rot_dim()`).
 * @param n_probes the number of clusters to search for each query
 * @param pq_dim
 *   The dimensionality of an encoded vector after compression by PQ.
 * @param n_queries the number of queries.
 * @param queries_offset
 *   An offset of the current query batch. It is used for feeding sample_filter with the
 *   correct query index.
 * @param metric the distance type.
 * @param codebook_kind Defines the way PQ codebooks have been trained.
 * @param topk the `k` in the select top-k.
 * @param max_samples the size of the output for a single query.
 * @param cluster_centers
 *   The device pointer to the cluster centers in the original space (NB: after rotation)
 *   [n_clusters, dim].
 * @param pq_centers
 *   The device pointer to the cluster centers in the PQ space
 *   [pq_dim, pq_book_size, pq_len] or [n_clusters, pq_book_size, pq_len].
 * @param pq_dataset
 *   The device pointer to the PQ index (data) [n_rows, ...].
 * @param cluster_labels
 *   The device pointer to the labels (clusters) for each query and probe [n_queries, n_probes].
 * @param _chunk_indices
 *   The device pointer to the data offsets for each query and probe [n_queries, n_probes].
 * @param queries
 *   The device pointer to the queries (NB: after rotation) [n_queries, dim].
 * @param index_list
 *   An optional device pointer to the enforced order of search [n_queries, n_probes].
 *   One can pass reordered indices here to try to improve data reading locality.
 * @param query_kth
 *   query_kths keep the current state of the filtering - atomically updated distances to the
 *   k-th closest neighbors for each query [n_queries].
 * @param sample_filter
 *   A filter that selects samples for a given query.
 * @param lut_scores
 *   The device pointer for storing the lookup table globally [gridDim.x, pq_dim << PqBits].
 *   Ignored when `EnableSMemLut == true`.
 * @param _out_scores
 *   The device pointer to the output scores
 *   [n_queries, max_samples] or [n_queries, n_probes, topk].
 * @param _out_indices
 *   The device pointer to the output indices [n_queries, n_probes, topk].
 *   These are the indices of the records as they appear in the database view formed by the probed
 *   clusters / defined by the `_chunk_indices`.
 *   The indices can have values within the range [0, max_samples).
 *   Ignored  when `Capacity == 0`.
 */
template <typename OutT, typename LutT>
__device__ void compute_similarity_impl(uint32_t dim,
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
                                        uint32_t* _out_indices)
{
  /* Shared memory:

    * lut_scores: lookup table (LUT) of size = `pq_dim << PqBits`  (when EnableSMemLut)
    * lut_end+:
       * base_diff: size = dim (which is equal to `pq_dim * pq_len`)  or dim*2
       * topk::warp_sort::mem_required - local topk temporary buffer (if necessary)
    * topk::block_sort: some amount of shared memory, but overlaps with the rest:
        block_sort only needs shared memory for `.done()` operation, which can come very last.
  */
  extern __shared__ __align__(256) uint8_t smem_buf[];  // NOLINT

  const uint32_t pq_len = dim / pq_dim;

  uint8_t* lut_end = nullptr;
  prepare_lut<LutT>(smem_buf, pq_dim, lut_scores, lut_end);

  for (int ib = blockIdx.x; ib < n_queries * n_probes; ib += gridDim.x) {
    if (ib >= gridDim.x) {
      // sync shared memory accesses on the second and further iterations
      __syncthreads();
    }
    uint32_t query_ix;
    uint32_t probe_ix;
    if (index_list == nullptr) {
      query_ix = ib % n_queries;
      probe_ix = ib / n_queries;
    } else {
      auto ordered_ix = index_list[ib];
      query_ix        = ordered_ix / n_probes;
      probe_ix        = ordered_ix % n_probes;
    }

    const uint32_t* chunk_indices = _chunk_indices + (n_probes * query_ix);
    const float* query            = queries + (dim * query_ix);
    OutT* out_scores;
    uint32_t* out_indices       = nullptr;
    uint32_t label              = cluster_labels[n_probes * query_ix + probe_ix];
    const float* cluster_center = cluster_centers + dim * label;

    store_calculated_distances<OutT>(_out_scores,
                                     _out_indices,
                                     probe_ix,
                                     n_probes,
                                     query_ix,
                                     topk,
                                     max_samples,
                                     out_scores,
                                     out_indices);
    precompute_base_diff(dim, lut_end, query, cluster_center);
    create_lut<LutT>(
      pq_dim, pq_len, label, pq_centers, codebook_kind, lut_scores, lut_end, query, cluster_center);
    compute_distances<OutT, LutT>(chunk_indices,
                                  query_kths,
                                  n_probes,
                                  probe_ix,
                                  query_ix,
                                  max_samples,
                                  topk,
                                  lut_end,
                                  pq_dim,
                                  pq_dataset,
                                  label,
                                  queries_offset,
                                  out_scores,
                                  out_indices,
                                  lut_scores,
                                  smem_buf,
                                  inds_ptrs,
                                  bitset_ptr,
                                  bitset_len,
                                  original_nbits);
  }
}

}  // namespace cuvs::neighbors::ivf_pq::detail
