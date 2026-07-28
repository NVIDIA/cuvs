/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/*
 * Brute-force prefiltered search with Roaring-bitmap filters.
 *
 * Both entry points implement a three-regime dispatch keyed on filter
 * selectivity s (known on the host from the filters' cardinalities — no
 * count kernels, no device synchronization):
 *
 *  - very sparse:   emit the filter's member ids straight into a CSR
 *                   structure (one kernel; indptr is free from host-side
 *                   cardinalities), compute distances at the nnz positions
 *                   with SDDMM, select_k over the sparse rows.
 *  - mid (shared filter only): gather the selected rows and run a dense
 *                   GEMM + select_k over [n_queries, |filter|] — computes
 *                   |filter| columns instead of n_rows. This is where the
 *                   bulk of the win lives (up to ~19x vs the dense+mask
 *                   path at 1-10% selectivity, 10M x 512d).
 *  - dense:         decompress to a flat bitset / bit matrix and delegate
 *                   to the existing bitset/bitmap pipeline, which is
 *                   already optimal above ~40-50% selectivity.
 *
 * The sparse/mid threshold is dimension-dependent (cusparse SDDMM
 * degrades with d, dense GEMM does not): measured crossovers on
 * RTX 5090 are ~3% at d=128 and ~0.1% at d=512.
 *
 * float32 only in this version (the SDDMM/GEMM plumbing is instantiated
 * for float; half support tracks the existing filtered-path limitation).
 */

#include <cuvs/core/roaring.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/roaring_filter.hpp>

#include "./knn_utils.cuh"

#include <raft/core/bitmap.cuh>
#include <raft/core/bitset.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/matrix/select_k.cuh>
#include <raft/sparse/convert/coo.cuh>
#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/linalg/sddmm.hpp>
#include <raft/sparse/matrix/select_k.cuh>

#include <rmm/device_uvector.hpp>

#include <thrust/fill.h>

#include <cstdint>
#include <vector>

namespace cuvs::neighbors::detail {

// defined in knn_brute_force.cuh (which includes this header first)
template <typename T, typename IdxT, typename BitsT, typename DistanceT>
void brute_force_search_filtered(
  raft::resources const& res,
  const cuvs::neighbors::brute_force::index<T, DistanceT>& idx,
  raft::device_matrix_view<const T, IdxT, raft::row_major> queries,
  const cuvs::neighbors::filtering::base_filter* filter,
  raft::device_matrix_view<IdxT, IdxT, raft::row_major> neighbors,
  raft::device_matrix_view<DistanceT, IdxT, raft::row_major> distances,
  std::optional<raft::device_vector_view<const DistanceT, IdxT>> query_norms = std::nullopt);

namespace roaring_detail {

/** Selectivity below which the SDDMM path beats gathering + dense GEMM. */
inline double sparse_threshold(int64_t dim) { return dim >= 256 ? 0.001 : 0.03; }
/** Selectivity above which the dense masked pipeline wins. */
constexpr double kDenseThreshold = 0.45;

__global__ inline void replicate_indices_kernel(const int64_t* ids,
                                                int64_t n_ids,
                                                int64_t total,
                                                int64_t* out)
{
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i < total) out[i] = ids[i % n_ids];
}

__global__ inline void gather_rows_kernel(const float* __restrict__ dataset,
                                          const int64_t* __restrict__ ids,
                                          int64_t n_ids,
                                          int64_t dim,
                                          float* __restrict__ out)
{
  int64_t row = blockIdx.x;
  if (row >= n_ids) return;
  const float* src = dataset + ids[row] * dim;
  float* dst       = out + row * dim;
  if ((dim & 3) == 0 && (reinterpret_cast<uintptr_t>(src) & 15) == 0) {
    auto* src4 = reinterpret_cast<const float4*>(src);
    auto* dst4 = reinterpret_cast<float4*>(dst);
    for (int64_t c = threadIdx.x; c < dim / 4; c += blockDim.x)
      dst4[c] = src4[c];
  } else {
    for (int64_t c = threadIdx.x; c < dim; c += blockDim.x)
      dst[c] = src[c];
  }
}

__global__ inline void gather_norms_kernel(const float* __restrict__ norms,
                                           const int64_t* __restrict__ ids,
                                           int64_t n_ids,
                                           float* __restrict__ out)
{
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i < n_ids) out[i] = norms[ids[i]];
}

/** Dense counterpart of epilogue_on_csr: dist[q][u] from raw dots. */
__global__ inline void dense_epilogue_kernel(float* __restrict__ dist,
                                             const float* __restrict__ q_norms,
                                             const float* __restrict__ r_norms,
                                             int64_t n_queries,
                                             int64_t n_cols,
                                             cuvs::distance::DistanceType metric)
{
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= n_queries * n_cols) return;
  int64_t q = i / n_cols;
  int64_t u = i % n_cols;
  float dot = dist[i];
  if (metric == cuvs::distance::DistanceType::L2Expanded) {
    dist[i] = -2.0f * dot + q_norms[q] + r_norms[u];
  } else if (metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
    dist[i] = raft::sqrt(-2.0f * dot + q_norms[q] + r_norms[u]);
  } else {  // CosineExpanded
    dist[i] = 1.0f - dot / (q_norms[q] * r_norms[u]);
  }
}

__global__ inline void remap_ids_kernel(const int64_t* __restrict__ pos,
                                        const int64_t* __restrict__ ids,
                                        int64_t n,
                                        int64_t* __restrict__ out)
{
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i < n) out[i] = pos[i] < 0 ? int64_t{-1} : ids[pos[i]];
}

/** Compute query norms the way the existing filtered path does. */
inline rmm::device_uvector<float> make_query_norms(
  raft::resources const& res,
  raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
  cuvs::distance::DistanceType metric)
{
  auto stream = raft::resource::get_cuda_stream(res);
  rmm::device_uvector<float> q_norms(queries.extent(0), stream);
  if (metric == cuvs::distance::DistanceType::CosineExpanded) {
    raft::linalg::rowNorm<raft::linalg::L2Norm, true>(q_norms.data(),
                                                      queries.data_handle(),
                                                      queries.extent(1),
                                                      queries.extent(0),
                                                      stream,
                                                      raft::sqrt_op{});
  } else {
    raft::linalg::rowNorm<raft::linalg::L2Norm, true>(q_norms.data(),
                                                      queries.data_handle(),
                                                      queries.extent(1),
                                                      queries.extent(0),
                                                      stream,
                                                      raft::identity_op{});
  }
  return q_norms;
}

/** Shared SDDMM + epilogue + sparse select_k over a prebuilt CSR pattern. */
inline void sparse_distances_select_k(
  raft::resources const& res,
  const cuvs::neighbors::brute_force::index<float, float>& idx,
  raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
  const int64_t* indptr,
  const int64_t* indices,
  int64_t nnz,
  raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
  raft::device_matrix_view<float, int64_t, raft::row_major> distances)
{
  auto stream       = raft::resource::get_cuda_stream(res);
  auto metric       = idx.metric();
  int64_t n_queries = queries.extent(0);
  int64_t n_rows    = idx.dataset().extent(0);
  int64_t dim       = idx.dataset().extent(1);

  rmm::device_uvector<float> values(nnz, stream);
  auto structure = raft::make_device_compressed_structure_view<int64_t, int64_t, int64_t>(
    const_cast<int64_t*>(indptr), const_cast<int64_t*>(indices), n_queries, n_rows, nnz);
  auto csr_view =
    raft::make_device_csr_matrix_view<float, int64_t, int64_t, int64_t>(values.data(), structure);

  auto dataset_view = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
    idx.dataset().data_handle(), n_rows, dim);
  float alpha_v = 1.0f, beta_v = 0.0f;
  auto alpha = raft::make_host_scalar_view<float>(&alpha_v);
  auto beta  = raft::make_host_scalar_view<float>(&beta_v);
  raft::sparse::linalg::sddmm(res,
                              queries,
                              dataset_view,
                              csr_view,
                              raft::linalg::Operation::NON_TRANSPOSE,
                              raft::linalg::Operation::TRANSPOSE,
                              alpha,
                              beta);

  if (metric == cuvs::distance::DistanceType::L2Expanded ||
      metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
      metric == cuvs::distance::DistanceType::CosineExpanded) {
    rmm::device_uvector<int64_t> rows(nnz, stream);
    raft::sparse::convert::csr_to_coo(
      const_cast<int64_t*>(indptr), n_queries, rows.data(), nnz, stream);
    auto q_norms = make_query_norms(res, queries, metric);
    cuvs::neighbors::detail::epilogue_on_csr(res,
                                             values.data(),
                                             nnz,
                                             rows.data(),
                                             const_cast<int64_t*>(indices),
                                             q_norms.data(),
                                             idx.norms().data_handle(),
                                             metric);
  }

  auto const_csr = raft::make_device_csr_matrix_view<const float, int64_t, int64_t, int64_t>(
    values.data(), structure);
  std::optional<raft::device_vector_view<const int64_t, int64_t>> no_opt = std::nullopt;
  bool select_min = cuvs::distance::is_min_close(metric);
  raft::sparse::matrix::select_k(res, const_csr, no_opt, distances, neighbors, select_min, true);
}

}  // namespace roaring_detail

/**
 * Shared-filter (one Roaring bitmap for all queries) brute-force search.
 */
inline void brute_force_search_roaring(
  raft::resources const& res,
  const cuvs::neighbors::brute_force::index<float, float>& idx,
  raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
  const cuvs::neighbors::filtering::roaring_filter& filter,
  raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
  raft::device_matrix_view<float, int64_t, raft::row_major> distances)
{
  using namespace roaring_detail;
  auto stream       = raft::resource::get_cuda_stream(res);
  auto metric       = idx.metric();
  int64_t n_queries = queries.extent(0);
  int64_t n_rows    = idx.dataset().extent(0);
  int64_t dim       = idx.dataset().extent(1);

  const cuvs::core::gpu_roaring& bitmap = *filter.bitmap_;
  RAFT_EXPECTS(!bitmap.negated, "roaring_filter: negated bitmaps are not supported");
  RAFT_EXPECTS(static_cast<int64_t>(bitmap.universe_size) == n_rows,
               "roaring_filter universe size must match the index size");
  int64_t n_selected = static_cast<int64_t>(bitmap.total_cardinality);
  double s           = static_cast<double>(n_selected) / static_cast<double>(n_rows);

  bool needs_norms = metric == cuvs::distance::DistanceType::L2Expanded ||
                     metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                     metric == cuvs::distance::DistanceType::CosineExpanded;
  RAFT_EXPECTS(idx.has_norms() || !needs_norms,
               "roaring_filter: index must carry norms for L2/Cosine metrics");

  if (n_selected == 0) {
    thrust::fill(raft::resource::get_thrust_policy(res),
                 neighbors.data_handle(),
                 neighbors.data_handle() + n_queries * neighbors.extent(1),
                 int64_t{-1});
    return;
  }

  // ---- dense regime: decompress and delegate to the bitset pipeline ----
  if (s >= kDenseThreshold) {
    auto bitset = cuvs::core::to_bitset(res, bitmap);
    auto bf     = cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>(bitset.view());
    brute_force_search_filtered<float, int64_t, uint32_t, float>(
      res, idx, queries, &bf, neighbors, distances, std::nullopt);
    return;
  }

  // member ids, emitted once per search from the containers
  const cuvs::core::gpu_roaring* bptr = &bitmap;
  rmm::device_uvector<int64_t> ids(n_selected, stream);
  cuvs::core::to_csr_indices(res, &bptr, 1, ids.data());

  // ---- very sparse: per-query CSR (replicated indices) + SDDMM ----
  if (s <= sparse_threshold(dim)) {
    int64_t nnz = n_queries * n_selected;
    std::vector<int64_t> h_indptr(n_queries + 1);
    for (int64_t q = 0; q <= n_queries; ++q)
      h_indptr[q] = q * n_selected;
    rmm::device_uvector<int64_t> indptr(n_queries + 1, stream);
    raft::update_device(indptr.data(), h_indptr.data(), h_indptr.size(), stream);
    rmm::device_uvector<int64_t> indices(nnz, stream);
    int64_t blocks = (nnz + 255) / 256;
    replicate_indices_kernel<<<static_cast<unsigned>(blocks), 256, 0, stream>>>(
      ids.data(), n_selected, nnz, indices.data());
    sparse_distances_select_k(
      res, idx, queries, indptr.data(), indices.data(), nnz, neighbors, distances);
    return;
  }

  // ---- mid regime: gather selected rows, dense GEMM, select_k, remap ----
  constexpr int64_t kChunk = int64_t{1} << 20;
  rmm::device_uvector<float> gathered(std::min(kChunk, n_selected) * dim, stream);
  rmm::device_uvector<float> dist(n_queries * n_selected, stream);

  for (int64_t c0 = 0; c0 < n_selected; c0 += kChunk) {
    int64_t uc = std::min(kChunk, n_selected - c0);
    gather_rows_kernel<<<static_cast<unsigned>(uc), 128, 0, stream>>>(
      idx.dataset().data_handle(), ids.data() + c0, uc, dim, gathered.data());
    // row-major dist[Q, U] tile = queries[Q, dim] x gathered[uc, dim]^T
    float alpha = 1.0f, beta = 0.0f;
    raft::linalg::gemm(res,
                       true,
                       false,
                       static_cast<int>(uc),
                       static_cast<int>(n_queries),
                       static_cast<int>(dim),
                       &alpha,
                       gathered.data(),
                       static_cast<int>(dim),
                       queries.data_handle(),
                       static_cast<int>(dim),
                       &beta,
                       dist.data() + c0,
                       static_cast<int>(n_selected),
                       stream);
  }

  if (needs_norms) {
    rmm::device_uvector<float> r_norms(n_selected, stream);
    int64_t blocks = (n_selected + 255) / 256;
    gather_norms_kernel<<<static_cast<unsigned>(blocks), 256, 0, stream>>>(
      idx.norms().data_handle(), ids.data(), n_selected, r_norms.data());
    auto q_norms  = make_query_norms(res, queries, metric);
    int64_t total = n_queries * n_selected;
    dense_epilogue_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(
      dist.data(), q_norms.data(), r_norms.data(), n_queries, n_selected, metric);
  }

  int64_t k = neighbors.extent(1);
  rmm::device_uvector<int64_t> positions(n_queries * k, stream);
  bool select_min = cuvs::distance::is_min_close(metric);
  raft::matrix::select_k<float, int64_t>(
    res,
    raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
      dist.data(), n_queries, n_selected),
    std::nullopt,
    distances,
    raft::make_device_matrix_view<int64_t, int64_t, raft::row_major>(
      positions.data(), n_queries, k),
    select_min,
    true);
  int64_t total = n_queries * k;
  remap_ids_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(
    positions.data(), ids.data(), total, neighbors.data_handle());
}

/**
 * Per-query-filter (one Roaring bitmap per query) brute-force search.
 */
inline void brute_force_search_roaring_matrix(
  raft::resources const& res,
  const cuvs::neighbors::brute_force::index<float, float>& idx,
  raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
  const cuvs::neighbors::filtering::roaring_matrix_filter& filter,
  raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
  raft::device_matrix_view<float, int64_t, raft::row_major> distances)
{
  using namespace roaring_detail;
  auto stream       = raft::resource::get_cuda_stream(res);
  auto metric       = idx.metric();
  int64_t n_queries = queries.extent(0);
  int64_t n_rows    = idx.dataset().extent(0);

  RAFT_EXPECTS(static_cast<int64_t>(filter.n_queries_) == n_queries,
               "roaring_matrix_filter must provide one bitmap per query");
  bool needs_norms = metric == cuvs::distance::DistanceType::L2Expanded ||
                     metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                     metric == cuvs::distance::DistanceType::CosineExpanded;
  RAFT_EXPECTS(idx.has_norms() || !needs_norms,
               "roaring_matrix_filter: index must carry norms for L2/Cosine metrics");

  // indptr comes straight off the host-side cardinalities
  std::vector<int64_t> h_indptr(n_queries + 1, 0);
  for (int64_t q = 0; q < n_queries; ++q) {
    const auto* b = filter.bitmaps_[q];
    RAFT_EXPECTS(!b->negated, "roaring_matrix_filter: negated bitmaps are not supported");
    RAFT_EXPECTS(static_cast<int64_t>(b->universe_size) == n_rows,
                 "roaring_matrix_filter universe size must match the index size");
    h_indptr[q + 1] = h_indptr[q] + static_cast<int64_t>(b->total_cardinality);
  }
  int64_t nnz = h_indptr[n_queries];
  double s    = static_cast<double>(nnz) / (static_cast<double>(n_queries) * n_rows);

  if (nnz == 0) {
    thrust::fill(raft::resource::get_thrust_policy(res),
                 neighbors.data_handle(),
                 neighbors.data_handle() + n_queries * neighbors.extent(1),
                 int64_t{-1});
    return;
  }

  // ---- dense regime: decompress to a [n_queries, n_rows] bit matrix ----
  if (s >= kDenseThreshold) {
    int64_t n_words = (n_queries * n_rows + 31) / 32;
    rmm::device_uvector<uint32_t> bits(n_words, stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(bits.data(), 0, n_words * sizeof(uint32_t), stream));
    cuvs::core::decompress_to_bitmap(
      res, filter.bitmaps_, static_cast<uint32_t>(n_queries), n_rows, bits.data());
    auto view = cuvs::core::bitmap_view<uint32_t, int64_t>(bits.data(), n_queries, n_rows);
    auto bf   = cuvs::neighbors::filtering::bitmap_filter<uint32_t, int64_t>(view);
    brute_force_search_filtered<float, int64_t, uint32_t, float>(
      res, idx, queries, &bf, neighbors, distances, std::nullopt);
    return;
  }

  // ---- sparse: batched CSR emission (one launch for all filters) ----
  rmm::device_uvector<int64_t> indptr(n_queries + 1, stream);
  raft::update_device(indptr.data(), h_indptr.data(), h_indptr.size(), stream);
  rmm::device_uvector<int64_t> indices(nnz, stream);
  cuvs::core::to_csr_indices(
    res, filter.bitmaps_, static_cast<uint32_t>(n_queries), indices.data());
  sparse_distances_select_k(
    res, idx, queries, indptr.data(), indices.data(), nnz, neighbors, distances);
}

}  // namespace cuvs::neighbors::detail
