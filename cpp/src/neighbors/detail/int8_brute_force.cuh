/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "./faiss_distance_utils.h"

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/selection/select_k.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/integer_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cublas_v2.h>
#include <cuda/iterator>
#include <thrust/for_each.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

namespace cuvs::neighbors::detail {

constexpr int kInt8BlockSize = 256;

inline cublasStatus_t int8_gemm(cublasHandle_t handle,
                                std::int8_t const* dataset,
                                std::int8_t const* queries,
                                std::int32_t* scores,
                                int64_t n_rows,
                                int64_t n_queries,
                                int64_t dim,
                                int64_t leading_dim)
{
  std::int32_t const alpha = 1;
  std::int32_t const beta  = 0;
  auto const status        = cublasGemmEx(handle,
                                   CUBLAS_OP_T,
                                   CUBLAS_OP_N,
                                   static_cast<int>(n_rows),
                                   static_cast<int>(n_queries),
                                   static_cast<int>(dim),
                                   &alpha,
                                   dataset,
                                   CUDA_R_8I,
                                   static_cast<int>(dim),
                                   queries,
                                   CUDA_R_8I,
                                   static_cast<int>(dim),
                                   &beta,
                                   scores,
                                   CUDA_R_32I,
                                   static_cast<int>(leading_dim),
                                   CUBLAS_COMPUTE_32I,
                                   CUBLAS_GEMM_DEFAULT);

  // Some cuBLAS versions do not provide an INT8 kernel for otherwise valid, irregular shapes.
  // Splitting only the rejected product keeps the common case to one GEMM.
  if (status != CUBLAS_STATUS_NOT_SUPPORTED) { return status; }
  if (n_queries > 16) {
    for (int64_t query = 0; query < n_queries; query += 16) {
      auto const result = int8_gemm(handle,
                                    dataset,
                                    queries + query * dim,
                                    scores + query * leading_dim,
                                    n_rows,
                                    std::min<int64_t>(16, n_queries - query),
                                    dim,
                                    leading_dim);
      if (result != CUBLAS_STATUS_SUCCESS) { return result; }
    }
    return CUBLAS_STATUS_SUCCESS;
  }
  if (n_rows >= 8) {
    auto const first_rows = n_rows / 8 * 4;
    auto const result =
      int8_gemm(handle, dataset, queries, scores, first_rows, n_queries, dim, leading_dim);
    if (result != CUBLAS_STATUS_SUCCESS) { return result; }
    return int8_gemm(handle,
                     dataset + first_rows * dim,
                     queries,
                     scores + first_rows,
                     n_rows - first_rows,
                     n_queries,
                     dim,
                     leading_dim);
  }
  return status;
}

RAFT_KERNEL int8_postprocess_kernel(std::int32_t* scores,
                                    int64_t n_rows,
                                    int64_t leading_dim,
                                    float const* query_norms,
                                    float const* dataset_norms,
                                    cuvs::distance::DistanceType metric,
                                    bool select_min)
{
  auto const query = static_cast<int64_t>(blockIdx.y);
  auto* row_scores = scores + query * leading_dim;
  auto* output     = reinterpret_cast<float*>(row_scores);
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; row < leading_dim;
       row += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    if (row >= n_rows) {
      output[row] =
        select_min ? std::numeric_limits<float>::infinity() : std::numeric_limits<float>::lowest();
      continue;
    }

    auto const dot = static_cast<float>(row_scores[row]);
    switch (metric) {
      case cuvs::distance::DistanceType::InnerProduct: output[row] = dot; break;
      case cuvs::distance::DistanceType::CosineExpanded: {
        auto const denominator = query_norms[query] * dataset_norms[row];
        output[row]            = denominator > 0.0f ? 1.0f - dot / denominator : 1.0f;
        break;
      }
      default:
        // Query norms are constant within each row; finish L2 only after the final top-k.
        output[row] = dataset_norms[row] - 2.0f * dot;
        break;
    }
  }
}

inline unsigned int8_grid_size(int64_t size)
{
  return static_cast<unsigned>(
    std::clamp<int64_t>(raft::ceildiv(size, int64_t{kInt8BlockSize}), 1, 65535));
}

inline bool int8_supported_metric(cuvs::distance::DistanceType metric)
{
  return metric == cuvs::distance::DistanceType::InnerProduct ||
         metric == cuvs::distance::DistanceType::L2Expanded ||
         metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
         metric == cuvs::distance::DistanceType::L2Unexpanded ||
         metric == cuvs::distance::DistanceType::L2SqrtUnexpanded ||
         metric == cuvs::distance::DistanceType::CosineExpanded;
}

template <typename IdxT, typename QueryLayoutT>
void int8_brute_force_search(
  raft::resources const& res,
  cuvs::neighbors::brute_force::index<std::int8_t, float> const& index,
  raft::device_matrix_view<const std::int8_t, int64_t, QueryLayoutT> queries,
  raft::device_matrix_view<IdxT, int64_t, raft::row_major> neighbors,
  raft::device_matrix_view<float, int64_t, raft::row_major> distances)
{
  auto const n = static_cast<int64_t>(index.size());
  auto const m = static_cast<int64_t>(queries.extent(0));
  auto const d = static_cast<int64_t>(index.dim());
  auto const k = static_cast<int64_t>(neighbors.extent(1));
  RAFT_EXPECTS(distances.extent(0) == queries.extent(0) &&
                 distances.extent(1) == neighbors.extent(1) &&
                 neighbors.extent(0) == queries.extent(0),
               "Output shapes must match the number of queries and k");
  RAFT_EXPECTS(queries.extent(1) == d, "Number of columns in queries must match brute force index");
  RAFT_EXPECTS(k >= 1 && k <= n, "k must satisfy 1 <= k <= index size");
  RAFT_EXPECTS(d > 0 && d % 4 == 0,
               "INT8 brute-force search requires a positive dimension divisible by 4");
  RAFT_EXPECTS(d <= std::numeric_limits<std::int32_t>::max() / (128 * 128),
               "INT8 brute-force dimension is too large for exact INT32 dot-product accumulation");
  RAFT_EXPECTS(int8_supported_metric(index.metric()),
               "INT8 brute-force search supports InnerProduct, L2, and CosineExpanded metrics");
  if (m == 0) { return; }

  auto const stream = raft::resource::get_cuda_stream(res);
  auto queries_storage =
    std::optional<raft::device_matrix<std::int8_t, int64_t, raft::row_major>>{};
  auto queries_ptr = queries.data_handle();
  if constexpr (std::is_same_v<QueryLayoutT, raft::col_major>) {
    queries_storage = raft::make_device_matrix<std::int8_t, int64_t, raft::row_major>(res, m, d);
    raft::copy(res, queries_storage->view(), queries);
    queries_ptr = queries_storage->data_handle();
  }

  auto const cosine      = index.metric() == cuvs::distance::DistanceType::CosineExpanded;
  auto const needs_norms = index.metric() != cuvs::distance::DistanceType::InnerProduct;
  rmm::device_uvector<float> query_norms(needs_norms ? m : 0, stream);
  rmm::device_uvector<float> dataset_norms(needs_norms && !index.has_norms() ? n : 0, stream);
  auto const* dataset_norms_ptr =
    index.has_norms() ? index.norms().data_handle() : dataset_norms.data();
  if (needs_norms) {
    auto compute_norms = [&](std::int8_t const* data, int64_t rows, float* norms) {
      raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
        res,
        raft::make_device_matrix_view<const std::int8_t, int64_t>(data, rows, d),
        raft::make_device_vector_view<float, int64_t>(norms, rows),
        [cosine] __device__(float norm) { return cosine ? sqrtf(norm) : norm; });
    };
    compute_norms(queries_ptr, m, query_norms.data());
    if (!index.has_norms()) {
      compute_norms(index.dataset().data_handle(), n, dataset_norms.data());
    }
  }

  size_t tile_rows{};
  size_t tile_cols{};
  cuvs::neighbors::detail::faiss_select::chooseTileSize(
    m, n, d, sizeof(float), tile_rows, tile_cols);
  tile_rows = std::max<size_t>(tile_rows, 1);
  tile_cols = std::max<size_t>(tile_cols, static_cast<size_t>(k));
  tile_cols = raft::round_up_safe(tile_cols, size_t{4});
  RAFT_EXPECTS(tile_rows <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
                 tile_cols <= static_cast<size_t>(std::numeric_limits<int>::max()),
               "INT8 brute-force tile is too large for cuBLAS");

  auto const n_tiles = raft::ceildiv(static_cast<size_t>(n), tile_cols);
  int64_t candidate_cols{};
  if (n_tiles > 1) {
    for (size_t tile = 0; tile < n_tiles; ++tile) {
      auto const first = tile * tile_cols;
      candidate_cols += std::min<int64_t>(k, n - static_cast<int64_t>(first));
    }
  }

  rmm::device_uvector<std::int32_t> scores(tile_rows * tile_cols, stream);
  rmm::device_uvector<float> candidate_distances(tile_rows * candidate_cols, stream);
  rmm::device_uvector<IdxT> candidate_indices(tile_rows * candidate_cols, stream);
  rmm::device_uvector<std::int8_t> padded_dataset(0, stream);

  auto handle           = raft::resource::get_cublas_handle(res);
  auto const select_min = cuvs::distance::is_min_close(index.metric());

  for (int64_t query_offset = 0; query_offset < m; query_offset += tile_rows) {
    auto const current_queries = std::min<int64_t>(tile_rows, m - query_offset);
    auto* distances_ptr        = distances.data_handle() + query_offset * k;
    auto* indices_ptr          = neighbors.data_handle() + query_offset * k;
    int64_t candidate_offset   = 0;
    for (int64_t dataset_offset = 0; dataset_offset < n; dataset_offset += tile_cols) {
      auto const current_rows = std::min<int64_t>(tile_cols, n - dataset_offset);
      auto const gemm_rows    = raft::round_up_safe(current_rows, int64_t{4});
      auto const* dataset_ptr = index.dataset().data_handle() + dataset_offset * d;
      if (gemm_rows != current_rows) {
        padded_dataset.resize(static_cast<size_t>(gemm_rows * d), stream);
        RAFT_CUDA_TRY(cudaMemsetAsync(
          padded_dataset.data(), 0, padded_dataset.size() * sizeof(std::int8_t), stream.get()));
        RAFT_CUDA_TRY(cudaMemcpyAsync(padded_dataset.data(),
                                      dataset_ptr,
                                      static_cast<size_t>(current_rows * d),
                                      cudaMemcpyDeviceToDevice,
                                      stream.get()));
        dataset_ptr = padded_dataset.data();
      }

      auto const status = int8_gemm(handle,
                                    dataset_ptr,
                                    queries_ptr + query_offset * d,
                                    scores.data(),
                                    gemm_rows,
                                    current_queries,
                                    d,
                                    tile_cols);
      if (status != CUBLAS_STATUS_SUCCESS) {
        RAFT_FAIL("INT8 brute-force cublasGemmEx failed with status %d", status);
      }

      auto const query_blocks = static_cast<unsigned>(current_queries);
      // Keep the block budget independent of the query tile size; each block strides within a row.
      auto const column_blocks = std::min(int8_grid_size(tile_cols), 65535u / query_blocks);
      dim3 const grid(column_blocks, query_blocks);
      int8_postprocess_kernel<<<grid, kInt8BlockSize, 0, stream.get()>>>(
        scores.data(),
        current_rows,
        tile_cols,
        needs_norms ? query_norms.data() + query_offset : nullptr,
        needs_norms ? dataset_norms_ptr + dataset_offset : nullptr,
        index.metric(),
        select_min);
      RAFT_CUDA_TRY(cudaPeekAtLastError());

      auto const current_k = std::min<int64_t>(k, current_rows);
      cuvs::selection::select_k(
        res,
        raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
          reinterpret_cast<float const*>(scores.data()), current_queries, tile_cols),
        std::nullopt,
        raft::make_device_matrix_view<float, int64_t, raft::row_major>(
          distances_ptr, current_queries, current_k),
        raft::make_device_matrix_view<IdxT, int64_t, raft::row_major>(
          indices_ptr, current_queries, current_k),
        select_min,
        true);

      if (n_tiles > 1) {
        // Group tile winners by query and convert tile-local IDs to dataset IDs.
        const float* in_distances = distances_ptr;
        const IdxT* in_indices    = indices_ptr;
        float* out_distances      = candidate_distances.data();
        IdxT* out_indices         = candidate_indices.data();
        auto count                = cuda::make_counting_iterator<int64_t>(0);
        thrust::for_each(raft::resource::get_thrust_policy(res),
                         count,
                         count + current_queries * current_k,
                         [=] __device__(int64_t i) {
                           auto const row           = i / current_k;
                           auto const col           = i % current_k;
                           auto const out_index     = row * candidate_cols + candidate_offset + col;
                           out_distances[out_index] = in_distances[i];
                           out_indices[out_index] =
                             in_indices[i] + static_cast<IdxT>(dataset_offset);
                         });
        candidate_offset += current_k;
      }
    }

    if (n_tiles > 1) {
      cuvs::selection::select_k(
        res,
        raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
          candidate_distances.data(), current_queries, candidate_cols),
        raft::make_device_matrix_view<const IdxT, int64_t, raft::row_major>(
          candidate_indices.data(), current_queries, candidate_cols),
        raft::make_device_matrix_view<float, int64_t, raft::row_major>(
          distances_ptr, current_queries, k),
        raft::make_device_matrix_view<IdxT, int64_t, raft::row_major>(
          indices_ptr, current_queries, k),
        select_min,
        true);
    }
  }

  if (needs_norms && !cosine) {
    auto values = raft::make_device_vector_view<float, int64_t>(distances.data_handle(), m * k);
    auto const* norms    = query_norms.data();
    bool const take_sqrt = index.metric() == cuvs::distance::DistanceType::L2SqrtExpanded ||
                           index.metric() == cuvs::distance::DistanceType::L2SqrtUnexpanded;
    raft::linalg::map_offset(
      res,
      values,
      [norms, k, take_sqrt] __device__(int64_t i, float score) {
        auto const distance = fmaxf(norms[i / k] + score, 0.0f);
        return take_sqrt ? sqrtf(distance) : distance;
      },
      raft::make_const_mdspan(values));
  }
}

}  // namespace cuvs::neighbors::detail
