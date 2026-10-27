/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "./faiss_distance_utils.h"
#include "./fused_l2_knn.cuh"

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/selection/select_k.hpp>

#include <raft/core/copy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/integer_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cublas_v2.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

namespace cuvs::neighbors::detail {

constexpr int kInt8BlockSize = 256;

__global__ void int8_col_major_to_row_major_kernel(std::int8_t const* input,
                                                   std::int8_t* output,
                                                   int64_t rows,
                                                   int64_t cols)
{
  auto const size = rows * cols;
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < size;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    auto const row = i / cols;
    auto const col = i - row * cols;
    output[i]      = input[col * rows + row];
  }
}
constexpr int kInt8WarpSize = 32;

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

__global__ void int8_row_norms_kernel(
  std::int8_t const* data, int64_t n_rows, int64_t dim, float* norms, bool take_sqrt)
{
  auto const lane = threadIdx.x % kInt8WarpSize;
  auto const warp =
    static_cast<int64_t>(blockIdx.x) * (blockDim.x / kInt8WarpSize) + threadIdx.x / kInt8WarpSize;
  auto const warp_count = static_cast<int64_t>(gridDim.x) * (blockDim.x / kInt8WarpSize);
  for (int64_t row = warp; row < n_rows; row += warp_count) {
    std::int32_t norm = 0;
    for (int64_t col = lane; col < dim; col += kInt8WarpSize) {
      auto const value = static_cast<std::int32_t>(data[row * dim + col]);
      norm += value * value;
    }
    for (int offset = kInt8WarpSize / 2; offset > 0; offset /= 2) {
      norm += __shfl_down_sync(0xffffffff, norm, offset);
    }
    if (lane == 0) {
      norms[row] = take_sqrt ? sqrtf(static_cast<float>(norm)) : static_cast<float>(norm);
    }
  }
}

inline void int8_row_norms(raft::resources const& res,
                           std::int8_t const* data,
                           int64_t n_rows,
                           int64_t dim,
                           float* norms,
                           bool take_sqrt)
{
  auto const warps_per_block = kInt8BlockSize / kInt8WarpSize;
  auto const blocks          = static_cast<unsigned>(
    std::clamp<int64_t>(raft::ceildiv(n_rows, int64_t{warps_per_block}), 1, 65535));
  auto const stream = raft::resource::get_cuda_stream(res);
  int8_row_norms_kernel<<<blocks, kInt8BlockSize, 0, stream.get()>>>(
    data, n_rows, dim, norms, take_sqrt);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

__global__ void int8_postprocess_kernel(std::int32_t* scores,
                                        int64_t n_queries,
                                        int64_t n_rows,
                                        int64_t leading_dim,
                                        float const* query_norms,
                                        float const* dataset_norms,
                                        cuvs::distance::DistanceType metric,
                                        bool select_min)
{
  auto const size = n_queries * leading_dim;
  auto* output    = reinterpret_cast<float*>(scores);
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < size;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    auto const query = index / leading_dim;
    auto const row   = index % leading_dim;
    if (row >= n_rows) {
      output[index] =
        select_min ? std::numeric_limits<float>::infinity() : std::numeric_limits<float>::lowest();
      continue;
    }

    auto const dot = static_cast<float>(scores[index]);
    switch (metric) {
      case cuvs::distance::DistanceType::InnerProduct: output[index] = dot; break;
      case cuvs::distance::DistanceType::CosineExpanded: {
        auto const denominator = query_norms[query] * dataset_norms[row];
        output[index]          = denominator > 0.0f ? 1.0f - dot / denominator : 1.0f;
        break;
      }
      case cuvs::distance::DistanceType::L2SqrtExpanded:
      case cuvs::distance::DistanceType::L2SqrtUnexpanded:
        output[index] = sqrtf(fmaxf(query_norms[query] + dataset_norms[row] - 2.0f * dot, 0.0f));
        break;
      default:
        output[index] = fmaxf(query_norms[query] + dataset_norms[row] - 2.0f * dot, 0.0f);
        break;
    }
  }
}

__global__ void int8_sqrt_distances_kernel(float* distances, int64_t size)
{
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < size;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    distances[i] = sqrtf(distances[i]);
  }
}

template <typename IdxT>
__global__ void int8_scatter_candidates_kernel(float const* input_distances,
                                               IdxT const* input_indices,
                                               int64_t n_queries,
                                               int64_t input_cols,
                                               int64_t output_cols,
                                               int64_t output_offset,
                                               int64_t index_offset,
                                               float* output_distances,
                                               IdxT* output_indices)
{
  auto const size = n_queries * input_cols;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < size;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    auto const row              = index / input_cols;
    auto const col              = index % input_cols;
    auto const out_index        = row * output_cols + output_offset + col;
    output_distances[out_index] = input_distances[index];
    output_indices[out_index]   = input_indices[index] + static_cast<IdxT>(index_offset);
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
void int8_cutlass_brute_force_search(
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
  RAFT_EXPECTS(k <= 64, "INT8 CUTLASS brute-force search requires k <= 64");
  RAFT_EXPECTS(index.metric() == cuvs::distance::DistanceType::L2Expanded ||
                 index.metric() == cuvs::distance::DistanceType::L2SqrtExpanded ||
                 index.metric() == cuvs::distance::DistanceType::L2Unexpanded ||
                 index.metric() == cuvs::distance::DistanceType::L2SqrtUnexpanded,
               "INT8 CUTLASS brute-force search supports only L2 metrics");
  RAFT_EXPECTS(d <= std::numeric_limits<std::int32_t>::max() / (127 * 127),
               "INT8 brute-force dimension is too large for exact INT32 accumulation");

  auto const stream = raft::resource::get_cuda_stream(res);
  auto queries_storage =
    std::optional<raft::device_matrix<std::int8_t, int64_t, raft::row_major>>{};
  auto queries_ptr = queries.data_handle();
  if constexpr (std::is_same_v<QueryLayoutT, raft::col_major>) {
    queries_storage = raft::make_device_matrix<std::int8_t, int64_t, raft::row_major>(res, m, d);
    int8_col_major_to_row_major_kernel<<<int8_grid_size(m * d), kInt8BlockSize, 0, stream.get()>>>(
      queries.data_handle(), queries_storage->data_handle(), m, d);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    queries_ptr = queries_storage->data_handle();
  }

  size_t workspace_size = 0;
  rmm::device_uvector<char> workspace(0, stream);
  auto run = [&] {
    fusedL2UnexpKnn<std::int8_t, std::int32_t, float, IdxT, false, true>(
      m,
      n,
      d,
      d,
      d,
      n,
      queries_ptr,
      index.dataset().data_handle(),
      false,
      distances.data_handle(),
      neighbors.data_handle(),
      k,
      stream.get(),
      workspace.data(),
      workspace_size);
  };
  run();
  if (workspace_size > 0) {
    workspace.resize(workspace_size, stream);
    run();
  }

  if (index.metric() == cuvs::distance::DistanceType::L2SqrtExpanded ||
      index.metric() == cuvs::distance::DistanceType::L2SqrtUnexpanded) {
    auto const size = m * k;
    int8_sqrt_distances_kernel<<<int8_grid_size(size), kInt8BlockSize, 0, stream.get()>>>(
      distances.data_handle(), size);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
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
  RAFT_EXPECTS(d % 4 == 0, "INT8 brute-force search requires the dimension to be a multiple of 4");
  RAFT_EXPECTS(d <= std::numeric_limits<std::int32_t>::max() / (127 * 127),
               "INT8 brute-force dimension is too large for exact INT32 dot-product accumulation");
  RAFT_EXPECTS(int8_supported_metric(index.metric()),
               "INT8 brute-force search supports InnerProduct, L2, and CosineExpanded metrics");

  auto const stream = raft::resource::get_cuda_stream(res);
  auto queries_storage =
    std::optional<raft::device_matrix<std::int8_t, int64_t, raft::row_major>>{};
  auto queries_ptr = queries.data_handle();
  if constexpr (std::is_same_v<QueryLayoutT, raft::col_major>) {
    queries_storage = raft::make_device_matrix<std::int8_t, int64_t, raft::row_major>(res, m, d);
    int8_col_major_to_row_major_kernel<<<int8_grid_size(m * d), kInt8BlockSize, 0, stream.get()>>>(
      queries.data_handle(), queries_storage->data_handle(), m, d);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    queries_ptr = queries_storage->data_handle();
  }

  auto const cosine      = index.metric() == cuvs::distance::DistanceType::CosineExpanded;
  auto const needs_norms = index.metric() != cuvs::distance::DistanceType::InnerProduct;
  rmm::device_uvector<float> query_norms(needs_norms ? m : 0, stream);
  rmm::device_uvector<float> dataset_norms(needs_norms && !index.has_norms() ? n : 0, stream);
  auto const* dataset_norms_ptr =
    index.has_norms() ? index.norms().data_handle() : dataset_norms.data();
  if (needs_norms) {
    int8_row_norms(res, queries_ptr, m, d, query_norms.data(), cosine);
    if (!index.has_norms()) {
      int8_row_norms(res, index.dataset().data_handle(), n, d, dataset_norms.data(), cosine);
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
  for (size_t tile = 0; tile < n_tiles; ++tile) {
    auto const first = tile * tile_cols;
    candidate_cols += std::min<int64_t>(k, n - static_cast<int64_t>(first));
  }

  rmm::device_uvector<std::int32_t> scores(tile_rows * tile_cols, stream);
  rmm::device_uvector<float> scratch_distances(tile_rows * k, stream);
  rmm::device_uvector<IdxT> scratch_indices(tile_rows * k, stream);
  rmm::device_uvector<float> candidate_distances(tile_rows * candidate_cols, stream);
  rmm::device_uvector<IdxT> candidate_indices(tile_rows * candidate_cols, stream);
  rmm::device_uvector<std::int8_t> padded_dataset(0, stream);

  auto handle = raft::resource::get_cublas_handle(res);
  RAFT_EXPECTS(cublasSetStream(handle, stream.get()) == CUBLAS_STATUS_SUCCESS,
               "INT8 brute-force cublasSetStream failed");
  auto const select_min = cuvs::distance::is_min_close(index.metric());

  for (int64_t query_offset = 0; query_offset < m; query_offset += tile_rows) {
    auto const current_queries = std::min<int64_t>(tile_rows, m - query_offset);
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

      int8_postprocess_kernel<<<int8_grid_size(current_queries * static_cast<int64_t>(tile_cols)),
                                kInt8BlockSize,
                                0,
                                stream.get()>>>(
        scores.data(),
        current_queries,
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
          scratch_distances.data(), current_queries, current_k),
        raft::make_device_matrix_view<IdxT, int64_t, raft::row_major>(
          scratch_indices.data(), current_queries, current_k),
        select_min,
        true);

      int8_scatter_candidates_kernel<<<int8_grid_size(current_queries * current_k),
                                       kInt8BlockSize,
                                       0,
                                       stream.get()>>>(scratch_distances.data(),
                                                       scratch_indices.data(),
                                                       current_queries,
                                                       current_k,
                                                       candidate_cols,
                                                       candidate_offset,
                                                       dataset_offset,
                                                       candidate_distances.data(),
                                                       candidate_indices.data());
      RAFT_CUDA_TRY(cudaPeekAtLastError());
      candidate_offset += current_k;
    }

    cuvs::selection::select_k(res,
                              raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
                                candidate_distances.data(), current_queries, candidate_cols),
                              raft::make_device_matrix_view<const IdxT, int64_t, raft::row_major>(
                                candidate_indices.data(), current_queries, candidate_cols),
                              raft::make_device_matrix_view<float, int64_t, raft::row_major>(
                                distances.data_handle() + query_offset * k, current_queries, k),
                              raft::make_device_matrix_view<IdxT, int64_t, raft::row_major>(
                                neighbors.data_handle() + query_offset * k, current_queries, k),
                              select_min,
                              true);
  }
}

}  // namespace cuvs::neighbors::detail
