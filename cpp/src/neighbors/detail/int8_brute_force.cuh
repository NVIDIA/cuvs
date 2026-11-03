/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/preprocessing/quantize/scalar.hpp>
#include <cuvs/selection/select_k.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/cublas_macros.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/map.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/reduce.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/integer_utils.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda/iterator>
#include <thrust/for_each.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>
#include <variant>

#include "./faiss_distance_utils.h"

namespace cuvs::neighbors::detail {

constexpr int kInt8BlockSize = 256;

template <typename T>
struct sq_affine_parameters {
  double scale;
  double shift;
  T const* shifts;
};

template <typename T>
sq_affine_parameters<T> get_sq_affine_parameters(
  cuvs::preprocessing::quantize::scalar::quantizer_const_view<T> quantizer, int64_t dim)
{
  using view_type = cuvs::preprocessing::quantize::scalar::quantizer_const_view<T>;
  sq_affine_parameters<T> affine{};
  if (auto const* range = std::get_if<typename view_type::range_type>(&quantizer.value)) {
    auto const lo = static_cast<double>(range->min_);
    auto const hi = static_cast<double>(range->max_);
    RAFT_EXPECTS(std::isfinite(lo) && std::isfinite(hi) && hi >= lo,
                 "SQ bounds must be finite with min <= max");
    constexpr double code_min = std::numeric_limits<int8_t>::min();
    constexpr double code_max = std::numeric_limits<int8_t>::max();
    affine.scale              = hi > lo ? (hi - lo) / (code_max - code_min) : 1.0;
    affine.shift              = lo - code_min * affine.scale;
  } else {
    auto const& shifts = std::get<typename view_type::shifts_type>(quantizer.value);
    RAFT_EXPECTS(shifts.shifts.extent(0) == dim, "SQ shifts must match the dataset dimension");
    affine.scale  = static_cast<double>(shifts.scale);
    affine.shifts = shifts.shifts.data_handle();
  }
  RAFT_EXPECTS(std::isfinite(affine.scale) && affine.scale > 0.0 && std::isfinite(affine.shift),
               "SQ scale must be positive and finite, and its shift must be finite");
  return affine;
}

template <typename T>
void compute_sq_statistics(raft::resources const& res,
                           raft::device_matrix_view<const int8_t, int64_t> codes,
                           sq_affine_parameters<T> affine,
                           cuvs::distance::DistanceType metric,
                           raft::device_vector_view<double, int64_t> norms,
                           raft::device_vector_view<double, int64_t> dot_corrections)
{
  auto const cosine = metric == cuvs::distance::DistanceType::CosineExpanded;
  if (cosine || metric == cuvs::distance::DistanceType::InnerProduct) {
    // For x = scale * code + shift, dot(x, y) = scale^2 * dot(code_x, code_y) + u_x + u_y.
    raft::linalg::reduce<raft::Apply::ALONG_ROWS>(
      res, codes, dot_corrections, 0.0, false, [affine] __device__(int8_t code, int64_t j) {
        auto const shift = affine.shifts ? static_cast<double>(affine.shifts[j]) : affine.shift;
        return shift * (affine.scale * code + 0.5 * shift);
      });
  }
  if (cosine) {
    raft::linalg::reduce<raft::Apply::ALONG_ROWS>(
      res,
      codes,
      norms,
      0.0,
      false,
      [affine] __device__(int8_t code, int64_t j) {
        auto const shift = affine.shifts ? static_cast<double>(affine.shifts[j]) : affine.shift;
        auto const value = fma(affine.scale, static_cast<double>(code), shift);
        return value * value;
      },
      raft::add_op{},
      [] __device__(double norm) { return sqrt(norm); });
  } else if (metric != cuvs::distance::DistanceType::InnerProduct) {
    raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(res, codes, norms);
  }
}

inline void int8_gemm(raft::resources const& res,
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
  try {
    raft::linalg::gemm(res,
                       true,
                       false,
                       static_cast<int>(n_rows),
                       static_cast<int>(n_queries),
                       static_cast<int>(dim),
                       &alpha,
                       dataset,
                       static_cast<int>(dim),
                       queries,
                       static_cast<int>(dim),
                       &beta,
                       scores,
                       static_cast<int>(leading_dim),
                       raft::resource::get_cuda_stream(res).get());
  } catch (raft::cublas_error const& error) {
    // Some cuBLASLt versions reject otherwise valid, irregular INT8 shapes. RAFT exposes
    // the cuBLAS status in the exception message; retry only unsupported products.
    if (std::string_view{error.what()}.find("CUBLAS_STATUS_NOT_SUPPORTED") ==
        std::string_view::npos) {
      throw;
    }
    if (n_queries > 16) {
      for (int64_t query = 0; query < n_queries; query += 16) {
        int8_gemm(res,
                  dataset,
                  queries + query * dim,
                  scores + query * leading_dim,
                  n_rows,
                  std::min<int64_t>(16, n_queries - query),
                  dim,
                  leading_dim);
      }
    } else if (n_rows >= 8) {
      auto const first_rows = n_rows / 8 * 4;
      int8_gemm(res, dataset, queries, scores, first_rows, n_queries, dim, leading_dim);
      int8_gemm(res,
                dataset + first_rows * dim,
                queries,
                scores + first_rows,
                n_rows - first_rows,
                n_queries,
                dim,
                leading_dim);
    } else {
      throw;
    }
  }
}

RAFT_KERNEL int8_postprocess_kernel(std::int32_t const* scores,
                                    float* output,
                                    int64_t n_rows,
                                    int64_t leading_dim,
                                    double const* query_norms,
                                    double const* dataset_norms,
                                    double const* query_corrections,
                                    double const* dataset_corrections,
                                    double scale,
                                    cuvs::distance::DistanceType metric)
{
  auto const query       = static_cast<int64_t>(blockIdx.y);
  auto const* row_scores = scores + query * leading_dim;
  auto* row_output       = output + query * n_rows;
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; row < n_rows;
       row += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    auto const integer_dot = row_scores[row];
    auto const dot         = static_cast<double>(integer_dot);
    switch (metric) {
      case cuvs::distance::DistanceType::InnerProduct:
        row_output[row] = static_cast<float>(scale * (scale * dot) + query_corrections[query] +
                                             dataset_corrections[row]);
        break;
      case cuvs::distance::DistanceType::CosineExpanded: {
        auto const denominator = query_norms[query] * dataset_norms[row];
        auto const corrected_dot =
          scale * (scale * dot) + query_corrections[query] + dataset_corrections[row];
        row_output[row] =
          denominator > 0.0 ? static_cast<float>(1.0 - corrected_dot / denominator) : 1.0f;
        break;
      }
      default: {
        // Cancel the exact integer terms before converting the distance to float for selection.
        auto const distance = static_cast<int64_t>(query_norms[query]) +
                              static_cast<int64_t>(dataset_norms[row]) -
                              2 * static_cast<int64_t>(integer_dot);
        row_output[row] = static_cast<float>(distance);
        break;
      }
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

template <typename T, typename IdxT>
void int8_brute_force_search(raft::resources const& res,
                             cuvs::neighbors::brute_force::device_sq_index<T> const& index,
                             raft::device_matrix_view<const T, int64_t> queries,
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

  auto const stream    = raft::resource::get_cuda_stream(res);
  auto const dataset   = index.dataset();
  auto const quantizer = dataset.const_manifest_view();
  auto const affine    = get_sq_affine_parameters(quantizer, d);
  auto queries_storage = raft::make_device_matrix<std::int8_t, int64_t>(res, m, d);
  cuvs::preprocessing::quantize::scalar::transform(res, quantizer, queries, queries_storage.view());
  auto const* queries_ptr = queries_storage.data_handle();

  auto const cosine            = index.metric() == cuvs::distance::DistanceType::CosineExpanded;
  auto const needs_norms       = index.metric() != cuvs::distance::DistanceType::InnerProduct;
  auto const needs_corrections = cosine || !needs_norms;
  rmm::device_uvector<double> query_norms(needs_norms ? m : 0, stream);
  rmm::device_uvector<double> query_corrections(needs_corrections ? m : 0, stream);
  compute_sq_statistics(
    res,
    raft::make_const_mdspan(queries_storage.view()),
    affine,
    index.metric(),
    raft::make_device_vector_view<double, int64_t>(query_norms.data(), query_norms.size()),
    raft::make_device_vector_view<double, int64_t>(query_corrections.data(),
                                                   query_corrections.size()));

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
  rmm::device_uvector<float> compact_scores(0, stream);
  rmm::device_uvector<float> candidate_distances(tile_rows * candidate_cols, stream);
  rmm::device_uvector<IdxT> candidate_indices(tile_rows * candidate_cols, stream);
  rmm::device_uvector<std::int8_t> padded_dataset(0, stream);

  auto const select_min = cuvs::distance::is_min_close(index.metric());

  for (int64_t query_offset = 0; query_offset < m; query_offset += tile_rows) {
    auto const current_queries = std::min<int64_t>(tile_rows, m - query_offset);
    auto* distances_ptr        = distances.data_handle() + query_offset * k;
    auto* indices_ptr          = neighbors.data_handle() + query_offset * k;
    int64_t candidate_offset   = 0;
    for (int64_t dataset_offset = 0; dataset_offset < n; dataset_offset += tile_cols) {
      auto const current_rows = std::min<int64_t>(tile_cols, n - dataset_offset);
      auto const gemm_rows    = raft::round_up_safe(current_rows, int64_t{4});
      auto const* dataset_ptr = dataset.data_const_view().data_handle() + dataset_offset * d;
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

      int8_gemm(res,
                dataset_ptr,
                queries_ptr + query_offset * d,
                scores.data(),
                gemm_rows,
                current_queries,
                d,
                tile_cols);

      auto* output_scores = reinterpret_cast<float*>(scores.data());
      if (current_rows != static_cast<int64_t>(tile_cols)) {
        compact_scores.resize(static_cast<size_t>(current_queries * current_rows), stream);
        output_scores = compact_scores.data();
      }
      auto const query_blocks = static_cast<unsigned>(current_queries);
      // Keep the block budget independent of the query tile size; each block strides within a row.
      auto const column_blocks = std::min(int8_grid_size(current_rows), 65535u / query_blocks);
      dim3 const grid(column_blocks, query_blocks);
      int8_postprocess_kernel<<<grid, kInt8BlockSize, 0, stream.get()>>>(
        scores.data(),
        output_scores,
        current_rows,
        tile_cols,
        needs_norms ? query_norms.data() + query_offset : nullptr,
        needs_norms ? index.norms().data_handle() + dataset_offset : nullptr,
        needs_corrections ? query_corrections.data() + query_offset : nullptr,
        needs_corrections ? index.dot_corrections().data_handle() + dataset_offset : nullptr,
        affine.scale,
        index.metric());
      RAFT_CUDA_TRY(cudaPeekAtLastError());

      auto const current_k = std::min<int64_t>(k, current_rows);
      cuvs::selection::select_k(
        res,
        raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
          output_scores, current_queries, current_rows),
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
    auto const scale     = affine.scale;
    bool const take_sqrt = index.metric() == cuvs::distance::DistanceType::L2SqrtExpanded ||
                           index.metric() == cuvs::distance::DistanceType::L2SqrtUnexpanded;
    raft::linalg::map(
      res,
      values,
      [take_sqrt, scale] __device__(float score) {
        auto const distance = scale * (scale * static_cast<double>(score));
        return static_cast<float>(take_sqrt ? sqrt(distance) : distance);
      },
      raft::make_const_mdspan(values));
  }
}

}  // namespace cuvs::neighbors::detail
