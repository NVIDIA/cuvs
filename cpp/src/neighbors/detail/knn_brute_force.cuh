/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/knn_merge_parts.hpp>
#include <cuvs/selection/select_k.hpp>

#include "../../distance/detail/distance_ops/l2_exp.cuh"
#include "./faiss_distance_utils.h"
#include "./fused_l2_knn.cuh"
#include "./haversine_distance.cuh"
#include "./knn_utils.cuh"

#include <raft/core/bitmap.cuh>
#include <raft/core/copy.cuh>
#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/transpose.cuh>
#include <raft/matrix/gather.cuh>
#include <raft/matrix/init.cuh>
#include <raft/sparse/convert/coo.cuh>
#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/linalg/masked_matmul.cuh>
#include <raft/sparse/matrix/select_k.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/popc.cuh>

#include <cuda/iterator>
#include <cuda_fp16.h>
#include <rmm/cuda_device.hpp>
#include <rmm/device_uvector.hpp>
#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <optional>
#include <set>
#include <variant>

namespace cuvs::neighbors::detail {

/**
 * Calculates brute force knn, using a fixed memory budget
 * by tiling over both the rows and columns of pairwise_distances
 */
template <typename ElementType      = float,
          typename IndexType        = int64_t,
          typename DistanceT        = float,
          typename DistanceEpilogue = raft::identity_op>
void tiled_brute_force_knn(const raft::resources& handle,
                           const ElementType* search,  // size (m ,d)
                           const ElementType* index,   // size (n ,d)
                           size_t m,
                           size_t n,
                           size_t d,
                           size_t k,
                           DistanceT* distances,  // size (m, k)
                           IndexType* indices,    // size (m, k)
                           cuvs::distance::DistanceType metric,
                           DistanceT metric_arg                      = 2.0,
                           size_t max_row_tile_size                  = 0,
                           size_t max_col_tile_size                  = 0,
                           const DistanceT* precomputed_index_norms  = nullptr,
                           const DistanceT* precomputed_search_norms = nullptr,
                           const uint32_t* filter_bits               = nullptr,
                           DistanceEpilogue distance_epilogue        = raft::identity_op(),
                           cuvs::neighbors::filtering::FilterType filter_type =
                             cuvs::neighbors::filtering::FilterType::Bitmap)
{
  // Figure out the number of rows/cols to tile for
  size_t tile_rows = 0;
  size_t tile_cols = 0;
  auto stream      = raft::resource::get_cuda_stream(handle);

  cuvs::neighbors::detail::faiss_select::chooseTileSize(
    m, n, d, sizeof(DistanceT), tile_rows, tile_cols);

  // for unittesting, its convenient to be able to put a max size on the tiles
  // so we can test the tiling logic without having to use huge inputs.
  if (max_row_tile_size && (tile_rows > max_row_tile_size)) { tile_rows = max_row_tile_size; }
  if (max_col_tile_size && (tile_cols > max_col_tile_size)) { tile_cols = max_col_tile_size; }

  // tile_cols must be at least k items
  tile_cols = std::max(tile_cols, k);

  // stores pairwise distances for the current tile
  rmm::device_uvector<DistanceT> temp_distances(tile_rows * tile_cols, stream);

  // calculate norms for L2 expanded distances - this lets us avoid calculating
  // norms repeatedly per-tile, and just do once for the entire input
  auto pairwise_metric = metric;
  rmm::device_uvector<DistanceT> search_norms(0, stream);
  rmm::device_uvector<DistanceT> index_norms(0, stream);
  if (metric == cuvs::distance::DistanceType::L2Expanded ||
      metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
      metric == cuvs::distance::DistanceType::CosineExpanded) {
    if (!precomputed_search_norms) { search_norms.resize(m, stream); }
    if (!precomputed_index_norms) { index_norms.resize(n, stream); }
    // cosine needs the l2norm, where as l2 distances needs the squared norm
    if (metric == cuvs::distance::DistanceType::CosineExpanded) {
      if (!precomputed_search_norms) {
        raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
          handle,
          raft::make_device_matrix_view<const ElementType, IndexType, raft::row_major>(
            search, m, d),
          raft::make_device_vector_view<DistanceT, IndexType>(search_norms.data(), m),
          raft::sqrt_op{});
      }
      if (!precomputed_index_norms) {
        raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
          handle,
          raft::make_device_matrix_view<const ElementType, IndexType, raft::row_major>(index, n, d),
          raft::make_device_vector_view<DistanceT, IndexType>(index_norms.data(), n),
          raft::sqrt_op{});
      }
    } else {
      if (!precomputed_search_norms) {
        raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
          handle,
          raft::make_device_matrix_view<const ElementType, IndexType, raft::row_major>(
            search, m, d),
          raft::make_device_vector_view<DistanceT, IndexType>(search_norms.data(), m));
      }
      if (!precomputed_index_norms) {
        raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
          handle,
          raft::make_device_matrix_view<const ElementType, IndexType, raft::row_major>(index, n, d),
          raft::make_device_vector_view<DistanceT, IndexType>(index_norms.data(), n));
      }
    }
    pairwise_metric = cuvs::distance::DistanceType::InnerProduct;
  }

  // if we're tiling over columns, we need additional buffers for temporary output
  // distances/indices
  size_t num_col_tiles = raft::ceildiv(n, tile_cols);
  size_t temp_out_cols = k * num_col_tiles;

  // the final column tile could have less than 'k' items in it
  // in which case the number of columns here is too high in the temp output.
  // adjust if necessary
  auto last_col_tile_size = n % tile_cols;
  if (last_col_tile_size && (last_col_tile_size < k)) { temp_out_cols -= k - last_col_tile_size; }

  // if we have less than k items in the index, we should fill out the result
  // to indicate that we are missing items (and match behaviour in faiss)
  if (n < k) {
    raft::matrix::fill(handle,
                       raft::make_device_matrix_view(distances, m, k),
                       std::numeric_limits<DistanceT>::lowest());

    if constexpr (std::is_signed_v<IndexType>) {
      raft::matrix::fill(handle, raft::make_device_matrix_view(indices, m, k), IndexType{-1});
    }
  }

  rmm::device_uvector<DistanceT> temp_out_distances(tile_rows * temp_out_cols, stream);
  rmm::device_uvector<IndexType> temp_out_indices(tile_rows * temp_out_cols, stream);

  bool select_min = cuvs::distance::is_min_close(metric);

  for (size_t i = 0; i < m; i += tile_rows) {
    size_t current_query_size = std::min(tile_rows, m - i);

    for (size_t j = 0; j < n; j += tile_cols) {
      size_t current_centroid_size = std::min(tile_cols, n - j);
      size_t current_k             = std::min(current_centroid_size, k);

      // calculate the top-k elements for the current tile, by calculating the
      // full pairwise distance for the tile - and then selecting the top-k from that
      cuvs::distance::pairwise_distance(
        handle,
        raft::make_device_matrix_view<const ElementType, int64_t>(
          search + i * d, current_query_size, d),
        raft::make_device_matrix_view<const ElementType, int64_t>(
          index + j * d, current_centroid_size, d),
        raft::make_device_matrix_view<DistanceT, int64_t>(
          temp_distances.data(), current_query_size, current_centroid_size),
        pairwise_metric,
        metric_arg);

      if (metric == cuvs::distance::DistanceType::L2Expanded ||
          metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
        auto row_norms = precomputed_search_norms ? precomputed_search_norms : search_norms.data();
        auto col_norms = precomputed_index_norms ? precomputed_index_norms : index_norms.data();
        auto dist      = temp_distances.data();
        bool sqrt      = metric == cuvs::distance::DistanceType::L2SqrtExpanded;

        raft::linalg::map_offset(
          handle,
          raft::make_device_vector_view(dist, current_query_size * current_centroid_size),
          [=] __device__(IndexType idx) {
            IndexType row = i + (idx / current_centroid_size);
            IndexType col = j + (idx % current_centroid_size);

            cuvs::distance::detail::ops::l2_exp_cutlass_op<DistanceT, DistanceT> l2_op(sqrt);
            auto val = l2_op(row_norms[row], col_norms[col], dist[idx]);
            return distance_epilogue(val, row, col);
          });
      } else if (metric == cuvs::distance::DistanceType::CosineExpanded) {
        auto row_norms = precomputed_search_norms ? precomputed_search_norms : search_norms.data();
        auto col_norms = precomputed_index_norms ? precomputed_index_norms : index_norms.data();
        auto dist      = temp_distances.data();

        raft::linalg::map_offset(
          handle,
          raft::make_device_vector_view(dist, current_query_size * current_centroid_size),
          [=] __device__(IndexType idx) {
            IndexType row = i + (idx / current_centroid_size);
            IndexType col = j + (idx % current_centroid_size);
            auto val      = DistanceT(1.0) - dist[idx] / DistanceT(row_norms[row] * col_norms[col]);
            return distance_epilogue(val, row, col);
          });
      } else {
        // if we're not l2 distance, and we have a distance epilogue - run it now
        if constexpr (!std::is_same_v<DistanceEpilogue, raft::identity_op>) {
          auto distances_ptr = temp_distances.data();
          raft::linalg::map_offset(
            handle,
            raft::make_device_vector_view(temp_distances.data(),
                                          current_query_size * current_centroid_size),
            [=] __device__(size_t idx) {
              IndexType row = i + (idx / current_centroid_size);
              IndexType col = j + (idx % current_centroid_size);
              return distance_epilogue(distances_ptr[idx], row, col);
            });
        }
      }

      auto distances_ptr        = temp_distances.data();
      auto count                = cuda::make_counting_iterator<IndexType>(0);
      DistanceT masked_distance = select_min ? std::numeric_limits<DistanceT>::infinity()
                                             : std::numeric_limits<DistanceT>::lowest();

      if (filter_bits != nullptr) {
        size_t n_cols = filter_type == cuvs::neighbors::filtering::FilterType::Bitmap ? n : 0;
        thrust::for_each(raft::resource::get_thrust_policy(handle),
                         count,
                         count + current_query_size * current_centroid_size,
                         [=] __device__(IndexType idx) {
                           IndexType row      = i + (idx / current_centroid_size);
                           IndexType col      = j + (idx % current_centroid_size);
                           IndexType g_idx    = row * n_cols + col;
                           IndexType item_idx = (g_idx) >> 5;
                           uint32_t bit_idx   = (g_idx) & 31;
                           uint32_t filter    = filter_bits[item_idx];
                           if ((filter & (uint32_t(1) << bit_idx)) == 0) {
                             distances_ptr[idx] = masked_distance;
                           }
                         });
      }

      std::optional<raft::device_matrix_view<const IndexType, int64_t, raft::row_major>>
        null_optional = std::nullopt;
      cuvs::selection::select_k(
        handle,
        raft::make_device_matrix_view<const DistanceT, int64_t, raft::row_major>(
          temp_distances.data(), current_query_size, current_centroid_size),
        null_optional,
        raft::make_device_matrix_view<DistanceT, int64_t, raft::row_major>(
          distances + i * k, current_query_size, current_k),
        raft::make_device_matrix_view<IndexType, int64_t, raft::row_major>(
          indices + i * k, current_query_size, current_k),
        select_min,
        true);

      // if we're tiling over columns, we need to do a couple things to fix up
      // the output of select_k
      // 1. The column id's in the output are relative to the tile, so we need
      // to adjust the column ids by adding the column the tile starts at (j)
      // 2. select_k writes out output in a row-major format, which means we
      // can't just concat the output of all the tiles and do a select_k on the
      // concatenation.
      // Fix both of these problems in a single pass here
      if (tile_cols != n) {
        const DistanceT* in_distances = distances + i * k;
        const IndexType* in_indices   = indices + i * k;
        DistanceT* out_distances      = temp_out_distances.data();
        IndexType* out_indices        = temp_out_indices.data();

        auto count = cuda::make_counting_iterator<IndexType>(0);
        thrust::for_each(raft::resource::get_thrust_policy(handle),
                         count,
                         count + current_query_size * current_k,
                         [=] __device__(IndexType i) {
                           IndexType row = i / current_k, col = i % current_k;
                           IndexType out_index = row * temp_out_cols + j * k / tile_cols + col;

                           out_distances[out_index] = in_distances[i];
                           out_indices[out_index]   = in_indices[i] + j;
                         });
      }
    }

    if (tile_cols != n) {
      // select the actual top-k items here from the temporary output
      cuvs::selection::select_k(
        handle,
        raft::make_device_matrix_view<const DistanceT, int64_t, raft::row_major>(
          temp_out_distances.data(), current_query_size, temp_out_cols),
        raft::make_device_matrix_view<const IndexType, int64_t, raft::row_major>(
          temp_out_indices.data(), current_query_size, temp_out_cols),
        raft::make_device_matrix_view<DistanceT, int64_t, raft::row_major>(
          distances + i * k, current_query_size, k),
        raft::make_device_matrix_view<IndexType, int64_t, raft::row_major>(
          indices + i * k, current_query_size, k),
        select_min,
        true);
    }
  }
}

/**
 * Search the kNN for the k-nearest neighbors of a set of query vectors
 * @param[in] input vector of device device memory array pointers to search
 * @param[in] sizes vector of memory sizes for each device array pointer in input
 * @param[in] D number of cols in input and search_items
 * @param[in] search_items set of vectors to query for neighbors
 * @param[in] n        number of items in search_items
 * @param[out] res_I    pointer to device memory for returning k nearest indices
 * @param[out] res_D    pointer to device memory for returning k nearest distances
 * @param[in] k        number of neighbors to query
 * @param[in] userStream the main cuda stream to use
 * @param[in] internalStreams optional when n_params > 0, the index partitions can be
 *        queried in parallel using these streams. Note that n_int_streams also
 *        has to be > 0 for these to be used and their cardinality does not need
 *        to correspond to n_parts.
 * @param[in] n_int_streams size of internalStreams. When this is <= 0, only the
 *        user stream will be used.
 * @param[in] rowMajorIndex are the index arrays in row-major layout?
 * @param[in] rowMajorQuery are the query array in row-major layout?
 * @param[in] translations translation ids for indices when index rows represent
 *        non-contiguous partitions
 * @param[in] metric corresponds to the cuvs::distance::DistanceType enum (default is L2Expanded)
 * @param[in] metricArg metric argument to use. Corresponds to the p arg for lp norm
 */
template <typename IntType  = int,
          typename IdxType  = std::int64_t,
          typename value_t  = float,
          typename DistType = float>
void brute_force_knn_impl(
  raft::resources const& handle,
  std::vector<value_t*>& input,
  std::vector<IntType>& sizes,
  IntType D,
  value_t* search_items,
  IntType n,
  IdxType* res_I,
  DistType* res_D,
  IntType k,
  bool rowMajorIndex                  = true,
  bool rowMajorQuery                  = true,
  std::vector<IdxType>* translations  = nullptr,
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded,
  DistType metricArg                  = 0,
  std::vector<DistType*>* input_norms = nullptr,
  const DistType* search_norms        = nullptr)
{
  auto userStream = raft::resource::get_cuda_stream(handle);

  ASSERT(input.size() == sizes.size(), "input and sizes vectors should be the same size");

  std::vector<IdxType> id_ranges;
  if (translations != nullptr) {
    // use the given translations
    id_ranges.insert(id_ranges.end(), translations->begin(), translations->end());
  } else if (input.size() > 1) {
    // If we don't have explicit translations
    // for offsets of the indices, build them
    // from the local partitions
    IdxType total_n = 0;
    for (size_t i = 0; i < input.size(); i++) {
      id_ranges.push_back(total_n);
      total_n += sizes[i];
    }
  }

  rmm::device_uvector<IdxType> trans(0, userStream);
  if (id_ranges.size() > 0) {
    trans.resize(id_ranges.size(), userStream);
    raft::copy(handle,
               raft::make_device_vector_view(trans.data(), id_ranges.size()),
               raft::make_host_vector_view(id_ranges.data(), id_ranges.size()));
  }

  rmm::device_uvector<DistType> all_D(0, userStream);
  rmm::device_uvector<IdxType> all_I(0, userStream);

  DistType* out_D = res_D;
  IdxType* out_I  = res_I;

  if (input.size() > 1) {
    all_D.resize(input.size() * k * n, userStream);
    all_I.resize(input.size() * k * n, userStream);

    out_D = all_D.data();
    out_I = all_I.data();
  }

  // currently we don't support col_major inside tiled_brute_force_knn, because
  // of limitations of the pairwise_distance API:
  // 1) paiwise_distance takes a single 'isRowMajor' parameter - and we have
  // multiple options here (like rowMajorQuery/rowMajorIndex)
  // 2) because of tiling, we need to be able to set a custom stride in the PW
  // api, which isn't supported
  // Instead, transpose the input matrices if they are passed as col-major.
  auto search = search_items;
  rmm::device_uvector<value_t> search_row_major(0, userStream);
  if (!rowMajorQuery) {
    search_row_major.resize(n * D, userStream);
    raft::linalg::transpose(handle, search, search_row_major.data(), n, D, userStream);
    search = search_row_major.data();
  }

  // transpose into a temporary buffer if necessary
  rmm::device_uvector<value_t> index_row_major(0, userStream);
  if (!rowMajorIndex) {
    size_t total_size = 0;
    for (auto size : sizes) {
      total_size += size;
    }
    index_row_major.resize(total_size * D, userStream);
  }

  // Make other streams from pool wait on main stream
  raft::resource::wait_stream_pool_on_stream(handle);

  size_t total_rows_processed = 0;
  for (size_t i = 0; i < input.size(); i++) {
    DistType* out_d_ptr = out_D + (i * k * n);
    IdxType* out_i_ptr  = out_I + (i * k * n);

    auto stream = raft::resource::get_next_usable_stream(handle, i);

    if (k <= 64 && rowMajorQuery == rowMajorIndex && rowMajorQuery == true &&
        (metric == cuvs::distance::DistanceType::L2Unexpanded ||
         metric == cuvs::distance::DistanceType::L2SqrtUnexpanded ||
         metric == cuvs::distance::DistanceType::L2Expanded ||
         metric == cuvs::distance::DistanceType::L2SqrtExpanded)) {
      fusedL2Knn(D,
                 out_i_ptr,
                 out_d_ptr,
                 input[i],
                 search_items,
                 sizes[i],
                 n,
                 k,
                 rowMajorIndex,
                 rowMajorQuery,
                 stream,
                 metric,
                 input_norms ? (*input_norms)[i] : nullptr,
                 search_norms);

      // Perform necessary post-processing
      if (metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
          metric == cuvs::distance::DistanceType::L2SqrtUnexpanded ||
          metric == cuvs::distance::DistanceType::LpUnexpanded) {
        DistType p = 0.5;  // standard l2
        if (metric == cuvs::distance::DistanceType::LpUnexpanded) p = 1.0 / metricArg;
        raft::linalg::map(
          handle,
          raft::make_device_vector_view<DistType, int64_t>(res_D, n * k),
          [p] __device__(DistType input) { return powf(fabsf(input), p); },
          raft::make_const_mdspan(
            raft::make_device_vector_view<const DistType, int64_t>(res_D, n * k)));
      }
    } else {
      switch (metric) {
        case cuvs::distance::DistanceType::Haversine:
          ASSERT(D == 2,
                 "Haversine distance requires 2 dimensions "
                 "(latitude / longitude).");

          haversine_knn(out_i_ptr, out_d_ptr, input[i], search_items, sizes[i], n, k, stream);
          break;
        default:
          // Create a new handle with the current stream from the stream pool
          raft::resources stream_pool_handle(handle);
          raft::resource::set_cuda_stream(stream_pool_handle, stream);

          auto index = input[i];
          if (!rowMajorIndex) {
            index = index_row_major.data() + total_rows_processed * D;
            total_rows_processed += sizes[i];
            raft::linalg::transpose(handle, input[i], index, sizes[i], D, stream);
          }

          tiled_brute_force_knn<value_t, IdxType>(stream_pool_handle,
                                                  search,
                                                  index,
                                                  n,
                                                  sizes[i],
                                                  D,
                                                  k,
                                                  out_d_ptr,
                                                  out_i_ptr,
                                                  metric,
                                                  metricArg,
                                                  0,
                                                  0,
                                                  input_norms ? (*input_norms)[i] : nullptr,
                                                  search_norms);
          break;
      }
    }

    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  // Sync internal streams if used. We don't need to
  // sync the user stream because we'll already have
  // fully serial execution.
  raft::resource::sync_stream_pool(handle);

  if (input.size() > 1 || translations != nullptr) {
    // This is necessary for proper index translations. If there are
    // no translations or partitions to combine, it can be skipped.
    knn_merge_parts(
      handle,
      raft::make_device_matrix_view<const DistType, int64_t>(out_D, n, input.size() * k),
      raft::make_device_matrix_view<const IdxType, int64_t>(out_I, n, input.size() * k),
      raft::make_device_matrix_view<DistType, int64_t>(res_D, n, k),
      raft::make_device_matrix_view<IdxType, int64_t>(res_I, n, k),
      raft::make_device_vector_view<IdxType, int64_t>(trans.data(), input.size()));
  }
};

template <typename T,
          typename IdxT,
          typename DistanceT    = float,
          typename QueryLayoutT = raft::row_major>
void brute_force_search(
  raft::resources const& res,
  const cuvs::neighbors::brute_force::index<T, DistanceT>& idx,
  raft::device_matrix_view<const T, int64_t, QueryLayoutT> queries,
  raft::device_matrix_view<IdxT, int64_t, raft::row_major> neighbors,
  raft::device_matrix_view<DistanceT, int64_t, raft::row_major> distances,
  std::optional<raft::device_vector_view<const DistanceT, int64_t>> query_norms = std::nullopt)
{
  RAFT_EXPECTS(neighbors.extent(1) == distances.extent(1), "Value of k must match for outputs");
  RAFT_EXPECTS(idx.dataset().extent(1) == queries.extent(1),
               "Number of columns in queries must match brute force index");

  auto k = neighbors.extent(1);
  auto d = idx.dataset().extent(1);

  std::vector<T*> dataset    = {const_cast<T*>(idx.dataset().data_handle())};
  std::vector<int64_t> sizes = {idx.dataset().extent(0)};
  std::vector<DistanceT*> norms;
  if (idx.has_norms()) { norms.push_back(const_cast<DistanceT*>(idx.norms().data_handle())); }

  brute_force_knn_impl<int64_t, IdxT, T, DistanceT>(
    res,
    dataset,
    sizes,
    d,
    const_cast<T*>(queries.data_handle()),
    queries.extent(0),
    neighbors.data_handle(),
    distances.data_handle(),
    k,
    true,
    std::is_same_v<QueryLayoutT, raft::row_major>,
    nullptr,
    idx.metric(),
    idx.metric_arg(),
    norms.size() ? &norms : nullptr,
    query_norms ? query_norms->data_handle() : nullptr);
}

/**
 * @brief The three strategies available for a filtered brute-force search.
 *
 *  - `sddmm`:  evaluate only the passing (query, row) pairs via a masked matmul. Cost
 *              grows with the number of passing rows.
 *  - `gather`: compact the passing rows into a dense [n_pass x dim] matrix and search
 *              that. Cost grows with the passing fraction, over a fixed setup cost.
 *  - `dense`:  GEMM against the whole dataset, then mask. Cost is independent of the
 *              filter.
 */
enum class filtered_search_path { sddmm, gather, dense };

/**
 * @brief Pick the cheapest search path for a filtered brute-force query.
 *
 * The thresholds are empirical.
 *
 * @param[in] n_dataset   rows in the dataset
 * @param[in] dim         columns in the dataset
 * @param[in] selectivity fraction of (query, row) pairs the filter passes
 * @param[in] k           neighbors requested
 * @param[in] passing     rows passing the filter, if the filter passes the same rows for
 *                        every query. A bitmap passes a different set per query, so it has
 *                        no such count, and cannot use the gather path.
 */
inline filtered_search_path select_filtered_search_path(
  int64_t n_dataset, int64_t dim, double selectivity, int64_t k, std::optional<int64_t> passing)
{
  // SDDMM competes against a fixed setup cost, so it stops paying off at an absolute row
  // count rather than at a fraction of the dataset.
  constexpr int64_t kSddmmMaxPassingRows = 10000;
  constexpr double kSddmmMaxSelectivity  = 0.03;
  // Gathering trades memory traffic (proportional to dim) against a GEMM (proportional to
  // n_dataset), and cannot beat simply GEMMing everything once most rows pass.
  constexpr double kGatherSelectivityScale = 5.5;
  constexpr double kGatherMaxSelectivity   = 0.5;
  // Below this, the gather setup costs more than the entire dense search.
  constexpr double kGatherMinDataset = 200000.0;

  const auto n_dataset_d = static_cast<double>(n_dataset);

  // SDDMM has to beat whichever of the other two is available.
  const bool sddmm_beats_dense = selectivity < kSddmmMaxSelectivity;
  if (!passing) {
    return sddmm_beats_dense ? filtered_search_path::sddmm : filtered_search_path::dense;
  }
  const int64_t n_pass = *passing;
  if (n_pass < kSddmmMaxPassingRows && sddmm_beats_dense) { return filtered_search_path::sddmm; }

  // Fraction of the dense search left over after paying the gather setup.
  const double amortized = 1.0 - kGatherMinDataset / n_dataset_d;
  const double gather_max_selectivity =
    amortized *
    std::min(kGatherSelectivityScale / std::sqrt(static_cast<double>(dim)), kGatherMaxSelectivity);

  // n_pass > k keeps the compacted dataset big enough to hold k neighbors.
  if (amortized > 0.0 && selectivity < gather_max_selectivity && n_pass > k) {
    return filtered_search_path::gather;
  }
  return filtered_search_path::dense;
}

/**
 * @brief Filtered search over the rows a bitset selects, by compacting them first.
 *
 * Bitset only: a bitmap selects different rows per query, so there is no single set of
 * rows to compact.
 */
template <typename T, typename IdxT, typename BitsT, typename DistanceT = float>
void brute_force_search_gathered(
  raft::resources const& res,
  const cuvs::neighbors::brute_force::index<T, DistanceT>& idx,
  raft::device_matrix_view<const T, IdxT, raft::row_major> queries,
  const cuvs::core::bitset_view<BitsT, IdxT>& filter_view,
  IdxT n_pass,
  raft::device_matrix_view<IdxT, IdxT, raft::row_major> neighbors,
  raft::device_matrix_view<DistanceT, IdxT, raft::row_major> distances,
  std::optional<raft::device_vector_view<const DistanceT, IdxT>> query_norms = std::nullopt)
{
  auto stream    = raft::resource::get_cuda_stream(res);
  auto policy    = raft::resource::get_thrust_policy(res);
  IdxT n_queries = queries.extent(0);
  IdxT n_dataset = idx.dataset().extent(0);
  IdxT dim       = idx.dataset().extent(1);
  IdxT k         = neighbors.extent(1);

  // 1. enumerate the rows the filter keeps
  rmm::device_uvector<IdxT> passing(n_pass, stream);
  thrust::copy_if(policy,
                  thrust::make_counting_iterator<IdxT>(0),
                  thrust::make_counting_iterator<IdxT>(n_dataset),
                  passing.data(),
                  [filter_view] __device__(IdxT i) { return filter_view.test(i); });

  // 2. compact those rows, and their norms, into a dense dataset
  rmm::device_uvector<T> gathered(static_cast<size_t>(n_pass) * dim, stream);
  raft::matrix::gather(
    res,
    raft::make_device_matrix_view<const T, IdxT, raft::row_major>(
      idx.dataset().data_handle(), n_dataset, dim),
    raft::make_device_vector_view<const IdxT, IdxT>(passing.data(), n_pass),
    raft::make_device_matrix_view<T, IdxT, raft::row_major>(gathered.data(), n_pass, dim));

  rmm::device_uvector<DistanceT> gathered_norms(idx.has_norms() ? n_pass : 0, stream);
  if (idx.has_norms()) {
    thrust::gather(policy,
                   passing.data(),
                   passing.data() + n_pass,
                   idx.norms().data_handle(),
                   gathered_norms.data());
  }

  // 3. ordinary unfiltered search over the compacted dataset
  rmm::device_uvector<IdxT> compact_neighbors(static_cast<size_t>(n_queries) * k, stream);
  std::vector<T*> dataset    = {gathered.data()};
  std::vector<int64_t> sizes = {static_cast<int64_t>(n_pass)};
  std::vector<DistanceT*> norms;
  if (idx.has_norms()) { norms.push_back(gathered_norms.data()); }

  brute_force_knn_impl<int64_t, IdxT, T, DistanceT>(
    res,
    dataset,
    sizes,
    dim,
    const_cast<T*>(queries.data_handle()),
    n_queries,
    compact_neighbors.data(),
    distances.data_handle(),
    k,
    true,
    true,
    nullptr,
    idx.metric(),
    idx.metric_arg(),
    norms.size() ? &norms : nullptr,
    query_norms ? query_norms->data_handle() : nullptr);

  // 4. translate compacted row numbers back into the original dataset's numbering
  thrust::gather(policy,
                 compact_neighbors.data(),
                 compact_neighbors.data() + static_cast<size_t>(n_queries) * k,
                 passing.data(),
                 neighbors.data_handle());
}

template <typename T, typename IdxT, typename BitsT, typename DistanceT = float>
void brute_force_search_filtered(
  raft::resources const& res,
  const cuvs::neighbors::brute_force::index<T, DistanceT>& idx,
  raft::device_matrix_view<const T, IdxT, raft::row_major> queries,
  const cuvs::neighbors::filtering::base_filter* filter,
  raft::device_matrix_view<IdxT, IdxT, raft::row_major> neighbors,
  raft::device_matrix_view<DistanceT, IdxT, raft::row_major> distances,
  std::optional<raft::device_vector_view<const DistanceT, IdxT>> query_norms = std::nullopt)
{
  auto metric = idx.metric();

  RAFT_EXPECTS(neighbors.extent(1) == distances.extent(1), "Value of k must match for outputs");
  RAFT_EXPECTS(idx.dataset().extent(1) == queries.extent(1),
               "Number of columns in queries must match brute force index");
  RAFT_EXPECTS(metric == cuvs::distance::DistanceType::InnerProduct ||
                 metric == cuvs::distance::DistanceType::L2Expanded ||
                 metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                 metric == cuvs::distance::DistanceType::CosineExpanded,
               "Only Euclidean, IP, and Cosine are supported!");

  RAFT_EXPECTS(idx.has_norms() || !(metric == cuvs::distance::DistanceType::L2Expanded ||
                                    metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                                    metric == cuvs::distance::DistanceType::CosineExpanded),
               "Index must has norms when using Euclidean, IP, and Cosine!");

  IdxT n_queries                                     = queries.extent(0);
  IdxT n_dataset                                     = idx.dataset().extent(0);
  IdxT dim                                           = idx.dataset().extent(1);
  IdxT k                                             = neighbors.extent(1);
  cuvs::neighbors::filtering::FilterType filter_type = filter->get_filter_type();

  auto stream = raft::resource::get_cuda_stream(res);

  std::optional<std::variant<const cuvs::core::bitmap_view<BitsT, IdxT>,
                             const cuvs::core::bitset_view<BitsT, IdxT>>>
    filter_view;

  IdxT nnz_h = 0;
  // A bitset passes the same rows for every query, so it has a row count; a bitmap passes a
  // different set per query and has none.
  std::optional<IdxT> n_pass;

  const BitsT* filter_data = nullptr;

  if (filter_type == cuvs::neighbors::filtering::FilterType::Bitmap) {
    auto actual_filter =
      dynamic_cast<const cuvs::neighbors::filtering::bitmap_filter<BitsT, int64_t>*>(filter);
    filter_view.emplace(actual_filter->view());
    nnz_h = actual_filter->view().count(res);
  } else if (filter_type == cuvs::neighbors::filtering::FilterType::Bitset) {
    auto actual_filter =
      dynamic_cast<const cuvs::neighbors::filtering::bitset_filter<BitsT, int64_t>*>(filter);
    filter_view.emplace(actual_filter->view());
    n_pass = actual_filter->view().count(res);
    nnz_h  = n_queries * (*n_pass);
  } else {
    RAFT_FAIL("Unsupported sample filter type");
  }

  std::visit([&](const auto& actual_view) { filter_data = actual_view.data(); }, *filter_view);

  const double selectivity =
    static_cast<double>(nnz_h) / (static_cast<double>(n_queries) * static_cast<double>(n_dataset));
  const auto path = select_filtered_search_path(n_dataset, dim, selectivity, k, n_pass);

  if (path == filtered_search_path::gather) {
    auto bitset_view = std::get<const cuvs::core::bitset_view<BitsT, IdxT>>(*filter_view);
    brute_force_search_gathered<T, IdxT, BitsT, DistanceT>(
      res, idx, queries, bitset_view, *n_pass, neighbors, distances, query_norms);
  } else if (path == filtered_search_path::dense) {
    raft::resources stream_pool_handle(res);
    raft::resource::set_cuda_stream(stream_pool_handle, stream);
    auto idx_norm = idx.has_norms() ? const_cast<DistanceT*>(idx.norms().data_handle()) : nullptr;

    tiled_brute_force_knn<T, IdxT, DistanceT>(stream_pool_handle,
                                              queries.data_handle(),
                                              idx.dataset().data_handle(),
                                              n_queries,
                                              n_dataset,
                                              dim,
                                              k,
                                              distances.data_handle(),
                                              neighbors.data_handle(),
                                              metric,
                                              DistanceT{2.0},
                                              0,
                                              0,
                                              idx_norm,
                                              nullptr,
                                              filter_data,
                                              raft::identity_op(),
                                              filter_type);
  } else {
    auto csr = raft::make_device_csr_matrix<DistanceT, IdxT>(res, n_queries, n_dataset, nnz_h);
    std::visit([&](const auto& actual_view) { actual_view.to_csr(res, csr); }, *filter_view);

    // create filter csr view
    auto compressed_csr_view = csr.structure_view();
    rmm::device_uvector<IdxT> rows(compressed_csr_view.get_nnz(), stream);
    raft::sparse::convert::csr_to_coo(compressed_csr_view.get_indptr().data(),
                                      compressed_csr_view.get_n_rows(),
                                      rows.data(),
                                      compressed_csr_view.get_nnz(),
                                      stream);
    auto dataset_view = raft::make_device_matrix_view<const T, IdxT, raft::row_major>(
      idx.dataset().data_handle(), n_dataset, dim);

    auto csr_view = raft::make_device_csr_matrix_view<DistanceT, IdxT, IdxT, IdxT>(
      csr.get_elements().data(), compressed_csr_view);

    std::visit(
      [&](const auto& actual_view) {
        raft::sparse::linalg::masked_matmul(res, queries, dataset_view, actual_view, csr_view);
      },
      *filter_view);

    // post process
    std::optional<raft::device_vector<DistanceT, IdxT>> query_norms_;
    if (metric == cuvs::distance::DistanceType::L2Expanded ||
        metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
        metric == cuvs::distance::DistanceType::CosineExpanded) {
      if (metric == cuvs::distance::DistanceType::CosineExpanded) {
        if (!query_norms) {
          query_norms_ = raft::make_device_vector<DistanceT, IdxT>(res, n_queries);
          raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
            res,
            raft::make_device_matrix_view<const T, IdxT, raft::row_major>(
              queries.data_handle(), n_queries, dim),
            query_norms_->view(),
            raft::sqrt_op{});
        }
      } else {
        if (!query_norms) {
          query_norms_ = raft::make_device_vector<DistanceT, IdxT>(res, n_queries);
          raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
            res,
            raft::make_device_matrix_view<const T, IdxT, raft::row_major>(
              queries.data_handle(), n_queries, dim),
            query_norms_->view());
        }
      }
      cuvs::neighbors::detail::epilogue_on_csr(
        res,
        csr.get_elements().data(),
        compressed_csr_view.get_nnz(),
        rows.data(),
        compressed_csr_view.get_indices().data(),
        query_norms ? query_norms->data_handle() : query_norms_->data_handle(),
        idx.norms().data_handle(),
        metric);
    }

    // select k
    auto const_csr_view = raft::make_device_csr_matrix_view<const DistanceT, IdxT, IdxT, IdxT>(
      csr.get_elements().data(), compressed_csr_view);
    std::optional<raft::device_vector_view<const IdxT, IdxT>> no_opt = std::nullopt;
    bool select_min = cuvs::distance::is_min_close(metric);
    raft::sparse::matrix::select_k(
      res, const_csr_view, no_opt, distances, neighbors, select_min, true);
  }

  return;
}

template <typename T, typename IdxT, typename DistT, typename LayoutT>
void search(raft::resources const& res,
            const cuvs::neighbors::brute_force::index<T, DistT>& idx,
            raft::device_matrix_view<const T, int64_t, LayoutT> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<DistT, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter_ref)
{
  try {
    auto& sample_filter =
      dynamic_cast<const cuvs::neighbors::filtering::none_sample_filter&>(sample_filter_ref);
    return brute_force_search<T, int64_t, DistT>(res, idx, queries, neighbors, distances);
  } catch (const std::bad_cast&) {
  }
  if constexpr (std::is_same_v<LayoutT, raft::col_major>) {
    RAFT_FAIL("filtered search isn't available with col_major queries yet");
  } else {
    try {
      auto& sample_filter =
        dynamic_cast<const cuvs::neighbors::filtering::bitmap_filter<uint32_t, int64_t>&>(
          sample_filter_ref);
      return brute_force_search_filtered<T, int64_t, uint32_t, DistT>(
        res, idx, queries, &sample_filter, neighbors, distances);
    } catch (const std::bad_cast&) {
    }

    try {
      auto& sample_filter =
        dynamic_cast<const cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>&>(
          sample_filter_ref);
      return brute_force_search_filtered<T, int64_t, uint32_t, DistT>(
        res, idx, queries, &sample_filter, neighbors, distances);
    } catch (const std::bad_cast&) {
      RAFT_FAIL("Unsupported sample filter type");
    }
  }
}

template <typename T, typename DistT, typename AccessorT, typename LayoutT = raft::row_major>
cuvs::neighbors::brute_force::index<T, DistT> build(
  raft::resources const& res,
  mdspan<const T, matrix_extent<int64_t>, LayoutT, AccessorT> dataset,
  cuvs::distance::DistanceType metric,
  DistT metric_arg)
{
  // certain distance metrics can benefit by pre-calculating the norms for the index dataset
  // which lets us avoid calculating these at query time
  std::optional<raft::device_vector<DistT, int64_t>> norms;

  if (metric == cuvs::distance::DistanceType::L2Expanded ||
      metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
      metric == cuvs::distance::DistanceType::CosineExpanded) {
    auto dataset_storage = std::optional<device_matrix<T, int64_t, LayoutT>>{};
    auto dataset_view    = [&res, &dataset_storage, dataset]() {
      if constexpr (std::is_same_v<decltype(dataset),
                                      raft::device_matrix_view<const T, int64_t, row_major>>) {
        return dataset;
      } else {
        dataset_storage =
          make_device_matrix<T, int64_t, LayoutT>(res, dataset.extent(0), dataset.extent(1));
        raft::copy(res, dataset_storage->view(), dataset);
        return raft::make_const_mdspan(dataset_storage->view());
      }
    }();

    norms = raft::make_device_vector<DistT, int64_t>(res, dataset.extent(0));
    // cosine needs the l2norm, where as l2 distances needs the squared norm
    if (metric == cuvs::distance::DistanceType::CosineExpanded) {
      raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
        res, dataset_view, norms->view(), raft::sqrt_op{});
    } else {
      raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
        res, dataset_view, norms->view());
    }
  }

  return cuvs::neighbors::brute_force::index<T, DistT>(
    res, dataset, std::move(norms), metric, metric_arg);
}
}  // namespace cuvs::neighbors::detail
