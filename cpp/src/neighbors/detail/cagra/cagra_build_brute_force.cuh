/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/knn_merge_parts.hpp>
#include <cuvs/selection/select_k.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/linalg/map.cuh>

#include <algorithm>
#include <optional>
#include <type_traits>
#include <utility>

namespace cuvs::neighbors::cagra::detail {

// Choose equal query/reference chunks from the full free workspace. In addition to the
// two data chunks and results, budget norms and a conservative upper bound for the
// pairwise distances and intermediate top-k buffers used by brute-force search.
inline size_t brute_force_host_chunk_size(
  size_t free_bytes, size_t rows, size_t dim, size_t element_size, size_t top_k)
{
  // Two translation offsets for the direct partition merge.
  free_bytes -= std::min(free_bytes, 2 * sizeof(int64_t));
  auto const data_bytes_per_row = 2 * dim * element_size;
  // Per query, 7 = 1 top_k list for local results + 2 lists in each of the two
  // merge buffers + 2 lists reserved for selection/sorting scratch. The first
  // 5 lists are explicit allocations; the final 2 are a conservative allowance,
  // not an exact scratch requirement of the selection library.
  auto const result_bytes_per_row = 7 * top_k * (sizeof(int64_t) + sizeof(float));
  // Bound the query tile by the whole dataset (up to the search limit of 1024),
  // rather than the unknown chunk size. This may overestimate scratch, never underestimate it.
  auto const scratch_bytes_per_row =
    std::min(rows, size_t{1024}) * (sizeof(int64_t) + 2 * sizeof(float)) + 2 * sizeof(float);
  auto const bytes_per_row = data_bytes_per_row + result_bytes_per_row + scratch_bytes_per_row;
  return std::min(rows, free_bytes / bytes_per_row);
}

template <typename T, typename IdxT, typename Layout, typename Accessor>
void build_brute_force_host_knn_graph(
  raft::resources const& res,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, Layout, Accessor> dataset,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> knn_graph,
  graph_build_params::brute_force_params params)
{
  auto const rows       = static_cast<size_t>(dataset.extent(0));
  auto const dim        = dataset.extent(1);
  auto const degree     = static_cast<size_t>(knn_graph.extent(1));
  auto const top_k      = degree + 1;
  auto const chunk_size = brute_force_host_chunk_size(
    raft::resource::get_workspace_free_bytes(res), rows, dim, sizeof(T), top_k);
  RAFT_EXPECTS(chunk_size >= top_k,
               "CAGRA: brute-force host chunk size must be at least graph degree + 1; "
               "increase the available workspace");
  auto mr = raft::resource::get_large_workspace_resource_ref(res);
  auto queries =
    raft::make_device_mdarray<T>(res, mr, raft::make_extents<int64_t>(chunk_size, dim));
  auto references =
    raft::make_device_mdarray<T>(res, mr, raft::make_extents<int64_t>(chunk_size, dim));
  auto neighbors =
    raft::make_device_mdarray<int64_t>(res, mr, raft::make_extents<int64_t>(chunk_size, top_k));
  auto distances =
    raft::make_device_mdarray<float>(res, mr, raft::make_extents<int64_t>(chunk_size, top_k));
  auto candidate_neighbors =
    raft::make_device_mdarray<int64_t>(res, mr, raft::make_extents<int64_t>(chunk_size, 2 * top_k));
  auto candidate_distances =
    raft::make_device_mdarray<float>(res, mr, raft::make_extents<int64_t>(chunk_size, 2 * top_k));
  auto merged_neighbors =
    raft::make_device_mdarray<int64_t>(res, mr, raft::make_extents<int64_t>(chunk_size, 2 * top_k));
  auto merged_distances =
    raft::make_device_mdarray<float>(res, mr, raft::make_extents<int64_t>(chunk_size, 2 * top_k));
  auto translations   = raft::make_device_mdarray<int64_t>(res, mr, raft::make_extents<int64_t>(2));
  auto host_neighbors = raft::make_host_matrix<int64_t, int64_t>(chunk_size, top_k);
  bool const select_min = cuvs::distance::is_min_close(params.build_params.metric);

  bool const needs_packing = dataset.stride(1) != 1 || dataset.stride(0) != dim;
  std::optional<raft::host_matrix<T, int64_t>> packed_host_chunk;
  if (needs_packing) {
    packed_host_chunk.emplace(raft::make_host_matrix<T, int64_t>(chunk_size, dim));
  }
  auto copy_chunk = [&](size_t offset, size_t count, auto destination) {
    if (!needs_packing) {
      auto source = raft::make_host_matrix_view<const T, int64_t>(
        dataset.data_handle() + offset * dim, count, dim);
      raft::copy(res, destination, source);
    } else {
      // Pack on the host so arbitrary strides never require device staging storage.
      auto source =
        raft::make_host_matrix_view<T, int64_t>(packed_host_chunk->data_handle(), count, dim);
      for (size_t row = 0; row < count; ++row) {
        for (int64_t col = 0; col < dim; ++col) {
          source(row, col) = dataset(offset + row, col);
        }
      }
      raft::copy(res, destination, source);
      // The same packing buffer is reused by the next chunk transfer.
      raft::resource::sync_stream(res);
    }
  };

  for (size_t query_offset = 0; query_offset < rows; query_offset += chunk_size) {
    auto const query_rows = std::min(chunk_size, rows - query_offset);
    auto query_view =
      raft::make_device_matrix_view<T, int64_t>(queries.data_handle(), query_rows, dim);
    copy_chunk(query_offset, query_rows, query_view);
    size_t merged_k = 0;

    for (size_t reference_offset = 0; reference_offset < rows; reference_offset += chunk_size) {
      auto const reference_rows = std::min(chunk_size, rows - reference_offset);
      auto const local_k        = std::min(top_k, reference_rows);
      auto reference_view =
        raft::make_device_matrix_view<T, int64_t>(references.data_handle(), reference_rows, dim);
      copy_chunk(reference_offset, reference_rows, reference_view);
      auto index = cuvs::neighbors::brute_force::build(
        res, params.build_params, raft::make_const_mdspan(reference_view));
      // knn_merge_parts supports ascending distances and k <= 1024. Write the
      // local search directly after the accumulated list for its partitioned layout.
      bool const direct_merge =
        select_min && top_k <= 1024 && merged_k == top_k && local_k == top_k;
      auto* local_neighbors = direct_merge ? merged_neighbors.data_handle() + query_rows * top_k
                                           : neighbors.data_handle();
      auto* local_distances = direct_merge ? merged_distances.data_handle() + query_rows * top_k
                                           : distances.data_handle();
      auto neighbor_view =
        raft::make_device_matrix_view<int64_t, int64_t>(local_neighbors, query_rows, local_k);
      auto distance_view =
        raft::make_device_matrix_view<float, int64_t>(local_distances, query_rows, local_k);
      cuvs::neighbors::brute_force::search(res,
                                           params.search_params,
                                           index,
                                           raft::make_const_mdspan(query_view),
                                           neighbor_view,
                                           distance_view);
      if (direct_merge) {
        auto const offset = static_cast<int64_t>(reference_offset);
        raft::linalg::map_offset(res, translations.view(), [offset] __device__(int64_t part) {
          return part == 0 ? int64_t{0} : offset;
        });
        cuvs::neighbors::knn_merge_parts(res,
                                         raft::make_device_matrix_view<const float, int64_t>(
                                           merged_distances.data_handle(), 2 * query_rows, top_k),
                                         raft::make_device_matrix_view<const int64_t, int64_t>(
                                           merged_neighbors.data_handle(), 2 * query_rows, top_k),
                                         raft::make_device_matrix_view<float, int64_t>(
                                           candidate_distances.data_handle(), query_rows, top_k),
                                         raft::make_device_matrix_view<int64_t, int64_t>(
                                           candidate_neighbors.data_handle(), query_rows, top_k),
                                         translations.view());
        // Both buffers have capacity for two full lists; swapping never allocates.
        std::swap(merged_distances, candidate_distances);
        std::swap(merged_neighbors, candidate_neighbors);
        continue;
      }
      if (merged_k == 0) {
        // The first chunk already has global IDs (offset zero), so select directly.
        cuvs::selection::select_k(res,
                                  raft::make_const_mdspan(distance_view),
                                  std::optional{raft::make_const_mdspan(neighbor_view)},
                                  raft::make_device_matrix_view<float, int64_t>(
                                    merged_distances.data_handle(), query_rows, local_k),
                                  raft::make_device_matrix_view<int64_t, int64_t>(
                                    merged_neighbors.data_handle(), query_rows, local_k),
                                  select_min,
                                  true);
        merged_k = local_k;
        continue;
      }
      auto const candidate_k = merged_k + local_k;
      auto const next_k      = std::min(top_k, candidate_k);
      // Translate the local IDs and concatenate using existing RAFT operations.
      raft::linalg::map(res,
                        neighbor_view,
                        raft::add_const_op<int64_t>{static_cast<int64_t>(reference_offset)},
                        raft::make_const_mdspan(neighbor_view));
      auto append_candidates = [&](auto source, auto* destination, size_t column_offset) {
        using value_type = std::remove_pointer_t<decltype(destination)>;
        auto target      = raft::make_device_strided_matrix_view<value_type, int64_t>(
          destination + column_offset, query_rows, source.extent(1), candidate_k);
        raft::copy(res, target, source);
      };
      append_candidates(raft::make_device_matrix_view<const float, int64_t>(
                          merged_distances.data_handle(), query_rows, merged_k),
                        candidate_distances.data_handle(),
                        0);
      append_candidates(raft::make_device_matrix_view<const int64_t, int64_t>(
                          merged_neighbors.data_handle(), query_rows, merged_k),
                        candidate_neighbors.data_handle(),
                        0);
      append_candidates(
        raft::make_const_mdspan(distance_view), candidate_distances.data_handle(), merged_k);
      append_candidates(
        raft::make_const_mdspan(neighbor_view), candidate_neighbors.data_handle(), merged_k);
      auto candidate_distance_view = raft::make_device_matrix_view<const float, int64_t>(
        candidate_distances.data_handle(), query_rows, candidate_k);
      auto candidate_neighbor_view = raft::make_device_matrix_view<const int64_t, int64_t>(
        candidate_neighbors.data_handle(), query_rows, candidate_k);
      auto merged_distance_view = raft::make_device_matrix_view<float, int64_t>(
        merged_distances.data_handle(), query_rows, next_k);
      auto merged_neighbor_view = raft::make_device_matrix_view<int64_t, int64_t>(
        merged_neighbors.data_handle(), query_rows, next_k);
      cuvs::selection::select_k(res,
                                candidate_distance_view,
                                std::optional{candidate_neighbor_view},
                                merged_distance_view,
                                merged_neighbor_view,
                                select_min,
                                true);
      merged_k = next_k;
    }
    RAFT_EXPECTS(merged_k == top_k, "CAGRA: brute-force kNN did not return enough candidates");
    auto host_neighbor_view = raft::make_host_matrix_view<int64_t, int64_t>(
      host_neighbors.data_handle(), query_rows, top_k);
    auto merged_neighbor_view = raft::make_device_matrix_view<const int64_t, int64_t>(
      merged_neighbors.data_handle(), query_rows, top_k);
    raft::copy(res, host_neighbor_view, merged_neighbor_view);
    raft::resource::sync_stream(res);
    for (size_t row = 0; row < query_rows; ++row) {
      size_t output_rank = 0;
      for (size_t rank = 0; rank < top_k && output_rank < degree; ++rank) {
        auto const neighbor = host_neighbor_view(row, rank);
        RAFT_EXPECTS(neighbor >= 0 && static_cast<size_t>(neighbor) < rows,
                     "CAGRA: brute-force kNN returned an invalid neighbor");
        if (neighbor == static_cast<int64_t>(query_offset + row)) { continue; }
        knn_graph(query_offset + row, output_rank++) = static_cast<IdxT>(neighbor);
      }
      RAFT_EXPECTS(output_rank == degree,
                   "CAGRA: brute-force kNN did not return enough non-self neighbors");
    }
  }
}

// Exact kNN construction. Request one extra
// neighbor and remove self by ID, including when duplicate vectors tie with self.
template <typename T, typename IdxT, typename Layout, typename Accessor>
void build_knn_graph(raft::resources const& res,
                     raft::mdspan<const T, raft::matrix_extent<int64_t>, Layout, Accessor> dataset,
                     raft::host_matrix_view<IdxT, int64_t, raft::row_major> knn_graph,
                     graph_build_params::brute_force_params params)
{
  auto const partition_size = static_cast<size_t>(dataset.extent(0));
  auto const graph_degree   = static_cast<size_t>(knn_graph.extent(1));
  RAFT_EXPECTS(knn_graph.extent(0) == dataset.extent(0),
               "CAGRA: kNN graph and dataset must have the same number of rows");
  RAFT_EXPECTS(graph_degree > 0 && partition_size > graph_degree,
               "CAGRA: kNN graph degree must be positive and smaller than the dataset size");
  auto const metric = params.build_params.metric;
  RAFT_EXPECTS(metric == cuvs::distance::DistanceType::L2Expanded ||
                 metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                 metric == cuvs::distance::DistanceType::InnerProduct ||
                 metric == cuvs::distance::DistanceType::CosineExpanded,
               "CAGRA: brute-force kNN graph build supports only L2Expanded, L2SqrtExpanded, "
               "InnerProduct and CosineExpanded metrics");
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
    if constexpr (raft::is_host_mdspan_v<decltype(dataset)>) {
      build_brute_force_host_knn_graph(res, dataset, knn_graph, params);
    } else {
      // Borrow contiguous device data. Strided inputs need a contiguous device copy
      // for the brute-force index, which must stay alive across all query batches.
      std::optional<raft::device_matrix<T, int64_t>> device_dataset;
      auto dataset_view = [&]() {
        if constexpr (raft::is_device_mdspan_v<decltype(dataset)>) {
          if (dataset.stride(1) == 1 && dataset.stride(0) == dataset.extent(1)) {
            return raft::make_device_matrix_view<const T, int64_t>(
              dataset.data_handle(), dataset.extent(0), dataset.extent(1));
          }
        }
        device_dataset.emplace(
          raft::make_device_matrix<T, int64_t>(res, partition_size, dataset.extent(1)));
        raft::copy(res, device_dataset->view(), dataset);
        return raft::make_const_mdspan(device_dataset->view());
      }();
      auto brute_force_index =
        cuvs::neighbors::brute_force::build(res, params.build_params, dataset_view);

      // Size result buffers from the full remaining workspace after building the index.
      // Allocate them from the large workspace resource, as in the other graph builders.
      auto free_bytes     = raft::resource::get_workspace_free_bytes(res);
      auto const top_k    = graph_degree + 1;
      auto bytes_per_row  = top_k * (sizeof(int64_t) + sizeof(float));
      bool const fused_l2 = top_k <= 64 && (metric == cuvs::distance::DistanceType::L2Expanded ||
                                            metric == cuvs::distance::DistanceType::L2SqrtExpanded);
      // Fused L2 allocates norm workspace for both inputs; tiled search reuses index norms.
      if (fused_l2) { free_bytes -= std::min(free_bytes, partition_size * sizeof(float)); }
      if (metric != cuvs::distance::DistanceType::InnerProduct) { bytes_per_row += sizeof(float); }
      if (!fused_l2) {
        // Tiled search (including inner product for any k) also allocates pairwise
        // distances, intermediate top-k lists, and selection/sorting scratch.
        // Bound each tile/list by a full index row to avoid depending on tile sizing.
        bytes_per_row += partition_size * (2 * sizeof(int64_t) + 3 * sizeof(float)) +
                         2 * top_k * (sizeof(int64_t) + sizeof(float));
      }
      auto const max_batch_size = std::min(partition_size, free_bytes / bytes_per_row);
      RAFT_EXPECTS(max_batch_size > 0,
                   "CAGRA: insufficient workspace for a brute-force kNN query batch");
      auto workspace_mr     = raft::resource::get_large_workspace_resource_ref(res);
      auto device_neighbors = raft::make_device_mdarray<int64_t>(
        res, workspace_mr, raft::make_extents<int64_t>(max_batch_size, top_k));
      auto device_distances = raft::make_device_mdarray<float>(
        res, workspace_mr, raft::make_extents<int64_t>(max_batch_size, top_k));
      auto host_neighbors = raft::make_host_matrix<int64_t, int64_t>(max_batch_size, top_k);

      for (size_t offset = 0; offset < partition_size; offset += max_batch_size) {
        auto const batch_size = std::min(max_batch_size, partition_size - offset);
        auto queries          = raft::make_device_matrix_view<const T, int64_t>(
          dataset_view.data_handle() + offset * dataset.extent(1), batch_size, dataset.extent(1));
        auto neighbors = raft::make_device_matrix_view<int64_t, int64_t>(
          device_neighbors.data_handle(), batch_size, top_k);
        auto distances = raft::make_device_matrix_view<float, int64_t>(
          device_distances.data_handle(), batch_size, top_k);
        auto neighbors_host = raft::make_host_matrix_view<int64_t, int64_t>(
          host_neighbors.data_handle(), batch_size, top_k);
        cuvs::neighbors::brute_force::search(
          res, params.search_params, brute_force_index, queries, neighbors, distances);
        raft::copy(res, neighbors_host, neighbors);
        raft::resource::sync_stream(res);
        for (size_t row = 0; row < batch_size; ++row) {
          size_t output_rank = 0;
          for (size_t rank = 0; rank < top_k; ++rank) {
            auto const neighbor = neighbors_host(row, rank);
            if (neighbor == static_cast<int64_t>(offset + row)) { continue; }
            RAFT_EXPECTS(neighbor >= 0 && static_cast<size_t>(neighbor) < partition_size,
                         "CAGRA: brute-force kNN returned an invalid neighbor");
            knn_graph(offset + row, output_rank++) = static_cast<IdxT>(neighbor);
            if (output_rank == graph_degree) { break; }
          }
          RAFT_EXPECTS(output_rank == graph_degree,
                       "CAGRA: brute-force kNN did not return enough non-self neighbors");
        }
      }
    }
  } else {
    RAFT_FAIL("CAGRA: brute-force kNN build supports only float and half datasets");
  }
}

}  // namespace cuvs::neighbors::cagra::detail
