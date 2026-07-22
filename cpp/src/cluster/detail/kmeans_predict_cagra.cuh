/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Cluster assignment via CAGRA: assign each data point to nearest centroid using an
 * approximate nearest neighbor search (CAGRA) over the centroids instead of brute force.
 * Used for scaling IVF training when the number of clusters K is very large.
 *
 * Shared helpers (build index on centroids, 1-NN search -> labels) are used by:
 * - assign_nearest_centroid_cagra_with_index_reuse (k-means fit: optional index reuse on shifting
 *   centroids)
 * - ivf_pq extend / build post-fit predict (fixed centroids: build_cagra_index_for_centroids +
 *   assign_nearest_centroid_cagra)
 */
#pragma once

#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>
#include <raft/util/cudart_utils.hpp>

#include <algorithm>
#include <optional>

namespace cuvs::cluster::kmeans::detail {

/** Default search params for 1-NN centroid assignment (max_queries = auto). */
inline cuvs::neighbors::cagra::search_params default_cagra_centroid_search_params()
{
  cuvs::neighbors::cagra::search_params p;
  p.max_queries = 0;  // auto
  return p;
}

/**
 * @brief Build a CAGRA index on centroid vectors (shared by extend,
 * assign_nearest_centroid_cagra callers, and assign_nearest_centroid_cagra_with_index_reuse).
 */
inline cuvs::neighbors::cagra::index<float, uint32_t> build_cagra_index_for_centroids(
  raft::resources const& handle,
  cuvs::cluster::kmeans::balanced_params const& params,
  raft::device_matrix_view<const float, int64_t, raft::row_major> centroids)
{
  using namespace cuvs::neighbors::cagra;
  int64_t n_clusters  = centroids.extent(0);
  size_t graph_degree = std::min<size_t>(64, std::max<int64_t>(1, n_clusters - 1));
  size_t inter_degree = std::min<size_t>(128, std::max<int64_t>(1, n_clusters - 1));
  index_params build_params;
  build_params.metric                    = params.metric;
  build_params.graph_degree              = graph_degree;
  build_params.intermediate_graph_degree = inter_degree;
  build_params.attach_dataset_on_build   = true;
  return build(handle, build_params, centroids);
}

/**
 * @brief Assign each query to its nearest centroid using an existing CAGRA index built on
 * centroids. Writes cluster labels (and optional per-query distances). Queries must be float
 * row-major [n_queries, dim].
 *
 * Uses explicit row_major matrix view and raw label pointer so this compiles when
 * raft::make_device_*_view returns layout_c_contiguous mdspan (not assignable to the view types
 * required by cagra::search / device_vector_view).
 */
template <typename LabelT>
void assign_nearest_centroid_cagra(
  raft::resources const& handle,
  cuvs::neighbors::cagra::search_params const& search_params,
  cuvs::neighbors::cagra::index<float, uint32_t> const& cagra_index,
  raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
  LabelT* labels_out,
  int64_t n_labels,
  float* distances_out = nullptr)
{
  using namespace cuvs::neighbors::cagra;
  int64_t n_rows = queries.extent(0);
  RAFT_EXPECTS(n_labels == n_rows,
               "assign_nearest_centroid_cagra: labels length must match n_queries");
  auto neighbors = raft::make_device_matrix<uint32_t, int64_t>(handle, n_rows, 1);
  auto distances = raft::make_device_matrix<float, int64_t>(handle, n_rows, 1);
  search(handle, search_params, cagra_index, queries, neighbors.view(), distances.view());
  auto neighbors_col =
    raft::make_device_vector_view<const uint32_t, int64_t>(neighbors.data_handle(), n_rows);
  auto labels_view = raft::make_device_vector_view<LabelT, int64_t>(labels_out, n_rows);
  raft::linalg::map(
    handle, raft::make_const_mdspan(neighbors_col), labels_view, raft::cast_op<LabelT>());
  if (distances_out != nullptr) {
    raft::copy(
      handle,
      raft::make_device_vector_view<float, int64_t>(distances_out, n_rows),
      raft::make_device_vector_view<const float, int64_t>(distances.data_handle(), n_rows));
  }
}

/** Minimum number of clusters to use ANN for k-means fit (below this, brute is faster). */
constexpr uint32_t kMinClustersForAnnFit = 5000;

/**
 * @brief Assign each row to nearest centroid using CAGRA, reusing or rebuilding the index.
 *
 * When rebuild is true (or index is empty), builds the index on current centroids and stores it
 * in *index_opt. Otherwise skips build and searches with the existing index: centroid vectors in
 * memory may have shifted since that build (k-means M-step), so the graph still indexes a stale
 * snapshot — assignments are intentionally approximate between rebuilds.
 *
 * Used by k-means fit when use_ann_for_build_fit and n_clusters >= kMinClustersForAnnFit.
 */
template <typename IdxT, typename LabelT>
void assign_nearest_centroid_cagra_with_index_reuse(
  raft::resources const& handle,
  cuvs::cluster::kmeans::balanced_params const& params,
  const float* centers,
  IdxT n_clusters,
  IdxT dim,
  const float* dataset,
  IdxT n_rows,
  LabelT* labels,
  std::optional<cuvs::neighbors::cagra::index<float, uint32_t>>* index_opt,
  bool rebuild)
{
  RAFT_EXPECTS(
    centers != nullptr && dataset != nullptr && labels != nullptr && index_opt != nullptr,
    "assign_nearest_centroid_cagra_with_index_reuse: null argument");
  RAFT_EXPECTS(n_clusters >= 1 && dim >= 1 && n_rows >= 1,
               "assign_nearest_centroid_cagra_with_index_reuse: bad extents");

  raft::device_matrix_view<const float, int64_t, raft::row_major> centers_view(
    centers, static_cast<int64_t>(n_clusters), static_cast<int64_t>(dim));
  raft::device_matrix_view<const float, int64_t, raft::row_major> queries_view(
    dataset, static_cast<int64_t>(n_rows), static_cast<int64_t>(dim));

  if (rebuild || !index_opt->has_value()) {
    *index_opt = build_cagra_index_for_centroids(handle, params, centers_view);
  }

  assign_nearest_centroid_cagra(handle,
                                default_cagra_centroid_search_params(),
                                index_opt->value(),
                                queries_view,
                                labels,
                                static_cast<int64_t>(n_rows),
                                nullptr);
}

}  // namespace cuvs::cluster::kmeans::detail
