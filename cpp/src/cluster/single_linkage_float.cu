/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/single_linkage.cuh"
#include <cuvs/cluster/agglomerative.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

namespace cuvs::cluster::agglomerative {

void single_linkage(raft::resources const& handle,
                    raft::device_matrix_view<const float, int, raft::row_major> X,
                    raft::device_matrix_view<int, int, raft::row_major> dendrogram,
                    raft::device_vector_view<int, int> labels,
                    cuvs::distance::DistanceType metric,
                    size_t n_clusters,
                    cuvs::cluster::agglomerative::Linkage linkage,
                    std::optional<int> c,
                    std::optional<raft::device_vector_view<float, int>> distances)
{
  if (linkage == Linkage::KNN_GRAPH) {
    detail::single_linkage<float, int, Linkage::KNN_GRAPH>(
      handle, X, dendrogram, labels, metric, n_clusters, c, distances);
  } else {
    detail::single_linkage<float, int, Linkage::PAIRWISE>(
      handle, X, dendrogram, labels, metric, n_clusters, c, distances);
  }
}

namespace helpers {
void build_linkage(
  raft::resources const& handle,
  raft::device_matrix_view<const float, int64_t, raft::row_major> X,
  std::variant<linkage_graph_params::distance_params,
               linkage_graph_params::mutual_reachability_params> linkage_graph_params,
  cuvs::distance::DistanceType metric,
  raft::device_coo_matrix_view<float, int64_t, int64_t, size_t> out_mst,
  raft::device_matrix_view<int64_t, int64_t> out_dendrogram,
  raft::device_vector_view<float, int64_t> out_distances,
  raft::device_vector_view<int64_t, int64_t> out_sizes,
  std::optional<raft::device_vector_view<float, int64_t>> core_dists)
{
  /**
   * Construct MST sorted by weights
   */
  if (std::holds_alternative<
        cuvs::cluster::agglomerative::helpers::linkage_graph_params::mutual_reachability_params>(
        linkage_graph_params)) {
    RAFT_EXPECTS(core_dists.has_value(),
                 "core distances must be pre-allocated to build the linkage with mutual "
                 "reachability distances");
    auto core_dists_mdspan = core_dists.value();
    RAFT_EXPECTS(core_dists_mdspan.extent(0) == X.extent(0),
                 "core_dists doesn't have expected size");
    auto mr_params = std::get<
      cuvs::cluster::agglomerative::helpers::linkage_graph_params::mutual_reachability_params>(
      linkage_graph_params);
    detail::build_mr_linkage<float, int64_t>(handle,
                                             X,
                                             mr_params.min_samples,
                                             mr_params.alpha,
                                             metric,
                                             core_dists_mdspan,
                                             out_mst,
                                             out_dendrogram,
                                             out_distances,
                                             out_sizes,
                                             mr_params.all_neighbors_params);
  } else {
    auto dist_params =
      std::get<cuvs::cluster::agglomerative::helpers::linkage_graph_params::distance_params>(
        linkage_graph_params);
    if (dist_params.dist_type == cuvs::cluster::agglomerative::Linkage::KNN_GRAPH) {
      detail::build_dist_linkage<float,
                                 int64_t,
                                 size_t,
                                 cuvs::cluster::agglomerative::Linkage::KNN_GRAPH>(
        handle, X, dist_params.c, metric, out_mst, out_dendrogram, out_distances, out_sizes);
    } else {
      detail::
        build_dist_linkage<float, int64_t, size_t, cuvs::cluster::agglomerative::Linkage::PAIRWISE>(
          handle, X, dist_params.c, metric, out_mst, out_dendrogram, out_distances, out_sizes);
    }
  }
}

void build_linkage(
  raft::resources const& handle,
  raft::host_matrix_view<const float, int64_t, raft::row_major> X,
  std::variant<linkage_graph_params::distance_params,
               linkage_graph_params::mutual_reachability_params> linkage_graph_params,
  cuvs::distance::DistanceType metric,
  raft::device_coo_matrix_view<float, int64_t, int64_t, size_t> out_mst,
  raft::device_matrix_view<int64_t, int64_t> out_dendrogram,
  raft::device_vector_view<float, int64_t> out_distances,
  raft::device_vector_view<int64_t, int64_t> out_sizes,
  std::optional<raft::device_vector_view<float, int64_t>> core_dists)
{
  /**
   * Construct MST sorted by weights
   */
  if (std::holds_alternative<
        cuvs::cluster::agglomerative::helpers::linkage_graph_params::mutual_reachability_params>(
        linkage_graph_params)) {
    RAFT_EXPECTS(core_dists.has_value(),
                 "core distances must be pre-allocated to build the linkage with mutual "
                 "reachability distances");
    auto core_dists_mdspan = core_dists.value();
    RAFT_EXPECTS(core_dists_mdspan.extent(0) == X.extent(0),
                 "core_dists doesn't have expected size");
    auto mr_params = std::get<
      cuvs::cluster::agglomerative::helpers::linkage_graph_params::mutual_reachability_params>(
      linkage_graph_params);
    detail::build_mr_linkage<float, int64_t>(handle,
                                             X,
                                             mr_params.min_samples,
                                             mr_params.alpha,
                                             metric,
                                             core_dists_mdspan,
                                             out_mst,
                                             out_dendrogram,
                                             out_distances,
                                             out_sizes,
                                             mr_params.all_neighbors_params);
  } else {
    RAFT_FAIL("data must be on device memory to build linkage with distance params");
  }
}
}  // namespace helpers
}  // namespace cuvs::cluster::agglomerative
