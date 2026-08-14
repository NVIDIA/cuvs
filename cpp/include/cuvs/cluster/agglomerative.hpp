/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/all_neighbors.hpp>
#include <optional>
#include <variant>

#include <cuvs/core/export.hpp>
#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

namespace CUVS_EXPORT cuvs {
namespace cluster {
namespace agglomerative {

// constant to indirectly control the number of neighbors. k = sqrt(n) + c. default to 15
constexpr int DEFAULT_CONST_C = 15;

/**
 * @defgroup agglomerative_params agglomerative clustering hyperparameters
 * @{
 */

/**
 * Determines the method for computing the minimum spanning tree (MST)
 */
enum Linkage {

  /**
   * Use a pairwise distance matrix as input to the mst. This
   * is very fast and the best option for fairly small datasets (~50k data points)
   */
  PAIRWISE = 0,

  /**
   * Construct a KNN graph as input to the mst and provide additional
   * edges if the mst does not converge. This is slower but scales
   * to very large datasets.
   */
  KNN_GRAPH = 1
};

/**
 * @}
 */

/**
 * Simple container object for consolidating linkage results. This closely
 * mirrors the trained instance variables populated in
 * Scikit-learn's AgglomerativeClustering estimator.
 * @tparam idx_t
 */
template <typename idx_t>
class single_linkage_output {
 public:
  idx_t m;
  idx_t n_clusters;

  idx_t n_leaves;
  idx_t n_connected_components;

  // TODO: These will be made private in a future release
  idx_t* labels;    // size: m
  idx_t* children;  // size: (m-1, 2)

  raft::device_vector_view<idx_t> get_labels()
  {
    return raft::make_device_vector_view<idx_t>(labels, m);
  }

  raft::device_matrix_view<idx_t> get_children()
  {
    return raft::make_device_matrix_view<idx_t>(children, m - 1, 2);
  }
};

/**
 * @defgroup single_linkage single-linkage clustering APIs
 * @{
 */
/**
 * Single-linkage clustering, capable of constructing a KNN graph to
 * scale the algorithm beyond the n^2 memory consumption of implementations
 * that use the fully-connected graph of pairwise distances by connecting
 * a knn graph when k is not large enough to connect it.

 * @param[in] handle raft handle
 * @param[in] X dense input matrix in row-major layout
 * @param[out] dendrogram output dendrogram (size [n_rows - 1] * 2)
 * @param[out] labels output labels vector (size n_rows)
 * @param[in] metric distance metric to use when constructing connectivities graph
 * @param[in] n_clusters number of clusters to assign data samples
 * @param[in] linkage strategy for constructing the linkage. PAIRWISE uses more memory but can be
 faster for
 *                    smaller datasets. KNN_GRAPH allows the memory usage to be controlled (using
 parameter c)
 *                    at the expense of potentially additional minimum spanning tree iterations.
 * @param[in] c a constant used when constructing linkage from knn graph. Allows the indirect
 control of k. The algorithm will set `k = log(n) + c`
 */
void single_linkage(
  raft::resources const& handle,
  raft::device_matrix_view<const float, int, raft::row_major> X,
  raft::device_matrix_view<int, int, raft::row_major> dendrogram,
  raft::device_vector_view<int, int> labels,
  cuvs::distance::DistanceType metric,
  size_t n_clusters,
  cuvs::cluster::agglomerative::Linkage linkage = cuvs::cluster::agglomerative::Linkage::KNN_GRAPH,
  std::optional<int> c                          = std::make_optional<int>(DEFAULT_CONST_C));

namespace helpers {

namespace linkage_graph_params {
/** Specialized parameters to build the KNN graph with regular distances */
struct distance_params {
  /** a constant used when constructing linkage from knn graph. Allows the indirect control of k.
   * The algorithm will set `k = log(n) + c` */
  int c = DEFAULT_CONST_C;

  /** strategy for constructing the linkage. PAIRWISE uses more memory but can be faster for smaller
   * datasets. KNN_GRAPH allows the memory usage to be controlled (using parameter c) */
  cuvs::cluster::agglomerative::Linkage dist_type =
    cuvs::cluster::agglomerative::Linkage::KNN_GRAPH;
};

/** Specialized parameters to build the Mutual Reachability graph */
struct mutual_reachability_params {
  /** this neighborhood will be selected for core distances. */
  int min_samples;

  /** weight applied when internal distance is chosen for mutual reachability (value of 1.0 disables
   * the weighting) */
  float alpha = 1.0;

  /**
   * Parameters for building the mutual reachability graph using an underlying KNN algorithm.
   *
   * The all-neighbors graph construction algorithm enables building the mutual reachability
   * graph on datasets larger than device memory by:
   *   1. Partitioning the dataset into overlapping clusters,
   *   2. Computing local KNN graphs within each cluster, and
   *   3. Merging the local graphs into a single global graph.
   *
   * Key fields:
   * - graph_build_params: Selects the KNN construction method (Brute Force or NN Descent)
   *   and controls algorithm-specific parameters.
   *
   * - n_clusters: Number of partitions (batches) to split the data into. Larger `n_clusters`
   *   reduces memory usage but may reduce accuracy if `overlap_factor` is too low.
   *   Recommended starting value: `n_clusters = 4`. Increase progressively (4 → 8 → 16 ...)
   *   to reduce memory usage at the cost of some accuracy. This is independent of `overlap_factor`
   *   as long as `overlap_factor < n_clusters`.
   *
   * - overlap_factor: Number of nearest clusters each data point is assigned to. Higher
   *   `overlap_factor` improves accuracy at the cost of memory and performance.
   *   Recommended starting value: `overlap_factor = 2`. Increase gradually (2 → 3 → 4 ...)
   *   for better accuracy with higher device memory usage.
   *
   * - metric: Distance metric to use when computing nearest neighbors.
   */
  cuvs::neighbors::all_neighbors::all_neighbors_params all_neighbors_params{
    cuvs::neighbors::graph_build_params::brute_force_params{}};
};
}  // namespace linkage_graph_params

/**
 * Given a dataset, builds the KNN graph, connects graph components and builds a linkage
 * (dendrogram). Returns the Minimum Spanning Tree edges sorted by weight and the dendrogram.
 * @param[in] handle raft handle for resource reuse
 * @param[in] X data points on device memory (size n_rows * d)
 * @param[in] linkage_graph_params
 *   Parameters controlling how the KNN graph is built. This can be either:
 *   - distance_params: standard distance-based KNN graph construction for traditional
 *     agglomerative clustering.
 *   - mutual_reachability_params: parameters to compute a mutual reachability graph for
 *     density-aware hierarchical clustering (e.g. HDBSCAN).
 * @param[in] metric distance metric to use
 * @param[out] out_mst output MST sorted by edge weights (size n_rows - 1)
 * @param[out] dendrogram output dendrogram (size [n_rows - 1] * 2)
 * @param[out] out_distances distances for output
 * @param[out] out_sizes cluster sizes of output
 * @param[out] core_dists (optional) core distances (size m). Must be supplied in the Mutual
 * Reachability space
 */
void build_linkage(
  raft::resources const& handle,
  raft::device_matrix_view<const float, int64_t, raft::row_major> X,
  std::variant<linkage_graph_params::distance_params,
               linkage_graph_params::mutual_reachability_params> linkage_graph_params,
  cuvs::distance::DistanceType metric,
  raft::device_coo_matrix_view<float, int64_t, int64_t, size_t> out_mst,
  raft::device_matrix_view<int64_t, int64_t> dendrogram,
  raft::device_vector_view<float, int64_t> out_distances,
  raft::device_vector_view<int64_t, int64_t> out_sizes,
  std::optional<raft::device_vector_view<float, int64_t>> core_dists);

/**
 * Given a dataset, builds the KNN graph, connects graph components and builds a linkage
 * (dendrogram). Returns the Minimum Spanning Tree edges sorted by weight and the dendrogram.
 * @param[in] handle raft handle for resource reuse
 * @param[in] X data points on host memory (size n_rows * d)
 * @param[in] linkage_graph_params
 *   Parameters controlling how the KNN graph is built. This can be either:
 *   - distance_params: standard distance-based KNN graph construction for traditional
 *     agglomerative clustering.
 *   - mutual_reachability_params: parameters to compute a mutual reachability graph for
 *     density-aware hierarchical clustering (e.g. HDBSCAN).
 * @param[in] metric distance metric to use
 * @param[out] out_mst output MST sorted by edge weights (size n_rows - 1)
 * @param[out] dendrogram output dendrogram (size [n_rows - 1] * 2)
 * @param[out] out_distances distances for output
 * @param[out] out_sizes cluster sizes of output
 * @param[out] core_dists (optional) core distances (size m). Must be supplied in the Mutual
 * Reachability space
 */
void build_linkage(
  raft::resources const& handle,
  raft::host_matrix_view<const float, int64_t, raft::row_major> X,
  std::variant<linkage_graph_params::distance_params,
               linkage_graph_params::mutual_reachability_params> linkage_graph_params,
  cuvs::distance::DistanceType metric,
  raft::device_coo_matrix_view<float, int64_t, int64_t, size_t> out_mst,
  raft::device_matrix_view<int64_t, int64_t> dendrogram,
  raft::device_vector_view<float, int64_t> out_distances,
  raft::device_vector_view<int64_t, int64_t> out_sizes,
  std::optional<raft::device_vector_view<float, int64_t>> core_dists);

/**
 * Build dendrogram from a Minimum Spanning Tree (MST).
 *
 * This function takes a sorted MST (represented as edges with source, destination, and weights)
 * and constructs a dendrogram (hierarchical clustering tree) on the host.
 *
 * @param[in] handle The raft resources handle
 * @param[in] rows Source nodes of the MST edges (device memory, size: nnz)
 * @param[in] cols Destination nodes of the MST edges (device memory, size: nnz)
 * @param[in] data Edge weights/distances of the MST (device memory, size: nnz)
 * @param[out] children Output dendrogram children array (device memory, size: nnz * 2)
 *                      Each pair of consecutive elements represents the two children
 *                      merged at each step of the hierarchy
 * @param[out] out_delta Output distances/heights at which clusters are merged (device memory, size:
 * nnz)
 * @param[out] out_size Output cluster sizes at each merge step (device memory, size: nnz)
 */
void build_dendrogram(raft::resources const& handle,
                      raft::device_vector_view<const int64_t, int64_t> rows,
                      raft::device_vector_view<const int64_t, int64_t> cols,
                      raft::device_vector_view<const float, int64_t> data,
                      raft::device_matrix_view<int64_t, int64_t, raft::row_major> children,
                      raft::device_vector_view<float, int64_t> out_delta,
                      raft::device_vector_view<int64_t, int64_t> out_size);

}  // namespace helpers
/**
 * @}
 */
}  // namespace agglomerative
}  // namespace cluster
}  // namespace CUVS_EXPORT cuvs
