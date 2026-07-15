/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/nn_descent.hpp>

#include <cuvs/core/export.hpp>
#include <variant>

namespace CUVS_EXPORT cuvs {
namespace neighbors {
namespace all_neighbors {
// For re-exporting into all_neighbors namespace
namespace graph_build_params = cuvs::neighbors::graph_build_params;
/**
 * @defgroup all_neighbors_cpp_params The all-neighbors algorithm parameters.
 * @{
 */

using GraphBuildParams = std::variant<graph_build_params::ivf_pq_params,
                                      graph_build_params::nn_descent_params,
                                      graph_build_params::brute_force_params>;

/**
 * @brief Parameters used to build an all-neighbors graph (find nearest neighbors for all the
 * training vectors).
 * For scalability, the all-neighbors graph construction algorithm partitions a set of training
 * vectors into overlapping clusters, computes a local knn graph on each cluster, and merges the
 * local graphs into a single global graph.
 * Device memory usage and accuracy can be configured by changing the `overlap_factor` and
 * `n_clusters`.
 * The algorithm used to build each local graph is also configurable.
 *
 */
struct all_neighbors_params {
  /** Parameters for knn graph building algorithm
   * Approximate nearest neighbors methods or a brute force approach are supported to build the knn
   * graph. Currently supported options are 'IVF-PQ', 'NN Descent', or 'Brute Force'. IVF-PQ is more
   * accurate, but slower compared to NN Descent. Note that 'Brute Force' can also be approximate if
   * n_clusters > 1.
   *
   * Set ivf_pq_params, nn_descent_params, or brute_force_params to select the graph build
   * algorithm and control their parameters.
   *
   * @code{.cpp}
   * all_neighbors::index_params params;
   * // 1. Choose IVF-PQ algorithm
   * params.graph_build_params = all_neighbors::graph_build_params::ivf_pq_params{};
   *
   * // 2. Choose NN Descent algorithm for kNN graph construction
   * params.graph_build_params = all_neighbors::graph_build_params::nn_descent_params{};
   *
   * // 3. Choose Brute Force algorithm for kNN graph construction
   * params.graph_build_params = all_neighbors::graph_build_params::brute_force_params{};
   *
   * @endcode
   */
  GraphBuildParams graph_build_params;

  /**
   * Number of nearest clusters each data point will be assigned to in the batching algorithm.
   * Start with `overlap_factor = 2` and gradually increase (2->3->4 ...) for better accuracy at the
   * cost of device memory usage.
   */
  size_t overlap_factor = 2;

  /**
   * Number of total clusters (aka batches) to split the data into. If set to 1, algorithm creates
   * an all-neighbors graph without batching.
   * Start with `n_clusters = 4` and increase (4 → 8 → 16...) for less device memory usage at the
   * cost of accuracy. This is independent from `overlap_factor` as long as `overlap_factor` <
   * `n_clusters`.
   *
   * The ratio of `overlap_factor / n_clusters` determines device memory usage.
   * Approximately `(overlap_factor / n_clusters) * num_rows_in_entire_data` number of rows will
   * be put on device memory at once.
   * E.g. between `(overlap_factor / n_clusters)` = 2/10 and 2/20, the latter will use less device
   * memory.
   *
   * Larger `overlap_factor` results in better accuracy of the final all-neighbors knn
   * graph. E.g. While using similar device memory, `(overlap_factor / n_clusters)` = 4/20
   * will have better accuracy than 2/10 at the cost of performance.
   *
   */
  size_t n_clusters = 1;  // defaults to not batching

  /** Metric used. */
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded;
};

/** @} */

/**
 * @defgroup all_neighbors_cpp_build The all-neighbors knn graph build
 * @{
 */

/**
 * @brief Builds an approximate all-neighbors knn graph  (find nearest neighbors for all the
 * training vectors)
 *
 * Usage example:
 * @code{.cpp}
 *  using namespace cuvs::neighbors;
 *  // use default index parameters
 *  all_neighbors::all_neighbors_params params;
 *  auto indices = raft::make_device_matrix<int64_t, int64_t>(handle, n_row, k);
 *  auto distances = raft::make_device_matrix<float, int64_t>(handle, n_row, k);
 *  all_neighbors::build(res, params, dataset, indices.view(), distances.view());
 * @endcode
 *
 * @param[in] handle raft::resources is an object managing resources
 * @param[in] params an instance of all_neighbors::all_neighbors_params that are parameters
 *               to build all-neighbors knn graph
 * @param[in] dataset raft::host_matrix_view input dataset expected to be located
 *                in host memory
 * @param[out] indices nearest neighbor indices of shape [n_row x k]
 * @param[out] distances nearest neighbor distances [n_row x k]
 * @param[out] core_distances array for core distances of size [n_row]. Requires distances matrix to
 * compute core_distances. If core_distances is given, the resulting indices and distances will be
 * mutual reachability space.
 * @param[in] alpha distance scaling parameter as used in robust single linkage.
 */
void build(
  const raft::resources& handle,
  const all_neighbors_params& params,
  raft::host_matrix_view<const float, int64_t, row_major> dataset,
  raft::device_matrix_view<int64_t, int64_t, row_major> indices,
  std::optional<raft::device_matrix_view<float, int64_t, row_major>> distances      = std::nullopt,
  std::optional<raft::device_vector_view<float, int64_t, row_major>> core_distances = std::nullopt,
  float alpha                                                                       = 1.0);

/**
 * @brief Builds an approximate all-neighbors knn graph (find nearest neighbors for all the training
 * vectors) params.n_clusters should be 1 for data on device. To use a larger params.n_clusters for
 * efficient device memory usage, put data on host RAM.
 *
 * Usage example:
 * @code{.cpp}
 *  using namespace cuvs::neighbors;
 *  // use default index parameters
 *  all_neighbors::all_neighbors_params params;
 *  auto indices = raft::make_device_matrix<int64_t, int64_t>(handle, n_row, k);
 *  auto distances = raft::make_device_matrix<float, int64_t>(handle, n_row, k);
 *  all_neighbors::build(res, params, dataset, indices.view(), distances.view());
 * @endcode
 *
 * @param[in] handle raft::resources is an object managing resources
 * @param[in] params an instance of all_neighbors::all_neighbors_params that are parameters
 *               to build all-neighbors knn graph
 * @param[in] dataset raft::device_matrix_view input dataset expected to be located
 *                in device memory
 * @param[out] indices nearest neighbor indices of shape [n_row x k]
 * @param[out] distances nearest neighbor distances [n_row x k]
 * @param[out] core_distances array for core distances of size [n_row]. Requires distances matrix to
 * compute core_distances. If core_distances is given, the resulting indices and distances will be
 * mutual reachability space.
 * @param[in] alpha distance scaling parameter as used in robust single linkage.
 */
void build(
  const raft::resources& handle,
  const all_neighbors_params& params,
  raft::device_matrix_view<const float, int64_t, row_major> dataset,
  raft::device_matrix_view<int64_t, int64_t, row_major> indices,
  std::optional<raft::device_matrix_view<float, int64_t, row_major>> distances      = std::nullopt,
  std::optional<raft::device_vector_view<float, int64_t, row_major>> core_distances = std::nullopt,
  float alpha                                                                       = 1.0);

/** @} */
}  // namespace all_neighbors
}  // namespace neighbors
}  // namespace CUVS_EXPORT cuvs
