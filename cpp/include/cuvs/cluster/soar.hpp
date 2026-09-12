/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/export.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <cstdint>

namespace CUVS_EXPORT cuvs {
namespace cluster {
namespace soar {

/**
 * @defgroup soar_params SOAR hyperparameters
 * @{
 */

/**
 * Simple object to specify hyper-parameters for SOAR assignment.
 */
struct params {
  /**
   * Weight of the projection of the secondary residual onto the primary residual in the SOAR
   * loss. Larger values penalize secondary centroids whose residual is aligned with the primary
   * residual, favoring complementary assignments. `0` reduces the loss to plain squared distance,
   * which the primary centroid itself minimizes, so nothing is spilled. Default: 1.0.
   */
  float lambda = 1.0f;
};

/**
 * @}
 */

/**
 * @defgroup soar_predict SOAR assignment
 * @{
 */

/**
 * @brief Assign a secondary ("spilled") cluster to each row of the dataset.
 *
 * SOAR (Spilling with Orthogonality-Amplified Residuals) picks, for each vector, a second
 * centroid that complements the primary assignment instead of merely being the next-closest
 * one. It minimizes the loss of Theorem 3.1 of https://arxiv.org/abs/2404.00774: for a vector
 * `x` with primary residual `r = x - centroids[labels[i]]`,
 *
 *   `score(c) = ||x - c||^2 + lambda * (dot(r / ||r||, x - c))^2`
 *
 * and `soar_labels[i]` is the centroid minimizing that score. Indexing a vector under both its
 * primary and its secondary centroid improves recall for queries near a partition boundary.
 *
 * Only float32 data and uint32 labels are supported.
 *
 * The primary centroid is not excluded from the search, so `soar_labels[i] == labels[i]` is a
 * possible (and meaningful) result: it says that no other centroid is worth spilling to, which
 * is the common case for vectors in the interior of a cluster. Callers that treat SOAR as a
 * strictly second posting list should test for this case and skip those rows.
 *
 * Scratch memory scales as `n_rows * n_clusters * 4` bytes because scores against all centroids
 * are materialized at once and are not tiled. Process the dataset in row batches to bound the
 * peak device memory usage.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *   #include <cuvs/cluster/soar.hpp>
 *   using namespace cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::balanced_params kmeans_params;
 *   int64_t n_features = 15, n_clusters = 100;
 *   auto centroids = raft::make_device_matrix<float, int64_t>(handle, n_clusters, n_features);
 *
 *   // primary assignments, e.g. from balanced k-means
 *   kmeans::fit(handle,
 *               kmeans_params,
 *               dataset,
 *               centroids.view());
 *   ...
 *   auto labels = raft::make_device_vector<uint32_t, int64_t>(handle, dataset.extent(0));
 *
 *   kmeans::predict(handle,
 *                   kmeans_params,
 *                   dataset,
 *                   raft::make_const_mdspan(centroids.view()),
 *                   labels.view());
 *   ...
 *   // secondary assignments
 *   cuvs::cluster::soar::params soar_params;
 *   auto soar_labels = raft::make_device_vector<uint32_t, int64_t>(handle, dataset.extent(0));
 *
 *   soar::predict(handle,
 *                 soar_params,
 *                 dataset,
 *                 raft::make_const_mdspan(centroids.view()),
 *                 raft::make_const_mdspan(labels.view()),
 *                 soar_labels.view());
 *   // soar_labels now holds one secondary centroid id per row
 * @endcode
 *
 * @param[in]  handle       The raft handle.
 * @param[in]  params       Parameters for SOAR assignment.
 * @param[in]  dataset      The dataset. The data must be in row-major format.
 *                          [dim = n_rows x n_features]
 * @param[in]  centroids    Cluster centroids. The data must be in row-major format.
 *                          [dim = n_clusters x n_features]
 * @param[in]  labels       Index of the primary cluster each row belongs to, as produced by
 *                          k-means prediction. Every value must be in `[0, n_clusters)`.
 *                          [len = n_rows]
 * @param[out] soar_labels  Index of the secondary cluster each row is spilled to.
 *                          [len = n_rows]
 */
void predict(raft::resources const& handle,
             const soar::params& params,
             raft::device_matrix_view<const float, int64_t> dataset,
             raft::device_matrix_view<const float, int64_t> centroids,
             raft::device_vector_view<const uint32_t, int64_t> labels,
             raft::device_vector_view<uint32_t, int64_t> soar_labels);

/**
 * @}
 */

}  // namespace soar
}  // namespace cluster
}  // namespace CUVS_EXPORT cuvs
