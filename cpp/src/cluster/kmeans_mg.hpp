/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/core/export.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <optional>
#include <vector>

namespace CUVS_EXPORT cuvs {
namespace cluster {
namespace kmeans {
namespace mg {

/**
 * @brief MNMG/SNMG k-means fit.
 *
 * Each rank supplies its local data as one or more partitions. Callers that
 * already have a single mdspan per rank can pass a one-element vector.
 *
 * @param[in]     handle              The raft handle (must have NCCL comms or
 *                                    SNMG clique resources initialized).
 * @param[in]     params              K-means parameters.
 * @param[in]     X_parts             Per-partition local data on this rank.
 *                                    Each entry is [n_rows_i x n_features].
 * @param[in]     sample_weight_parts Optional per-partition row weights with
 *                                    one vector per data partition.
 * @param[inout]  centroids           Device matrix [n_clusters x n_features].
 *                                    On entry, used as the initial centers
 *                                    when params.init == InitMethod::Array.
 *                                    On return, holds the converged centroids.
 * @param[out]    inertia             Host scalar receiving the final
 *                                    clustering cost.
 * @param[out]    n_iter              Host scalar receiving the iteration
 *                                    count at which the run terminated.
 */
void fit(
  raft::resources const& handle,
  const cuvs::cluster::kmeans::params& params,
  const std::vector<raft::device_matrix_view<const float, int>>& X_parts,
  const std::optional<std::vector<raft::device_vector_view<const float, int>>>& sample_weight_parts,
  raft::device_matrix_view<float, int> centroids,
  raft::host_scalar_view<float> inertia,
  raft::host_scalar_view<int> n_iter);

void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         const std::vector<raft::device_matrix_view<const float, int64_t>>& X_parts,
         const std::optional<std::vector<raft::device_vector_view<const float, int64_t>>>&
           sample_weight_parts,
         raft::device_matrix_view<float, int64_t> centroids,
         raft::host_scalar_view<float> inertia,
         raft::host_scalar_view<int64_t> n_iter);

void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         const std::vector<raft::device_matrix_view<const double, int>>& X_parts,
         const std::optional<std::vector<raft::device_vector_view<const double, int>>>&
           sample_weight_parts,
         raft::device_matrix_view<double, int> centroids,
         raft::host_scalar_view<double> inertia,
         raft::host_scalar_view<int> n_iter);

void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         const std::vector<raft::device_matrix_view<const double, int64_t>>& X_parts,
         const std::optional<std::vector<raft::device_vector_view<const double, int64_t>>>&
           sample_weight_parts,
         raft::device_matrix_view<double, int64_t> centroids,
         raft::host_scalar_view<double> inertia,
         raft::host_scalar_view<int64_t> n_iter);

void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         const std::vector<raft::host_matrix_view<const float, int64_t>>& X_parts,
         const std::optional<std::vector<raft::host_vector_view<const float, int64_t>>>&
           sample_weight_parts,
         raft::device_matrix_view<float, int64_t> centroids,
         raft::host_scalar_view<float> inertia,
         raft::host_scalar_view<int64_t> n_iter);

void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         const std::vector<raft::host_matrix_view<const double, int64_t>>& X_parts,
         const std::optional<std::vector<raft::host_vector_view<const double, int64_t>>>&
           sample_weight_parts,
         raft::device_matrix_view<double, int64_t> centroids,
         raft::host_scalar_view<double> inertia,
         raft::host_scalar_view<int64_t> n_iter);

/**
 * @brief Single-mdspan convenience overloads. The mdspan is wrapped into a
 * one-element vector and routed through the vector-of-partitions overload
 * above. Host-mdspan variants also handle SNMG-vs-MNMG dispatch internally.
 */
void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::device_matrix_view<const float, int> X,
         std::optional<raft::device_vector_view<const float, int>> sample_weight,
         raft::device_matrix_view<float, int> centroids,
         raft::host_scalar_view<float> inertia,
         raft::host_scalar_view<int> n_iter);

void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::device_matrix_view<const float, int64_t> X,
         std::optional<raft::device_vector_view<const float, int64_t>> sample_weight,
         raft::device_matrix_view<float, int64_t> centroids,
         raft::host_scalar_view<float> inertia,
         raft::host_scalar_view<int64_t> n_iter);

void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::device_matrix_view<const double, int> X,
         std::optional<raft::device_vector_view<const double, int>> sample_weight,
         raft::device_matrix_view<double, int> centroids,
         raft::host_scalar_view<double> inertia,
         raft::host_scalar_view<int> n_iter);

void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::device_matrix_view<const double, int64_t> X,
         std::optional<raft::device_vector_view<const double, int64_t>> sample_weight,
         raft::device_matrix_view<double, int64_t> centroids,
         raft::host_scalar_view<double> inertia,
         raft::host_scalar_view<int64_t> n_iter);

void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::host_matrix_view<const float, int64_t> X,
         std::optional<raft::host_vector_view<const float, int64_t>> sample_weight,
         raft::device_matrix_view<float, int64_t> centroids,
         raft::host_scalar_view<float> inertia,
         raft::host_scalar_view<int64_t> n_iter);

void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::host_matrix_view<const double, int64_t> X,
         std::optional<raft::host_vector_view<const double, int64_t>> sample_weight,
         raft::device_matrix_view<double, int64_t> centroids,
         raft::host_scalar_view<double> inertia,
         raft::host_scalar_view<int64_t> n_iter);

}  // namespace mg
}  // namespace kmeans
}  // namespace cluster
}  // namespace CUVS_EXPORT cuvs
