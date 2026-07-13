/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "./detail/kmeans_mg.cuh"

#include <cuvs/cluster/kmeans.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <optional>
#include <vector>

// Multi-GPU k-means fit overloads that take several local partitions per rank.
// The single-partition (single mdspan) overloads live with the rest of the
// public `fit` overloads in kmeans_fit_double.cu, which dispatch to the
// multi-GPU backend based on the resources attached to the handle.
namespace cuvs::cluster::kmeans {

void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         const std::vector<raft::device_matrix_view<const double, int>>& X_parts,
         const std::optional<std::vector<raft::device_vector_view<const double, int>>>&
           sample_weight_parts,
         raft::device_matrix_view<double, int> centroids,
         raft::host_scalar_view<double> inertia,
         raft::host_scalar_view<int> n_iter)
{
  cuvs::cluster::kmeans::mg::detail::mnmg_fit<double, int>(
    handle, params, X_parts, sample_weight_parts, centroids, inertia, n_iter);
}

void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         const std::vector<raft::device_matrix_view<const double, int64_t>>& X_parts,
         const std::optional<std::vector<raft::device_vector_view<const double, int64_t>>>&
           sample_weight_parts,
         raft::device_matrix_view<double, int64_t> centroids,
         raft::host_scalar_view<double> inertia,
         raft::host_scalar_view<int64_t> n_iter)
{
  cuvs::cluster::kmeans::mg::detail::mnmg_fit<double, int64_t>(
    handle, params, X_parts, sample_weight_parts, centroids, inertia, n_iter);
}

void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         const std::vector<raft::host_matrix_view<const double, int64_t>>& X_parts,
         const std::optional<std::vector<raft::host_vector_view<const double, int64_t>>>&
           sample_weight_parts,
         raft::device_matrix_view<double, int64_t> centroids,
         raft::host_scalar_view<double> inertia,
         raft::host_scalar_view<int64_t> n_iter)
{
  cuvs::cluster::kmeans::mg::detail::mnmg_fit<double, int64_t>(
    handle, params, X_parts, sample_weight_parts, centroids, inertia, n_iter);
}

}  // namespace cuvs::cluster::kmeans
