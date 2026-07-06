/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "./detail/kmeans_mg.cuh"

#include <cuvs/cluster/kmeans.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/comms.hpp>
#include <raft/core/resource/multi_gpu.hpp>
#include <raft/core/resources.hpp>

namespace cuvs::cluster::kmeans::mg {

void fit(
  raft::resources const& handle,
  const cuvs::cluster::kmeans::params& params,
  const std::vector<raft::device_matrix_view<const float, int>>& X_parts,
  const std::optional<std::vector<raft::device_vector_view<const float, int>>>& sample_weight_parts,
  raft::device_matrix_view<float, int> centroids,
  raft::host_scalar_view<float> inertia,
  raft::host_scalar_view<int> n_iter)
{
  cuvs::cluster::kmeans::mg::detail::mnmg_fit<float, int>(
    handle, params, X_parts, sample_weight_parts, centroids, inertia, n_iter);
}

void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         const std::vector<raft::device_matrix_view<const float, int64_t>>& X_parts,
         const std::optional<std::vector<raft::device_vector_view<const float, int64_t>>>&
           sample_weight_parts,
         raft::device_matrix_view<float, int64_t> centroids,
         raft::host_scalar_view<float> inertia,
         raft::host_scalar_view<int64_t> n_iter)
{
  cuvs::cluster::kmeans::mg::detail::mnmg_fit<float, int64_t>(
    handle, params, X_parts, sample_weight_parts, centroids, inertia, n_iter);
}

void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         const std::vector<raft::host_matrix_view<const float, int64_t>>& X_parts,
         const std::optional<std::vector<raft::host_vector_view<const float, int64_t>>>&
           sample_weight_parts,
         raft::device_matrix_view<float, int64_t> centroids,
         raft::host_scalar_view<float> inertia,
         raft::host_scalar_view<int64_t> n_iter)
{
  cuvs::cluster::kmeans::mg::detail::mnmg_fit<float, int64_t>(
    handle, params, X_parts, sample_weight_parts, centroids, inertia, n_iter);
}

// Single-mdspan convenience overloads.
void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::device_matrix_view<const float, int> X,
         std::optional<raft::device_vector_view<const float, int>> sample_weight,
         raft::device_matrix_view<float, int> centroids,
         raft::host_scalar_view<float> inertia,
         raft::host_scalar_view<int> n_iter)
{
  std::vector<raft::device_matrix_view<const float, int>> X_parts{X};
  std::optional<std::vector<raft::device_vector_view<const float, int>>> sw_parts;
  if (sample_weight.has_value()) {
    sw_parts = std::vector<raft::device_vector_view<const float, int>>{*sample_weight};
  }
  cuvs::cluster::kmeans::mg::fit(handle, params, X_parts, sw_parts, centroids, inertia, n_iter);
}

void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::device_matrix_view<const float, int64_t> X,
         std::optional<raft::device_vector_view<const float, int64_t>> sample_weight,
         raft::device_matrix_view<float, int64_t> centroids,
         raft::host_scalar_view<float> inertia,
         raft::host_scalar_view<int64_t> n_iter)
{
  std::vector<raft::device_matrix_view<const float, int64_t>> X_parts{X};
  std::optional<std::vector<raft::device_vector_view<const float, int64_t>>> sw_parts;
  if (sample_weight.has_value()) {
    sw_parts = std::vector<raft::device_vector_view<const float, int64_t>>{*sample_weight};
  }
  cuvs::cluster::kmeans::mg::fit(handle, params, X_parts, sw_parts, centroids, inertia, n_iter);
}

void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::host_matrix_view<const float, int64_t> X,
         std::optional<raft::host_vector_view<const float, int64_t>> sample_weight,
         raft::device_matrix_view<float, int64_t> centroids,
         raft::host_scalar_view<float> inertia,
         raft::host_scalar_view<int64_t> n_iter)
{
  if (raft::resource::is_multi_gpu(handle)) {
    cuvs::cluster::kmeans::mg::detail::batched_fit_omp<float, int64_t>(
      handle, params, X, sample_weight, centroids, inertia, n_iter);
    return;
  }
  std::vector<raft::host_matrix_view<const float, int64_t>> X_parts{X};
  std::optional<std::vector<raft::host_vector_view<const float, int64_t>>> sw_parts;
  if (sample_weight.has_value()) {
    sw_parts = std::vector<raft::host_vector_view<const float, int64_t>>{*sample_weight};
  }
  cuvs::cluster::kmeans::mg::fit(handle, params, X_parts, sw_parts, centroids, inertia, n_iter);
}

}  // namespace cuvs::cluster::kmeans::mg
