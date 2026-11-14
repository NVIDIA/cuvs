/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/superkmeans.cuh"

#include <cuvs/cluster/kmeans.hpp>

namespace cuvs::cluster::kmeans {

void fit(const raft::resources& handle,
         cuvs::cluster::kmeans::superkmeans_params const& params,
         raft::device_matrix_view<const float, int64_t> X,
         raft::device_matrix_view<float, int64_t> centroids,
         std::optional<raft::host_scalar_view<float>> inertia)
{
  detail::superkmeans::fit<float>(handle, params, X, centroids, std::nullopt, inertia);
}

void fit(const raft::resources& handle,
         cuvs::cluster::kmeans::superkmeans_params const& params,
         raft::device_matrix_view<const half, int64_t> X,
         raft::device_matrix_view<float, int64_t> centroids,
         std::optional<raft::host_scalar_view<float>> inertia)
{
  detail::superkmeans::fit<half>(handle, params, X, centroids, std::nullopt, inertia);
}

void predict(const raft::resources& handle,
             cuvs::cluster::kmeans::superkmeans_params const& params,
             raft::device_matrix_view<const float, int64_t> X,
             raft::device_matrix_view<const float, int64_t> centroids,
             raft::device_vector_view<uint32_t, int64_t> labels)
{
  detail::superkmeans::predict<float>(handle, params, X, centroids, labels);
}

void predict(const raft::resources& handle,
             cuvs::cluster::kmeans::superkmeans_params const& params,
             raft::device_matrix_view<const half, int64_t> X,
             raft::device_matrix_view<const float, int64_t> centroids,
             raft::device_vector_view<uint32_t, int64_t> labels)
{
  detail::superkmeans::predict<half>(handle, params, X, centroids, labels);
}

void fit_predict(const raft::resources& handle,
                 cuvs::cluster::kmeans::superkmeans_params const& params,
                 raft::device_matrix_view<const float, int64_t> X,
                 raft::device_matrix_view<float, int64_t> centroids,
                 raft::device_vector_view<uint32_t, int64_t> labels,
                 std::optional<raft::host_scalar_view<float>> inertia)
{
  detail::superkmeans::fit<float>(handle, params, X, centroids, labels, inertia);
}

void fit_predict(const raft::resources& handle,
                 cuvs::cluster::kmeans::superkmeans_params const& params,
                 raft::device_matrix_view<const half, int64_t> X,
                 raft::device_matrix_view<float, int64_t> centroids,
                 raft::device_vector_view<uint32_t, int64_t> labels,
                 std::optional<raft::host_scalar_view<float>> inertia)
{
  detail::superkmeans::fit<half>(handle, params, X, centroids, labels, inertia);
}

}  // namespace cuvs::cluster::kmeans
