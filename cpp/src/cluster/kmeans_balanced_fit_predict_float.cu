/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// clang-format off
#include "kmeans_balanced_impl_fit_predict.cuh"
#include "../neighbors/detail/ann_utils.cuh"
#include <raft/core/resources.hpp>
// clang-format on

namespace cuvs::cluster::kmeans {

void fit_predict(const raft::resources& handle,
                 cuvs::cluster::kmeans::balanced_params const& params,
                 raft::device_matrix_view<const float, int64_t> X,
                 raft::device_matrix_view<float, int64_t> centroids,
                 raft::device_vector_view<uint32_t, int64_t> labels,
                 std::optional<raft::host_scalar_view<float>> inertia,
                 std::optional<raft::device_matrix_view<float, int64_t>> parent_centroids,
                 std::optional<raft::device_vector_view<uint32_t, int64_t>> centroid_to_parent)
{
  cuvs::cluster::kmeans_balanced::fit_predict(
    handle, params, X, centroids, labels, inertia, parent_centroids, centroid_to_parent);
}
}  // namespace cuvs::cluster::kmeans
