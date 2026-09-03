/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kmeans.cuh"
#include "kmeans_impl.cuh"
#include <raft/core/resources.hpp>

namespace cuvs::cluster::kmeans {

#define INSTANTIATE_PREDICT(DataT, IndexT, LabelT)                              \
  template void predict<DataT, IndexT, LabelT>(                                 \
    raft::resources const& handle,                                              \
    const kmeans::params& params,                                               \
    raft::device_matrix_view<const DataT, IndexT> X,                            \
    std::optional<raft::device_vector_view<const DataT, IndexT>> sample_weight, \
    raft::device_matrix_view<const DataT, IndexT> centroids,                    \
    raft::device_vector_view<LabelT, IndexT> labels,                            \
    bool normalize_weight,                                                      \
    raft::host_scalar_view<DataT> inertia);

INSTANTIATE_PREDICT(float, int, int)
INSTANTIATE_PREDICT(float, int, int64_t)
INSTANTIATE_PREDICT(float, int64_t, int)
INSTANTIATE_PREDICT(float, int64_t, int64_t)

#undef INSTANTIATE_PREDICT

void predict(raft::resources const& handle,
             const kmeans::params& params,
             raft::device_matrix_view<const float, int> X,
             std::optional<raft::device_vector_view<const float, int>> sample_weight,
             raft::device_matrix_view<const float, int> centroids,
             raft::device_vector_view<int, int> labels,
             bool normalize_weight,
             raft::host_scalar_view<float> inertia)

{
  cuvs::cluster::kmeans::predict<float, int, int>(
    handle, params, X, sample_weight, centroids, labels, normalize_weight, inertia);
}

void predict(raft::resources const& handle,
             const kmeans::params& params,
             raft::device_matrix_view<const float, int> X,
             std::optional<raft::device_vector_view<const float, int>> sample_weight,
             raft::device_matrix_view<const float, int> centroids,
             raft::device_vector_view<int64_t, int> labels,
             bool normalize_weight,
             raft::host_scalar_view<float> inertia)
{
  cuvs::cluster::kmeans::predict<float, int, int64_t>(
    handle, params, X, sample_weight, centroids, labels, normalize_weight, inertia);
}

void predict(raft::resources const& handle,
             const kmeans::params& params,
             raft::device_matrix_view<const float, int64_t> X,
             std::optional<raft::device_vector_view<const float, int64_t>> sample_weight,
             raft::device_matrix_view<const float, int64_t> centroids,
             raft::device_vector_view<int, int64_t> labels,
             bool normalize_weight,
             raft::host_scalar_view<float> inertia)
{
  cuvs::cluster::kmeans::predict<float, int64_t, int>(
    handle, params, X, sample_weight, centroids, labels, normalize_weight, inertia);
}

void predict(raft::resources const& handle,
             const kmeans::params& params,
             raft::device_matrix_view<const float, int64_t> X,
             std::optional<raft::device_vector_view<const float, int64_t>> sample_weight,
             raft::device_matrix_view<const float, int64_t> centroids,
             raft::device_vector_view<int64_t, int64_t> labels,
             bool normalize_weight,
             raft::host_scalar_view<float> inertia)

{
  cuvs::cluster::kmeans::predict<float, int64_t, int64_t>(
    handle, params, X, sample_weight, centroids, labels, normalize_weight, inertia);
}
}  // namespace cuvs::cluster::kmeans
