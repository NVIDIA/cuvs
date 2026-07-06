/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kmeans.cuh"
#include "kmeans_impl.cuh"
#include <raft/core/logger.hpp>
#include <raft/core/resource/comms.hpp>
#include <raft/core/resource/multi_gpu.hpp>
#include <raft/core/resources.hpp>

namespace cuvs::cluster::kmeans {

#define INSTANTIATE_FIT(DataT, IndexT)                                          \
  template void fit<DataT, IndexT>(                                             \
    raft::resources const& handle,                                              \
    const kmeans::params& params,                                               \
    raft::device_matrix_view<const DataT, IndexT> X,                            \
    std::optional<raft::device_vector_view<const DataT, IndexT>> sample_weight, \
    raft::device_matrix_view<DataT, IndexT> centroids,                          \
    raft::host_scalar_view<DataT> inertia,                                      \
    raft::host_scalar_view<IndexT> n_iter);

INSTANTIATE_FIT(float, int)
INSTANTIATE_FIT(float, int64_t)

#undef INSTANTIATE_FIT

void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::device_matrix_view<const float, int> X,
         std::optional<raft::device_vector_view<const float, int>> sample_weight,
         raft::device_matrix_view<float, int> centroids,
         raft::host_scalar_view<float> inertia,
         raft::host_scalar_view<int> n_iter)
{
  cuvs::cluster::kmeans::fit<float, int>(
    handle, params, X, sample_weight, centroids, inertia, n_iter);
}

void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::device_matrix_view<const float, int64_t> X,
         std::optional<raft::device_vector_view<const float, int64_t>> sample_weight,
         raft::device_matrix_view<float, int64_t> centroids,
         raft::host_scalar_view<float> inertia,
         raft::host_scalar_view<int64_t> n_iter)
{
  cuvs::cluster::kmeans::fit<float, int64_t>(
    handle, params, X, sample_weight, centroids, inertia, n_iter);
}

void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::host_matrix_view<const float, int64_t> X,
         std::optional<raft::host_vector_view<const float, int64_t>> sample_weight,
         raft::device_matrix_view<float, int64_t> centroids,
         raft::host_scalar_view<float> inertia,
         raft::host_scalar_view<int64_t> n_iter)
{
  if (raft::resource::is_multi_gpu(handle) || raft::resource::comms_initialized(handle)) {
    RAFT_LOG_WARN(
      "Multi-GPU handle detected on single-GPU kmeans::fit() entry; "
      "falling back to single-GPU. Use cuvs::cluster::kmeans::mg::fit(...) for multi-GPU.");
  }
  cuvs::cluster::kmeans::detail::fit<float, int64_t>(
    handle, params, X, sample_weight, centroids, inertia, n_iter);
}

}  // namespace cuvs::cluster::kmeans
