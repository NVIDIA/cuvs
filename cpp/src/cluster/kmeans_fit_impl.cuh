/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "kmeans.cuh"

#ifdef CUVS_BUILD_MG_ALGOS
#include "detail/kmeans_mg.cuh"
#endif

#include <raft/core/resource/comms.hpp>
#include <raft/core/resource/multi_gpu.hpp>

#include <optional>
#include <vector>

namespace cuvs::cluster::kmeans {

template <typename DataT, typename IndexT>
void fit(raft::resources const& handle,
         const kmeans::params& params,
         raft::device_matrix_view<const DataT, IndexT> X,
         std::optional<raft::device_vector_view<const DataT, IndexT>> sample_weight,
         raft::device_matrix_view<DataT, IndexT> centroids,
         raft::host_scalar_view<DataT> inertia,
         raft::host_scalar_view<IndexT> n_iter)
{
#ifdef CUVS_BUILD_MG_ALGOS
  if (raft::resource::is_multi_gpu(handle) || raft::resource::comms_initialized(handle)) {
    std::vector<raft::device_matrix_view<const DataT, IndexT>> X_parts{X};
    std::optional<std::vector<raft::device_vector_view<const DataT, IndexT>>> sw_parts;
    if (sample_weight.has_value()) {
      sw_parts = std::vector<raft::device_vector_view<const DataT, IndexT>>{*sample_weight};
    }
    cuvs::cluster::kmeans::mg::detail::mnmg_fit<DataT, IndexT>(
      handle, params, X_parts, sw_parts, centroids, inertia, n_iter);
    return;
  }
#endif
  cuvs::cluster::kmeans::detail::kmeans_fit<DataT, IndexT>(
    handle, params, X, sample_weight, centroids, inertia, n_iter);
}

template <typename DataT, typename IndexT>
void fit(raft::resources const& handle,
         const kmeans::params& params,
         raft::host_matrix_view<const DataT, IndexT> X,
         std::optional<raft::host_vector_view<const DataT, IndexT>> sample_weight,
         raft::device_matrix_view<DataT, IndexT> centroids,
         raft::host_scalar_view<DataT> inertia,
         raft::host_scalar_view<IndexT> n_iter)
{
#ifdef CUVS_BUILD_MG_ALGOS
  if (raft::resource::is_multi_gpu(handle)) {
    cuvs::cluster::kmeans::mg::detail::batched_fit_omp<DataT, IndexT>(
      handle, params, X, sample_weight, centroids, inertia, n_iter);
    return;
  }
  if (raft::resource::comms_initialized(handle)) {
    std::vector<raft::host_matrix_view<const DataT, IndexT>> X_parts{X};
    std::optional<std::vector<raft::host_vector_view<const DataT, IndexT>>> sw_parts;
    if (sample_weight.has_value()) {
      sw_parts = std::vector<raft::host_vector_view<const DataT, IndexT>>{*sample_weight};
    }
    cuvs::cluster::kmeans::mg::detail::mnmg_fit<DataT, IndexT>(
      handle, params, X_parts, sw_parts, centroids, inertia, n_iter);
    return;
  }
#endif
  cuvs::cluster::kmeans::detail::fit<DataT, IndexT>(
    handle, params, X, sample_weight, centroids, inertia, n_iter);
}

}  // namespace cuvs::cluster::kmeans
