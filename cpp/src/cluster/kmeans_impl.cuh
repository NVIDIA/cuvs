/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "kmeans.cuh"

#include <raft/core/logger.hpp>
#include <raft/core/resource/comms.hpp>

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
  if (raft::resource::comms_initialized(handle)) {
    RAFT_LOG_WARN(
      "Multi-GPU handle detected on single-GPU kmeans::fit() entry; "
      "falling back to single-GPU. Use cuvs::cluster::kmeans::mg::fit(...) for multi-GPU.");
  }
  cuvs::cluster::kmeans::detail::kmeans_fit<DataT, IndexT>(
    handle, params, X, sample_weight, centroids, inertia, n_iter);
}

template <typename DataT, typename IndexT>
void predict(raft::resources const& handle,
             const kmeans::params& params,
             raft::device_matrix_view<const DataT, IndexT> X,
             std::optional<raft::device_vector_view<const DataT, IndexT>> sample_weight,
             raft::device_matrix_view<const DataT, IndexT> centroids,
             raft::device_vector_view<IndexT, IndexT> labels,
             bool normalize_weight,
             raft::host_scalar_view<DataT> inertia)
{
  cuvs::cluster::kmeans::detail::kmeans_predict<DataT, IndexT>(
    handle, params, X, sample_weight, centroids, labels, normalize_weight, inertia);
}

}  // namespace cuvs::cluster::kmeans
