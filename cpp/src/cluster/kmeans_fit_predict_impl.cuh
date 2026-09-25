/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuvs/cluster/kmeans.hpp>

#include <raft/core/device_mdarray.hpp>

namespace cuvs::cluster::kmeans {

template <typename DataT, typename IndexT>
void fit_predict(raft::resources const& handle,
                 const kmeans::params& pams,
                 raft::device_matrix_view<const DataT, IndexT> X,
                 std::optional<raft::device_vector_view<const DataT, IndexT>> sample_weight,
                 std::optional<raft::device_matrix_view<DataT, IndexT>> centroids,
                 raft::device_vector_view<IndexT, IndexT> labels,
                 raft::host_scalar_view<DataT> inertia,
                 raft::host_scalar_view<IndexT> n_iter)
{
  if (!centroids.has_value()) {
    auto n_features = X.extent(1);
    auto centroids_matrix =
      raft::make_device_matrix<DataT, IndexT>(handle, pams.n_clusters, n_features);
    cuvs::cluster::kmeans::fit(
      handle, pams, X, sample_weight, centroids_matrix.view(), inertia, n_iter);
    cuvs::cluster::kmeans::predict(handle,
                                   pams,
                                   X,
                                   sample_weight,
                                   raft::make_const_mdspan(centroids_matrix.view()),
                                   labels,
                                   true,
                                   inertia);
  } else {
    cuvs::cluster::kmeans::fit(handle, pams, X, sample_weight, centroids.value(), inertia, n_iter);
    cuvs::cluster::kmeans::predict(handle,
                                   pams,
                                   X,
                                   sample_weight,
                                   raft::make_const_mdspan(centroids.value()),
                                   labels,
                                   true,
                                   inertia);
  }
}

}  // namespace cuvs::cluster::kmeans
