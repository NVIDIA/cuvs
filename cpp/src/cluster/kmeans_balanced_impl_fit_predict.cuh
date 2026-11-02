/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/cluster/kmeans.hpp>

namespace cuvs::cluster::kmeans_balanced {

template <typename DataT, typename MathT, typename IndexT, typename LabelT>
void fit_predict(
  const raft::resources& handle,
  cuvs::cluster::kmeans::balanced_params const& params,
  raft::device_matrix_view<const DataT, IndexT> X,
  raft::device_matrix_view<MathT, IndexT> centroids,
  raft::device_vector_view<LabelT, IndexT> labels,
  std::optional<raft::host_scalar_view<MathT>> inertia                         = std::nullopt,
  std::optional<raft::device_matrix_view<MathT, IndexT>> parent_centroids      = std::nullopt,
  std::optional<raft::device_vector_view<uint32_t, IndexT>> centroid_to_parent = std::nullopt)
{
  auto centroids_const = raft::make_device_matrix_view<const MathT, IndexT>(
    centroids.data_handle(), centroids.extent(0), centroids.extent(1));
  cuvs::cluster::kmeans::fit(
    handle, params, X, centroids, inertia, parent_centroids, centroid_to_parent);
  cuvs::cluster::kmeans::predict(handle, params, X, centroids_const, labels);
}

}  // namespace cuvs::cluster::kmeans_balanced
