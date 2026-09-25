/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/cluster/kmeans.hpp>

namespace cuvs::cluster::kmeans_balanced {

template <typename DataT, typename MathT, typename IndexT, typename LabelT>
void fit_predict(const raft::resources& handle,
                 cuvs::cluster::kmeans::balanced_params const& params,
                 raft::device_matrix_view<const DataT, IndexT> X,
                 raft::device_matrix_view<MathT, IndexT> centroids,
                 raft::device_vector_view<LabelT, IndexT> labels)
{
  auto centroids_const = raft::make_device_matrix_view<const MathT, IndexT>(
    centroids.data_handle(), centroids.extent(0), centroids.extent(1));
  cuvs::cluster::kmeans::fit(handle, params, X, centroids);
  cuvs::cluster::kmeans::predict(handle, params, X, centroids_const, labels);
}

}  // namespace cuvs::cluster::kmeans_balanced
