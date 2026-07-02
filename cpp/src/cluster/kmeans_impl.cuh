/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "kmeans.cuh"

namespace cuvs::cluster::kmeans {

template <typename DataT, typename IndexT>
void fit_main(raft::resources const& handle,
              const kmeans::params& params,
              raft::device_matrix_view<const DataT, IndexT> X,
              raft::device_vector_view<const DataT, IndexT> sample_weights,
              raft::device_matrix_view<DataT, IndexT> centroids,
              raft::host_scalar_view<DataT> inertia,
              raft::host_scalar_view<IndexT> n_iter,
              rmm::device_uvector<char>& workspace)
{
  cuvs::cluster::kmeans::params p = params;
  p.init                          = kmeans::params::InitMethod::Array;
  RAFT_EXPECTS(sample_weights.extent(0) == X.extent(0),
               "invalid parameter (sample_weight!=n_samples)");
  auto sw = std::make_optional(sample_weights);
  cuvs::cluster::kmeans::detail::kmeans_fit(
    handle, p, X, sw, centroids, inertia, n_iter, std::ref(workspace));
}

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
    cuvs::cluster::kmeans::mg::fit(handle, params, X, sample_weight, centroids, inertia, n_iter);
  } else {
    cuvs::cluster::kmeans::detail::kmeans_fit<DataT, IndexT>(
      handle, params, X, sample_weight, centroids, inertia, n_iter);
  }
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

template <typename DataT, typename IndexT>
void fit(raft::resources const& handle,
         const kmeans::params& params,
         raft::host_matrix_view<const DataT, IndexT> X,
         std::optional<raft::host_vector_view<const DataT, IndexT>> sample_weight,
         raft::device_matrix_view<DataT, IndexT> centroids,
         raft::host_scalar_view<DataT> inertia,
         raft::host_scalar_view<IndexT> n_iter)
{
  cuvs::cluster::kmeans::detail::fit<DataT, IndexT>(
    handle, params, X, sample_weight, centroids, inertia, n_iter);
}

}  // namespace cuvs::cluster::kmeans
