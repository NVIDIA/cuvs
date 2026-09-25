/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "detail/kmeans_balanced.cuh"
#include "kmeans.cuh"

#ifdef CUVS_BUILD_MG_ALGOS
#include "detail/kmeans_mg.cuh"
#endif

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/comms.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
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
void fit_predict(raft::resources const& handle,
                 const kmeans::params& params,
                 raft::device_matrix_view<const DataT, IndexT> X,
                 std::optional<raft::device_vector_view<const DataT, IndexT>> sample_weight,
                 std::optional<raft::device_matrix_view<DataT, IndexT>> centroids,
                 raft::device_vector_view<IndexT, IndexT> labels,
                 raft::host_scalar_view<DataT> inertia,
                 raft::host_scalar_view<IndexT> n_iter)
{
  if (!centroids.has_value()) {
    auto centroids_matrix =
      raft::make_device_matrix<DataT, IndexT>(handle, params.n_clusters, X.extent(1));
    cuvs::cluster::kmeans::fit(
      handle, params, X, sample_weight, centroids_matrix.view(), inertia, n_iter);
    cuvs::cluster::kmeans::predict(handle,
                                   params,
                                   X,
                                   sample_weight,
                                   raft::make_const_mdspan(centroids_matrix.view()),
                                   labels,
                                   true,
                                   inertia);
  } else {
    cuvs::cluster::kmeans::fit(
      handle, params, X, sample_weight, centroids.value(), inertia, n_iter);
    cuvs::cluster::kmeans::predict(handle,
                                   params,
                                   X,
                                   sample_weight,
                                   raft::make_const_mdspan(centroids.value()),
                                   labels,
                                   true,
                                   inertia);
  }
}

}  // namespace cuvs::cluster::kmeans

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

namespace helpers {

template <typename DataT,
          typename MathT,
          typename IndexT,
          typename LabelT,
          typename CounterT,
          typename MappingOpT>
void build_clusters(const raft::resources& handle,
                    const cuvs::cluster::kmeans::balanced_params& params,
                    raft::device_matrix_view<const DataT, IndexT> X,
                    raft::device_matrix_view<MathT, IndexT> centroids,
                    raft::device_vector_view<LabelT, IndexT> labels,
                    raft::device_vector_view<CounterT, IndexT> cluster_sizes,
                    MappingOpT mapping_op,
                    std::optional<raft::device_vector_view<const MathT>> X_norm)
{
  RAFT_EXPECTS(X.extent(0) == labels.extent(0),
               "Number of rows in dataset and labels are different");
  RAFT_EXPECTS(X.extent(1) == centroids.extent(1),
               "Number of features in dataset and centroids are different");
  RAFT_EXPECTS(centroids.extent(0) == cluster_sizes.extent(0),
               "Number of rows in centroids and clusyer_sizes are different");

  cuvs::cluster::kmeans::detail::build_clusters(
    handle,
    params,
    X.extent(1),
    X.data_handle(),
    X.extent(0),
    centroids.extent(0),
    centroids.data_handle(),
    labels.data_handle(),
    cluster_sizes.data_handle(),
    mapping_op,
    raft::resource::get_workspace_resource_ref(handle),
    X_norm.has_value() ? X_norm.value().data_handle() : nullptr);
}

}  // namespace helpers
}  // namespace cuvs::cluster::kmeans_balanced
