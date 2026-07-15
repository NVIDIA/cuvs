/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "detail/kmeans_balanced.cuh"
#include <cuvs/cluster/kmeans.hpp>
#include <raft/core/resource/device_memory_resource.hpp>

namespace cuvs::cluster::kmeans_balanced::helpers {

/**
 * @brief Randomly initialize centers and apply expectation-maximization-balancing iterations
 *
 * This is essentially the non-hierarchical balanced k-means algorithm which is used by the
 * hierarchical algorithm once to build the mesoclusters and once per mesocluster to build the fine
 * clusters.
 *
 * @tparam DataT Type of the input data.
 * @tparam MathT Type of the centroids and mapped data.
 * @tparam IndexT Type used for indexing.
 * @tparam LabelT Type of the output labels.
 * @tparam CounterT Counter type supported by CUDA's native atomicAdd.
 * @tparam MappingOpT Type of the mapping function.
 * @param[in]  handle        The raft resources
 * @param[in]  params        Structure containing the hyper-parameters
 * @param[in]  X             Training instances to cluster. The data must be in row-major format.
 *                           [dim = n_samples x n_features]
 * @param[out] centroids     The output centroids [dim = n_clusters x n_features]
 * @param[out] labels        The output labels [dim = n_samples]
 * @param[out] cluster_sizes Size of each cluster [dim = n_clusters]
 * @param[in]  mapping_op    (optional) Functor to convert from the input datatype to the
 *                           arithmetic datatype. If DataT == MathT, this must be the identity.
 * @param[in]  X_norm        (optional) Dataset's row norms [dim = n_samples]
 */
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

}  // namespace cuvs::cluster::kmeans_balanced::helpers
