/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../neighbors/detail/ann_utils.cuh"
#include "kmeans_balanced_build_clusters_impl.cuh"

#include <raft/core/resources.hpp>

namespace cuvs::cluster::kmeans_balanced::helpers {

void build_clusters(const raft::resources& handle,
                    const cuvs::cluster::kmeans::balanced_params& params,
                    raft::device_matrix_view<const float, int64_t> X,
                    raft::device_matrix_view<float, int64_t> centroids,
                    raft::device_vector_view<uint32_t, int64_t> labels,
                    raft::device_vector_view<uint32_t, int64_t> cluster_sizes,
                    cuvs::spatial::knn::detail::utils::mapping<float> mapping_op,
                    std::optional<raft::device_vector_view<const float>> X_norm)
{
  build_clusters<float,
                 float,
                 int64_t,
                 uint32_t,
                 uint32_t,
                 cuvs::spatial::knn::detail::utils::mapping<float>>(
    handle, params, X, centroids, labels, cluster_sizes, mapping_op, X_norm);
}

}  // namespace cuvs::cluster::kmeans_balanced::helpers
