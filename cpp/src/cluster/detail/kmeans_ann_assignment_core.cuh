/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Approximate nearest-centroid assignment for Lloyd's k-means (cuvs::cluster::kmeans::fit/predict).
 *
 * A CAGRA index is built over the current centroids at the start of EVERY Lloyd iteration (it is
 * never reused across iterations, because the centroids move each iteration), and
 * each sample is assigned to the nearest centroid found by a top-1 search of that index. The
 * search is the `Top1nnBackend::Cagra` backend of `top_1_nn` (distance/top_1_nn.cuh), selected by
 * passing that backend and the index to `minClusterAndDistanceCompute`; everything else in the
 * E-step and M-step is the existing Lloyd code. The index always matches the centroids used in
 * the iteration.
 *
 * Only float data with the L2Expanded / L2SqrtExpanded metrics is supported.
 */
#pragma once

#include "../../distance/detail/top_1_nn_cagra.cuh"

#include <cuvs/cluster/kmeans.hpp>

#include <raft/core/logger.hpp>
#include <raft/core/resources.hpp>

#include <type_traits>

namespace cuvs::cluster::kmeans::detail {

/** Throws if the ANN-related fields of `params` are invalid. Only checked when ANN is enabled. */
inline void validate_ann_params(cuvs::cluster::kmeans::params const& params)
{
  RAFT_EXPECTS(params.ann_graph_degree > 0, "ann_graph_degree must be > 0");
  RAFT_EXPECTS(params.ann_intermediate_graph_degree > 0,
               "ann_intermediate_graph_degree must be > 0");
  RAFT_EXPECTS(params.ann_itopk_size > 0, "ann_itopk_size must be > 0");
  RAFT_EXPECTS(params.ann_search_width > 0, "ann_search_width must be > 0");
}

/**
 * @brief Whether ANN assignment should be used for this call. Validates the ANN params when the
 * flag is set; unsupported configurations log a warning and return false (brute force is used).
 */
template <typename DataT>
bool use_ann_assignment(cuvs::cluster::kmeans::params const& params, int64_t n_clusters)
{
  if (!params.use_ann_for_assignment) { return false; }
  validate_ann_params(params);
  if constexpr (!std::is_same_v<DataT, float>) {
    RAFT_LOG_WARN("kmeans: use_ann_for_assignment requires float data; using brute force.");
    return false;
  } else {
    if (params.metric != cuvs::distance::DistanceType::L2Expanded &&
        params.metric != cuvs::distance::DistanceType::L2SqrtExpanded) {
      RAFT_LOG_WARN("kmeans: use_ann_for_assignment requires an L2 metric; using brute force.");
      return false;
    }
    if (n_clusters < 2) {
      RAFT_LOG_WARN("kmeans: use_ann_for_assignment requires n_clusters >= 2; using brute force.");
      return false;
    }
    return true;
  }
}

/**
 * @brief (Re)builds @p ann_index over `n_clusters` x `n_features` row-major centroids at
 * @p centroids. Only called when use_ann_assignment<DataT>() is true, which implies DataT = float.
 */
template <typename DataT>
void build_ann_index(raft::resources const& handle,
                     cuvs::cluster::kmeans::params const& params,
                     cuvs::distance::detail::top_1_nn_cagra_index& ann_index,
                     const DataT* centroids,
                     int64_t n_clusters,
                     int64_t n_features)
{
  if constexpr (std::is_same_v<DataT, float>) {
    cuvs::distance::detail::top_1_nn_cagra_params cagra_params;
    cagra_params.graph_degree              = params.ann_graph_degree;
    cagra_params.intermediate_graph_degree = params.ann_intermediate_graph_degree;
    cagra_params.itopk_size                = params.ann_itopk_size;
    cagra_params.search_width              = params.ann_search_width;
    ann_index.build(handle, cagra_params, centroids, n_clusters, n_features);
  } else {
    RAFT_FAIL("ANN centroid assignment only supports float data");
  }
}

}  // namespace cuvs::cluster::kmeans::detail
