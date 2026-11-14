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
 * Only float data is supported. The metric (L2Expanded / L2SqrtExpanded) is already enforced by
 * fit/predict.
 */
#pragma once

#include "../../distance/detail/top_1_nn_cagra.cuh"

#include <cuvs/cluster/kmeans.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <optional>
#include <type_traits>
#include <variant>

namespace cuvs::cluster::kmeans::detail {

/**
 * @brief Whether ANN assignment should be used for this call. Validates the ANN params when the
 * flag is set; unsupported configurations log a warning and return false (brute force is used).
 */
template <typename DataT>
bool use_ann_assignment(cuvs::cluster::kmeans::params const& params,
                        int64_t n_clusters,
                        int64_t n_features)
{
  if (!params.use_ann_for_assignment) { return false; }
  if constexpr (!std::is_same_v<DataT, float>) {
    RAFT_LOG_WARN("kmeans: use_ann_for_assignment requires float data; using brute force.");
    return false;
  } else {
    if (n_clusters < 2) {
      RAFT_LOG_WARN("kmeans: use_ann_for_assignment requires n_clusters >= 2; using brute force.");
      return false;
    }
    // The centroids are a device dataset, for which an ACE graph build is not supported.
    if (params.ann_params.has_value() &&
        std::holds_alternative<cuvs::neighbors::cagra::graph_build_params::ace_params>(
          params.ann_params->index_params.graph_build_params)) {
      RAFT_LOG_WARN(
        "kmeans: use_ann_for_assignment does not support ACE graph build params; using brute "
        "force.");
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
    // The user's CAGRA parameters are passed through as they are; an unset optional falls back to
    // the defaults (NN-descent graph build, itopk_size = 32).
    const auto ann_params =
      params.ann_params.value_or(cuvs::cluster::kmeans::ann_assignment_params{});
    ann_index.build(
      handle, ann_params.index_params, ann_params.search_params, centroids, n_clusters, n_features);
  } else {
    RAFT_FAIL("ANN centroid assignment only supports float data");
  }
}

/**
 * @brief Multi-GPU counterpart of build_ann_index: every rank ends up with the same CAGRA index
 * over the (identical on all ranks) `n_clusters` x `n_features` centroids, but the graph is built
 * only once.
 *
 * @p root builds the graph and broadcasts it through @p comms (`mnmg_comms`-like: `bcast(T*,
 * count, root)`); every other rank attaches the received graph to its own copy of the centroids.
 * The CAGRA graph is the only part of the index that is expensive to produce. All ranks must call
 * this together, with identical centroids and params, even ranks that hold no data.
 */
template <typename DataT, typename CommsT>
void build_ann_index_broadcast(raft::resources const& handle,
                               cuvs::cluster::kmeans::params const& params,
                               CommsT const& comms,
                               int rank,
                               int root,
                               cuvs::distance::detail::top_1_nn_cagra_index& ann_index,
                               const DataT* centroids,
                               int64_t n_clusters,
                               int64_t n_features)
{
  if constexpr (std::is_same_v<DataT, float>) {
    const auto ann_params =
      params.ann_params.value_or(cuvs::cluster::kmeans::ann_assignment_params{});
    auto stream = raft::resource::get_cuda_stream(handle);

    // The graph degree is only known to the rank that built the graph (it depends on the CAGRA
    // build params), so it is broadcast ahead of the graph itself.
    auto degree_dev = raft::make_device_scalar<int>(handle, 0);
    if (rank == root) {
      ann_index.build(handle,
                      ann_params.index_params,
                      ann_params.search_params,
                      centroids,
                      n_clusters,
                      n_features);
      int degree = static_cast<int>(ann_index.graph().extent(1));
      raft::copy(degree_dev.data_handle(), &degree, 1, stream);
    }
    comms.bcast(degree_dev.data_handle(), std::size_t{1}, root);
    int degree = 0;
    raft::copy(&degree, degree_dev.data_handle(), 1, stream);
    raft::resource::sync_stream(handle);

    // The collectives have no uint32 type; the graph is broadcast as same-sized int32.
    if (rank == root) {
      auto* graph = const_cast<uint32_t*>(ann_index.graph().data_handle());
      comms.bcast(reinterpret_cast<int*>(graph),
                  static_cast<std::size_t>(n_clusters) * static_cast<std::size_t>(degree),
                  root);
    } else {
      auto graph = raft::make_device_matrix<uint32_t, int64_t>(handle, n_clusters, degree);
      comms.bcast(reinterpret_cast<int*>(graph.data_handle()),
                  static_cast<std::size_t>(n_clusters) * static_cast<std::size_t>(degree),
                  root);
      ann_index.attach_graph(
        handle, ann_params.search_params, centroids, n_clusters, n_features, std::move(graph));
    }
    raft::resource::sync_stream(handle);
  } else {
    RAFT_FAIL("ANN centroid assignment only supports float data");
  }
}

}  // namespace cuvs::cluster::kmeans::detail
