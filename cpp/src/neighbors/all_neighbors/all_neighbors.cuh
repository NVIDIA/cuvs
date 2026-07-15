/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "../detail/reachability.cuh"
#include "all_neighbors_batched.cuh"
#include <cuvs/neighbors/all_neighbors.hpp>
#include <raft/matrix/shift.cuh>
#include <raft/util/cudart_utils.hpp>
#include <unordered_set>

namespace cuvs::neighbors::all_neighbors::detail {
using namespace cuvs::neighbors;

GRAPH_BUILD_ALGO check_params_validity(const all_neighbors_params& params,
                                       bool do_mutual_reachability_dist)
{
  using DT = cuvs::distance::DistanceType;

  // InnerProduct is not supported for mutual reachability distance, because mutual reachability
  // distance takes "max" of core distances and pairwise distance.
  static const std::unordered_set<DT> mrd_allowed_metrics = {
    DT::L2Expanded, DT::L2SqrtExpanded, DT::CosineExpanded};

  static const std::unordered_set<DT> bf_allowed_metrics = {DT::L2Expanded,
                                                            DT::L2SqrtExpanded,
                                                            DT::CosineExpanded,
                                                            DT::L1,
                                                            DT::L2Unexpanded,
                                                            DT::L2SqrtUnexpanded,
                                                            DT::InnerProduct,
                                                            DT::Linf,
                                                            DT::Canberra,
                                                            DT::LpUnexpanded,
                                                            DT::CorrelationExpanded,
                                                            DT::JensenShannon};

  static const std::unordered_set<DT> nnd_allowed_metrics = {
    DT::L2Expanded, DT::L2SqrtExpanded, DT::CosineExpanded, DT::InnerProduct};

  if (std::holds_alternative<graph_build_params::brute_force_params>(params.graph_build_params)) {
    if (do_mutual_reachability_dist) {
      RAFT_EXPECTS(
        mrd_allowed_metrics.count(params.metric),
        "Distance metric for all-neighbors build with brute force for computing mutual "
        "reachability distance should be L2Expanded, L2SqrtExpanded, or CosineExpanded.");
    } else {
      RAFT_EXPECTS(
        bf_allowed_metrics.count(params.metric),
        "Distance metric for all-neighbors build with brute force should be L2Expanded, "
        "L2SqrtExpanded, CosineExpanded, L1, L2Unexpanded, L2SqrtUnexpanded, InnerProduct, Linf, "
        "Canberra, LpUnexpanded, CorrelationExpanded, or JensenShannon.");
    }
    return GRAPH_BUILD_ALGO::BRUTE_FORCE;
  } else if (std::holds_alternative<graph_build_params::nn_descent_params>(
               params.graph_build_params)) {
    if (do_mutual_reachability_dist) {
      RAFT_EXPECTS(
        mrd_allowed_metrics.count(params.metric),
        "Distance metric for all-neighbors build with NN Descent for computing mutual reachability "
        "distance should be L2Expanded, L2SqrtExpanded, or CosineExpanded.");
    } else {
      RAFT_EXPECTS(nnd_allowed_metrics.count(params.metric),
                   "Distance metric for all-neighbors build with NN Descent should be L2Expanded, "
                   "L2SqrtExpanded, CosineExpanded, or InnerProduct.");
    }
    return GRAPH_BUILD_ALGO::NN_DESCENT;
  } else if (std::holds_alternative<graph_build_params::ivf_pq_params>(params.graph_build_params)) {
    RAFT_EXPECTS(params.metric == cuvs::distance::DistanceType::L2Expanded,
                 "Distance metric for all-neighbors build with IVFPQ should be L2Expanded");
    RAFT_EXPECTS(!do_mutual_reachability_dist,
                 "mutual reachability distance cannot be calculated using IVFPQ");
    return GRAPH_BUILD_ALGO::IVF_PQ;
  } else {
    RAFT_FAIL("Invalid all-neighbors build algo");
  }
}

// Single build (i.e. no batching) supports both host and device datasets
template <typename T, typename IdxT, typename Accessor, typename DistEpilogueT = raft::identity_op>
void single_build(
  const raft::resources& handle,
  const all_neighbors_params& params,
  mdspan<const T, matrix_extent<IdxT>, row_major, Accessor> dataset,
  raft::device_matrix_view<IdxT, IdxT, row_major> indices,
  std::optional<raft::device_matrix_view<T, IdxT, row_major>> distances = std::nullopt,
  DistEpilogueT dist_epilogue                                           = DistEpilogueT{})
{
  size_t num_rows = static_cast<size_t>(dataset.extent(0));
  size_t num_cols = static_cast<size_t>(dataset.extent(1));

  auto knn_builder = get_knn_builder<T, IdxT>(
    handle, params, num_rows, num_rows, indices.extent(1), indices, distances, dist_epilogue);

  knn_builder->prepare_build(dataset);
  knn_builder->build_knn(dataset);
}

template <typename T, typename IdxT>
void build(
  const raft::resources& handle,
  const all_neighbors_params& params,
  raft::host_matrix_view<const T, IdxT, row_major> dataset,
  raft::device_matrix_view<IdxT, IdxT, row_major> indices,
  std::optional<raft::device_matrix_view<T, IdxT, row_major>> distances      = std::nullopt,
  std::optional<raft::device_vector_view<T, IdxT, row_major>> core_distances = std::nullopt,
  T alpha                                                                    = 1.0)
{
  auto build_algo = check_params_validity(params, core_distances.has_value());

  RAFT_EXPECTS(dataset.extent(0) == indices.extent(0),
               "number of rows in dataset should be the same as number of rows in indices matrix");

  if (distances.has_value()) {
    RAFT_EXPECTS(indices.extent(0) == distances.value().extent(0) &&
                   indices.extent(1) == distances.value().extent(1),
                 "indices matrix and distances matrix has to be the same shape.");
  }

  if (core_distances.has_value()) {
    RAFT_EXPECTS(distances.has_value(),
                 "distances matrix should be allocated to get mutual reachability distance.");
  }

  std::unique_ptr<BatchBuildAux<IdxT>> aux_vectors;
  if (params.n_clusters == 1) {
    single_build(handle, params, dataset, indices, distances);
  } else {
    if (core_distances.has_value()) {
      aux_vectors = std::make_unique<BatchBuildAux<IdxT>>(
        params.n_clusters, dataset.extent(0), params.overlap_factor);
      batch_build(handle, params, dataset, indices, distances, aux_vectors.get());
    } else {
      batch_build(handle, params, dataset, indices, distances);
    }
  }

  // NN Descent doesn't include self loops. Shifted to keep it consistent with brute force and ivfpq
  bool need_shift = (build_algo == GRAPH_BUILD_ALGO::NN_DESCENT) &&
                    (params.metric != cuvs::distance::DistanceType::InnerProduct);

  if (need_shift) {
    raft::matrix::shift(handle, indices, 1);
    if (distances.has_value()) {
      raft::matrix::shift(handle, distances.value(), 1, std::make_optional<T>(0.0));
    }
  }

  if (core_distances.has_value()) {  // calculate mutual reachability distances
    size_t k        = indices.extent(1);
    size_t num_rows = core_distances.value().size();
    cuvs::neighbors::detail::reachability::core_distances<IdxT, T>(
      handle,
      distances.value().data_handle(),
      k,
      k,
      num_rows,
      core_distances.value().data_handle());

    using ReachabilityPP = cuvs::neighbors::detail::reachability::ReachabilityPostProcess<IdxT, T>;
    auto dist_epilogue   = ReachabilityPP{core_distances.value().data_handle(), alpha, num_rows};
    if (params.n_clusters == 1) {
      single_build(handle, params, dataset, indices, distances, dist_epilogue);
    } else {
      batch_build(handle, params, dataset, indices, distances, aux_vectors.get(), dist_epilogue);
    }

    if (need_shift) {
      raft::matrix::shift(handle, indices, 1);
      raft::matrix::shift(handle,
                          distances.value(),
                          raft::make_device_matrix_view<const T, IdxT>(
                            core_distances.value().data_handle(), num_rows, 1));
    }
  }
}

template <typename T, typename IdxT>
void build(
  const raft::resources& handle,
  const all_neighbors_params& params,
  raft::device_matrix_view<const T, IdxT, row_major> dataset,
  raft::device_matrix_view<IdxT, IdxT, row_major> indices,
  std::optional<raft::device_matrix_view<T, IdxT, row_major>> distances      = std::nullopt,
  std::optional<raft::device_vector_view<T, IdxT, row_major>> core_distances = std::nullopt,
  T alpha                                                                    = 1.0)
{
  auto build_algo = check_params_validity(params, core_distances.has_value());

  RAFT_EXPECTS(dataset.extent(0) == indices.extent(0),
               "number of rows in dataset should be the same as number of rows in indices matrix");

  if (distances.has_value()) {
    RAFT_EXPECTS(indices.extent(0) == distances.value().extent(0) &&
                   indices.extent(1) == distances.value().extent(1),
                 "indices matrix and distances matrix has to be the same shape.");
  }

  if (core_distances.has_value()) {
    RAFT_EXPECTS(distances.has_value(),
                 "distances matrix should be allocated to get mutual reachability distance.");
  }

  if (params.n_clusters > 1) {
    RAFT_FAIL(
      "Batched all-neighbors build is not supported with data on device. Put data on host for "
      "batch build.");
  } else {
    single_build(handle, params, dataset, indices, distances);
  }

  // NN Descent doesn't include self loops. Shifted to keep it consistent with brute force and ivfpq
  bool need_shift = (build_algo == GRAPH_BUILD_ALGO::NN_DESCENT) &&
                    (params.metric != cuvs::distance::DistanceType::InnerProduct);

  if (need_shift) {
    raft::matrix::shift(handle, indices, 1);
    if (distances.has_value()) {
      raft::matrix::shift(handle, distances.value(), 1, std::make_optional<T>(0.0));
    }
  }

  if (core_distances.has_value()) {
    size_t k        = indices.extent(1);
    size_t num_rows = core_distances.value().size();
    cuvs::neighbors::detail::reachability::core_distances<IdxT, T>(
      handle,
      distances.value().data_handle(),
      k,
      k,
      num_rows,
      core_distances.value().data_handle());

    using ReachabilityPP = cuvs::neighbors::detail::reachability::ReachabilityPostProcess<IdxT, T>;
    auto dist_epilogue   = ReachabilityPP{core_distances.value().data_handle(), alpha, num_rows};
    single_build(handle, params, dataset, indices, distances, dist_epilogue);

    if (need_shift) {
      raft::matrix::shift(handle, indices, 1);
      raft::matrix::shift(handle,
                          distances.value(),
                          raft::make_device_matrix_view<const T, IdxT>(
                            core_distances.value().data_handle(), num_rows, 1));
    }
  }
}
}  // namespace cuvs::neighbors::all_neighbors::detail
