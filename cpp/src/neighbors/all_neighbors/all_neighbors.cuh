/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "../detail/reachability.cuh"
#include "all_neighbors_batched.cuh"
#include <algorithm>
#include <cuvs/neighbors/all_neighbors.hpp>
#include <raft/matrix/shift.cuh>
#include <raft/util/cudart_utils.hpp>
#include <type_traits>
#include <unordered_set>

namespace cuvs::neighbors::all_neighbors::detail {
using namespace cuvs::neighbors;

template <typename T,
          typename IdxT,
          typename DatasetView,
          typename IndicesView,
          typename DistancesView,
          typename CoreView>
GRAPH_BUILD_ALGO check_params_validity(const all_neighbors_params& params,
                                       DatasetView dataset,
                                       IndicesView indices,
                                       const std::optional<DistancesView>& distances,
                                       const std::optional<CoreView>& core_distances)
{
  // 1. Check output memory-space consistency
  constexpr bool host_output =
    std::is_same_v<IndicesView, raft::host_matrix_view<IdxT, IdxT, row_major>>;
  constexpr bool device_output =
    std::is_same_v<IndicesView, raft::device_matrix_view<IdxT, IdxT, row_major>>;
  static_assert(host_output || device_output,
                "indices must be a host_matrix_view or device_matrix_view");
  if constexpr (host_output) {
    static_assert(std::is_same_v<DistancesView, raft::host_matrix_view<T, IdxT, row_major>>,
                  "distances must be a host_matrix_view when indices are on host");
    static_assert(std::is_same_v<CoreView, raft::host_vector_view<T, IdxT, row_major>>,
                  "core_distances must be a host_vector_view when indices are on host");
  } else {
    static_assert(std::is_same_v<DistancesView, raft::device_matrix_view<T, IdxT, row_major>>,
                  "distances must be a device_matrix_view when indices are on device");
    static_assert(std::is_same_v<CoreView, raft::device_vector_view<T, IdxT, row_major>>,
                  "core_distances must be a device_vector_view when indices are on device");
  }

  // 2. Check shape consistency
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

  // 3. Check metric / algorithm validity
  const bool do_mutual_reachability_dist = core_distances.has_value();
  using DT                               = cuvs::distance::DistanceType;

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

// Full build (i.e. no batching) for output indices/distances on device memory.
template <typename T, typename IdxT, typename Accessor, typename DistEpilogueT = raft::identity_op>
void full_build(
  const raft::resources& handle,
  const all_neighbors_params& params,
  mdspan<const T, matrix_extent<IdxT>, row_major, Accessor> dataset,
  raft::device_matrix_view<IdxT, IdxT, row_major> indices,
  std::optional<raft::device_matrix_view<T, IdxT, row_major>> distances = std::nullopt,
  DistEpilogueT dist_epilogue                                           = DistEpilogueT{})
{
  size_t num_rows = static_cast<size_t>(dataset.extent(0));
  size_t num_cols = static_cast<size_t>(dataset.extent(1));

  auto knn_builder =
    get_knn_builder<T, IdxT>(handle, params, num_rows, num_rows, indices.extent(1), dist_epilogue);

  knn_builder->prepare_build(dataset);
  knn_builder->build_knn(dataset, std::nullopt, global_graph_view<T, IdxT>{indices, distances});
}

// Full build (i.e. no batching) for output indices/distances on host memory.
template <typename T, typename IdxT, typename Accessor, typename DistEpilogueT = raft::identity_op>
void full_build_host(
  const raft::resources& handle,
  const all_neighbors_params& params,
  mdspan<const T, matrix_extent<IdxT>, row_major, Accessor> dataset,
  raft::host_matrix_view<IdxT, IdxT, row_major> indices,
  std::optional<raft::host_matrix_view<T, IdxT, row_major>> distances = std::nullopt,
  DistEpilogueT dist_epilogue                                         = DistEpilogueT{})
{
  size_t num_rows = static_cast<size_t>(dataset.extent(0));
  size_t k        = static_cast<size_t>(indices.extent(1));

  // kNN algorithms require device-resident output graph
  auto indices_d = raft::make_device_matrix<IdxT, IdxT>(handle, num_rows, k);
  std::optional<raft::device_matrix<T, IdxT>> distances_d;
  std::optional<raft::device_matrix_view<T, IdxT, row_major>> distances_d_view;
  if (distances.has_value()) {
    distances_d.emplace(raft::make_device_matrix<T, IdxT>(handle, num_rows, k));
    distances_d_view = distances_d.value().view();
  }

  full_build(handle, params, dataset, indices_d.view(), distances_d_view, dist_epilogue);

  raft::copy(handle, indices, indices_d.view());
  if (distances.has_value()) { raft::copy(handle, distances.value(), distances_d.value().view()); }
  raft::resource::sync_stream(handle);
}

// Host counterparts of raft::matrix::shift used to insert the self-references.
template <typename IdxT>
void host_shift_self_indices(raft::host_matrix_view<IdxT, IdxT, row_major> indices)
{
  size_t num_rows = static_cast<size_t>(indices.extent(0));
  size_t k        = static_cast<size_t>(indices.extent(1));
#pragma omp parallel for
  for (size_t i = 0; i < num_rows; i++) {
    IdxT* row = indices.data_handle() + i * k;
    std::shift_right(row, row + k, 1);
    row[0] = static_cast<IdxT>(i);
  }
}

// Mirrors use cases of raft::matrix::shift, the distance side adjustments for inserting
// self-references. Shift each row of distances right by 1. Column 0 is filled from col0(i) when
// provided (e.g. the core distances), otherwise with fill_val.
template <typename T, typename IdxT>
void host_shift_distances(
  raft::host_matrix_view<T, IdxT, row_major> distances,
  std::optional<T> fill_val,
  std::optional<raft::host_vector_view<const T, IdxT, row_major>> col0 = std::nullopt)
{
  size_t num_rows = static_cast<size_t>(distances.extent(0));
  size_t k        = static_cast<size_t>(distances.extent(1));
#pragma omp parallel for
  for (size_t i = 0; i < num_rows; i++) {
    T* row = distances.data_handle() + i * k;
    std::shift_right(row, row + k, 1);
    row[0] = col0.has_value() ? col0.value()(i) : fill_val.value();
  }
}

// Inserts the self-reference for the kNN graph
template <typename T,
          typename IdxT,
          typename IndicesView,
          typename DistancesView,
          typename CoreView = raft::host_vector_view<const T, IdxT>>
void shift_indices_distances(const raft::resources& handle,
                             IndicesView indices,
                             std::optional<DistancesView> distances = std::nullopt,
                             std::optional<CoreView> core_distances = std::nullopt)
{
  constexpr bool host_output =
    std::is_same_v<IndicesView, raft::host_matrix_view<IdxT, IdxT, row_major>>;

  if constexpr (host_output) {
    host_shift_self_indices<IdxT>(indices);
    if (distances.has_value()) {
      if (core_distances.has_value()) {
        host_shift_distances<T, IdxT>(
          distances.value(), std::nullopt, raft::make_const_mdspan(core_distances.value()));
      } else {
        host_shift_distances<T, IdxT>(distances.value(), std::make_optional<T>(0.0));
      }
    }
  } else {
    raft::matrix::shift(handle, indices, 1);
    if (distances.has_value()) {
      if (core_distances.has_value()) {
        raft::matrix::shift(handle,
                            distances.value(),
                            raft::make_device_matrix_view<const T, IdxT>(
                              core_distances.value().data_handle(), indices.extent(0), IdxT{1}));
      } else {
        raft::matrix::shift(handle, distances.value(), 1, std::make_optional<T>(0.0));
      }
    }
  }
}

// Builds an all-neighbors knn graph with the dataset on host. The output (indices/distances/
// core_distances) can be either on device or host.
template <typename T,
          typename IdxT,
          typename IndicesView,
          typename DistancesView,
          typename CoreView>
void build(const raft::resources& handle,
           const all_neighbors_params& params,
           raft::host_matrix_view<const T, IdxT, row_major> dataset,
           IndicesView indices,
           std::optional<DistancesView> distances = std::nullopt,
           std::optional<CoreView> core_distances = std::nullopt,
           T alpha                                = 1.0)
{
  constexpr bool host_output =
    std::is_same_v<IndicesView, raft::host_matrix_view<IdxT, IdxT, row_major>>;

  auto build_algo =
    check_params_validity<T, IdxT>(params, dataset, indices, distances, core_distances);

  // Runs the no-batching build into the (device or host) output, selected at compile time.
  auto run_full_build = [&](auto epilogue) {
    if constexpr (host_output) {
      full_build_host(handle, params, dataset, indices, distances, epilogue);
    } else {
      full_build(handle, params, dataset, indices, distances, epilogue);
    }
  };

  std::unique_ptr<BatchBuildAux<IdxT>> aux_vectors;
  if (params.n_clusters == 1) {
    run_full_build(raft::identity_op{});
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
  if (need_shift) { shift_indices_distances<T, IdxT>(handle, indices, distances); }

  if (core_distances.has_value()) {  // calculate mutual reachability distances
    size_t k        = indices.extent(1);
    size_t num_rows = core_distances.value().size();

    std::optional<raft::device_vector<T, IdxT>> core_dists_d;
    const T* core_dists_ptr = nullptr;

    if constexpr (host_output) {
      auto core_dists      = core_distances.value();
      auto distances_value = distances.value();
#pragma omp parallel for
      for (size_t r = 0; r < num_rows; r++) {
        core_dists(r) = distances_value(r, k - 1);
      }

      if (params.n_clusters > 1 && raft::resource::is_multi_gpu(handle)) {
        // Multi-GPU batched builds redistribute core distances to each GPU through host inside
        // multi_gpu_batch_build
        core_dists_ptr = core_dists.data_handle();
      } else {
        core_dists_d.emplace(raft::make_device_vector<T, IdxT>(handle, num_rows));
        raft::copy(handle,
                   core_dists_d.value().view(),
                   raft::make_host_vector_view<const T, IdxT>(core_dists.data_handle(), num_rows));
        core_dists_ptr = core_dists_d.value().data_handle();
      }
    } else {
      cuvs::neighbors::detail::reachability::core_distances<IdxT, T>(
        handle,
        distances.value().data_handle(),
        k,
        k,
        num_rows,
        core_distances.value().data_handle());
      core_dists_ptr = core_distances.value().data_handle();
    }

    using ReachabilityPP = cuvs::neighbors::detail::reachability::ReachabilityPostProcess<IdxT, T>;
    auto dist_epilogue   = ReachabilityPP{core_dists_ptr, alpha, num_rows};
    if (params.n_clusters == 1) {
      run_full_build(dist_epilogue);
    } else {
      batch_build(handle, params, dataset, indices, distances, aux_vectors.get(), dist_epilogue);
    }

    if (need_shift) {
      shift_indices_distances<T, IdxT>(handle, indices, distances, core_distances);
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
  auto build_algo =
    check_params_validity<T, IdxT>(params, dataset, indices, distances, core_distances);

  if (params.n_clusters > 1) {
    RAFT_FAIL(
      "Batched all-neighbors build is not supported with data on device. Put data on host for "
      "batch build.");
  } else {
    full_build(handle, params, dataset, indices, distances);
  }

  // NN Descent doesn't include self loops. Shifted to keep it consistent with brute force and ivfpq
  bool need_shift = (build_algo == GRAPH_BUILD_ALGO::NN_DESCENT) &&
                    (params.metric != cuvs::distance::DistanceType::InnerProduct);
  if (need_shift) { shift_indices_distances<T, IdxT>(handle, indices, distances); }

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
    full_build(handle, params, dataset, indices, distances, dist_epilogue);

    if (need_shift) {
      shift_indices_distances<T, IdxT>(handle, indices, distances, core_distances);
    }
  }
}
}  // namespace cuvs::neighbors::all_neighbors::detail
