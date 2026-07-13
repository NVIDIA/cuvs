/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "../detail/reachability.cuh"
#include "all_neighbors_batched.cuh"
#include <cuvs/neighbors/all_neighbors.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/matrix/shift.cuh>
#include <raft/util/cudart_utils.hpp>
#include <unordered_set>

namespace cuvs::neighbors::all_neighbors::detail {
using namespace cuvs::neighbors;

// Host-side shift: shifts columns to the right by k, fills first column with row IDs (for indices)
// or 0 (for distances)
template <typename Mdspan>
void host_shift_columns(Mdspan in_out,
                        size_t k,
                        std::optional<typename Mdspan::element_type> fill_value = std::nullopt)
{
  size_t n_rows = in_out.extent(0);
  size_t n_cols = in_out.extent(1);
  RAFT_EXPECTS(n_cols > k, "Shift size k should be smaller than the number of columns in matrix.");

  // Shift columns to the right by k
#pragma omp parallel for
  for (size_t i = 0; i < n_rows; i++) {
    // Copy columns from right to left
    for (size_t j = n_cols - 1; j >= k; j--) {
      in_out(i, j) = in_out(i, j - k);
    }
    // Fill first k columns
    for (size_t j = 0; j < k; j++) {
      if (fill_value.has_value()) {
        in_out(i, j) = fill_value.value();
      } else {
        // Fill with row ID (for indices) or 0 (for distances)
        in_out(i, j) = static_cast<typename Mdspan::element_type>(i);
      }
    }
  }
}

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
template <typename T,
          typename IdxT,
          typename DatasetMdspan,
          typename IndicesMdspan,
          typename DistancesMdspan = raft::device_matrix_view<T, IdxT, row_major>,
          typename DistEpilogueT   = raft::identity_op>
void single_build(const raft::resources& handle,
                  const all_neighbors_params& params,
                  DatasetMdspan dataset,
                  IndicesMdspan indices,
                  std::optional<DistancesMdspan> distances = std::nullopt,
                  DistEpilogueT dist_epilogue              = DistEpilogueT{})
{
  size_t num_rows = static_cast<size_t>(dataset.extent(0));
  size_t num_cols = static_cast<size_t>(dataset.extent(1));
  size_t k        = indices.extent(1);

  constexpr bool indices_are_host = raft::is_output_host_mdspan_t<IndicesMdspan>::value;

  // Builder requires device-side arrays. Allocate device buffers when user provides host arrays.
  std::optional<raft::device_matrix<IdxT, IdxT>> indices_d;
  std::optional<raft::device_matrix<T, IdxT>> distances_d;

  auto indices_d_view = [&]() -> raft::device_matrix_view<IdxT, IdxT> {
    if constexpr (indices_are_host) {
      indices_d.emplace(raft::make_device_matrix<IdxT, IdxT>(handle, num_rows, k));
      return indices_d.value().view();
    } else {
      return raft::make_device_matrix_view<IdxT, IdxT>(indices.data_handle(), num_rows, k);
    }
  }();

  auto distances_d_opt_view = [&]() -> std::optional<raft::device_matrix_view<T, IdxT>> {
    if constexpr (indices_are_host) {
      if (distances.has_value()) {
        distances_d.emplace(raft::make_device_matrix<T, IdxT>(handle, num_rows, k));
        return distances_d.value().view();
      }
    } else {
      return distances;
    }
    return std::nullopt;
  }();

  auto knn_builder = get_knn_builder<T, IdxT>(handle,
                                              params,
                                              num_rows,
                                              num_rows,
                                              indices.extent(1),
                                              indices_d_view,
                                              distances_d_opt_view,
                                              dist_epilogue);

  knn_builder->prepare_build(dataset);
  knn_builder->build_knn(dataset);

  if constexpr (indices_are_host) {
    raft::copy(indices.data_handle(),
               indices_d_view.data_handle(),
               num_rows * k,
               raft::resource::get_cuda_stream(handle));
    if (distances.has_value()) {
      raft::copy(distances.value().data_handle(),
                 distances_d.value().data_handle(),
                 num_rows * k,
                 raft::resource::get_cuda_stream(handle));
    }
    raft::resource::sync_stream(handle);
  }
}

template <typename T,
          typename IdxT,
          typename IndicesMdspan,
          typename DistancesMdspan = raft::device_matrix_view<T, IdxT, row_major>,
          typename CoreDistMdspan  = raft::device_vector_view<T, IdxT, row_major>>
void build(const raft::resources& handle,
           const all_neighbors_params& params,
           raft::host_matrix_view<const T, IdxT, row_major> dataset,
           IndicesMdspan indices,
           std::optional<DistancesMdspan> distances     = std::nullopt,
           std::optional<CoreDistMdspan> core_distances = std::nullopt,
           T alpha                                      = 1.0)
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

  constexpr bool outputs_are_host = raft::is_output_host_mdspan_t<IndicesMdspan>::value;

  std::unique_ptr<BatchBuildAux<IdxT>> aux_vectors;
  if (params.n_clusters == 1) {
    single_build<T, IdxT>(handle, params, dataset, indices, distances);
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
    if constexpr (outputs_are_host) {
      host_shift_columns(indices, 1, std::nullopt);  // fill with row ID
      if (distances.has_value()) {
        host_shift_columns(distances.value(), 1, std::make_optional<T>(0.0));
      }
    } else {
      auto indices_d = raft::make_device_matrix_view<IdxT, IdxT>(
        const_cast<IdxT*>(indices.data_handle()), indices.extent(0), indices.extent(1));
      raft::matrix::shift(handle, indices_d, 1);
      if (distances.has_value()) {
        auto distances_d =
          raft::make_device_matrix_view<T, IdxT>(const_cast<T*>(distances.value().data_handle()),
                                                 distances.value().extent(0),
                                                 distances.value().extent(1));
        raft::matrix::shift(handle, distances_d, 1, std::make_optional<T>(0.0));
      }
    }
  }

  if (core_distances.has_value()) {  // calculate mutual reachability distances
    size_t k        = indices.extent(1);
    size_t num_rows = core_distances.value().size();

    std::optional<raft::device_vector<T, IdxT>> core_distances_tmp;
    T* core_distances_d_ptr;
    if constexpr (outputs_are_host) {
      // Core distance is the last column (index k-1) of distances
#pragma omp parallel for
      for (size_t i = 0; i < num_rows; i++) {
        core_distances.value()(i) = distances.value()(i, k - 1);
      }
      core_distances_tmp.emplace(raft::make_device_vector<T, IdxT>(handle, num_rows));
      core_distances_d_ptr = core_distances_tmp.value().data_handle();
      raft::copy(core_distances_d_ptr,
                 core_distances.value().data_handle(),
                 num_rows,
                 raft::resource::get_cuda_stream(handle));
    } else {
      core_distances_d_ptr = core_distances.value().data_handle();
      cuvs::neighbors::detail::reachability::core_distances<IdxT, T>(
        handle, distances.value().data_handle(), k, k, num_rows, core_distances_d_ptr);
    }

    using ReachabilityPP = cuvs::neighbors::detail::reachability::ReachabilityPostProcess<IdxT, T>;
    auto dist_epilogue   = ReachabilityPP{core_distances_d_ptr, alpha, num_rows};
    if (params.n_clusters == 1) {
      single_build<T, IdxT>(handle, params, dataset, indices, distances, dist_epilogue);
    } else {
      batch_build(handle, params, dataset, indices, distances, aux_vectors.get(), dist_epilogue);
    }

    if (need_shift) {
      if constexpr (outputs_are_host) {
        host_shift_columns(indices, 1, std::nullopt);
        size_t n_cols = distances.value().extent(1);
#pragma omp parallel for
        for (size_t i = 0; i < num_rows; i++) {
          for (size_t j = n_cols - 1; j >= 1; j--) {
            distances.value()(i, j) = distances.value()(i, j - 1);
          }
          distances.value()(i, 0) = core_distances.value()(i);
        }
      } else {
        auto indices_d = raft::make_device_matrix_view<IdxT, IdxT>(
          const_cast<IdxT*>(indices.data_handle()), indices.extent(0), indices.extent(1));
        raft::matrix::shift(handle, indices_d, 1);
        auto distances_d =
          raft::make_device_matrix_view<T, IdxT>(const_cast<T*>(distances.value().data_handle()),
                                                 distances.value().extent(0),
                                                 distances.value().extent(1));
        raft::matrix::shift(handle,
                            distances_d,
                            raft::make_device_matrix_view<const T, IdxT>(
                              core_distances.value().data_handle(), num_rows, 1));
      }
    }
  }
}

template <typename T,
          typename IdxT,
          typename IndicesMdspan,
          typename DistancesMdspan = raft::device_matrix_view<T, IdxT, row_major>,
          typename CoreDistMdspan  = raft::device_vector_view<T, IdxT, row_major>>
void build(const raft::resources& handle,
           const all_neighbors_params& params,
           raft::device_matrix_view<const T, IdxT, row_major> dataset,
           IndicesMdspan indices,
           std::optional<DistancesMdspan> distances     = std::nullopt,
           std::optional<CoreDistMdspan> core_distances = std::nullopt,
           T alpha                                      = 1.0)
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
    single_build<T, IdxT>(handle, params, dataset, indices, distances);
  }

  // NN Descent doesn't include self loops. Shifted to keep it consistent with brute force and ivfpq
  bool need_shift = (build_algo == GRAPH_BUILD_ALGO::NN_DESCENT) &&
                    (params.metric != cuvs::distance::DistanceType::InnerProduct);

  constexpr bool outputs_are_host = raft::is_output_host_mdspan_t<IndicesMdspan>::value;
  if (need_shift) {
    if constexpr (outputs_are_host) {
      host_shift_columns(indices, 1, std::nullopt);  // fill first column with row ID
      if (distances.has_value()) {                   // fill first column with 0
        host_shift_columns(distances.value(), 1, std::make_optional<T>(0.0));
      }
    } else {
      auto indices_d = raft::make_device_matrix_view<IdxT, IdxT>(
        const_cast<IdxT*>(indices.data_handle()), indices.extent(0), indices.extent(1));
      raft::matrix::shift(handle, indices_d, 1);
      if (distances.has_value()) {
        auto distances_d =
          raft::make_device_matrix_view<T, IdxT>(const_cast<T*>(distances.value().data_handle()),
                                                 distances.value().extent(0),
                                                 distances.value().extent(1));
        raft::matrix::shift(handle, distances_d, 1, std::make_optional<T>(0.0));
      }
    }
  }

  if (core_distances.has_value()) {
    size_t k        = indices.extent(1);
    size_t num_rows = core_distances.value().size();

    std::optional<raft::device_vector<T, IdxT>> core_distances_tmp;
    T* core_distances_d_ptr;
    if constexpr (outputs_are_host) {
#pragma omp parallel for
      for (size_t i = 0; i < num_rows; i++) {
        core_distances.value()(i) = distances.value()(i, k - 1);
      }
      core_distances_tmp.emplace(raft::make_device_vector<T, IdxT>(handle, num_rows));
      core_distances_d_ptr = core_distances_tmp.value().data_handle();
      raft::copy(core_distances_d_ptr,
                 core_distances.value().data_handle(),
                 num_rows,
                 raft::resource::get_cuda_stream(handle));
    } else {
      core_distances_d_ptr = core_distances.value().data_handle();
      cuvs::neighbors::detail::reachability::core_distances<IdxT, T>(
        handle, distances.value().data_handle(), k, k, num_rows, core_distances_d_ptr);
    }

    using ReachabilityPP = cuvs::neighbors::detail::reachability::ReachabilityPostProcess<IdxT, T>;
    auto dist_epilogue   = ReachabilityPP{core_distances_d_ptr, alpha, num_rows};
    single_build<T, IdxT>(handle, params, dataset, indices, distances, dist_epilogue);

    if (need_shift) {
      if constexpr (outputs_are_host) {
        host_shift_columns(indices, 1, std::nullopt);
        size_t n_cols = distances.value().extent(1);
#pragma omp parallel for
        for (size_t i = 0; i < num_rows; i++) {
          for (size_t j = n_cols - 1; j >= 1; j--) {
            distances.value()(i, j) = distances.value()(i, j - 1);
          }
          distances.value()(i, 0) = core_distances.value()(i);
        }
      } else {
        auto indices_d = raft::make_device_matrix_view<IdxT, IdxT>(
          const_cast<IdxT*>(indices.data_handle()), indices.extent(0), indices.extent(1));
        raft::matrix::shift(handle, indices_d, 1);
        auto distances_d =
          raft::make_device_matrix_view<T, IdxT>(const_cast<T*>(distances.value().data_handle()),
                                                 distances.value().extent(0),
                                                 distances.value().extent(1));
        raft::matrix::shift(handle,
                            distances_d,
                            raft::make_device_matrix_view<const T, IdxT>(
                              core_distances.value().data_handle(), num_rows, 1));
      }
    }
  }
}
}  // namespace cuvs::neighbors::all_neighbors::detail
