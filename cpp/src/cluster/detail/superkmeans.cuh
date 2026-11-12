/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Leonardo Xavier Kuffo Rivero
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/*
 * Copyright (c) 2025 Leonardo Xavier Kuffo Rivero
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file thirdparty/LICENSES/LICENSE.superkmeans
 */
#pragma once

#include "../../distance/pruned_top_1_nn.cuh"
#include "../../neighbors/detail/ann_utils.cuh"
#include "../kmeans_balanced.cuh"

#include <cuvs/cluster/kmeans.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>
#include <raft/linalg/map_then_reduce.cuh>
#include <raft/matrix/sample_rows.cuh>
#include <raft/random/rng_state.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <type_traits>

namespace cuvs::cluster::kmeans::detail::superkmeans {

namespace pruned_nn = cuvs::distance::detail::pruned_nn;

// Pruning works on whole dimension tiles past the first filter's, and only pays off with enough
// centroids to discard; below that every iteration assigns with the full 1-NN.
constexpr int64_t kMinPrunedK   = 256;
constexpr int64_t kMinPrunedDim = 1024;

inline bool prunes(int64_t n_clusters, int64_t dim)
{
  return dim > kMinPrunedDim && n_clusters > kMinPrunedK;
}

/** Budget the per-stream survivor lists and filter inputs together, as in balanced prediction. */
template <typename T>
auto calc_minibatch_size(const raft::resources& handle, int64_t n_rows) -> int64_t
{
  std::size_t mem_per_row = pruned_nn::kCentroidBatchSize * sizeof(int64_t) + sizeof(int32_t);
  if constexpr (!std::is_same_v<T, half>) { mem_per_row += pruned_nn::kPartialDim * sizeof(half); }
  const auto pool_size    = raft::resource::is_stream_pool_initialized(handle)
                              ? static_cast<int64_t>(raft::resource::get_stream_pool_size(handle))
                              : 0;
  const auto n_streams    = std::min({1 + pool_size, pruned_nn::kMaxStreams, n_rows});
  const auto free_ws_size = raft::resource::get_workspace_free_bytes(handle);
  const auto available_ws_size =
    std::min<std::size_t>((free_ws_size / std::size_t{10}) * std::size_t{8}, std::size_t{1} << 29);
  // Prefer whole 1K-row batches; under tight budgets retain float-buffer alignment.
  constexpr std::size_t kAlignedRows = pruned_nn::kAlignment / sizeof(float);
  auto batch_size         = std::max(kAlignedRows, available_ws_size / (n_streams * mem_per_row));
  const auto row_multiple = batch_size >= 1024 ? std::size_t{1024} : kAlignedRows;
  batch_size              = batch_size / row_multiple * row_multiple;
  return static_cast<int64_t>(std::min(batch_size, static_cast<std::size_t>(n_rows)));
}

/** Sum of squared distances of the rows to their assigned centroids. */
template <typename T>
auto inertia(const raft::resources& handle,
             raft::device_matrix_view<const T, int64_t> X,
             raft::device_matrix_view<const float, int64_t> centroids,
             raft::device_vector_view<const uint32_t, int64_t> labels) -> float
{
  auto dist           = raft::make_device_vector<float, int64_t>(handle, X.extent(0));
  cudaStream_t stream = raft::resource::get_cuda_stream(handle).get();
  pruned_nn::distance_to_assigned(handle,
                                  X.data_handle(),
                                  X.extent(0),
                                  centroids.data_handle(),
                                  X.extent(1),
                                  labels.data_handle(),
                                  dist.data_handle());
  auto sum = raft::make_device_scalar<float>(handle, 0.0f);
  raft::linalg::mapThenSumReduce(
    sum.data_handle(), dist.size(), raft::identity_op{}, stream, dist.data_handle());
  float out = 0;
  raft::copy(&out, sum.data_handle(), 1, stream);
  raft::resource::sync_stream(handle);
  return out;
}

template <typename T>
void validate(const cuvs::cluster::kmeans::superkmeans_params& params,
              raft::device_matrix_view<const T, int64_t> X,
              raft::device_matrix_view<const float, int64_t> centroids)
{
  RAFT_EXPECTS(params.metric == cuvs::distance::DistanceType::L2Expanded,
               "SuperKMeans supports only the L2Expanded metric");
  RAFT_EXPECTS(centroids.extent(0) > 0, "The number of centroids must be strictly positive");
  RAFT_EXPECTS(centroids.extent(1) == X.extent(1),
               "Number of features in dataset and centroids are different");
}

/** Each row's closest centroid over all dims. */
template <typename T>
void predict(const raft::resources& handle,
             const cuvs::cluster::kmeans::superkmeans_params& params,
             raft::device_matrix_view<const T, int64_t> X,
             raft::device_matrix_view<const float, int64_t> centroids,
             raft::device_vector_view<uint32_t, int64_t> labels)
{
  validate(params, X, centroids);
  RAFT_EXPECTS(labels.extent(0) == X.extent(0),
               "Number of rows in dataset and labels are different");
  cuvs::cluster::kmeans::balanced_params assign_params;
  assign_params.metric = params.metric;
  cuvs::cluster::kmeans_balanced::predict(handle,
                                          assign_params,
                                          X,
                                          centroids,
                                          labels,
                                          cuvs::spatial::knn::detail::utils::mapping<float>{});
}

/**
 * The pruned assignment (cuvs::distance::pruned_top_1_nn) and what it keeps across iterations: the
 * row norms, the row order, and the workspace. Float filter rows are prepared per batch.
 * The centroids the 1-NN reads are T: for half rows, a half copy of the float centroids, which is
 * also the filter's.
 */
template <typename T>
class pruned_assignment {
 public:
  pruned_assignment(const raft::resources& handle,
                    raft::device_matrix_view<const T, int64_t> X,
                    int64_t n_clusters)
    : handle_(handle),
      X_(X),
      n_rows_(X.extent(0)),
      dim_(X.extent(1)),
      n_clusters_(n_clusters),
      filter_dim_(std::is_same_v<T, half> ? dim_ : pruned_nn::kPartialDim),
      row_norms_(raft::make_device_vector<float, int64_t>(handle, n_rows_)),
      centroid_norms_(raft::make_device_vector<float, int64_t>(handle, n_clusters)),
      dist_(raft::make_device_vector<float, int64_t>(handle, n_rows_)),
      row_order_(raft::make_device_vector<uint32_t, int64_t>(handle, n_rows_)),
      filter_centroids_(raft::make_device_matrix<half, int64_t>(handle, n_clusters, filter_dim_)),
      params_{.row_batch_size = calc_minibatch_size<T>(handle, n_rows_)},
      workspace_(raft::make_device_vector<char, int64_t>(
        handle,
        cuvs::distance::pruned_top_1_nn_workspace_size<T>(handle, n_rows_, params_.row_batch_size)))
  {
    // The input rows do not change between iterations, so cache their filter norms once.
    pruned_nn::squared_norms(
      handle, X.data_handle(), n_rows_, dim_, pruned_nn::kPartialDim, row_norms_.data_handle());
  }

  /**
   * Updates `labels`, each row's centroid from the previous assignment, for `centroids`.
   * `order_next`: another call follows, so order the rows for it.
   */
  void assign(raft::device_matrix_view<const float, int64_t> centroids,
              raft::device_vector_view<uint32_t, int64_t> labels,
              bool order_next)
  {
    pruned_nn::squared_norms(handle_,
                             centroids.data_handle(),
                             n_clusters_,
                             dim_,
                             pruned_nn::kPartialDim,
                             centroid_norms_.data_handle(),
                             filter_centroids_.data_handle(),
                             filter_dim_);
    const T* kernel_centroids = nullptr;
    if constexpr (std::is_same_v<T, half>) {
      kernel_centroids = filter_centroids_.data_handle();
    } else {
      kernel_centroids = centroids.data_handle();
    }
    params_.next_row_order = order_next ? row_order_.data_handle() : nullptr;
    cuvs::distance::pruned_top_1_nn<T>(handle_,
                                       {labels.data_handle(), dist_.data_handle()},
                                       X_.data_handle(),
                                       kernel_centroids,
                                       filter_centroids_.data_handle(),
                                       row_norms_.data_handle(),
                                       centroid_norms_.data_handle(),
                                       n_rows_,
                                       n_clusters_,
                                       dim_,
                                       params_,
                                       workspace_.data_handle(),
                                       workspace_.size());
    if (order_next) { params_.row_order = row_order_.data_handle(); }
  }

 private:
  const raft::resources& handle_;
  raft::device_matrix_view<const T, int64_t> X_;
  int64_t n_rows_, dim_, n_clusters_, filter_dim_;
  raft::device_vector<float, int64_t> row_norms_, centroid_norms_, dist_;
  raft::device_vector<uint32_t, int64_t> row_order_;
  raft::device_matrix<half, int64_t> filter_centroids_;
  cuvs::distance::PrunedTop1nnParams params_;
  raft::device_vector<char, int64_t> workspace_;
};

/**
 * Lloyd's iterations from rows sampled at random, a fixed number of them. Each iteration assigns
 * every row (the full 1-NN when pruning does not apply) and sets each centroid to the mean of its
 * rows; an empty cluster keeps its centroid. `labels` and `inertia`, when given, are for the
 * returned centroids.
 */
template <typename T>
void fit(const raft::resources& handle,
         const cuvs::cluster::kmeans::superkmeans_params& params,
         raft::device_matrix_view<const T, int64_t> X,
         raft::device_matrix_view<float, int64_t> centroids,
         std::optional<raft::device_vector_view<uint32_t, int64_t>> labels,
         std::optional<raft::host_scalar_view<float>> inertia_out)
{
  const int64_t n_rows     = X.extent(0);
  const int64_t dim        = X.extent(1);
  const int64_t n_clusters = centroids.extent(0);
  validate(params, X, raft::make_const_mdspan(centroids));
  RAFT_EXPECTS(n_rows >= n_clusters, "SuperKMeans needs at least as many rows as clusters");
  RAFT_EXPECTS(!labels.has_value() || labels->extent(0) == n_rows,
               "Number of rows in dataset and labels are different");
  RAFT_EXPECTS(params.n_iters > 0, "SuperKMeans needs at least one iteration");
  const bool pruned = prunes(n_clusters, dim);
  if (pruned) {
    // Validate before sampling centroids or allocating scratch: eligible inputs must use pruning.
    RAFT_EXPECTS(
      dim > pruned_nn::kPartialDim && (dim - pruned_nn::kPartialDim) % pruned_nn::kDimTileSize == 0,
      "SuperKMeans pruning needs a positive tail after %u dims in tiles of %u",
      pruned_nn::kPartialDim,
      pruned_nn::kDimTileSize);
    const auto aligned = [](const void* ptr) {
      return reinterpret_cast<uintptr_t>(ptr) % pruned_nn::kAlignment == 0;
    };
    RAFT_EXPECTS(aligned(X.data_handle()) && aligned(centroids.data_handle()),
                 "SuperKMeans pruning needs 16-byte-aligned dataset and centroid pointers");
    RAFT_EXPECTS(cuvs::distance::is_pruned_top_1_nn_available(dim),
                 "SuperKMeans pruning kernel is unavailable on this build or device");
  }

  auto stream     = raft::resource::get_cuda_stream(handle);
  auto mapping_op = cuvs::spatial::knn::detail::utils::mapping<float>{};

  // The start centroids: rows sampled at random
  raft::random::RngState random_state(params.rng_state.seed);
  if constexpr (std::is_same_v<T, float>) {
    raft::matrix::sample_rows(handle, random_state, X, centroids);
  } else {
    auto start = raft::make_device_matrix<T, int64_t>(handle, n_clusters, dim);
    raft::matrix::sample_rows(handle, random_state, X, start.view());
    raft::linalg::map(handle, centroids, mapping_op, raft::make_const_mdspan(start.view()));
  }

  auto assigned = raft::make_device_vector<uint32_t, int64_t>(handle, n_rows);
  auto sizes    = raft::make_device_vector<uint32_t, int64_t>(handle, n_clusters);
  auto means    = raft::make_device_matrix<float, int64_t>(handle, n_clusters, dim);
  auto assign   = [&]() {
    predict(handle, params, X, raft::make_const_mdspan(centroids), assigned.view());
  };

  std::optional<pruned_assignment<T>> pruned_assign;
  if (pruned) { pruned_assign.emplace(handle, X, n_clusters); }

  for (uint32_t iter = 0; iter < params.n_iters; iter++) {
    if (pruned_assign.has_value() && iter > 0) {
      // The last pruned call orders no rows: the final labels come from a full assignment
      pruned_assign->assign(
        raft::make_const_mdspan(centroids), assigned.view(), iter + 1 < params.n_iters);
    } else {
      assign();
    }
    cuvs::cluster::kmeans_balanced::helpers::calc_centers_and_sizes(
      handle,
      X,
      raft::make_const_mdspan(assigned.view()),
      means.view(),
      sizes.view(),
      true,
      mapping_op);
    const float* m    = means.data_handle();
    const float* c    = centroids.data_handle();
    const uint32_t* z = sizes.data_handle();
    raft::linalg::map_offset(
      handle, centroids, [=] __device__(int64_t i) { return z[i / dim] ? m[i] : c[i]; });
  }

  if (labels.has_value() || inertia_out.has_value()) {
    assign();
    if (inertia_out.has_value()) {
      *inertia_out->data_handle() = inertia(
        handle, X, raft::make_const_mdspan(centroids), raft::make_const_mdspan(assigned.view()));
    }
    if (labels.has_value()) {
      raft::copy(labels->data_handle(), assigned.data_handle(), n_rows, stream);
    }
  }
}

}  // namespace cuvs::cluster::kmeans::detail::superkmeans
