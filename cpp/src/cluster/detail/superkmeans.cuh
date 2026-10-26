/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../../neighbors/detail/ann_utils.cuh"
#include "../kmeans_balanced.cuh"
#include "superkmeans/cutile/fused_filter_tile.hpp"
#include "superkmeans/prune.cuh"

#include <cuvs/cluster/kmeans.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>
#include <raft/linalg/map_then_reduce.cuh>
#include <raft/matrix/sample_rows.cuh>
#include <raft/random/rng_state.hpp>
#include <raft/util/cudart_utils.hpp>

#include <algorithm>
#include <optional>
#include <type_traits>

namespace cuvs::cluster::kmeans::detail::superkmeans {

// Pruning works on whole dimension tiles and only pays off with enough centroids to discard; below
// that every iteration assigns with the full 1-NN.
constexpr int64_t kMinPrunedK = 256;

inline bool prunes(int64_t n_clusters, int64_t dim)
{
  return dim >= kDimTileSize && n_clusters > kMinPrunedK;
}

/** Sum of squared distances of the rows to their assigned centroids. */
template <typename T>
auto inertia(const raft::resources& handle,
             raft::device_matrix_view<const T, int64_t> X,
             raft::device_matrix_view<const float, int64_t> centroids,
             raft::device_vector_view<const uint32_t, int64_t> labels) -> float
{
  auto dist = raft::make_device_vector<float, int64_t>(handle, X.extent(0));
  distance_to_assigned(handle,
                       X.data_handle(),
                       X.extent(0),
                       centroids.data_handle(),
                       X.extent(1),
                       labels.data_handle(),
                       dist.data_handle());
  cudaStream_t stream = raft::resource::get_cuda_stream(handle).get();
  auto sum            = raft::make_device_scalar<float>(handle, 0.0f);
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

/** The fused first filter's tile on this device, or nullopt when the build or device has none. */
inline auto filter_tile() -> std::optional<fused_filter_tile>
{
#if CUVS_CUTILE_ENABLED
  return fused_filter_tile_config();
#else
  return std::nullopt;
#endif
}

/** The first `cols` dims of n rows (row stride `stride`) as half, packed (row stride `cols`). */
template <typename S>
void copy_first_columns(
  const raft::resources& handle, const S* in, int64_t n, int64_t stride, int64_t cols, half* out)
{
  raft::linalg::map_offset(
    handle, raft::make_device_vector_view(out, n * cols), [=] __device__(int64_t i) {
      return static_cast<half>(in[i / cols * stride + i % cols]);
    });
}

/**
 * The pruned assignment: from each row's current centroid, the fused partial GEMM over the first
 * kDimTileSize dims with the first filter, then the pruning kernel on the survivors' tail dims.
 * Rows go in batches of kRowBatchSize against centroid batches of kCentroidBatchSize. The filter
 * runs on FP16: float rows are filtered on half copies of their first dims, and the centroids the
 * kernels read are T, a half copy of the float centroids for half rows.
 */
template <typename T>
class pruned_assignment {
 public:
  pruned_assignment(const raft::resources& handle,
                    raft::device_matrix_view<const T, int64_t> X,
                    int64_t n_clusters,
                    fused_filter_tile tile)
    : handle_(handle),
      X_(X),
      n_rows_(X.extent(0)),
      dim_(X.extent(1)),
      n_clusters_(n_clusters),
      filter_dim_(std::is_same_v<T, half> ? dim_ : kDimTileSize),
      tile_(tile),
      n_col_tiles_(raft::ceildiv<int64_t>(kCentroidBatchSize, tile.tile_n)),
      ratios_(raft::make_device_vector<float, int64_t>(handle, dim_ + 1)),
      row_norms_(raft::make_device_vector<float, int64_t>(handle, n_rows_)),
      centroid_norms_(raft::make_device_vector<float, int64_t>(handle, n_clusters)),
      dist_(raft::make_device_vector<float, int64_t>(handle, n_rows_)),
      filter_rows_(raft::make_device_matrix<half, int64_t>(
        handle, std::is_same_v<T, half> ? 0 : n_rows_, filter_dim_)),
      filter_centroids_(raft::make_device_matrix<half, int64_t>(handle, n_clusters, filter_dim_)),
      packed_(raft::make_device_vector<int64_t, int64_t>(
        handle, std::min(n_rows_, kRowBatchSize) * n_col_tiles_ * tile.tile_n)),
      seg_count_(raft::make_device_vector<int32_t, int64_t>(
        handle, std::min(n_rows_, kRowBatchSize) * n_col_tiles_))
  {
    const auto ratios = adsampling_ratios(dim_);
    ratio_            = ratios[kDimTileSize];
    raft::copy(
      ratios_.data_handle(), ratios.data(), ratios.size(), raft::resource::get_cuda_stream(handle));
    squared_norms(handle, X.data_handle(), n_rows_, dim_, kDimTileSize, row_norms_.data_handle());
    if constexpr (!std::is_same_v<T, half>) {
      copy_columns(X.data_handle(), n_rows_, filter_rows_.view());
    }
    // The host ratios are released when the copy is done
    raft::resource::sync_stream(handle);
  }

  /** Updates `labels`, each row's centroid from the previous assignment, for `centroids`. */
  void assign(raft::device_matrix_view<const float, int64_t> centroids,
              raft::device_vector_view<uint32_t, int64_t> labels)
  {
#if CUVS_CUTILE_ENABLED
    // The filter's FP16 copy of the centroids' first dims; for half rows, all dims, which is also
    // the copy the other kernels read
    copy_columns(centroids.data_handle(), n_clusters_, filter_centroids_.view());
    const T* kernel_centroids = nullptr;
    const half* filter_rows   = nullptr;
    if constexpr (std::is_same_v<T, half>) {
      kernel_centroids = filter_centroids_.data_handle();
      filter_rows      = X_.data_handle();
    } else {
      kernel_centroids = centroids.data_handle();
      filter_rows      = filter_rows_.data_handle();
    }
    squared_norms(
      handle_, kernel_centroids, n_clusters_, dim_, kDimTileSize, centroid_norms_.data_handle());

    for (int64_t i = 0; i < n_rows_; i += kRowBatchSize) {
      const int64_t n_rows = std::min(kRowBatchSize, n_rows_ - i);
      const T* rows        = X_.data_handle() + i * dim_;
      uint32_t* row_labels = labels.data_handle() + i;
      float* row_dist      = dist_.data_handle() + i;
      // Each row's bound: its distance to its current centroid
      distance_to_assigned(handle_, rows, n_rows, kernel_centroids, dim_, row_labels, row_dist);
      for (int64_t j = 0; j < n_clusters_; j += kCentroidBatchSize) {
        const int64_t n_centroids = std::min<int64_t>(kCentroidBatchSize, n_clusters_ - j);
        launch_fused_filter(handle_,
                            filter_rows + i * filter_dim_,
                            static_cast<int>(n_rows),
                            filter_centroids_.data_handle() + j * filter_dim_,
                            static_cast<int>(n_centroids),
                            static_cast<int>(filter_dim_),
                            row_norms_.data_handle() + i,
                            centroid_norms_.data_handle() + j,
                            row_dist,
                            ratio_,
                            static_cast<int>(kDimTileSize),
                            packed_.data_handle(),
                            seg_count_.data_handle());
        prune(handle_,
              rows,
              n_rows,
              kernel_centroids + j * dim_,
              static_cast<uint32_t>(j),
              dim_,
              kDimTileSize,
              ratios_.data_handle(),
              first_filter_lists{
                packed_.data_handle(),
                seg_count_.data_handle(),
                static_cast<uint32_t>(raft::ceildiv<int64_t>(n_centroids, tile_.tile_n)),
                static_cast<uint32_t>(tile_.tile_n)},
              row_labels,
              row_dist);
      }
    }
#else
    RAFT_FAIL("SuperKMeans pruning needs a build with cuTile");
#endif
  }

 private:
  /** The first filter_dim_ dims of n rows as half, packed into `out`. */
  template <typename S>
  void copy_columns(const S* in, int64_t n, raft::device_matrix_view<half, int64_t> out)
  {
    copy_first_columns(handle_, in, n, dim_, filter_dim_, out.data_handle());
  }

  const raft::resources& handle_;
  raft::device_matrix_view<const T, int64_t> X_;
  int64_t n_rows_, dim_, n_clusters_, filter_dim_;
  fused_filter_tile tile_;
  int64_t n_col_tiles_;
  float ratio_;
  raft::device_vector<float, int64_t> ratios_, row_norms_, centroid_norms_, dist_;
  raft::device_matrix<half, int64_t> filter_rows_, filter_centroids_;
  raft::device_vector<int64_t, int64_t> packed_;
  raft::device_vector<int32_t, int64_t> seg_count_;
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
  RAFT_EXPECTS(!prunes(n_clusters, dim) || dim % kDimTileSize == 0,
               "SuperKMeans with pruning (n_clusters > %ld) needs dim to be a multiple of %u",
               kMinPrunedK,
               kDimTileSize);
  // Without the filter's kernels on this build or device, every iteration assigns in full
  const auto tile   = prunes(n_clusters, dim) ? filter_tile() : std::nullopt;
  const bool pruned = tile.has_value();

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
  if (pruned) { pruned_assign.emplace(handle, X, n_clusters, *tile); }

  for (uint32_t iter = 0; iter < params.n_iters; iter++) {
    if (pruned_assign.has_value() && iter > 0) {
      pruned_assign->assign(raft::make_const_mdspan(centroids), assigned.view());
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
