/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../../neighbors/detail/ann_utils.cuh"
#include "../kmeans_balanced.cuh"

#include <cuvs/cluster/kmeans.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>
#include <raft/matrix/sample_rows.cuh>
#include <raft/random/rng_state.hpp>
#include <raft/util/cudart_utils.hpp>

#include <thrust/reduce.h>

#include <algorithm>
#include <optional>
#include <type_traits>

namespace cuvs::cluster::kmeans::detail::superkmeans {

// Pruning works on whole dimension tiles and only pays off with enough centroids to discard; below
// that every iteration assigns with the full 1-NN.
constexpr int64_t kDimTileSize = 256;
constexpr int64_t kMinPrunedK  = 256;

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
  const int64_t dim = X.extent(1);
  auto dist         = raft::make_device_vector<float, int64_t>(handle, X.extent(0));
  const T* x        = X.data_handle();
  const float* c    = centroids.data_handle();
  const uint32_t* l = labels.data_handle();
  raft::linalg::map_offset(handle, dist.view(), [=] __device__(int64_t row) {
    const float* center = c + static_cast<int64_t>(l[row]) * dim;
    float sum           = 0;
    for (int64_t j = 0; j < dim; j++) {
      const float diff = static_cast<float>(x[row * dim + j]) - center[j];
      sum += diff * diff;
    }
    return sum;
  });
  return thrust::reduce(raft::resource::get_thrust_policy(handle),
                        dist.data_handle(),
                        dist.data_handle() + dist.size(),
                        0.0f);
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
  if (prunes(n_clusters, dim)) {
    RAFT_FAIL("SuperKMeans with pruning (dim >= %ld and n_clusters > %ld) is not implemented yet",
              kDimTileSize,
              kMinPrunedK);
  }

  auto stream     = raft::resource::get_cuda_stream(handle);
  auto mapping_op = cuvs::spatial::knn::detail::utils::mapping<float>{};

  // The start centroids: rows sampled at random, the same draw as kmeans::fit's Random init.
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

  for (uint32_t iter = 0; iter < params.n_iters; iter++) {
    assign();
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
