/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.h"

#include <cuvs/cluster/kmeans.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/matrix/sample_rows.cuh>
#include <raft/random/make_blobs.cuh>
#include <raft/random/rng_state.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/cuda_stream_pool.hpp>

#include <cuda_fp16.h>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace cuvs {

namespace {

constexpr int64_t kRows     = 4000;
constexpr int64_t kDim      = 200;  // below one dimension tile: no pruning
constexpr int64_t kClusters = 300;

double sq_dist(const float* a, const float* b, int64_t dim)
{
  double s = 0;
  for (int64_t j = 0; j < dim; j++) {
    const double d = double(a[j]) - double(b[j]);
    s += d * d;
  }
  return s;
}

double sq_norm(const float* a, int64_t dim)
{
  double s = 0;
  for (int64_t j = 0; j < dim; j++) {
    s += double(a[j]) * a[j];
  }
  return s;
}

/**
 * Host SuperKMeans without pruning: Lloyd's from the given start centroids, an empty cluster
 * keeping its centroid.
 */
void reference_superkmeans(const std::vector<float>& X,
                           int64_t n_rows,
                           int64_t dim,
                           int64_t n_clusters,
                           uint32_t n_iters,
                           std::vector<float> start,
                           std::vector<float>& centroids,
                           std::vector<uint32_t>& labels,
                           double& inertia)
{
  centroids = std::move(start);
  labels.resize(n_rows);
  auto assign = [&]() {
    inertia = 0;
    for (int64_t r = 0; r < n_rows; r++) {
      double best = std::numeric_limits<double>::max();
      for (int64_t c = 0; c < n_clusters; c++) {
        const double d = sq_dist(&X[r * dim], &centroids[c * dim], dim);
        if (d < best) {
          best      = d;
          labels[r] = c;
        }
      }
      inertia += best;
    }
  };
  std::vector<double> sums(n_clusters * dim);
  std::vector<uint32_t> sizes(n_clusters);
  for (uint32_t iter = 0; iter < n_iters; iter++) {
    assign();
    std::fill(sums.begin(), sums.end(), 0.0);
    std::fill(sizes.begin(), sizes.end(), 0u);
    for (int64_t r = 0; r < n_rows; r++) {
      sizes[labels[r]]++;
      for (int64_t j = 0; j < dim; j++) {
        sums[labels[r] * dim + j] += X[r * dim + j];
      }
    }
    for (int64_t c = 0; c < n_clusters; c++) {
      if (sizes[c] == 0) { continue; }
      for (int64_t j = 0; j < dim; j++) {
        centroids[c * dim + j] = float(sums[c * dim + j] / sizes[c]);
      }
    }
  }
  assign();
}

/** Well separated blobs as T, and the same values as float on the host. */
template <typename T>
auto make_data(raft::handle_t& handle, std::vector<float>& host)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle).get();
  auto blobs          = raft::make_device_matrix<float, int64_t>(handle, kRows, kDim);
  auto blob_labels    = raft::make_device_vector<int64_t, int64_t>(handle, kRows);
  raft::random::make_blobs<float, int64_t>(blobs.data_handle(),
                                           blob_labels.data_handle(),
                                           kRows,
                                           kDim,
                                           kClusters,
                                           stream,
                                           true,
                                           nullptr,
                                           nullptr,
                                           0.1f,
                                           true,
                                           -1.0f,
                                           1.0f,
                                           uint64_t{1234});
  auto X = raft::make_device_matrix<T, int64_t>(handle, kRows, kDim);
  raft::linalg::unaryOp(
    X.data_handle(), blobs.data_handle(), kRows * kDim, raft::cast_op<T>(), stream);
  raft::linalg::unaryOp(
    blobs.data_handle(), X.data_handle(), kRows * kDim, raft::cast_op<float>(), stream);
  host.resize(kRows * kDim);
  raft::copy(host.data(), blobs.data_handle(), host.size(), stream);
  raft::resource::sync_stream(handle);
  return X;
}

/** The start centroids SuperKMeans draws for `seed`: rows sampled with raft::matrix::sample_rows.
 */
std::vector<float> start_centroids(raft::handle_t& handle,
                                   const std::vector<float>& host,
                                   uint64_t seed)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle).get();
  auto X              = raft::make_device_matrix<float, int64_t>(handle, kRows, kDim);
  auto start          = raft::make_device_matrix<float, int64_t>(handle, kClusters, kDim);
  raft::copy(X.data_handle(), host.data(), host.size(), stream);
  raft::matrix::sample_rows(
    handle, raft::random::RngState(seed), raft::make_const_mdspan(X.view()), start.view());
  std::vector<float> out(kClusters * kDim);
  raft::copy(out.data(), start.data_handle(), out.size(), stream);
  raft::resource::sync_stream(handle);
  return out;
}

template <typename T>
void fit_predict(raft::handle_t& handle,
                 raft::device_matrix_view<const T, int64_t> X,
                 uint32_t n_iters,
                 std::vector<float>& centroids,
                 std::vector<uint32_t>& labels,
                 float& inertia)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle).get();
  cuvs::cluster::kmeans::superkmeans_params params;
  params.n_iters   = n_iters;
  auto d_centroids = raft::make_device_matrix<float, int64_t>(handle, kClusters, kDim);
  auto d_labels    = raft::make_device_vector<uint32_t, int64_t>(handle, kRows);
  cuvs::cluster::kmeans::fit_predict(
    handle, params, X, d_centroids.view(), d_labels.view(), raft::make_host_scalar_view(&inertia));
  centroids.resize(kClusters * kDim);
  labels.resize(kRows);
  raft::copy(centroids.data(), d_centroids.data_handle(), centroids.size(), stream);
  raft::copy(labels.data(), d_labels.data_handle(), labels.size(), stream);
  raft::resource::sync_stream(handle);
}

}  // namespace

/* Without pruning SuperKMeans is Lloyd's algorithm from random start rows. The GPU assignment
 * rounds its GEMM operands to TF32, so a run on float data drifts from an exact reference once
 * near-ties go the other way: the test compares against the reference where that cannot happen,
 * and checks the invariants of Lloyd's output otherwise.
 */
TEST(KmeansSuperKMeans, Fallback)
{
  raft::handle_t handle;

  // Half values are exact in TF32 and one iteration assigns against the start rows, so the start
  // rows, the assignment and the update must match the reference.
  {
    std::vector<float> h_X, centroids, ref_centroids;
    std::vector<uint32_t> labels, ref_labels;
    float inertia;
    double ref_inertia;
    auto X = make_data<half>(handle, h_X);
    fit_predict<half>(handle, raft::make_const_mdspan(X.view()), 1, centroids, labels, inertia);
    reference_superkmeans(
      h_X,
      kRows,
      kDim,
      kClusters,
      1,
      start_centroids(handle, h_X, cuvs::cluster::kmeans::superkmeans_params{}.rng_state.seed),
      ref_centroids,
      ref_labels,
      ref_inertia);
    for (size_t i = 0; i < centroids.size(); i++) {
      ASSERT_NEAR(centroids[i], ref_centroids[i], 1e-5f) << "centroid " << i / kDim;
    }
    int64_t same_label = 0;
    for (int64_t r = 0; r < kRows; r++) {
      same_label += labels[r] == ref_labels[r];
    }
    EXPECT_GE(double(same_label) / kRows, 0.999);
    EXPECT_NEAR(inertia, ref_inertia, 1e-4 * ref_inertia);
  }

  // Float data, several iterations: each row's label is its nearest centroid (up to the TF32
  // rounding of the distances), the inertia is that of the labels, and each centroid is the mean of
  // its rows.
  {
    constexpr uint32_t n_iters = 5;
    std::vector<float> h_X, centroids;
    std::vector<uint32_t> labels;
    float inertia;
    auto X = make_data<float>(handle, h_X);
    fit_predict<float>(
      handle, raft::make_const_mdspan(X.view()), n_iters, centroids, labels, inertia);

    double host_inertia = 0;
    for (int64_t r = 0; r < kRows; r++) {
      const float* row = &h_X[r * kDim];
      const float* own = &centroids[int64_t(labels[r]) * kDim];
      double best      = std::numeric_limits<double>::max();
      for (int64_t c = 0; c < kClusters; c++) {
        best = std::min(best, sq_dist(row, &centroids[c * kDim], kDim));
      }
      const double own_dist = sq_dist(row, own, kDim);
      ASSERT_LE(own_dist, best + 1e-3 * (sq_norm(row, kDim) + sq_norm(own, kDim))) << "row " << r;
      host_inertia += own_dist;
    }
    EXPECT_NEAR(inertia, host_inertia, 1e-4 * host_inertia);

    std::vector<double> sums(kClusters * kDim, 0.0);
    std::vector<int64_t> sizes(kClusters, 0);
    for (int64_t r = 0; r < kRows; r++) {
      sizes[labels[r]]++;
      for (int64_t j = 0; j < kDim; j++) {
        sums[int64_t(labels[r]) * kDim + j] += h_X[r * kDim + j];
      }
    }
    for (int64_t c = 0; c < kClusters; c++) {
      if (sizes[c] == 0) { continue; }
      for (int64_t j = 0; j < kDim; j++) {
        ASSERT_NEAR(centroids[c * kDim + j], sums[c * kDim + j] / sizes[c], 1e-4)
          << "centroid " << c;
      }
    }

    // predict on the returned centroids gives fit_predict's labels.
    auto d_centroids_fp = raft::make_device_matrix<float, int64_t>(handle, kClusters, kDim);
    auto d_labels       = raft::make_device_vector<uint32_t, int64_t>(handle, kRows);
    cudaStream_t stream = raft::resource::get_cuda_stream(handle).get();
    raft::copy(d_centroids_fp.data_handle(), centroids.data(), centroids.size(), stream);
    cuvs::cluster::kmeans::predict(handle,
                                   cuvs::cluster::kmeans::superkmeans_params{},
                                   raft::make_const_mdspan(X.view()),
                                   raft::make_const_mdspan(d_centroids_fp.view()),
                                   d_labels.view());
    std::vector<uint32_t> predicted(kRows);
    raft::copy(predicted.data(), d_labels.data_handle(), kRows, stream);
    raft::resource::sync_stream(handle);
    EXPECT_EQ(predicted, labels);

    // fit runs the same iterations; its centroids may differ in the last bits (atomic sums).
    cuvs::cluster::kmeans::superkmeans_params params;
    params.n_iters    = n_iters;
    auto d_centroids  = raft::make_device_matrix<float, int64_t>(handle, kClusters, kDim);
    float fit_inertia = 0;
    cuvs::cluster::kmeans::fit(handle,
                               params,
                               raft::make_const_mdspan(X.view()),
                               d_centroids.view(),
                               raft::make_host_scalar_view(&fit_inertia));
    EXPECT_NEAR(fit_inertia, inertia, 1e-3 * inertia);
  }
}

/* With more than 256 clusters and a multiple of 256 dims, the iterations after the first prune
 * (on a build and device with the filter's kernels; otherwise they assign in full). The pruning
 * bound is a heuristic without a rotation, so the result is compared with Lloyd's from the same
 * start rows rather than exactly. The labels and inertia come from a full assignment.
 */
TEST(KmeansSuperKMeans, Pruned)
{
  // Three row batches, the last partial; two centroid batches, the second partial; and two tail
  // dim tiles (one re-filter)
  constexpr int64_t n_rows     = 300000;
  constexpr int64_t dim        = 768;
  constexpr int64_t n_clusters = 1100;
  constexpr uint32_t n_iters   = 8;
  constexpr uint64_t seed      = 42;
  raft::handle_t handle;
  cudaStream_t stream = raft::resource::get_cuda_stream(handle).get();

  auto blobs       = raft::make_device_matrix<float, int64_t>(handle, n_rows, dim);
  auto blob_labels = raft::make_device_vector<int64_t, int64_t>(handle, n_rows);
  raft::random::make_blobs<float, int64_t>(blobs.data_handle(),
                                           blob_labels.data_handle(),
                                           n_rows,
                                           dim,
                                           n_clusters,
                                           stream,
                                           true,
                                           nullptr,
                                           nullptr,
                                           1.0f,
                                           true,
                                           -2.0f,
                                           2.0f,
                                           uint64_t{1234});
  auto X_half = raft::make_device_matrix<half, int64_t>(handle, n_rows, dim);
  raft::linalg::unaryOp(
    X_half.data_handle(), blobs.data_handle(), n_rows * dim, raft::cast_op<half>(), stream);
  // Lloyd's runs on the half values too, so both data types are compared with the same reference
  raft::linalg::unaryOp(
    blobs.data_handle(), X_half.data_handle(), n_rows * dim, raft::cast_op<float>(), stream);
  auto X = raft::make_const_mdspan(blobs.view());

  // Lloyd's from SuperKMeans' start rows (its own Random init draws other rows for the same seed)
  auto centroids = raft::make_device_matrix<float, int64_t>(handle, n_clusters, dim);
  raft::matrix::sample_rows(handle, raft::random::RngState(seed), X, centroids.view());
  cuvs::cluster::kmeans::params lloyd;
  lloyd.n_clusters    = n_clusters;
  lloyd.init          = cuvs::cluster::kmeans::params::Array;
  lloyd.max_iter      = n_iters;
  lloyd.tol           = 1e-30;  // must be positive; this small, all iterations run
  float lloyd_inertia = 0;
  int64_t lloyd_iters = 0;
  cuvs::cluster::kmeans::fit(handle,
                             lloyd,
                             X,
                             std::nullopt,
                             centroids.view(),
                             raft::make_host_scalar_view(&lloyd_inertia),
                             raft::make_host_scalar_view(&lloyd_iters));

  cuvs::cluster::kmeans::superkmeans_params params;
  params.n_iters   = n_iters;
  params.rng_state = raft::random::RngState(seed);
  auto labels      = raft::make_device_vector<uint32_t, int64_t>(handle, n_rows);
  float inertia    = 0;
  cuvs::cluster::kmeans::fit_predict(
    handle, params, X, centroids.view(), labels.view(), raft::make_host_scalar_view(&inertia));
  EXPECT_NEAR(inertia, lloyd_inertia, 0.01 * lloyd_inertia);

  // With a stream pool the row batches run on two streams
  raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(2));
  float pooled_inertia = 0;
  cuvs::cluster::kmeans::fit_predict(handle,
                                     params,
                                     X,
                                     centroids.view(),
                                     labels.view(),
                                     raft::make_host_scalar_view(&pooled_inertia));
  EXPECT_NEAR(pooled_inertia, lloyd_inertia, 0.01 * lloyd_inertia);

  float half_inertia = 0;
  cuvs::cluster::kmeans::fit_predict(handle,
                                     params,
                                     raft::make_const_mdspan(X_half.view()),
                                     centroids.view(),
                                     labels.view(),
                                     raft::make_host_scalar_view(&half_inertia));
  EXPECT_NEAR(half_inertia, lloyd_inertia, 0.01 * lloyd_inertia);
}

TEST(KmeansSuperKMeans, RejectsUnsupported)
{
  raft::handle_t handle;
  cuvs::cluster::kmeans::superkmeans_params params;
  // Pruning works on whole 256-dim tiles.
  auto X_odd     = raft::make_device_matrix<float, int64_t>(handle, 1000, 384);
  auto centroids = raft::make_device_matrix<float, int64_t>(handle, 300, 384);
  EXPECT_THROW(cuvs::cluster::kmeans::fit(
                 handle, params, raft::make_const_mdspan(X_odd.view()), centroids.view()),
               raft::logic_error);

  auto X               = raft::make_device_matrix<float, int64_t>(handle, 1000, 256);
  auto small_centroids = raft::make_device_matrix<float, int64_t>(handle, 8, 256);
  params.metric        = cuvs::distance::DistanceType::InnerProduct;
  EXPECT_THROW(cuvs::cluster::kmeans::fit(
                 handle, params, raft::make_const_mdspan(X.view()), small_centroids.view()),
               raft::logic_error);

  params.metric = cuvs::distance::DistanceType::L2Expanded;
  auto too_many = raft::make_device_matrix<float, int64_t>(handle, 1001, 256);
  EXPECT_THROW(
    cuvs::cluster::kmeans::fit(handle, params, raft::make_const_mdspan(X.view()), too_many.view()),
    raft::logic_error);
}

}  // namespace cuvs
