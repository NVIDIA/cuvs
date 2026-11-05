/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include "kmeans_test_blobs.cuh"

#include <cuvs/cluster/kmeans.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <optional>
#include <vector>

namespace cuvs {

namespace {

struct fit_result {
  float inertia;
  int n_iter;
  std::vector<int> labels;  // from kmeans::predict with the same params
  float predict_inertia;
  std::vector<float> centroids;  // [n_clusters x n_cols], row-major
};

template <typename T>
struct ann_fixture {
  raft::resources handle;
  int n_rows;
  int n_cols;
  int n_clusters;
  kmeans_blob_inputs<T> data;

  ann_fixture(int n_rows_, int n_cols_, int n_clusters_)
    : n_rows(n_rows_),
      n_cols(n_cols_),
      n_clusters(n_clusters_),
      data(make_kmeans_blob_inputs<T>(handle, n_rows_, n_cols_, n_clusters_, true))
  {
  }

  cuvs::cluster::kmeans::params base_params() const
  {
    cuvs::cluster::kmeans::params p;
    p.n_clusters = n_clusters;
    p.max_iter   = 20;
    p.tol        = 1e-6;
    p.n_init     = 1;
    p.init       = cuvs::cluster::kmeans::params::Random;
    p.rng_state  = raft::random::RngState(7);
    return p;
  }

  fit_result run(cuvs::cluster::kmeans::params const& p,
                 bool host_batched                                       = false,
                 std::optional<raft::device_vector_view<const T, int>> w = std::nullopt)
  {
    auto stream    = raft::resource::get_cuda_stream(handle);
    auto centroids = raft::make_device_matrix<T, int>(handle, n_clusters, n_cols);
    T inertia      = 0;
    int n_iter     = 0;
    auto X = raft::make_device_matrix_view<const T, int>(data.d_X.data_handle(), n_rows, n_cols);
    if (host_batched) {
      auto hX =
        raft::make_host_matrix_view<const T, int64_t>(data.h_X->data_handle(), n_rows, n_cols);
      auto cview =
        raft::make_device_matrix_view<T, int64_t>(centroids.data_handle(), n_clusters, n_cols);
      int64_t n_iter64 = 0;
      cuvs::cluster::kmeans::fit(handle,
                                 p,
                                 hX,
                                 std::nullopt,
                                 cview,
                                 raft::make_host_scalar_view(&inertia),
                                 raft::make_host_scalar_view(&n_iter64));
      n_iter = static_cast<int>(n_iter64);
    } else {
      cuvs::cluster::kmeans::fit(handle,
                                 p,
                                 X,
                                 w,
                                 centroids.view(),
                                 raft::make_host_scalar_view(&inertia),
                                 raft::make_host_scalar_view(&n_iter));
    }
    auto labels    = raft::make_device_vector<int, int>(handle, n_rows);
    T pred_inertia = 0;
    cuvs::cluster::kmeans::predict(
      handle,
      p,
      X,
      w,
      raft::make_device_matrix_view<const T, int>(centroids.data_handle(), n_clusters, n_cols),
      labels.view(),
      false,
      raft::make_host_scalar_view(&pred_inertia));
    std::vector<int> h_labels(n_rows);
    raft::update_host(h_labels.data(), labels.data_handle(), n_rows, stream);
    raft::resource::sync_stream(handle, stream);
    std::vector<float> h_centroids(static_cast<size_t>(n_clusters) * n_cols);
    for (size_t i = 0; i < h_centroids.size(); ++i) {
      h_centroids[i] = 0;
    }
    std::vector<T> tmp(h_centroids.size());
    raft::update_host(tmp.data(), centroids.data_handle(), tmp.size(), stream);
    raft::resource::sync_stream(handle, stream);
    for (size_t i = 0; i < tmp.size(); ++i) {
      h_centroids[i] = static_cast<float>(tmp[i]);
    }
    return {
      static_cast<float>(inertia), n_iter, h_labels, static_cast<float>(pred_inertia), h_centroids};
  }
};

double agreement(std::vector<int> const& a, std::vector<int> const& b)
{
  size_t same = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    same += (a[i] == b[i]);
  }
  return static_cast<double>(same) / static_cast<double>(a.size());
}

}  // namespace

TEST(KmeansAnn, DisabledByDefault)
{
  cuvs::cluster::kmeans::params p;
  EXPECT_FALSE(p.use_ann_for_assignment);
}

TEST(KmeansAnn, FitMatchesBruteForceQuality)
{
  ann_fixture<float> f(20000, 32, 300);
  auto brute                 = f.base_params();
  auto ann                   = f.base_params();
  ann.use_ann_for_assignment = true;

  auto r_brute = f.run(brute);
  auto r_ann   = f.run(ann);
  EXPECT_LE(r_ann.inertia, r_brute.inertia * 1.02f)
    << "n_iter ann=" << r_ann.n_iter << " brute=" << r_brute.n_iter;
  EXPECT_LE(r_ann.predict_inertia, r_brute.predict_inertia * 1.02f);
  EXPECT_GE(agreement(r_ann.labels, r_brute.labels), 0.98);
  // Approximate assignments make the cost noisy; fit must not stop on the first cost uptick
  // (an early version stopped after 2 iterations where brute force needed 10).
  EXPECT_GE(r_ann.n_iter, r_brute.n_iter / 2);
}

// The inertia reported by the ANN predict uses the distances returned by the CAGRA search. They
// must equal the exact squared distances from each point to its assigned centroid.
TEST(KmeansAnn, ReportedInertiaMatchesAssignedLabels)
{
  ann_fixture<float> f(20000, 32, 300);
  auto ann                   = f.base_params();
  ann.use_ann_for_assignment = true;
  auto r                     = f.run(ann);

  const float* x = f.data.h_X->data_handle();
  double exact   = 0;
  for (int i = 0; i < f.n_rows; ++i) {
    const float* c = r.centroids.data() + static_cast<size_t>(r.labels[i]) * f.n_cols;
    for (int d = 0; d < f.n_cols; ++d) {
      const double diff = static_cast<double>(x[static_cast<size_t>(i) * f.n_cols + d]) - c[d];
      exact += diff * diff;
    }
  }
  EXPECT_NEAR(r.predict_inertia, exact, 1e-3 * exact);
}

TEST(KmeansAnn, UserTunableParams)
{
  ann_fixture<float> f(10000, 16, 200);
  auto brute                        = f.base_params();
  auto ann                          = f.base_params();
  ann.use_ann_for_assignment        = true;
  ann.ann_graph_degree              = 32;
  ann.ann_intermediate_graph_degree = 48;
  ann.ann_itopk_size                = 128;
  ann.ann_search_width              = 2;
  auto r_brute                      = f.run(brute);
  auto r_ann                        = f.run(ann);
  EXPECT_LE(r_ann.inertia, r_brute.inertia * 1.02f);
}

TEST(KmeansAnn, TinyNumberOfClustersClampsDegrees)
{
  for (int k : {2, 3, 5}) {
    ann_fixture<float> f(2000, 8, k);
    auto ann                   = f.base_params();
    ann.use_ann_for_assignment = true;  // default degrees (64/128) exceed k - 1
    auto r_ann                 = f.run(ann);
    auto r_brute               = f.run(f.base_params());
    EXPECT_LE(r_ann.inertia, r_brute.inertia * 1.02f) << "k=" << k;
  }
}

TEST(KmeansAnn, SingleClusterFallsBackToBruteForce)
{
  ann_fixture<float> f(1000, 8, 1);
  auto ann                   = f.base_params();
  ann.use_ann_for_assignment = true;
  auto r_ann                 = f.run(ann);
  auto r_brute               = f.run(f.base_params());
  EXPECT_NEAR(r_ann.inertia, r_brute.inertia, 1e-3f * r_brute.inertia);
}

TEST(KmeansAnn, DoubleFallsBackToBruteForce)
{
  ann_fixture<double> f(2000, 8, 20);
  auto ann                   = f.base_params();
  ann.use_ann_for_assignment = true;
  auto r_ann                 = f.run(ann);
  auto r_brute               = f.run(f.base_params());
  EXPECT_NEAR(r_ann.inertia, r_brute.inertia, 1e-6 * r_brute.inertia);
  EXPECT_EQ(r_ann.labels, r_brute.labels);
}

TEST(KmeansAnn, InvalidParamsThrow)
{
  ann_fixture<float> f(1000, 8, 10);
  auto p                   = f.base_params();
  p.use_ann_for_assignment = true;

  auto bad           = p;
  bad.ann_itopk_size = 0;
  EXPECT_THROW(f.run(bad), raft::logic_error);
  bad                  = p;
  bad.ann_graph_degree = 0;
  EXPECT_THROW(f.run(bad), raft::logic_error);
  bad                               = p;
  bad.ann_intermediate_graph_degree = 0;
  EXPECT_THROW(f.run(bad), raft::logic_error);
  bad                  = p;
  bad.ann_search_width = 0;
  EXPECT_THROW(f.run(bad), raft::logic_error);

  // Invalid ANN params are ignored when ANN is disabled.
  bad                        = p;
  bad.use_ann_for_assignment = false;
  bad.ann_itopk_size         = 0;
  EXPECT_NO_THROW(f.run(bad));
}

TEST(KmeansAnn, MultipleSeedsAndSampleWeights)
{
  ann_fixture<float> f(10000, 16, 150);
  auto weights = raft::make_device_vector<float, int>(f.handle, f.n_rows);
  fill_kmeans_test_weights(f.handle, weights.view(), kmeans_weight_mode::mild_nonuniform);
  auto w = std::make_optional(
    raft::make_device_vector_view<const float, int>(weights.data_handle(), f.n_rows));

  // Random init on many close blobs makes the final inertia vary by tens of percent across seeds
  // (best-of-n_init compounds this), so ANN vs brute-force trajectories can land in different local
  // minima. With a single seed the two agree closely; with n_init = 2 only a loose bound is
  // meaningful. This test checks that the weighted and multi-seed paths work (a broken path would
  // be off by far more).
  for (int n_init : {1, 2}) {
    auto brute                 = f.base_params();
    brute.n_init               = n_init;
    auto ann                   = brute;
    ann.use_ann_for_assignment = true;
    auto r_brute               = f.run(brute, false, w);
    auto r_ann                 = f.run(ann, false, w);
    const float tolerance      = n_init == 1 ? 1.02f : 1.10f;
    EXPECT_LE(r_ann.inertia, r_brute.inertia * tolerance)
      << "n_init=" << n_init << " ann=" << r_ann.inertia << " brute=" << r_brute.inertia;
  }
}

TEST(KmeansAnn, HostBatchedFit)
{
  ann_fixture<float> f(12000, 16, 150);
  auto brute                  = f.base_params();
  brute.device_buffer_samples = 2500;  // several batches per iteration
  auto ann                    = brute;
  ann.use_ann_for_assignment  = true;
  auto r_brute                = f.run(brute, true);
  auto r_ann                  = f.run(ann, true);
  EXPECT_LE(r_ann.inertia, r_brute.inertia * 1.02f);
}

}  // namespace cuvs
