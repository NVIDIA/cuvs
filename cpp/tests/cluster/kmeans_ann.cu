/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include "kmeans_test_blobs.cuh"

#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/neighbors/cagra.hpp>

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

template <typename T>
struct KmeansAnnInputs {
  int n_row;
  int n_col;
  int n_clusters;
  T tol;  // allowed relative increase of the ANN inertia over the brute-force inertia
  bool weighted;
};

template <typename T>
struct KmeansAnnResult {
  T inertia;          // inertia reported by fit
  T predict_inertia;  // inertia reported by predict on the fitted centroids
  std::vector<int> labels;
};

// Runs fit and predict with `params` on the blobs in `bi`.
template <typename T>
KmeansAnnResult<T> fit_predict(raft::resources const& handle,
                               cuvs::cluster::kmeans::params const& params,
                               kmeans_blob_inputs<T> const& bi,
                               std::optional<raft::device_vector_view<const T, int>> weights,
                               bool host_batched = false)
{
  auto stream     = raft::resource::get_cuda_stream(handle);
  const int n_row = bi.d_X.extent(0);
  const int n_col = bi.d_X.extent(1);
  auto X          = raft::make_device_matrix_view<const T, int>(bi.d_X.data_handle(), n_row, n_col);
  auto centroids  = raft::make_device_matrix<T, int>(handle, params.n_clusters, n_col);
  T inertia       = 0;
  int n_iter      = 0;
  if (host_batched) {
    int64_t n_iter64 = 0;
    cuvs::cluster::kmeans::fit(
      handle,
      params,
      raft::make_host_matrix_view<const T, int64_t>(bi.h_X->data_handle(), n_row, n_col),
      std::nullopt,
      raft::make_device_matrix_view<T, int64_t>(centroids.data_handle(), params.n_clusters, n_col),
      raft::make_host_scalar_view(&inertia),
      raft::make_host_scalar_view(&n_iter64));
  } else {
    cuvs::cluster::kmeans::fit(handle,
                               params,
                               X,
                               weights,
                               centroids.view(),
                               raft::make_host_scalar_view(&inertia),
                               raft::make_host_scalar_view(&n_iter));
  }

  auto labels    = raft::make_device_vector<int, int>(handle, n_row);
  T predict_cost = 0;
  cuvs::cluster::kmeans::predict(
    handle,
    params,
    X,
    weights,
    raft::make_device_matrix_view<const T, int>(centroids.data_handle(), params.n_clusters, n_col),
    labels.view(),
    false,
    raft::make_host_scalar_view(&predict_cost));
  std::vector<int> h_labels(n_row);
  raft::update_host(h_labels.data(), labels.data_handle(), n_row, stream);
  raft::resource::sync_stream(handle, stream);
  return {inertia, predict_cost, h_labels};
}

cuvs::cluster::kmeans::params make_params(int n_clusters, bool use_ann)
{
  cuvs::cluster::kmeans::params params;
  params.n_clusters             = n_clusters;
  params.max_iter               = 20;
  params.tol                    = 1e-6;
  params.n_init                 = 1;
  params.init                   = cuvs::cluster::kmeans::params::Random;
  params.rng_state              = raft::random::RngState(7);
  params.use_ann_for_assignment = use_ann;
  return params;
}

double label_agreement(std::vector<int> const& a, std::vector<int> const& b)
{
  size_t same = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    same += (a[i] == b[i]);
  }
  return static_cast<double>(same) / static_cast<double>(a.size());
}

}  // namespace

// ANN assignment gives the same quality as brute force: the inertia of fit and of predict, and the
// labels, match.
template <typename T>
class KmeansAnnTest : public ::testing::TestWithParam<KmeansAnnInputs<T>> {
 protected:
  void testRun()
  {
    testparams = ::testing::TestWithParam<KmeansAnnInputs<T>>::GetParam();

    auto bi = make_kmeans_blob_inputs<T>(
      handle, testparams.n_row, testparams.n_col, testparams.n_clusters, false);
    std::optional<raft::device_vector_view<const T, int>> weights = std::nullopt;
    auto d_weights =
      raft::make_device_vector<T, int>(handle, testparams.weighted ? testparams.n_row : 0);
    if (testparams.weighted) {
      fill_kmeans_test_weights(handle, d_weights.view(), kmeans_weight_mode::mild_nonuniform);
      weights =
        raft::make_device_vector_view<const T, int>(d_weights.data_handle(), testparams.n_row);
    }

    auto brute = fit_predict<T>(handle, make_params(testparams.n_clusters, false), bi, weights);
    auto ann   = fit_predict<T>(handle, make_params(testparams.n_clusters, true), bi, weights);

    inertia_ratio         = ann.inertia / brute.inertia;
    predict_inertia_ratio = ann.predict_inertia / brute.predict_inertia;
    agreement             = label_agreement(ann.labels, brute.labels);
  }

  void SetUp() override { testRun(); }

  raft::resources handle;
  KmeansAnnInputs<T> testparams;
  T inertia_ratio;
  T predict_inertia_ratio;
  double agreement;
};

const std::vector<KmeansAnnInputs<float>> inputsf2 = {{20000, 32, 300, 0.02f, false},
                                                      {10000, 16, 150, 0.02f, true},
                                                      {10000, 16, 200, 0.02f, false},
                                                      {2000, 8, 2, 0.02f, false},
                                                      {2000, 8, 5, 0.02f, false}};

typedef KmeansAnnTest<float> KmeansAnnTestF;
TEST_P(KmeansAnnTestF, Result)
{
  EXPECT_LE(inertia_ratio, 1 + testparams.tol);
  EXPECT_LE(predict_inertia_ratio, 1 + testparams.tol);
  EXPECT_GE(agreement, 0.98);
}

INSTANTIATE_TEST_CASE_P(KmeansAnnTests, KmeansAnnTestF, ::testing::ValuesIn(inputsf2));

// Configurations the ANN assignment does not support use brute force, so the result is identical to
// the brute-force one.
template <typename T>
void expect_brute_force_fallback(int n_row,
                                 int n_col,
                                 int n_clusters,
                                 cuvs::cluster::kmeans::params ann_params)
{
  raft::resources handle;
  auto bi    = make_kmeans_blob_inputs<T>(handle, n_row, n_col, n_clusters, false);
  auto brute = fit_predict<T>(handle, make_params(n_clusters, false), bi, std::nullopt);
  auto ann   = fit_predict<T>(handle, ann_params, bi, std::nullopt);
  EXPECT_EQ(ann.labels, brute.labels);
  EXPECT_FLOAT_EQ(ann.inertia, brute.inertia);
}

TEST(KmeansAnnFallback, SingleCluster)
{
  expect_brute_force_fallback<float>(1000, 8, 1, make_params(1, true));
}

TEST(KmeansAnnFallback, DoubleData)
{
  expect_brute_force_fallback<double>(2000, 8, 20, make_params(20, true));
}

TEST(KmeansAnnFallback, UnalignedNumberOfFeatures)
{
  expect_brute_force_fallback<float>(2000, 10, 20, make_params(20, true));
}

TEST(KmeansAnnFallback, AceGraphBuild)
{
  auto params = make_params(20, true);
  params.ann_index_params.graph_build_params =
    cuvs::neighbors::cagra::graph_build_params::ace_params();
  expect_brute_force_fallback<float>(2000, 8, 20, params);
}

// The CAGRA parameters are used as given: a custom graph/search configuration reaches the same
// quality, and a non-L2 metric in them is overridden.
TEST(KmeansAnn, UserCagraParams)
{
  raft::resources handle;
  auto bi    = make_kmeans_blob_inputs<float>(handle, 10000, 16, 200, false);
  auto brute = fit_predict<float>(handle, make_params(200, false), bi, std::nullopt);

  auto params                                       = make_params(200, true);
  params.ann_index_params.graph_degree              = 32;
  params.ann_index_params.intermediate_graph_degree = 48;
  params.ann_index_params.metric                    = cuvs::distance::DistanceType::InnerProduct;
  params.ann_search_params.itopk_size               = 128;
  params.ann_search_params.search_width             = 2;
  auto ann = fit_predict<float>(handle, params, bi, std::nullopt);
  EXPECT_LE(ann.inertia, brute.inertia * 1.02f);
}

// fit on host data processes the samples in batches.
TEST(KmeansAnn, HostBatchedFit)
{
  raft::resources handle;
  auto bi                            = make_kmeans_blob_inputs<float>(handle, 12000, 16, 150);
  auto brute_params                  = make_params(150, false);
  auto ann_params                    = make_params(150, true);
  brute_params.device_buffer_samples = 2500;
  ann_params.device_buffer_samples   = 2500;
  auto brute = fit_predict<float>(handle, brute_params, bi, std::nullopt, true);
  auto ann   = fit_predict<float>(handle, ann_params, bi, std::nullopt, true);
  EXPECT_LE(ann.inertia, brute.inertia * 1.02f);
}

}  // namespace cuvs
