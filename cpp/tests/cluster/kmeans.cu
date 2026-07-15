/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include "kmeans_test_blobs.cuh"

#include <cuvs/cluster/kmeans.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/stats/adjusted_rand_index.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <optional>
#include <vector>

namespace cuvs {

template <typename T>
struct KmeansInputs {
  int n_row;
  int n_col;
  int n_clusters;
  T tol;
  bool weighted;
};

// template <typename DataT, typename IndexT>
// void run_cluster_cost(const raft::resources& handle,
//                       raft::device_vector_view<DataT, IndexT> minClusterDistance,
//                       rmm::device_uvector<char>& workspace,
//                       raft::device_scalar_view<DataT> clusterCost)
//{
//   cuvs::cluster::kmeans::cluster_cost(
//     handle, minClusterDistance, workspace, clusterCost, raft::add_op{});
// }

template <typename T>
class KmeansTest : public ::testing::TestWithParam<KmeansInputs<T>> {
 protected:
  KmeansTest()
    : d_labels(0, raft::resource::get_cuda_stream(handle)),
      d_labels_ref(0, raft::resource::get_cuda_stream(handle)),
      d_centroids(0, raft::resource::get_cuda_stream(handle)),
      d_sample_weight(0, raft::resource::get_cuda_stream(handle))
  {
  }

  //  void apiTest()
  //  {
  //    testparams = ::testing::TestWithParam<KmeansInputs<T>>::GetParam();
  //
  //    auto stream                = raft::resource::get_cuda_stream(handle);
  //    int n_samples              = testparams.n_row;
  //    int n_features             = testparams.n_col;
  //    params.n_clusters          = testparams.n_clusters;
  //    params.tol                 = testparams.tol;
  //    params.n_init              = 1;
  //    params.rng_state.seed      = 1;
  //    params.oversampling_factor = 0;
  //
  //    raft::random::RngState rng(params.rng_state.seed, params.rng_state.type);
  //
  //    auto X      = raft::make_device_matrix<T, int>(handle, n_samples, n_features);
  //    auto labels = raft::make_device_vector<int, int>(handle, n_samples);
  //
  //    raft::random::make_blobs<T, int>(X.data_handle(),
  //                                     labels.data_handle(),
  //                                     n_samples,
  //                                     n_features,
  //                                     params.n_clusters,
  //                                     stream,
  //                                     true,
  //                                     nullptr,
  //                                     nullptr,
  //                                     T(1.0),
  //                                     false,
  //                                     (T)-10.0f,
  //                                     (T)10.0f,
  //                                     (uint64_t)1234);
  //    d_labels.resize(n_samples, stream);
  //    d_labels_ref.resize(n_samples, stream);
  //    d_centroids.resize(params.n_clusters * n_features, stream);
  //    raft::copy(d_labels_ref.data(), labels.data_handle(), n_samples, stream);
  //    rmm::device_uvector<T> d_sample_weight(n_samples, stream);
  //    thrust::fill(
  //      thrust::cuda::par.on(stream), d_sample_weight.data(), d_sample_weight.data() + n_samples,
  //      1);
  //    auto weight_view =
  //      raft::make_device_vector_view<const T, int>(d_sample_weight.data(), n_samples);
  //
  //    T inertia  = 0;
  //    int n_iter = 0;
  //    rmm::device_uvector<char> workspace(0, stream);
  //    rmm::device_uvector<T> L2NormBuf_OR_DistBuf(0, stream);
  //    rmm::device_uvector<T> inRankCp(0, stream);
  //    auto X_view = raft::make_const_mdspan(X.view());
  //    auto centroids_view =
  //      raft::make_device_matrix_view<T, int>(d_centroids.data(), params.n_clusters, n_features);
  //    auto miniX = raft::make_device_matrix<T, int>(handle, n_samples / 4, n_features);
  //
  //    // Initialize kmeans on a portion of X
  //    raft::cluster::kmeans::shuffle_and_gather(
  //      handle,
  //      X_view,
  //      raft::make_device_matrix_view<T, int>(miniX.data_handle(), miniX.extent(0),
  //      miniX.extent(1)), miniX.extent(0), params.rng_state.seed);
  //
  //    raft::cluster::kmeans::init_plus_plus(
  //      handle, params, raft::make_const_mdspan(miniX.view()), centroids_view, workspace);
  //
  //    auto minClusterDistance = raft::make_device_vector<T, int>(handle, n_samples);
  //    auto minClusterAndDistance =
  //      raft::make_device_vector<raft::KeyValuePair<int, T>, int>(handle, n_samples);
  //    auto L2NormX           = raft::make_device_vector<T, int>(handle, n_samples);
  //    auto clusterCostBefore = raft::make_device_scalar<T>(handle, 0);
  //    auto clusterCostAfter  = raft::make_device_scalar<T>(handle, 0);
  //
  //    raft::linalg::rowNorm(L2NormX.data_handle(),
  //                          X.data_handle(),
  //                          X.extent(1),
  //                          X.extent(0),
  //                          raft::linalg::L2Norm,
  //                          true,
  //                          stream);
  //
  //    raft::cluster::kmeans::min_cluster_distance(handle,
  //                                                X_view,
  //                                                centroids_view,
  //                                                minClusterDistance.view(),
  //                                                L2NormX.view(),
  //                                                L2NormBuf_OR_DistBuf,
  //                                                params.metric,
  //                                                params.batch_samples,
  //                                                params.batch_centroids,
  //                                                workspace);
  //
  //    run_cluster_cost(handle, minClusterDistance.view(), workspace, clusterCostBefore.view());
  //
  //    // Run a fit of kmeans
  //    raft::cluster::kmeans::fit_main(handle,
  //                                    params,
  //                                    X_view,
  //                                    weight_view,
  //                                    centroids_view,
  //                                    raft::make_host_scalar_view(&inertia),
  //                                    raft::make_host_scalar_view(&n_iter),
  //                                    workspace);
  //
  //    // Check that the cluster cost decreased
  //    raft::cluster::kmeans::min_cluster_distance(handle,
  //                                                X_view,
  //                                                centroids_view,
  //                                                minClusterDistance.view(),
  //                                                L2NormX.view(),
  //                                                L2NormBuf_OR_DistBuf,
  //                                                params.metric,
  //                                                params.batch_samples,
  //                                                params.batch_centroids,
  //                                                workspace);
  //
  //    run_cluster_cost(handle, minClusterDistance.view(), workspace, clusterCostAfter.view());
  //    T h_clusterCostBefore = T(0);
  //    T h_clusterCostAfter  = T(0);
  //    raft::update_host(&h_clusterCostBefore, clusterCostBefore.data_handle(), 1, stream);
  //    raft::update_host(&h_clusterCostAfter, clusterCostAfter.data_handle(), 1, stream);
  //    ASSERT_TRUE(h_clusterCostAfter < h_clusterCostBefore);
  //
  //    // Count samples in clusters using 2 methods and compare them
  //    // Fill minClusterAndDistance
  //    raft::cluster::kmeans::min_cluster_and_distance(
  //      handle,
  //      X_view,
  //      raft::make_device_matrix_view<const T, int>(
  //        d_centroids.data(), params.n_clusters, n_features),
  //      minClusterAndDistance.view(),
  //      L2NormX.view(),
  //      L2NormBuf_OR_DistBuf,
  //      params.metric,
  //      params.batch_samples,
  //      params.batch_centroids,
  //      workspace);
  //    raft::cluster::kmeans::KeyValueIndexOp<int, T> conversion_op;
  //    thrust::transform_iterator<raft::cluster::kmeans::KeyValueIndexOp<int, T>,
  //                               raft::KeyValuePair<int, T>*>
  //      itr(minClusterAndDistance.data_handle(), conversion_op);
  //
  //    auto sampleCountInCluster = raft::make_device_vector<T, int>(handle, params.n_clusters);
  //    auto weigthInCluster      = raft::make_device_vector<T, int>(handle, params.n_clusters);
  //    auto newCentroids = raft::make_device_matrix<T, int>(handle, params.n_clusters, n_features);
  //    raft::cluster::kmeans::update_centroids(handle,
  //                                            X_view,
  //                                            weight_view,
  //                                            raft::make_device_matrix_view<const T, int>(
  //                                              d_centroids.data(), params.n_clusters,
  //                                              n_features),
  //                                            itr,
  //                                            weigthInCluster.view(),
  //                                            newCentroids.view());
  //    raft::cluster::kmeans::count_samples_in_cluster(handle,
  //                                                    params,
  //                                                    X_view,
  //                                                    L2NormX.view(),
  //                                                    newCentroids.view(),
  //                                                    workspace,
  //                                                    sampleCountInCluster.view());
  //
  //    ASSERT_TRUE(devArrMatch(sampleCountInCluster.data_handle(),
  //                            weigthInCluster.data_handle(),
  //                            params.n_clusters,
  //                            CompareApprox<T>(params.tol)));
  //  }

  void basicTest()
  {
    testparams = ::testing::TestWithParam<KmeansInputs<T>>::GetParam();

    int n_samples              = testparams.n_row;
    int n_features             = testparams.n_col;
    params.n_clusters          = testparams.n_clusters;
    params.tol                 = testparams.tol;
    params.n_init              = 5;
    params.rng_state.seed      = 1;
    params.oversampling_factor = 0;

    auto stream = raft::resource::get_cuda_stream(handle);
    auto bi     = make_kmeans_blob_inputs<T>(
      handle, n_samples, n_features, params.n_clusters, /* with_host_mirror */ false);

    d_labels.resize(n_samples, stream);
    d_labels_ref.resize(n_samples, stream);
    d_centroids.resize(params.n_clusters * n_features, stream);

    std::optional<raft::device_vector_view<const T, int>> d_sw = std::nullopt;
    auto d_centroids_view =
      raft::make_device_matrix_view<T, int>(d_centroids.data(), params.n_clusters, n_features);
    if (testparams.weighted) {
      d_sample_weight.resize(n_samples, stream);
      auto d_sw_view = raft::make_device_vector_view<T, int>(d_sample_weight.data(), n_samples);
      fill_kmeans_test_weights(handle, d_sw_view, kmeans_weight_mode::uniform);
      d_sw = std::make_optional(
        raft::make_device_vector_view<const T, int>(d_sample_weight.data(), n_samples));
    }

    raft::copy(d_labels_ref.data(), bi.d_labels_ref.data_handle(), n_samples, stream);

    T inertia   = 0;
    int n_iter  = 0;
    auto X_view = raft::make_const_mdspan(bi.d_X.view());

    cuvs::cluster::kmeans::fit_predict(
      handle,
      params,
      X_view,
      d_sw,
      d_centroids_view,
      raft::make_device_vector_view<int, int>(d_labels.data(), n_samples),
      raft::make_host_scalar_view<T>(&inertia),
      raft::make_host_scalar_view<int>(&n_iter));

    raft::resource::sync_stream(handle, stream);

    score = raft::stats::adjusted_rand_index(
      d_labels_ref.data(), d_labels.data(), n_samples, raft::resource::get_cuda_stream(handle));

    if (score < 1.0) {
      std::stringstream ss;
      ss << "Expected: " << raft::arr2Str(d_labels_ref.data(), 25, "d_labels_ref", stream);
      std::cout << (ss.str().c_str()) << '\n';
      ss.str(std::string());
      ss << "Actual: " << raft::arr2Str(d_labels.data(), 25, "d_labels", stream);
      std::cout << (ss.str().c_str()) << '\n';
      std::cout << "Score = " << score << '\n';
    }
  }

  void SetUp() override
  {
    basicTest();
    //    apiTest();
  }

 protected:
  raft::resources handle;
  KmeansInputs<T> testparams;
  rmm::device_uvector<int> d_labels;
  rmm::device_uvector<int> d_labels_ref;
  rmm::device_uvector<T> d_centroids;
  rmm::device_uvector<T> d_sample_weight;
  double score;
  cuvs::cluster::kmeans::params params;
};

const std::vector<KmeansInputs<float>> inputsf2 = {{1000, 32, 5, 0.0001f, true},
                                                   {1000, 32, 5, 0.0001f, false},
                                                   {1000, 100, 20, 0.0001f, true},
                                                   {1000, 100, 20, 0.0001f, false},
                                                   {10000, 32, 10, 0.0001f, true},
                                                   {10000, 32, 10, 0.0001f, false},
                                                   {10000, 100, 50, 0.0001f, true},
                                                   {10000, 100, 50, 0.0001f, false},
                                                   {10000, 500, 100, 0.0001f, true},
                                                   {10000, 500, 100, 0.0001f, false}};

const std::vector<KmeansInputs<double>> inputsd2 = {{1000, 32, 5, 0.0001, true},
                                                    {1000, 32, 5, 0.0001, false},
                                                    {1000, 100, 20, 0.0001, true},
                                                    {1000, 100, 20, 0.0001, false},
                                                    {10000, 32, 10, 0.0001, true},
                                                    {10000, 32, 10, 0.0001, false},
                                                    {10000, 100, 50, 0.0001, true},
                                                    {10000, 100, 50, 0.0001, false},
                                                    {10000, 500, 100, 0.0001, true},
                                                    {10000, 500, 100, 0.0001, false}};

typedef KmeansTest<float> KmeansTestF;

TEST_P(KmeansTestF, Result) { ASSERT_TRUE(score == 1.0); }

INSTANTIATE_TEST_CASE_P(KmeansTests, KmeansTestF, ::testing::ValuesIn(inputsf2));

// ============================================================================
// Batched KMeans Tests (fit + predict with host data)
// ============================================================================

template <typename T>
struct KmeansBatchedInputs {
  int n_row;
  int n_col;
  int n_clusters;
  T tol;
  bool weighted;
  int streaming_batch_size;
};

template <typename T>
class KmeansFitBatchedTest : public ::testing::TestWithParam<KmeansBatchedInputs<T>> {
 protected:
  KmeansFitBatchedTest() = default;

  void prepareBlobInputs()
  {
    testparams     = ::testing::TestWithParam<KmeansBatchedInputs<T>>::GetParam();
    int n_samples  = testparams.n_row;
    int n_features = testparams.n_col;
    int n_clusters = testparams.n_clusters;

    auto bi = make_kmeans_blob_inputs<T>(handle, n_samples, n_features, n_clusters);
    d_X.emplace(std::move(bi.d_X));
    d_labels_ref.emplace(std::move(bi.d_labels_ref));
    h_X = std::move(bi.h_X);

    if (testparams.weighted) {
      d_sample_weight.emplace(raft::make_device_vector<T, int>(handle, n_samples));
      fill_kmeans_test_weights(handle, d_sample_weight->view(), kmeans_weight_mode::uniform);
    } else {
      d_sample_weight.reset();
    }
  }

  std::optional<raft::device_vector_view<const T, int>> d_sw_view() const
  {
    if (!d_sample_weight.has_value()) return std::nullopt;
    return std::make_optional(raft::make_const_mdspan(d_sample_weight->view()));
  }

  void fitBatchedTest()
  {
    int n_samples              = testparams.n_row;
    int n_features             = testparams.n_col;
    params.n_clusters          = testparams.n_clusters;
    params.tol                 = testparams.tol;
    params.rng_state.seed      = 1;
    params.oversampling_factor = 0;

    auto stream = raft::resource::get_cuda_stream(handle);

    d_labels.emplace(raft::make_device_vector<int, int>(handle, n_samples));
    d_centroids.emplace(raft::make_device_matrix<T, int>(handle, params.n_clusters, n_features));
    d_centroids_ref.emplace(
      raft::make_device_matrix<T, int>(handle, params.n_clusters, n_features));

    raft::random::RngState rng(params.rng_state.seed);
    raft::random::uniform(
      handle, rng, d_centroids->data_handle(), params.n_clusters * n_features, T(-1), T(1));
    raft::copy(d_centroids_ref->data_handle(),
               d_centroids->data_handle(),
               params.n_clusters * n_features,
               stream);

    auto d_centroids_view = raft::make_device_matrix_view<T, int64_t>(
      d_centroids->data_handle(), params.n_clusters, n_features);

    auto d_sw = d_sw_view();

    params.init     = cuvs::cluster::kmeans::params::Array;
    params.max_iter = 20;

    T ref_inertia  = 0;
    int ref_n_iter = 0;
    cuvs::cluster::kmeans::fit(handle,
                               params,
                               raft::make_const_mdspan(d_X->view()),
                               d_sw,
                               d_centroids_ref->view(),
                               raft::make_host_scalar_view<T>(&ref_inertia),
                               raft::make_host_scalar_view<int>(&ref_n_iter));

    cuvs::cluster::kmeans::params batched_params = params;
    batched_params.streaming_batch_size          = testparams.streaming_batch_size;

    std::optional<raft::host_vector_view<const T, int64_t>> h_sw = std::nullopt;
    auto h_sample_weight = raft::make_host_vector<T, int64_t>(testparams.weighted ? n_samples : 0);
    if (testparams.weighted) {
      std::fill_n(h_sample_weight.data_handle(), n_samples, T(1));
      h_sw = std::make_optional(raft::make_const_mdspan(h_sample_weight.view()));
    }

    T inertia      = 0;
    int64_t n_iter = 0;

    cuvs::cluster::kmeans::fit(handle,
                               batched_params,
                               raft::make_const_mdspan(h_X->view()),
                               h_sw,
                               d_centroids_view,
                               raft::make_host_scalar_view<T>(&inertia),
                               raft::make_host_scalar_view<int64_t>(&n_iter));

    raft::resource::sync_stream(handle, stream);

    centroids_match = devArrMatch(d_centroids_ref->data_handle(),
                                  d_centroids->data_handle(),
                                  params.n_clusters,
                                  n_features,
                                  CompareApprox<T>(T(1e-2)),
                                  stream);

    T ref_pred_inertia = 0;
    cuvs::cluster::kmeans::predict(handle,
                                   params,
                                   raft::make_const_mdspan(d_X->view()),
                                   d_sw,
                                   raft::make_const_mdspan(d_centroids_ref->view()),
                                   d_labels_ref->view(),
                                   true,
                                   raft::make_host_scalar_view<T>(&ref_pred_inertia));

    T pred_inertia = 0;
    cuvs::cluster::kmeans::predict(handle,
                                   params,
                                   raft::make_const_mdspan(d_X->view()),
                                   d_sw,
                                   raft::make_const_mdspan(d_centroids->view()),
                                   d_labels->view(),
                                   true,
                                   raft::make_host_scalar_view<T>(&pred_inertia));

    raft::resource::sync_stream(handle, stream);

    score = raft::stats::adjusted_rand_index(d_labels_ref->data_handle(),
                                             d_labels->data_handle(),
                                             n_samples,
                                             raft::resource::get_cuda_stream(handle));

    if (score < 0.99) {
      std::stringstream ss;
      ss << "Expected: " << raft::arr2Str(d_labels_ref->data_handle(), 25, "d_labels_ref", stream);
      std::cout << (ss.str().c_str()) << '\n';
      ss.str(std::string());
      ss << "Actual: " << raft::arr2Str(d_labels->data_handle(), 25, "d_labels", stream);
      std::cout << (ss.str().c_str()) << '\n';
      std::cout << "Score = " << score << '\n';
    }

    inertia_match = (std::abs(ref_inertia - inertia) < T(1e-2) * std::abs(ref_inertia));

    if (!inertia_match) {
      std::cout << "Inertia mismatch: ref=" << ref_inertia << " batched=" << inertia << '\n';
    }
  }

  T fitHostWithInitSize(int64_t init_size_value)
  {
    int n_features = testparams.n_col;
    int n_clusters = testparams.n_clusters;
    auto stream    = raft::resource::get_cuda_stream(handle);

    cuvs::cluster::kmeans::params p;
    p.n_clusters           = n_clusters;
    p.tol                  = testparams.tol;
    p.n_init               = 1;
    p.init                 = cuvs::cluster::kmeans::params::KMeansPlusPlus;
    p.max_iter             = 20;
    p.rng_state.seed       = 1;
    p.oversampling_factor  = 0;
    p.streaming_batch_size = testparams.streaming_batch_size;
    p.init_size            = init_size_value;

    auto d_centroids_buf = raft::make_device_matrix<T, int64_t>(handle, n_clusters, n_features);
    T inertia            = 0;
    int64_t n_iter       = 0;
    cuvs::cluster::kmeans::fit(
      handle,
      p,
      raft::make_const_mdspan(h_X->view()),
      std::optional<raft::host_vector_view<const T, int64_t>>{std::nullopt},
      d_centroids_buf.view(),
      raft::make_host_scalar_view<T>(&inertia),
      raft::make_host_scalar_view<int64_t>(&n_iter));
    raft::resource::sync_stream(handle, stream);
    return inertia;
  }

  void runInitSizeCompare()
  {
    int n_samples         = testparams.n_row;
    int n_clusters        = testparams.n_clusters;
    int default_init_size = std::min(3 * n_clusters, n_samples);

    T inertia_default  = fitHostWithInitSize(0);
    T inertia_explicit = fitHostWithInitSize(default_init_size);
    T inertia_full     = fitHostWithInitSize(n_samples);

    ASSERT_TRUE(std::isfinite(inertia_default));
    ASSERT_TRUE(std::isfinite(inertia_explicit));
    ASSERT_TRUE(std::isfinite(inertia_full));
    ASSERT_GT(inertia_default, T(0));
    ASSERT_GT(inertia_explicit, T(0));
    ASSERT_GT(inertia_full, T(0));

    const T rel = T(1e-5);

    // init_size = 0 must resolve to the documented default (min(3*k, n));
    // feeding that value explicitly should reproduce the same inertia.
    ASSERT_NEAR(inertia_default, inertia_explicit, std::abs(inertia_default) * rel);

    // Full-dataset seeding has at least as much information as the subsample
    // default, so the converged inertia should not be worse.
    ASSERT_LE(inertia_full, inertia_default * (T(1) + rel));
  }

  T fitKMeansPlusPlus(int n_init_value)
  {
    int n_features = testparams.n_col;
    int n_clusters = testparams.n_clusters;
    auto stream    = raft::resource::get_cuda_stream(handle);

    cuvs::cluster::kmeans::params p;
    p.n_clusters          = n_clusters;
    p.tol                 = testparams.tol;
    p.n_init              = n_init_value;
    p.init                = cuvs::cluster::kmeans::params::KMeansPlusPlus;
    p.max_iter            = 10;
    p.rng_state.seed      = 7;
    p.oversampling_factor = 0;

    auto d_centroids_buf = raft::make_device_matrix<T, int>(handle, n_clusters, n_features);
    T inertia            = 0;
    int n_iter           = 0;
    cuvs::cluster::kmeans::fit(handle,
                               p,
                               raft::make_const_mdspan(d_X->view()),
                               std::optional<raft::device_vector_view<const T, int>>{std::nullopt},
                               d_centroids_buf.view(),
                               raft::make_host_scalar_view<T>(&inertia),
                               raft::make_host_scalar_view<int>(&n_iter));
    raft::resource::sync_stream(handle, stream);
    return inertia;
  }

  void runMultiSeedCheck()
  {
    T inertia1 = fitKMeansPlusPlus(1);
    T inertia3 = fitKMeansPlusPlus(3);
    ASSERT_GT(inertia1, T(0));
    ASSERT_GT(inertia3, T(0));
    // n_init > 1 keeps the best trial, so it should not be worse than
    // n_init == 1.
    const T rel = T(1e-5);
    ASSERT_LE(inertia3, inertia1 * (T(1) + rel));
  }

  void runZeroCost()
  {
    int n_samples  = testparams.n_row;
    int n_features = testparams.n_col;
    int n_clusters = n_samples;
    auto stream    = raft::resource::get_cuda_stream(handle);

    auto d_centroids_buf = raft::make_device_matrix<T, int>(handle, n_clusters, n_features);
    raft::copy(d_centroids_buf.data_handle(),
               d_X->data_handle(),
               static_cast<size_t>(n_samples) * n_features,
               stream);

    cuvs::cluster::kmeans::params p;
    p.n_clusters          = n_clusters;
    p.tol                 = testparams.tol;
    p.n_init              = 1;
    p.init                = cuvs::cluster::kmeans::params::Array;
    p.max_iter            = 5;
    p.rng_state.seed      = 1;
    p.oversampling_factor = 0;

    T inertia  = 0;
    int n_iter = 0;
    ASSERT_NO_THROW(cuvs::cluster::kmeans::fit(
      handle,
      p,
      raft::make_const_mdspan(d_X->view()),
      std::optional<raft::device_vector_view<const T, int>>{std::nullopt},
      d_centroids_buf.view(),
      raft::make_host_scalar_view<T>(&inertia),
      raft::make_host_scalar_view<int>(&n_iter)));
    raft::resource::sync_stream(handle, stream);

    // The expanded L2 formula ||x||^2 - 2 x.c + ||c||^2 does not cancel to
    // exactly 0 even when x == c due to float roundoff. The largest residual
    // inertia observed across parameterized blob shapes is ~27.95; use an
    // absolute upper bound of 100 for headroom.
    ASSERT_LE(inertia, T(100));
  }

 protected:
  raft::resources handle;
  KmeansBatchedInputs<T> testparams;
  std::optional<raft::device_vector<int, int>> d_labels;
  std::optional<raft::device_vector<int, int>> d_labels_ref;
  std::optional<raft::device_matrix<T, int>> d_centroids;
  std::optional<raft::device_matrix<T, int>> d_centroids_ref;
  std::optional<raft::device_matrix<T, int>> d_X;
  std::optional<raft::device_vector<T, int>> d_sample_weight;
  std::optional<raft::host_matrix<T, int64_t>> h_X;
  double score;
  testing::AssertionResult centroids_match = testing::AssertionSuccess();
  bool inertia_match                       = false;
  cuvs::cluster::kmeans::params params;
};

// ============================================================================
// Test inputs for batched tests
// ============================================================================

const std::vector<KmeansBatchedInputs<float>> batched_inputsf2 = {
  {1000, 32, 5, 0.0001f, true, 256},
  {1000, 64, 5, 0.0001f, false, 500},
  {1000, 100, 20, 0.0001f, true, 30},
  {1000, 10, 20, 0.0001f, false, 30},
  {10000, 16, 10, 0.0001f, true, 1000},
  {10000, 96, 10, 0.0001f, false, 10000},
};

const std::vector<KmeansBatchedInputs<double>> batched_inputsd2 = {
  {1000, 32, 5, 0.0001, true, 256},
  {1000, 64, 5, 0.0001, false, 500},
  {1000, 100, 20, 0.0001, true, 30},
  {1000, 10, 20, 0.0001, false, 30},
  {10000, 16, 10, 0.0001, true, 1000},
  {10000, 96, 10, 0.0001, false, 10000},
};

// ============================================================================
// fit (host/batched) tests
// ============================================================================
typedef KmeansFitBatchedTest<float> KmeansFitBatchedTestF;
typedef KmeansFitBatchedTest<double> KmeansFitBatchedTestD;

TEST_P(KmeansFitBatchedTestF, Result)
{
  prepareBlobInputs();
  fitBatchedTest();
  ASSERT_TRUE(centroids_match);
  ASSERT_TRUE(score >= 0.99);
  ASSERT_TRUE(inertia_match);
  runInitSizeCompare();
  runMultiSeedCheck();
  runZeroCost();
}

TEST_P(KmeansFitBatchedTestD, Result)
{
  prepareBlobInputs();
  fitBatchedTest();
  ASSERT_TRUE(centroids_match);
  ASSERT_TRUE(score >= 0.99);
  ASSERT_TRUE(inertia_match);
  runInitSizeCompare();
  runMultiSeedCheck();
  runZeroCost();
}

INSTANTIATE_TEST_CASE_P(KmeansFitBatchedTests,
                        KmeansFitBatchedTestF,
                        ::testing::ValuesIn(batched_inputsf2));
INSTANTIATE_TEST_CASE_P(KmeansFitBatchedTests,
                        KmeansFitBatchedTestD,
                        ::testing::ValuesIn(batched_inputsd2));

}  // namespace cuvs
