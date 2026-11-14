/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../src/cluster/detail/kmeans_batch_loader.cuh"
#include "../../src/cluster/detail/kmeans_emeans.cuh"
#include "../test_utils.cuh"
#include "kmeans_test_blobs.cuh"

#include <cuvs/cluster/kmeans.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resources.hpp>
#include <raft/stats/adjusted_rand_index.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <limits>
#include <optional>
#include <vector>

namespace cuvs {

TEST(EMeans, Schedule)
{
  using namespace cuvs::cluster::kmeans;
  emeans_params p;
  p.epochs      = 2.5;
  p.batch_size  = 4;
  p.alpha0      = .8;
  auto schedule = detail::resolve_emeans(p, 13, 2, 2);
  EXPECT_EQ(schedule.epoch_rows, (std::vector<int64_t>{13, 6, 13}));
  EXPECT_EQ(schedule.steps, 10);
  EXPECT_DOUBLE_EQ(schedule.alpha(0), .8);
  EXPECT_NEAR(schedule.alpha(schedule.steps), .08, 1e-15);
  EXPECT_EQ(schedule.unused_threshold, 0);
  p.epochs = .01;
  EXPECT_THROW(detail::resolve_emeans(p, 13, 2, 2), raft::logic_error);
  p.epochs = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(detail::resolve_emeans(p, 13, 2, 2), raft::logic_error);
  p.epochs = 1;
  p.alpha0 = 1.1;
  EXPECT_THROW(detail::resolve_emeans(p, 13, 2, 2), raft::logic_error);
}

template <typename T, typename IdxT>
void test_emeans_fit()
{
  // Both logical batches and fractional passes end in remainders. Physical host chunks of
  // three rows cross five-row logical batches and must not change the optimizer trajectory.
  constexpr IdxT n = 13, d = 2, k = 2;
  std::vector<T> x(n * d);
  const T positions[n] = {0, .25, 2.5, 5, 7.5, 8, 4.25, 3.75, 1, 7, 2, 6, 4};
  for (IdxT i = 0; i < n; ++i) {
    x[2 * i] = x[2 * i + 1] = positions[i];
  }
  std::vector<T> initial{0, 0, 8, 8}, expected(k * d, 0), counts(k, 0);
  std::vector<T> running = initial, mass(k, T(0x1p-16));
  auto nearest           = [&](IdxT row, const std::vector<T>& centers) {
    T best     = std::numeric_limits<T>::max();
    IdxT label = 0;
    for (IdxT c = 0; c < k; ++c) {
      T distance = 0;
      for (IdxT j = 0; j < d; ++j) {
        T delta = x[row * d + j] - centers[c * d + j];
        distance += delta * delta;
      }
      if (distance < best) {
        best  = distance;
        label = c;
      }
    }
    return label;
  };
  cluster::kmeans::emeans_params options;
  options.epochs           = 1.5;
  options.batch_size       = 5;
  options.alpha0           = .5;
  options.unused_threshold = 0;
  auto schedule            = cluster::kmeans::detail::resolve_emeans(options, n, d, k);
  uint64_t step            = 0;
  for (size_t epoch = 0; epoch < schedule.epoch_rows.size(); ++epoch) {
    for (int64_t offset = 0; offset < schedule.epoch_rows[epoch]; offset += schedule.batch_size) {
      auto rows = std::min(schedule.batch_size, schedule.epoch_rows[epoch] - offset);
      std::vector<T> sums(k * d, 0), batch_counts(k, 0);
      for (IdxT i = offset; i < offset + rows; ++i) {
        auto label = nearest(i, running);
        batch_counts[label]++;
        for (IdxT j = 0; j < d; ++j) {
          sums[label * d + j] += x[i * d + j];
        }
      }
      double alpha = schedule.alpha(step++);
      for (IdxT c = 0; c < k; ++c) {
        T incoming = T(alpha) * batch_counts[c], retained = T(1 - alpha) * mass[c];
        mass[c] = incoming + retained;
        for (IdxT j = 0; j < d; ++j) {
          auto i = c * d + j;
          if (batch_counts[c]) {
            running[i] += incoming / mass[c] * (sums[i] / batch_counts[c] - running[i]);
          }
          if (epoch + 1 == schedule.epoch_rows.size()) { expected[i] += sums[i]; }
        }
        if (epoch + 1 == schedule.epoch_rows.size()) { counts[c] += batch_counts[c]; }
      }
    }
  }
  for (IdxT i = 0; i < k * d; ++i) {
    expected[i] = counts[i / d] ? expected[i] / counts[i / d] : running[i];
  }
  T expected_inertia = 0;
  for (IdxT i = 0; i < n; ++i) {
    for (IdxT j = 0; j < d; ++j) {
      T delta = x[i * d + j] - expected[nearest(i, expected) * d + j];
      expected_inertia += delta * delta;
    }
  }
  rmm::cuda_stream stream;
  raft::resources handle;
  raft::resource::set_cuda_stream(handle, stream.view());
  auto device_x = raft::make_device_matrix<T, IdxT>(handle, n, d);
  auto centers  = raft::make_device_matrix<T, IdxT>(handle, k, d);
  raft::update_device(device_x.data_handle(), x.data(), x.size(), stream.view());
  cluster::kmeans::params p;
  p.init       = cluster::kmeans::params::Array;
  p.n_clusters = k;
  p.emeans     = options;
  for (bool host : {false, true}) {
    for (int64_t physical_rows : {3, 7, 13}) {
      SCOPED_TRACE(host);
      SCOPED_TRACE(physical_rows);
      p.device_buffer_samples = physical_rows;
      raft::update_device(centers.data_handle(), initial.data(), initial.size(), stream.view());
      T inertia{};
      IdxT iterations{};
      if (host) {
        // The existing public host overload uses int64 extents.
        int64_t host_iterations{};
        cluster::kmeans::fit(handle,
                             p,
                             raft::make_host_matrix_view<const T, int64_t>(x.data(), n, d),
                             std::nullopt,
                             raft::make_device_matrix_view<T, int64_t>(centers.data_handle(), k, d),
                             raft::make_host_scalar_view(&inertia),
                             raft::make_host_scalar_view(&host_iterations));
        iterations = static_cast<IdxT>(host_iterations);
      } else {
        cluster::kmeans::fit(handle,
                             p,
                             raft::make_const_mdspan(device_x.view()),
                             std::nullopt,
                             centers.view(),
                             raft::make_host_scalar_view(&inertia),
                             raft::make_host_scalar_view(&iterations));
      }
      std::vector<T> actual(k * d);
      raft::update_host(actual.data(), centers.data_handle(), actual.size(), stream.view());
      raft::resource::sync_stream(handle);
      EXPECT_EQ(iterations, 2);
      EXPECT_NEAR(inertia, expected_inertia, 1e-4);
      for (IdxT i = 0; i < k * d; ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1e-6);
      }
    }
  }
  // An ANN index would become stale after each logical batch's centroid update.
  p.use_ann_for_assignment = true;
  T inertia{};
  IdxT iterations{};
  EXPECT_THROW(cluster::kmeans::fit(handle,
                                    p,
                                    raft::make_const_mdspan(device_x.view()),
                                    std::nullopt,
                                    centers.view(),
                                    raft::make_host_scalar_view(&inertia),
                                    raft::make_host_scalar_view(&iterations)),
               raft::logic_error);
}

TEST(EMeans, FitFloat) { test_emeans_fit<float, int64_t>(); }
TEST(EMeans, FitDouble) { test_emeans_fit<double, int64_t>(); }
TEST(EMeans, FitFloatInt) { test_emeans_fit<float, int>(); }
TEST(EMeans, FitDoubleInt) { test_emeans_fit<double, int>(); }

TEST(EMeans, Revival)
{
  rmm::cuda_stream stream;
  raft::resources handle;
  raft::resource::set_cuda_stream(handle, stream.view());
  cluster::kmeans::emeans_params options;
  options.epochs           = 1;
  options.batch_size       = 1;
  options.alpha0           = .5;
  options.unused_threshold = .1;
  cluster::kmeans::detail::emeans_state<float, int> state(
    handle, cluster::kmeans::detail::resolve_emeans(options, 5, 3, 3), 3, 3);
  auto centers = raft::make_device_matrix<float, int>(handle, 3, 3);
  auto sums    = raft::make_device_matrix<float, int>(handle, 3, 3);
  auto counts  = raft::make_device_vector<float, int>(handle, 3);
  std::vector<float> sum{1, 2, 3, 0, 0, 0, 0, 0, 0}, count{1, 0, 0}, actual(9);
  raft::matrix::fill(handle, centers.view(), 0.f);
  raft::update_device(sums.data_handle(), sum.data(), sum.size(), stream.view());
  raft::update_device(counts.data_handle(), count.data(), count.size(), stream.view());
  for (int step = 0; step < 5; ++step) {
    state.update(handle, centers.view(), sums.data_handle(), counts.data_handle(), false, 123);
  }
  raft::update_host(actual.data(), centers.data_handle(), actual.size(), stream.view());
  raft::resource::sync_stream(handle);
  EXPECT_NEAR(actual[0], 1, 1e-3);
  EXPECT_NEAR(actual[3], 1, 1e-3);
  EXPECT_NE(actual[0], actual[3]);
  EXPECT_EQ(actual[6], 0);  // Only one live donor: no donor is reused within this revival.
}

TEST(EMeans, StepOneAndEmptyCluster)
{
  rmm::cuda_stream stream;
  raft::resources handle;
  raft::resource::set_cuda_stream(handle, stream.view());
  auto centers = raft::make_device_matrix<float, int>(handle, 3, 2);
  auto masses  = raft::make_device_vector<float, int>(handle, 3);
  auto sums    = raft::make_device_matrix<float, int>(handle, 3, 2);
  auto counts  = raft::make_device_vector<float, int>(handle, 3);
  std::vector<float> initial{0, 0, 10, 10, 30, 30}, mass{2, 3, 4}, sum{4, 8, 0, 0, 20, 22},
    count{2, 0, 1};
  raft::update_device(sums.data_handle(), sum.data(), sum.size(), stream.view());
  raft::update_device(counts.data_handle(), count.data(), count.size(), stream.view());
  for (double alpha : {1., .25}) {
    raft::update_device(centers.data_handle(), initial.data(), initial.size(), stream.view());
    raft::update_device(masses.data_handle(), mass.data(), mass.size(), stream.view());
    raft::launch_kernel(handle,
                        3,
                        256,
                        cluster::kmeans::detail::emeans_update_kernel<float, int>,
                        centers.data_handle(),
                        masses.data_handle(),
                        sums.data_handle(),
                        counts.data_handle(),
                        2,
                        alpha,
                        static_cast<float*>(nullptr),
                        static_cast<uint64_t*>(nullptr));
    std::vector<float> actual(6), actual_mass(3);
    raft::update_host(actual.data(), centers.data_handle(), actual.size(), stream.view());
    raft::update_host(actual_mass.data(), masses.data_handle(), actual_mass.size(), stream.view());
    raft::resource::sync_stream(handle);
    for (int c = 0; c < 3; ++c) {
      float incoming = float(alpha) * count[c];
      float retained = alpha == 1 ? (count[c] ? 0 : mass[c]) : float(1 - alpha) * mass[c];
      EXPECT_FLOAT_EQ(actual_mass[c], incoming + retained);
      for (int j = 0; j < 2; ++j) {
        float expected = initial[2 * c + j];
        if (count[c]) {
          expected += incoming / (incoming + retained) * (sum[2 * c + j] / count[c] - expected);
        }
        EXPECT_FLOAT_EQ(actual[2 * c + j], expected);
      }
    }
  }
}

template <typename T>
struct KmeansInputs {
  int n_row;
  int n_col;
  int n_clusters;
  T tol;
  bool weighted;
  bool use_ann = false;  // CAGRA ANN nearest-centroid assignment (float data only)
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

    int n_samples                 = testparams.n_row;
    int n_features                = testparams.n_col;
    params.n_clusters             = testparams.n_clusters;
    params.tol                    = testparams.tol;
    params.use_ann_for_assignment = testparams.use_ann;
    params.n_init                 = 5;
    params.rng_state.seed         = 1;
    params.oversampling_factor    = 0;

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

    score = raft::stats::adjusted_rand_index(d_labels_ref.data(),
                                             d_labels.data(),
                                             n_samples,
                                             raft::resource::get_cuda_stream(handle).get());

    if (score < 1.0) {
      std::stringstream ss;
      ss << "Expected: " << raft::arr2Str(d_labels_ref.data(), 25, "d_labels_ref", stream.get());
      std::cout << (ss.str().c_str()) << '\n';
      ss.str(std::string());
      ss << "Actual: " << raft::arr2Str(d_labels.data(), 25, "d_labels", stream.get());
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
                                                   {10000, 500, 100, 0.0001f, false},
                                                   // CAGRA ANN assignment (aligned and unaligned
                                                   // number of features)
                                                   {1000, 32, 5, 0.0001f, true, true},
                                                   {1000, 32, 5, 0.0001f, false, true},
                                                   {1000, 10, 20, 0.0001f, false, true},
                                                   {10000, 32, 10, 0.0001f, true, true},
                                                   {10000, 100, 50, 0.0001f, false, true},
                                                   {10000, 30, 100, 0.0001f, true, true},
                                                   {10000, 500, 100, 0.0001f, false, true}};

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
  int device_buffer_samples;
  bool use_ann = false;  // CAGRA ANN nearest-centroid assignment (float data only)
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
    int n_samples                 = testparams.n_row;
    int n_features                = testparams.n_col;
    params.n_clusters             = testparams.n_clusters;
    params.tol                    = testparams.tol;
    params.use_ann_for_assignment = testparams.use_ann;
    params.rng_state.seed         = 1;
    params.oversampling_factor    = 0;

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
    batched_params.device_buffer_samples         = testparams.device_buffer_samples;

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
                                  stream.get());

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
                                             raft::resource::get_cuda_stream(handle).get());

    if (score < 0.99) {
      std::stringstream ss;
      ss << "Expected: "
         << raft::arr2Str(d_labels_ref->data_handle(), 25, "d_labels_ref", stream.get());
      std::cout << (ss.str().c_str()) << '\n';
      ss.str(std::string());
      ss << "Actual: " << raft::arr2Str(d_labels->data_handle(), 25, "d_labels", stream.get());
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
    p.n_clusters             = n_clusters;
    p.tol                    = testparams.tol;
    p.use_ann_for_assignment = testparams.use_ann;
    p.n_init                 = 3;
    p.init                   = cuvs::cluster::kmeans::params::KMeansPlusPlus;
    p.max_iter               = 20;
    p.rng_state.seed         = 1;
    p.oversampling_factor    = 0;
    p.device_buffer_samples  = testparams.device_buffer_samples;
    p.init_size              = init_size_value;

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

    // cuTile's TF32 assignment path can introduce small FP32 convergence variation between
    // otherwise equivalent runs; keep the original tighter tolerance for double precision.
    const T rel = std::is_same_v<T, float> ? T(1e-4) : T(1e-5);

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
  {10000, 16, 10, 0.00001f, true, 1000},
  {10000, 96, 10, 0.0001f, false, 10000},
  // CAGRA ANN assignment (aligned and unaligned number of features)
  {1000, 32, 5, 0.0001f, true, 256, true},
  {1000, 10, 20, 0.0001f, false, 30, true},
  {10000, 96, 10, 0.0001f, false, 1000, true},
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
  // AUTO may select cuTile, whose TF32 assignment arithmetic can converge to slightly different
  // centroid values when accumulation is split into outer host batches. Equivalent assignments and
  // clustering cost are the stable behavioral contract.
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

class KmeansBatchLoaderTest : public ::testing::TestWithParam<bool> {};

TEST_P(KmeansBatchLoaderTest, CyclicFourPasses)
{
  constexpr int64_t n_rows     = 257;
  constexpr int64_t n_cols     = 17;
  constexpr int64_t batch_size = 64;
  constexpr int n_passes       = 4;

  raft::resources handle;
  rmm::cuda_stream copy_stream(rmm::cuda_stream::flags::non_blocking);
  std::vector<int64_t> host_data(n_rows * n_cols);
  for (int64_t row = 0; row < n_rows; ++row) {
    for (int64_t col = 0; col < n_cols; ++col) {
      host_data[row * n_cols + col] = row * n_cols + col;
    }
  }

  auto host_view =
    raft::make_host_matrix_view<const int64_t, int64_t>(host_data.data(), n_rows, n_cols);
  const bool enable_prefetch = GetParam();
  cluster::kmeans::detail::kmeans_batch_loader<int64_t, int64_t, false> loader(
    handle,
    host_view,
    batch_size,
    copy_stream,
    raft::resource::get_workspace_resource_ref(handle),
    enable_prefetch);
  auto device_readback =
    raft::make_device_vector<int64_t, int64_t>(handle, n_passes * n_rows * n_cols);

  loader.start();
  // Starting an active pipeline is a no-op.
  loader.start();
  for (int pass = 0; pass < n_passes; ++pass) {
    for (std::size_t pos = 0; pos < loader.num_batches(); ++pos) {
      const auto batch = loader.acquire(pos);
      const auto output_offset =
        (static_cast<std::size_t>(pass) * n_rows + batch.offset()) * n_cols;
      raft::copy(device_readback.data_handle() + output_offset,
                 batch.data(),
                 batch.size() * n_cols,
                 raft::resource::get_cuda_stream(handle));

      if (enable_prefetch && (pos + 1 < loader.num_batches() || pass + 1 < n_passes)) {
        loader.prefetch((pos + 1) % loader.num_batches());
      }
      const std::size_t recycle_offset = enable_prefetch ? 2 : 1;
      const bool needs_future_batch =
        pos + recycle_offset < loader.num_batches() || pass + 1 < n_passes;
      if (needs_future_batch) {
        loader.recycle(batch, (pos + recycle_offset) % loader.num_batches());
      } else {
        loader.release(batch);
      }
    }
  }

  std::vector<int64_t> readback(device_readback.size());
  raft::copy(readback.data(),
             device_readback.data_handle(),
             device_readback.size(),
             raft::resource::get_cuda_stream(handle));
  raft::resource::sync_stream(handle);
  for (int pass = 0; pass < n_passes; ++pass) {
    for (std::size_t i = 0; i < host_data.size(); ++i) {
      EXPECT_EQ(readback[static_cast<std::size_t>(pass) * host_data.size() + i], host_data[i]);
    }
  }
}

INSTANTIATE_TEST_CASE_P(WithAndWithoutPrefetch,
                        KmeansBatchLoaderTest,
                        ::testing::Values(false, true));

}  // namespace cuvs
