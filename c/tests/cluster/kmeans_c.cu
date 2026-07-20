/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_utils.cuh"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <rmm/device_uvector.hpp>

#include "../../src/core/interop.hpp"
#include <cuvs/cluster/kmeans.h>
#include <cuvs/core/c_api.h>

#include <cstdint>

namespace {

constexpr int64_t kNSamples  = 8;
constexpr int64_t kNFeatures = 2;
constexpr int kNClusters     = 2;

float kDataset[kNSamples][kNFeatures] = {
  {1.0f, 1.0f},
  {1.0f, 2.0f},
  {2.0f, 1.0f},
  {2.0f, 2.0f},
  {10.0f, 10.0f},
  {10.0f, 11.0f},
  {11.0f, 10.0f},
  {11.0f, 11.0f},
};

float kInitCentroids[kNClusters][kNFeatures] = {
  {0.0f, 0.0f},
  {12.0f, 12.0f},
};

float kExpectedCentroids[kNClusters * kNFeatures] = {1.5f, 1.5f, 10.5f, 10.5f};
int32_t kExpectedLabels[kNSamples]                = {0, 0, 0, 0, 1, 1, 1, 1};

// 8 points, each at squared distance 0.5 from its cluster mean -> 4.0.
constexpr double kExpectedInertia = 4.0;

// Type-erased dispatcher to exercise both the v1 and v2 entry points with
// shared test bodies.
struct kmeans_api_v1 {
  using params_t = cuvsKMeansParams_t;
  static cuvsError_t params_create(params_t* p) { return cuvsKMeansParamsCreate(p); }
  static cuvsError_t params_destroy(params_t p) { return cuvsKMeansParamsDestroy(p); }
  static cuvsError_t fit(cuvsResources_t res,
                         params_t params,
                         DLManagedTensor* dataset,
                         DLManagedTensor* centroids,
                         double* inertia,
                         int* n_iter)
  {
    return cuvsKMeansFit(res, params, dataset, NULL, centroids, inertia, n_iter);
  }
  static cuvsError_t predict(cuvsResources_t res,
                             params_t params,
                             DLManagedTensor* dataset,
                             DLManagedTensor* centroids,
                             DLManagedTensor* labels,
                             double* inertia)
  {
    return cuvsKMeansPredict(
      res, params, dataset, NULL, centroids, labels, false, inertia);
  }
};

struct kmeans_api_v2 {
  using params_t = cuvsKMeansParams_v2_t;
  static cuvsError_t params_create(params_t* p) { return cuvsKMeansParamsCreate_v2(p); }
  static cuvsError_t params_destroy(params_t p) { return cuvsKMeansParamsDestroy_v2(p); }
  static cuvsError_t fit(cuvsResources_t res,
                         params_t params,
                         DLManagedTensor* dataset,
                         DLManagedTensor* centroids,
                         double* inertia,
                         int* n_iter)
  {
    return cuvsKMeansFit_v2(res, params, dataset, NULL, centroids, inertia, n_iter);
  }
  static cuvsError_t predict(cuvsResources_t res,
                             params_t params,
                             DLManagedTensor* dataset,
                             DLManagedTensor* centroids,
                             DLManagedTensor* labels,
                             double* inertia)
  {
    return cuvsKMeansPredict_v2(
      res, params, dataset, NULL, centroids, labels, false, inertia);
  }
};

template <typename Api>
void test_fit_predict()
{
  raft::handle_t handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  rmm::device_uvector<float> dataset_d(kNSamples * kNFeatures, stream);
  rmm::device_uvector<float> centroids_d(kNClusters * kNFeatures, stream);
  rmm::device_uvector<int32_t> labels_d(kNSamples, stream);

  raft::copy(dataset_d.data(),
             reinterpret_cast<float const*>(kDataset),
             kNSamples * kNFeatures,
             stream);
  raft::copy(centroids_d.data(),
             reinterpret_cast<float const*>(kInitCentroids),
             kNClusters * kNFeatures,
             stream);

  cuvsResources_t res;
  ASSERT_EQ(cuvsResourcesCreate(&res), CUVS_SUCCESS);

  typename Api::params_t params;
  ASSERT_EQ(Api::params_create(&params), CUVS_SUCCESS);
  params->n_clusters           = kNClusters;
  params->max_iter             = 100;
  params->tol                  = 1e-6;
  params->init                 = Array;
  params->streaming_batch_size = 0;

  DLManagedTensor dataset_t{};
  cuvs::core::to_dlpack(
    raft::make_device_matrix_view<float, int64_t>(dataset_d.data(), kNSamples, kNFeatures),
    &dataset_t);

  DLManagedTensor centroids_t{};
  cuvs::core::to_dlpack(
    raft::make_device_matrix_view<float, int64_t>(centroids_d.data(), kNClusters, kNFeatures),
    &centroids_t);

  DLManagedTensor labels_t{};
  cuvs::core::to_dlpack(
    raft::make_device_vector_view<int32_t, int64_t>(labels_d.data(), kNSamples), &labels_t);

  double inertia         = -1.0;
  int n_iter             = -1;
  double predict_inertia = -1.0;
  double cluster_cost    = -1.0;

  ASSERT_EQ(Api::fit(res, params, &dataset_t, &centroids_t, &inertia, &n_iter), CUVS_SUCCESS);
  ASSERT_EQ(Api::predict(res, params, &dataset_t, &centroids_t, &labels_t, &predict_inertia),
            CUVS_SUCCESS);
  ASSERT_EQ(cuvsKMeansClusterCost(res, &dataset_t, &centroids_t, &cluster_cost), CUVS_SUCCESS);

  ASSERT_TRUE(cuvs::devArrMatchHost(kExpectedCentroids,
                                    centroids_d.data(),
                                    kNClusters * kNFeatures,
                                    cuvs::CompareApprox<float>(1e-4f)));
  ASSERT_TRUE(cuvs::devArrMatchHost(
    kExpectedLabels, labels_d.data(), kNSamples, cuvs::Compare<int32_t>()));

  EXPECT_GT(n_iter, 0);
  EXPECT_NEAR(inertia, kExpectedInertia, 1e-4);
  EXPECT_NEAR(predict_inertia, kExpectedInertia, 1e-4);
  EXPECT_NEAR(cluster_cost, kExpectedInertia, 1e-4);

  labels_t.deleter(&labels_t);
  centroids_t.deleter(&centroids_t);
  dataset_t.deleter(&dataset_t);

  ASSERT_EQ(Api::params_destroy(params), CUVS_SUCCESS);
  ASSERT_EQ(cuvsResourcesDestroy(res), CUVS_SUCCESS);
}

template <typename Api>
void test_fit_host()
{
  raft::handle_t handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  rmm::device_uvector<float> centroids_d(kNClusters * kNFeatures, stream);
  raft::copy(centroids_d.data(),
             reinterpret_cast<float const*>(kInitCentroids),
             kNClusters * kNFeatures,
             stream);

  cuvsResources_t res;
  ASSERT_EQ(cuvsResourcesCreate(&res), CUVS_SUCCESS);

  typename Api::params_t params;
  ASSERT_EQ(Api::params_create(&params), CUVS_SUCCESS);
  params->n_clusters           = kNClusters;
  params->max_iter             = 100;
  params->tol                  = 1e-6;
  params->init                 = Array;
  params->streaming_batch_size = 4;  // force at least 2 streamed batches

  DLManagedTensor dataset_t{};
  cuvs::core::to_dlpack(
    raft::make_host_matrix_view<float, int64_t>(
      reinterpret_cast<float*>(kDataset), kNSamples, kNFeatures),
    &dataset_t);

  DLManagedTensor centroids_t{};
  cuvs::core::to_dlpack(
    raft::make_device_matrix_view<float, int64_t>(centroids_d.data(), kNClusters, kNFeatures),
    &centroids_t);

  double inertia = -1.0;
  int n_iter     = -1;

  ASSERT_EQ(Api::fit(res, params, &dataset_t, &centroids_t, &inertia, &n_iter), CUVS_SUCCESS);

  ASSERT_TRUE(cuvs::devArrMatchHost(kExpectedCentroids,
                                    centroids_d.data(),
                                    kNClusters * kNFeatures,
                                    cuvs::CompareApprox<float>(1e-4f)));

  EXPECT_GT(n_iter, 0);
  EXPECT_NEAR(inertia, kExpectedInertia, 1e-4);

  centroids_t.deleter(&centroids_t);
  dataset_t.deleter(&dataset_t);

  ASSERT_EQ(Api::params_destroy(params), CUVS_SUCCESS);
  ASSERT_EQ(cuvsResourcesDestroy(res), CUVS_SUCCESS);
}

}  // namespace

TEST(KMeansC, FitPredict) { test_fit_predict<kmeans_api_v1>(); }
// TODO(cuVS 26.08): remove FitPredictV2 once `_v2` is promoted to the
// unsuffixed ABI -- it will be redundant with FitPredict at that point.
TEST(KMeansC, FitPredictV2) { test_fit_predict<kmeans_api_v2>(); }

TEST(KMeansC, FitHost) { test_fit_host<kmeans_api_v1>(); }
// TODO(cuVS 26.08): remove FitHostV2 once `_v2` is promoted to the
// unsuffixed ABI.
TEST(KMeansC, FitHostV2) { test_fit_host<kmeans_api_v2>(); }

TEST(KMeansC, ParamsCreateDestroy)
{
  cuvsKMeansParams_t params = nullptr;
  ASSERT_EQ(cuvsKMeansParamsCreate(&params), CUVS_SUCCESS);
  ASSERT_NE(params, nullptr);
  EXPECT_GT(params->n_clusters, 0);
  EXPECT_GT(params->max_iter, 0);
  ASSERT_EQ(cuvsKMeansParamsDestroy(params), CUVS_SUCCESS);
}

// TODO(cuVS 26.08): remove ParamsCreateDestroyV2 once cuvsKMeansParamsCreate_v2
// / cuvsKMeansParamsDestroy_v2 are promoted to the unsuffixed entry points and
// the `_v2` symbols are deleted from the public header.
TEST(KMeansC, ParamsCreateDestroyV2)
{
  cuvsKMeansParams_v2_t params = nullptr;
  ASSERT_EQ(cuvsKMeansParamsCreate_v2(&params), CUVS_SUCCESS);
  ASSERT_NE(params, nullptr);
  EXPECT_GT(params->n_clusters, 0);
  EXPECT_GT(params->max_iter, 0);
  ASSERT_EQ(cuvsKMeansParamsDestroy_v2(params), CUVS_SUCCESS);
}
