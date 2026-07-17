/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CUVS_BUILD_MG_ALGOS
#error "KMEANS_MG_C_TEST requires BUILD_MG_ALGOS"
#endif

#include "../../src/core/interop.hpp"
#include <cuvs/cluster/kmeans.h>
#include <cuvs/cluster/mg_kmeans.h>
#include <cuvs/core/c_api.h>

#include <gtest/gtest.h>

#include <raft/core/host_mdspan.hpp>

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

float kExpectedCentroids[kNClusters * kNFeatures] = {1.5f, 1.5f, 10.5f, 10.5f};

// 8 points, each at squared distance 0.5 from its cluster mean -> 4.0.
constexpr double kExpectedInertia = 4.0;

struct kmeans_mg_api {
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
    return cuvsMultiGpuKMeansFit(res, params, dataset, NULL, centroids, inertia, n_iter);
  }
};

template <typename Api>
void test_mg_fit_host()
{
  float centroids_h[kNClusters][kNFeatures] = {
    {0.0f, 0.0f},
    {12.0f, 12.0f},
  };

  cuvsResources_t res;
  int32_t device_ids[1] = {0};
  DLManagedTensor device_ids_t{};
  cuvs::core::to_dlpack(raft::make_host_vector_view<int32_t, int64_t>(device_ids, 1),
                        &device_ids_t);
  ASSERT_EQ(cuvsMultiGpuResourcesCreateWithDeviceIds(&res, &device_ids_t), CUVS_SUCCESS);
  device_ids_t.deleter(&device_ids_t);

  typename Api::params_t params;
  ASSERT_EQ(Api::params_create(&params), CUVS_SUCCESS);
  EXPECT_FALSE(params->device_buffer_prefetch);
  params->n_clusters           = kNClusters;
  params->max_iter             = 100;
  params->tol                  = 1e-6;
  params->init                 = Array;
  params->device_buffer_batch_size = 4;  // force at least two device buffers
  params->device_buffer_prefetch = true;

  DLManagedTensor dataset_t{};
  cuvs::core::to_dlpack(raft::make_host_matrix_view<float, int64_t>(
                          reinterpret_cast<float*>(kDataset), kNSamples, kNFeatures),
                        &dataset_t);

  DLManagedTensor centroids_t{};
  cuvs::core::to_dlpack(raft::make_host_matrix_view<float, int64_t>(
                          reinterpret_cast<float*>(centroids_h), kNClusters, kNFeatures),
                        &centroids_t);

  double inertia = -1.0;
  int n_iter     = -1;

  ASSERT_EQ(Api::fit(res, params, &dataset_t, &centroids_t, &inertia, &n_iter), CUVS_SUCCESS);

  auto* centroids_data = reinterpret_cast<float*>(centroids_h);
  for (int i = 0; i < kNClusters * kNFeatures; ++i) {
    EXPECT_NEAR(centroids_data[i], kExpectedCentroids[i], 1e-4f);
  }

  EXPECT_GT(n_iter, 0);
  EXPECT_NEAR(inertia, kExpectedInertia, 1e-4);

  centroids_t.deleter(&centroids_t);
  dataset_t.deleter(&dataset_t);

  ASSERT_EQ(Api::params_destroy(params), CUVS_SUCCESS);
  ASSERT_EQ(cuvsMultiGpuResourcesDestroy(res), CUVS_SUCCESS);
}

}  // namespace

TEST(KMeansMgC, FitHost) { test_mg_fit_host<kmeans_mg_api>(); }
