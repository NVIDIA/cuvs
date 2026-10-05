/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <cuda/stream>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>

#include "../../src/core/interop.hpp"
#include <cuvs/cluster/soar.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace {

constexpr int64_t kNRows     = 512;
constexpr int64_t kDim       = 16;
constexpr int64_t kNClusters = 24;

template <typename T>
DLManagedTensor make_matrix_tensor(T* data, int64_t rows, int64_t cols)
{
  DLManagedTensor tensor{};
  cuvs::core::to_dlpack(raft::make_device_matrix_view<T, int64_t, raft::row_major>(data, rows, cols),
                        &tensor);
  return tensor;
}

template <typename T>
DLManagedTensor make_vector_tensor(T* data, int64_t size)
{
  DLManagedTensor tensor{};
  cuvs::core::to_dlpack(raft::make_device_vector_view<T, int64_t>(data, size), &tensor);
  return tensor;
}

void free_tensor(DLManagedTensor& t)
{
  if (t.deleter) { t.deleter(&t); }
}

/** Uniform random matrix in [-1, 1], generated on the host so the tests are reproducible. */
std::vector<float> random_matrix(int64_t n_rows, int64_t dim, uint64_t seed)
{
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> data(n_rows * dim);
  std::generate(data.begin(), data.end(), [&]() { return dist(rng); });
  return data;
}

/** Index of the closest centroid in L2, i.e. what k-means prediction would produce. */
std::vector<uint32_t> nearest_centroids(const std::vector<float>& dataset,
                                        const std::vector<float>& centroids,
                                        int64_t n_rows,
                                        int64_t dim,
                                        int64_t n_clusters)
{
  std::vector<uint32_t> labels(n_rows);
  for (int64_t i = 0; i < n_rows; i++) {
    double best_distance = std::numeric_limits<double>::max();
    for (int64_t c = 0; c < n_clusters; c++) {
      double distance = 0.0;
      for (int64_t k = 0; k < dim; k++) {
        double diff = static_cast<double>(dataset[i * dim + k]) - centroids[c * dim + k];
        distance += diff * diff;
      }
      if (distance < best_distance) {
        best_distance = distance;
        labels[i]     = static_cast<uint32_t>(c);
      }
    }
  }
  return labels;
}

/**
 * `||x - c||^2 + lambda * (dot(r / ||r||, x - c))^2`, the loss the implementation minimizes over
 * all centroids, up to a per-row constant that does not move the argmin.
 */
double soar_score(const float* x,
                  const float* residual,
                  const float* centroid,
                  int64_t dim,
                  float lambda)
{
  double residual_norm = 0.0;
  for (int64_t k = 0; k < dim; k++) {
    residual_norm += static_cast<double>(residual[k]) * residual[k];
  }
  residual_norm = std::sqrt(residual_norm);

  double squared_distance = 0.0;
  double projection       = 0.0;
  for (int64_t k = 0; k < dim; k++) {
    double diff = static_cast<double>(x[k]) - centroid[k];
    squared_distance += diff * diff;
    projection += diff * (residual[k] / residual_norm);
  }

  return squared_distance + static_cast<double>(lambda) * projection * projection;
}

/**
 * Holds the random dataset, the primary k-means labels and the matching device buffers shared by
 * the tests below.
 */
struct SoarFixture {
  explicit SoarFixture(cuda::stream_ref stream)
    : dataset(kNRows * kDim, stream),
      centroids(kNClusters * kDim, stream),
      labels(kNRows, stream),
      soar_labels(kNRows, stream)
  {
    h_dataset   = random_matrix(kNRows, kDim, 1234ULL);
    h_centroids = random_matrix(kNClusters, kDim, 5678ULL);
    h_labels    = nearest_centroids(h_dataset, h_centroids, kNRows, kDim, kNClusters);

    raft::update_device(dataset.data(), h_dataset.data(), h_dataset.size(), stream);
    raft::update_device(centroids.data(), h_centroids.data(), h_centroids.size(), stream);
    raft::update_device(labels.data(), h_labels.data(), h_labels.size(), stream);
  }

  std::vector<float> h_dataset;
  std::vector<float> h_centroids;
  std::vector<uint32_t> h_labels;

  rmm::device_uvector<float> dataset;
  rmm::device_uvector<float> centroids;
  rmm::device_uvector<uint32_t> labels;
  rmm::device_uvector<uint32_t> soar_labels;
};

}  // namespace

TEST(SoarC, PredictMatchesExhaustiveHostSearch)
{
  raft::device_resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  SoarFixture fixture(stream);
  handle.sync_stream();

  cuvsResources_t res;
  ASSERT_EQ(cuvsResourcesCreate(&res), CUVS_SUCCESS);

  cuvsSoarParams_t params;
  ASSERT_EQ(cuvsSoarParamsCreate(&params), CUVS_SUCCESS);
  ASSERT_FLOAT_EQ(params->lambda, 1.0f) << "default lambda should match the C++ params";

  // Deliberately not the default, so the test fails if the wrapper drops lambda on its way to
  // the C++ implementation.
  params->lambda = 2.0f;

  auto dataset_t     = make_matrix_tensor(fixture.dataset.data(), kNRows, kDim);
  auto centroids_t   = make_matrix_tensor(fixture.centroids.data(), kNClusters, kDim);
  auto labels_t      = make_vector_tensor(fixture.labels.data(), kNRows);
  auto soar_labels_t = make_vector_tensor(fixture.soar_labels.data(), kNRows);

  ASSERT_EQ(cuvsSoarPredict(res, params, &dataset_t, &centroids_t, &labels_t, &soar_labels_t),
            CUVS_SUCCESS)
    << cuvsGetLastErrorText();

  ASSERT_EQ(cuvsStreamSync(res), CUVS_SUCCESS);

  std::vector<uint32_t> h_soar_labels(kNRows);
  raft::update_host(h_soar_labels.data(), fixture.soar_labels.data(), kNRows, stream);
  handle.sync_stream();

  // Compare losses rather than ids so the test is not fragile when two centroids tie.
  std::vector<float> residual(kDim);
  int64_t n_spilled = 0;
  for (int64_t i = 0; i < kNRows; i++) {
    ASSERT_LT(h_soar_labels[i], kNClusters) << "row " << i << " got an out-of-range label";
    if (h_soar_labels[i] != fixture.h_labels[i]) { ++n_spilled; }

    const float* x       = &fixture.h_dataset[i * kDim];
    const float* primary = &fixture.h_centroids[fixture.h_labels[i] * kDim];
    for (int64_t k = 0; k < kDim; k++) {
      residual[k] = x[k] - primary[k];
    }

    double best_score = std::numeric_limits<double>::max();
    for (int64_t c = 0; c < kNClusters; c++) {
      best_score = std::min(
        best_score,
        soar_score(x, residual.data(), &fixture.h_centroids[c * kDim], kDim, params->lambda));
    }

    double actual_score = soar_score(
      x, residual.data(), &fixture.h_centroids[h_soar_labels[i] * kDim], kDim, params->lambda);

    ASSERT_NEAR(actual_score, best_score, 1e-3 * std::max(1.0, std::abs(best_score)))
      << "row " << i << " was not assigned a loss-minimizing centroid";
  }

  // Guards against the loss check above going vacuous: if the fixture ever stopped spilling,
  // simply echoing the primary labels would satisfy it.
  EXPECT_GT(n_spilled, 0) << "expected at least one row to spill";

  free_tensor(dataset_t);
  free_tensor(centroids_t);
  free_tensor(labels_t);
  free_tensor(soar_labels_t);
  ASSERT_EQ(cuvsSoarParamsDestroy(params), CUVS_SUCCESS);
  ASSERT_EQ(cuvsResourcesDestroy(res), CUVS_SUCCESS);
}

TEST(SoarC, Int32LabelsMatchUint32Labels)
{
  raft::device_resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  SoarFixture fixture(stream);

  // The same primary labels, as int32, which is what cuvsKMeansPredict writes.
  std::vector<int32_t> h_labels_i32(fixture.h_labels.begin(), fixture.h_labels.end());
  rmm::device_uvector<int32_t> labels_i32(kNRows, stream);
  rmm::device_uvector<int32_t> soar_labels_i32(kNRows, stream);
  raft::update_device(labels_i32.data(), h_labels_i32.data(), h_labels_i32.size(), stream);
  handle.sync_stream();

  cuvsResources_t res;
  ASSERT_EQ(cuvsResourcesCreate(&res), CUVS_SUCCESS);
  cuvsSoarParams_t params;
  ASSERT_EQ(cuvsSoarParamsCreate(&params), CUVS_SUCCESS);

  auto dataset_t   = make_matrix_tensor(fixture.dataset.data(), kNRows, kDim);
  auto centroids_t = make_matrix_tensor(fixture.centroids.data(), kNClusters, kDim);

  auto labels_u32_t      = make_vector_tensor(fixture.labels.data(), kNRows);
  auto soar_labels_u32_t = make_vector_tensor(fixture.soar_labels.data(), kNRows);
  ASSERT_EQ(
    cuvsSoarPredict(res, params, &dataset_t, &centroids_t, &labels_u32_t, &soar_labels_u32_t),
    CUVS_SUCCESS)
    << cuvsGetLastErrorText();

  auto labels_i32_t      = make_vector_tensor(labels_i32.data(), kNRows);
  auto soar_labels_i32_t = make_vector_tensor(soar_labels_i32.data(), kNRows);
  ASSERT_EQ(
    cuvsSoarPredict(res, params, &dataset_t, &centroids_t, &labels_i32_t, &soar_labels_i32_t),
    CUVS_SUCCESS)
    << cuvsGetLastErrorText();

  ASSERT_EQ(cuvsStreamSync(res), CUVS_SUCCESS);

  std::vector<uint32_t> h_from_u32(kNRows);
  std::vector<int32_t> h_from_i32(kNRows);
  raft::update_host(h_from_u32.data(), fixture.soar_labels.data(), kNRows, stream);
  raft::update_host(h_from_i32.data(), soar_labels_i32.data(), kNRows, stream);
  handle.sync_stream();

  for (int64_t i = 0; i < kNRows; i++) {
    ASSERT_EQ(static_cast<int32_t>(h_from_u32[i]), h_from_i32[i])
      << "row " << i << " differs between the uint32 and int32 label paths";
  }

  free_tensor(dataset_t);
  free_tensor(centroids_t);
  free_tensor(labels_u32_t);
  free_tensor(soar_labels_u32_t);
  free_tensor(labels_i32_t);
  free_tensor(soar_labels_i32_t);
  ASSERT_EQ(cuvsSoarParamsDestroy(params), CUVS_SUCCESS);
  ASSERT_EQ(cuvsResourcesDestroy(res), CUVS_SUCCESS);
}

TEST(SoarC, RejectsInvalidInputs)
{
  raft::device_resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  SoarFixture fixture(stream);
  rmm::device_uvector<int32_t> soar_labels_i32(kNRows, stream);
  rmm::device_uvector<double> labels_f64(kNRows, stream);
  handle.sync_stream();

  cuvsResources_t res;
  ASSERT_EQ(cuvsResourcesCreate(&res), CUVS_SUCCESS);
  cuvsSoarParams_t params;
  ASSERT_EQ(cuvsSoarParamsCreate(&params), CUVS_SUCCESS);

  auto dataset_t     = make_matrix_tensor(fixture.dataset.data(), kNRows, kDim);
  auto centroids_t   = make_matrix_tensor(fixture.centroids.data(), kNClusters, kDim);
  auto labels_t      = make_vector_tensor(fixture.labels.data(), kNRows);
  auto soar_labels_t = make_vector_tensor(fixture.soar_labels.data(), kNRows);

  // labels and soar_labels must agree on dtype.
  auto soar_labels_i32_t = make_vector_tensor(soar_labels_i32.data(), kNRows);
  EXPECT_EQ(cuvsSoarPredict(res, params, &dataset_t, &centroids_t, &labels_t, &soar_labels_i32_t),
            CUVS_ERROR);

  // float64 is not a valid label dtype.
  auto labels_f64_t = make_vector_tensor(labels_f64.data(), kNRows);
  EXPECT_EQ(cuvsSoarPredict(res, params, &dataset_t, &centroids_t, &labels_f64_t, &labels_f64_t),
            CUVS_ERROR);

  // The centroid dimensionality has to match the dataset.
  auto narrow_centroids_t = make_matrix_tensor(fixture.centroids.data(), kNClusters, kDim - 1);
  EXPECT_EQ(
    cuvsSoarPredict(res, params, &dataset_t, &narrow_centroids_t, &labels_t, &soar_labels_t),
    CUVS_ERROR);

  // soar_labels has to be as long as the dataset.
  auto short_soar_labels_t = make_vector_tensor(fixture.soar_labels.data(), kNRows - 1);
  EXPECT_EQ(
    cuvsSoarPredict(res, params, &dataset_t, &centroids_t, &labels_t, &short_soar_labels_t),
    CUVS_ERROR);

  free_tensor(dataset_t);
  free_tensor(centroids_t);
  free_tensor(labels_t);
  free_tensor(soar_labels_t);
  free_tensor(soar_labels_i32_t);
  free_tensor(labels_f64_t);
  free_tensor(narrow_centroids_t);
  free_tensor(short_soar_labels_t);
  ASSERT_EQ(cuvsSoarParamsDestroy(params), CUVS_SUCCESS);
  ASSERT_EQ(cuvsResourcesDestroy(res), CUVS_SUCCESS);
}
