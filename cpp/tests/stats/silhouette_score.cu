/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "../test_utils.cuh"

#include <cuvs/distance/distance.hpp>
#include <cuvs/stats/silhouette_score.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>

namespace cuvs {
namespace stats {

// parameter structure definition
struct silhouetteScoreParam {
  int nRows;
  int nCols;
  int nLabels;
  cuvs::distance::DistanceType metric;
  int chunk;
  double tolerance;
};

// test fixture class
template <typename LabelT, typename DataT>
class silhouetteScoreTest : public ::testing::TestWithParam<silhouetteScoreParam> {
 protected:
  silhouetteScoreTest()
    : d_X(0, raft::resource::get_cuda_stream(handle)),
      sampleSilScore(0, raft::resource::get_cuda_stream(handle)),
      d_labels(0, raft::resource::get_cuda_stream(handle))
  {
  }

  void host_silhouette_score()
  {
    // generating random value test input
    std::vector<double> h_X(nElements, 0.0);
    std::vector<int> h_labels(nRows, 0);
    std::random_device rd;
    std::default_random_engine dre(nElements * nLabels);
    std::uniform_int_distribution<int> intGenerator(0, nLabels - 1);
    std::uniform_real_distribution<double> realGenerator(0, 100);

    std::generate(h_X.begin(), h_X.end(), [&]() { return realGenerator(dre); });
    std::generate(h_labels.begin(), h_labels.end(), [&]() { return intGenerator(dre); });

    // allocating and initializing memory to the GPU
    auto stream = raft::resource::get_cuda_stream(handle);
    d_X.resize(nElements, stream);
    d_labels.resize(nElements, stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(d_X.data(), 0, d_X.size() * sizeof(DataT), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(d_labels.data(), 0, d_labels.size() * sizeof(LabelT), stream));
    sampleSilScore.resize(nElements, stream);

    raft::update_device(d_X.data(), &h_X[0], (int)nElements, stream);
    raft::update_device(d_labels.data(), &h_labels[0], (int)nElements, stream);

    // finding the distance matrix

    rmm::device_uvector<double> d_distanceMatrix(nRows * nRows, stream);
    double* h_distanceMatrix = (double*)malloc(nRows * nRows * sizeof(double*));

    auto d_X_view = raft::make_device_matrix_view<const DataT, int64_t>(d_X.data(), nRows, nCols);
    cuvs::distance::pairwise_distance(
      handle,
      d_X_view,
      d_X_view,
      raft::make_device_matrix_view<DataT, int64_t>(d_distanceMatrix.data(), nRows, nRows),
      params.metric);

    raft::resource::sync_stream(handle, stream);

    raft::update_host(h_distanceMatrix, d_distanceMatrix.data(), nRows * nRows, stream);

    // finding the bincount array

    double* binCountArray = (double*)malloc(nLabels * sizeof(double*));
    memset(binCountArray, 0, nLabels * sizeof(double));

    for (int i = 0; i < nRows; ++i) {
      binCountArray[h_labels[i]] += 1;
    }

    // finding the average intra cluster distance for every element

    double* a = (double*)malloc(nRows * sizeof(double*));

    for (int i = 0; i < nRows; ++i) {
      int myLabel               = h_labels[i];
      double sumOfIntraClusterD = 0;

      for (int j = 0; j < nRows; ++j) {
        if (h_labels[j] == myLabel) { sumOfIntraClusterD += h_distanceMatrix[i * nRows + j]; }
      }

      if (binCountArray[myLabel] <= 1)
        a[i] = -1;
      else
        a[i] = sumOfIntraClusterD / (binCountArray[myLabel] - 1);
    }

    // finding the average inter cluster distance for every element

    double* b = (double*)malloc(nRows * sizeof(double*));

    for (int i = 0; i < nRows; ++i) {
      int myLabel          = h_labels[i];
      double minAvgInterCD = ULLONG_MAX;

      for (int j = 0; j < nLabels; ++j) {
        int curClLabel = j;
        if (curClLabel == myLabel) continue;
        double avgInterCD = 0;

        for (int k = 0; k < nRows; ++k) {
          if (h_labels[k] == curClLabel) { avgInterCD += h_distanceMatrix[i * nRows + k]; }
        }

        if (binCountArray[curClLabel])
          avgInterCD /= binCountArray[curClLabel];
        else
          avgInterCD = ULLONG_MAX;
        minAvgInterCD = min(minAvgInterCD, avgInterCD);
      }

      b[i] = minAvgInterCD;
    }

    // finding the silhouette score for every element

    double* truthSampleSilScore = (double*)malloc(nRows * sizeof(double*));
    for (int i = 0; i < nRows; ++i) {
      if (a[i] == -1)
        truthSampleSilScore[i] = 0;
      else if (a[i] == 0 && b[i] == 0)
        truthSampleSilScore[i] = 0;
      else
        truthSampleSilScore[i] = (b[i] - a[i]) / max(a[i], b[i]);
      truthSilhouetteScore += truthSampleSilScore[i];
    }

    truthSilhouetteScore /= nRows;
  }

  // the constructor
  void SetUp() override
  {
    // getting the parameters
    params = ::testing::TestWithParam<silhouetteScoreParam>::GetParam();

    nRows     = params.nRows;
    nCols     = params.nCols;
    nLabels   = params.nLabels;
    chunk     = params.chunk;
    nElements = nRows * nCols;

    host_silhouette_score();

    // calling the silhouette_score CUDA implementation
    computedSilhouetteScore = cuvs::stats::silhouette_score(
      handle,
      raft::make_device_matrix_view<const DataT>(d_X.data(), nRows, nCols),
      raft::make_device_vector_view<const LabelT>(d_labels.data(), nRows),
      std::make_optional(raft::make_device_vector_view(sampleSilScore.data(), nRows)),
      nLabels,
      params.metric);

    batchedSilhouetteScore = cuvs::stats::silhouette_score_batched(
      handle,
      raft::make_device_matrix_view<const DataT>(d_X.data(), nRows, nCols),
      raft::make_device_vector_view<const LabelT>(d_labels.data(), nRows),
      std::make_optional(raft::make_device_vector_view(sampleSilScore.data(), nRows)),
      nLabels,
      chunk,
      params.metric);
  }

  // declaring the data values
  raft::resources handle;
  silhouetteScoreParam params;
  int nLabels;
  rmm::device_uvector<DataT> d_X;
  rmm::device_uvector<DataT> sampleSilScore;
  rmm::device_uvector<LabelT> d_labels;
  int nRows;
  int nCols;
  int nElements;
  double truthSilhouetteScore    = 0;
  double computedSilhouetteScore = 0;
  double batchedSilhouetteScore  = 0;
  int chunk;
};

// setting test parameter values
const std::vector<silhouetteScoreParam> inputs = {
  {4, 2, 3, cuvs::distance::DistanceType::L2Expanded, 4, 0.00001},
  {4, 2, 2, cuvs::distance::DistanceType::L2SqrtUnexpanded, 2, 0.00001},
  {8, 8, 3, cuvs::distance::DistanceType::L2Unexpanded, 4, 0.00001},
  {11, 2, 5, cuvs::distance::DistanceType::L2Expanded, 3, 0.00001},
  {40, 2, 8, cuvs::distance::DistanceType::L2Expanded, 10, 0.00001},
  {12, 7, 3, cuvs::distance::DistanceType::CosineExpanded, 8, 0.00001},
  {7, 5, 5, cuvs::distance::DistanceType::L1, 2, 0.00001}};

// writing the test suite
typedef silhouetteScoreTest<int, double> silhouetteScoreTestClass;
TEST_P(silhouetteScoreTestClass, Result)
{
  ASSERT_NEAR(computedSilhouetteScore, truthSilhouetteScore, params.tolerance);
  ASSERT_NEAR(batchedSilhouetteScore, truthSilhouetteScore, params.tolerance);
}
INSTANTIATE_TEST_CASE_P(silhouetteScore, silhouetteScoreTestClass, ::testing::ValuesIn(inputs));

TEST(silhouetteScore, BatchedStreamPoolOrdering)
{
  constexpr int64_t n_rows = 4096;
  constexpr int64_t n_cols = 2;
  constexpr int n_labels   = 2;

  std::vector<float> X(n_rows * n_cols);
  std::vector<int> labels(n_rows);
  for (int64_t i = 0; i < n_rows; ++i) {
    X[2 * i]     = std::sin(0.01f * i);
    X[2 * i + 1] = std::cos(0.013f * i);
    labels[i]    = i % n_labels;
  }

  raft::resources handle;
  raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(4));
  auto stream = raft::resource::get_cuda_stream(handle);

  rmm::device_uvector<float> d_X(X.size(), stream);
  rmm::device_uvector<int> d_labels(labels.size(), stream);
  raft::update_device(d_X.data(), X.data(), X.size(), stream);
  raft::update_device(d_labels.data(), labels.data(), labels.size(), stream);

  auto X_view = raft::make_device_matrix_view<const float, int64_t>(d_X.data(), n_rows, n_cols);
  auto labels_view = raft::make_device_vector_view<const int, int64_t>(d_labels.data(), n_rows);
  constexpr auto metric = cuvs::distance::DistanceType::L2SqrtUnexpanded;

  auto expected =
    cuvs::stats::silhouette_score(handle, X_view, labels_view, std::nullopt, n_labels, metric);

  for (int repeat = 0; repeat < 8; ++repeat) {
    auto actual = cuvs::stats::silhouette_score_batched(
      handle, X_view, labels_view, std::nullopt, n_labels, n_rows, metric);
    ASSERT_NEAR(actual, expected, 1e-4f);
  }
}

}  // end namespace stats
}  // end namespace cuvs
