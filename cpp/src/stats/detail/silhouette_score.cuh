/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../core/reduction_launchers.hpp"
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/map.cuh>
#include <raft/linalg/map_then_reduce.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/reduce.cuh>
#include <raft/linalg/reduce_cols_by_key.cuh>
#include <raft/matrix/init.cuh>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_scalar.hpp>

#include <cub/device/device_histogram.cuh>

#include <math.h>

#include <algorithm>
#include <iostream>
#include <numeric>

namespace cuvs {
namespace stats {
namespace detail {

/**
 * @brief kernel that calculates the average intra-cluster distance for every sample data point and
 * updates the cluster distance to max value
 * @tparam DataT: type of the data samples
 * @tparam LabelT: type of the labels
 * @param sampleToClusterSumOfDistances: the pointer to the 2D array that contains the sum of
 * distances from every sample to every cluster (nRows x nLabels)
 * @param binCountArray: pointer to the 1D array that contains the count of samples per cluster (1 x
 * nLabels)
 * @param d_aArray: the pointer to the array of average intra-cluster distances for every sample in
 * device memory (1 x nRows)
 * @param labels: the pointer to the array containing labels for every data sample (1 x nRows)
 * @param nRows: number of data samples
 * @param nLabels: number of Labels
 * @param MAX_VAL: DataT specific upper limit
 */
template <typename DataT, typename LabelT>
RAFT_KERNEL populateAKernel(DataT* sampleToClusterSumOfDistances,
                            DataT* binCountArray,
                            DataT* d_aArray,
                            const LabelT* labels,
                            int nRows,
                            int nLabels,
                            const DataT MAX_VAL)
{
  // getting the current index
  int sampleIndex = threadIdx.x + blockIdx.x * blockDim.x;

  if (sampleIndex >= nRows) return;

  // sampleDistanceVector is an array that stores that particular row of the distanceMatrix
  DataT* sampleToClusterSumOfDistancesVector =
    &sampleToClusterSumOfDistances[sampleIndex * nLabels];

  LabelT sampleCluster = labels[sampleIndex];

  int sampleClusterIndex = (int)sampleCluster;

  if (binCountArray[sampleClusterIndex] - 1 <= 0) {
    d_aArray[sampleIndex] = -1;
    return;

  }

  else {
    d_aArray[sampleIndex] = (sampleToClusterSumOfDistancesVector[sampleClusterIndex]) /
                            (binCountArray[sampleClusterIndex] - 1);

    // modifying the sampleDistanceVector to give sample average distance
    sampleToClusterSumOfDistancesVector[sampleClusterIndex] = MAX_VAL;
  }
}

/**
 * @brief function to calculate the bincounts of number of samples in every label
 * @tparam DataT: type of the data samples
 * @tparam LabelT: type of the labels
 * @param labels: the pointer to the array containing labels for every data sample (1 x nRows)
 * @param binCountArray: pointer to the 1D array that contains the count of samples per cluster (1 x
 * nLabels)
 * @param nRows: number of data samples
 * @param nUniqueLabels: number of Labels
 * @param workspace: device buffer containing workspace memory
 * @param stream: the cuda stream where to launch this kernel
 */
template <typename DataT, typename LabelT>
void countLabels(const LabelT* labels,
                 DataT* binCountArray,
                 int nRows,
                 int nUniqueLabels,
                 rmm::device_uvector<char>& workspace,
                 cudaStream_t stream)
{
  int num_levels            = nUniqueLabels + 1;
  LabelT lower_level        = 0;
  LabelT upper_level        = nUniqueLabels;
  size_t temp_storage_bytes = 0;

  rmm::device_uvector<int> countArray(nUniqueLabels, stream);

  RAFT_CUDA_TRY(cub::DeviceHistogram::HistogramEven(nullptr,
                                                    temp_storage_bytes,
                                                    labels,
                                                    binCountArray,
                                                    num_levels,
                                                    lower_level,
                                                    upper_level,
                                                    nRows,
                                                    stream));

  workspace.resize(temp_storage_bytes, stream);

  RAFT_CUDA_TRY(cub::DeviceHistogram::HistogramEven(workspace.data(),
                                                    temp_storage_bytes,
                                                    labels,
                                                    binCountArray,
                                                    num_levels,
                                                    lower_level,
                                                    upper_level,
                                                    nRows,
                                                    stream));
}

/**
 * @brief structure that defines the division Lambda for elementwise op
 */
template <typename DataT>
struct DivOp {
  HDI DataT operator()(DataT a, int b, int c)
  {
    if (b == 0)
      return ULLONG_MAX;
    else
      return a / b;
  }
};

/**
 * @brief structure that defines the elementwise operation to calculate silhouette score using
 * params 'a' and 'b'
 */
template <typename DataT>
struct SilOp {
  HDI DataT operator()(DataT a, DataT b)
  {
    if (a == 0 && b == 0 || a == b)
      return 0;
    else if (a == -1)
      return 0;
    else if (a > b)
      return (b - a) / a;
    else
      return (b - a) / b;
  }
};

/**
 * @brief main function that returns the average silhouette score for a given set of data and its
 * clusterings
 * @tparam DataT: type of the data samples
 * @tparam LabelT: type of the labels
 * @param X_in: pointer to the input Data samples array (nRows x nCols)
 * @param nRows: number of data samples
 * @param nCols: number of features
 * @param labels: the pointer to the array containing labels for every data sample (1 x nRows)
 * @param nLabels: number of Labels
 * @param silhouette_scorePerSample: pointer to the array that is optionally taken in as input and
 * is populated with the silhouette score for every sample (1 x nRows)
 * @param stream: the cuda stream where to launch this kernel
 * @param metric: the numerical value that maps to the type of distance metric to be used in the
 * calculations
 */
template <typename DataT, typename LabelT>
DataT silhouette_score(
  raft::resources const& handle,
  const DataT* X_in,
  int nRows,
  int nCols,
  const LabelT* labels,
  int nLabels,
  DataT* silhouette_scorePerSample,
  cudaStream_t stream,
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded)
{
  ASSERT(nLabels >= 2 && nLabels <= (nRows - 1),
         "silhouette Score not defined for the given number of labels!");

  // compute the distance matrix
  rmm::device_uvector<DataT> distanceMatrix(nRows * nRows, stream);
  rmm::device_uvector<char> workspace(1, stream);

  auto X_in_view = raft::make_device_matrix_view<const DataT, int64_t>(X_in, nRows, nCols);

  cuvs::distance::pairwise_distance(
    handle,
    X_in_view,
    X_in_view,
    raft::make_device_matrix_view<DataT, int64_t>(distanceMatrix.data(), nRows, nRows),
    metric);

  // deciding on the array of silhouette scores for each dataPoint
  rmm::device_uvector<DataT> silhouette_scoreSamples(0, stream);
  DataT* perSampleSilScore = nullptr;
  if (silhouette_scorePerSample == nullptr) {
    silhouette_scoreSamples.resize(nRows, stream);
    perSampleSilScore = silhouette_scoreSamples.data();
  } else {
    perSampleSilScore = silhouette_scorePerSample;
  }
  raft::matrix::fill(
    handle, raft::make_device_vector_view<DataT, int>(perSampleSilScore, nRows), DataT(0));

  // getting the sample count per cluster
  rmm::device_uvector<DataT> binCountArray(nLabels, stream);
  raft::matrix::fill(
    handle, raft::make_device_vector_view<DataT, int>(binCountArray.data(), nLabels), DataT(0));
  countLabels(labels, binCountArray.data(), nRows, nLabels, workspace, stream);

  // calculating the sample-cluster-distance-sum-array
  rmm::device_uvector<DataT> sampleToClusterSumOfDistances(nRows * nLabels, stream);
  raft::matrix::fill(handle,
                     raft::make_device_vector_view<DataT, int>(sampleToClusterSumOfDistances.data(),
                                                               nRows * nLabels),
                     DataT(0));
  raft::linalg::reduce_cols_by_key(handle,
                                   raft::make_device_matrix_view<const DataT, int, raft::row_major>(
                                     distanceMatrix.data(), nRows, nRows),
                                   raft::make_device_vector_view<const LabelT, int>(labels, nRows),
                                   raft::make_device_matrix_view<DataT, int, raft::row_major>(
                                     sampleToClusterSumOfDistances.data(), nRows, nLabels),
                                   nLabels);

  // creating the a array and b array
  rmm::device_uvector<DataT> d_aArray(nRows, stream);
  rmm::device_uvector<DataT> d_bArray(nRows, stream);
  raft::matrix::fill(
    handle, raft::make_device_vector_view<DataT, int>(d_aArray.data(), nRows), DataT(0));
  raft::matrix::fill(
    handle, raft::make_device_vector_view<DataT, int>(d_bArray.data(), nRows), DataT(0));

  // kernel that populates the d_aArray
  // kernel configuration
  dim3 numThreadsPerBlock(32, 1, 1);
  dim3 numBlocks(raft::ceildiv<int>(nRows, numThreadsPerBlock.x), 1, 1);

  // calling the kernel
  populateAKernel<<<numBlocks, numThreadsPerBlock, 0, stream>>>(
    sampleToClusterSumOfDistances.data(),
    binCountArray.data(),
    d_aArray.data(),
    labels,
    nRows,
    nLabels,
    std::numeric_limits<DataT>::max());

  // elementwise dividing by bincounts
  rmm::device_uvector<DataT> averageDistanceBetweenSampleAndCluster(nRows * nLabels, stream);
  raft::matrix::fill(handle,
                     raft::make_device_vector_view<DataT, int>(
                       averageDistanceBetweenSampleAndCluster.data(), nRows * nLabels),
                     DataT(0));

  auto averageDistanceBetweenSampleAndClusterView = raft::make_device_matrix_view<DataT>(
    averageDistanceBetweenSampleAndCluster.data(), nRows, nLabels);
  auto sampleToClusterSumOfDistancesView = raft::make_device_matrix_view<const DataT>(
    sampleToClusterSumOfDistances.data(), nRows, nLabels);
  auto binCountArrayView =
    raft::make_device_vector_view<const DataT>(binCountArray.data(), nLabels);

  raft::linalg::matrix_vector_op<raft::Apply::ALONG_ROWS>(
    handle,
    sampleToClusterSumOfDistancesView,
    binCountArrayView,
    averageDistanceBetweenSampleAndClusterView,
    [] __device__(DataT a, DataT b) {
      if (b == 0)
        return static_cast<DataT>(ULLONG_MAX);
      else
        return a / b;
    });

  // calculating row-wise minimum
  cuvs::detail::reduce<raft::Apply::ALONG_ROWS>(
    handle,
    raft::make_device_matrix_view<const DataT, int, raft::row_major>(
      averageDistanceBetweenSampleAndCluster.data(), nRows, nLabels),
    raft::make_device_vector_view<DataT, int>(d_bArray.data(), nRows),
    std::numeric_limits<DataT>::max(),
    false,
    raft::identity_op{},
    raft::min_op{});

  // calculating the silhouette score per sample using the d_aArray and d_bArray
  raft::linalg::map(
    handle,
    raft::make_device_vector_view<DataT>(perSampleSilScore, nRows),
    SilOp<DataT>(),
    raft::make_const_mdspan(raft::make_device_vector_view<const DataT>(d_aArray.data(), nRows)),
    raft::make_const_mdspan(raft::make_device_vector_view<const DataT>(d_bArray.data(), nRows)));

  // calculating the sum of all the silhouette score
  rmm::device_scalar<DataT> d_avgSilhouetteScore(stream);
  raft::matrix::fill(
    handle, raft::make_device_vector_view<DataT, int>(d_avgSilhouetteScore.data(), 1), DataT(0));

  raft::linalg::mapThenSumReduce<DataT, raft::identity_op>(d_avgSilhouetteScore.data(),
                                                           nRows,
                                                           raft::identity_op(),
                                                           stream,
                                                           perSampleSilScore,
                                                           perSampleSilScore);

  DataT avgSilhouetteScore = d_avgSilhouetteScore.value(stream);

  raft::resource::sync_stream(handle, stream);

  avgSilhouetteScore /= nRows;

  return avgSilhouetteScore;
}

};  // namespace detail
};  // namespace stats
};  // namespace cuvs
