/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../distance/fused_distance_nn.cuh"
#include "../../distance/unfused_distance_nn.cuh"
#include "kmeans_common.cuh"

#include <raft/matrix/init.cuh>

namespace cuvs::cluster::kmeans::detail {

// Calculates a <key, value> pair for every sample in input 'X' where key is an
// index to an sample in 'centroids' (index of the nearest centroid) and 'value'
// is the distance between the sample and the 'centroids[key]'.
template <typename DataT, typename IndexT>
void minClusterAndDistanceCompute(
  raft::resources const& handle,
  raft::device_matrix_view<const DataT, IndexT> X,
  raft::device_matrix_view<const DataT, IndexT> centroids,
  raft::device_vector_view<raft::KeyValuePair<IndexT, DataT>, IndexT> minClusterAndDistance,
  raft::device_vector_view<const DataT, IndexT> L2NormX,
  rmm::device_uvector<DataT>& L2NormBuf_OR_DistBuf,
  cuvs::distance::DistanceType metric,
  int batch_samples,
  int batch_centroids,
  rmm::device_uvector<char>& workspace)
{
  cudaStream_t stream           = raft::resource::get_cuda_stream(handle);
  auto n_samples                = X.extent(0);
  auto n_features               = X.extent(1);
  auto n_clusters               = centroids.extent(0);
  const bool can_use_fused_path = metric == cuvs::distance::DistanceType::L2Expanded ||
                                  metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                                  metric == cuvs::distance::DistanceType::CosineExpanded;

  if (can_use_fused_path) {
    L2NormBuf_OR_DistBuf.resize(n_clusters, stream);
    auto centroidsNorm =
      raft::make_device_vector_view<DataT, IndexT>(L2NormBuf_OR_DistBuf.data(), n_clusters);

    if (metric == cuvs::distance::DistanceType::CosineExpanded) {
      raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
        handle, centroids, centroidsNorm, raft::sqrt_op{});
    } else {
      raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
        handle, centroids, centroidsNorm);
    }

    raft::KeyValuePair<IndexT, DataT> initial_value(0, std::numeric_limits<DataT>::max());
    raft::matrix::fill(handle, minClusterAndDistance, initial_value);

    const bool use_fused_path =
      use_fused<DataT, IndexT, IndexT>(handle, n_samples, n_clusters, n_features);

    if (use_fused_path) {
      workspace.resize((sizeof(int)) * n_samples, stream);

      cuvs::distance::fusedDistanceNNMinReduce<DataT, raft::KeyValuePair<IndexT, DataT>, IndexT>(
        minClusterAndDistance.data_handle(),
        X.data_handle(),
        centroids.data_handle(),
        L2NormX.data_handle(),
        centroidsNorm.data_handle(),
        n_samples,
        n_clusters,
        n_features,
        (void*)workspace.data(),
        metric != cuvs::distance::DistanceType::L2Expanded,
        false,
        true,
        metric,
        0.0f,
        stream);
    } else {
      auto dataBatchSize      = getDataBatchSize(batch_samples, n_samples);
      auto centroidsBatchSize = getCentroidsBatchSize(batch_centroids, n_clusters);

      // The unfused reduction indexes its distance matrix with IndexT.
      dataBatchSize =
        std::min(dataBatchSize, std::numeric_limits<IndexT>::max() / centroidsBatchSize);

      workspace.resize(sizeof(DataT) * dataBatchSize * centroidsBatchSize, stream);

      using KeyValueT          = raft::KeyValuePair<IndexT, DataT>;
      const bool tileCentroids = centroidsBatchSize < n_clusters;
      rmm::device_uvector<KeyValueT> batchMinClusterAndDistance(tileCentroids ? dataBatchSize : 0,
                                                                stream);

      for (IndexT dIdx = 0; dIdx < n_samples;) {
        auto ns                        = std::min(dataBatchSize, n_samples - dIdx);
        auto minClusterAndDistanceView = raft::make_device_vector_view<KeyValueT, IndexT>(
          minClusterAndDistance.data_handle() + dIdx, ns);

        for (IndexT cIdx = 0; cIdx < n_clusters;) {
          auto nc       = std::min(centroidsBatchSize, n_clusters - cIdx);
          auto batchMin = tileCentroids ? batchMinClusterAndDistance.data()
                                        : minClusterAndDistanceView.data_handle();

          cuvs::distance::unfusedDistanceNNMinReduce<DataT, DataT, KeyValueT, IndexT>(
            handle,
            batchMin,
            X.data_handle() + dIdx * n_features,
            centroids.data_handle() + cIdx * n_features,
            L2NormX.data_handle() + dIdx,
            centroidsNorm.data_handle() + cIdx,
            ns,
            nc,
            n_features,
            (void*)workspace.data(),
            metric != cuvs::distance::DistanceType::L2Expanded,
            tileCentroids,
            true,
            metric,
            0.0f,
            stream);

          if (tileCentroids) {
            // Convert tile-local centroid indices and merge the tile minima.
            auto batchMinView =
              raft::make_device_vector_view<const KeyValueT, IndexT>(batchMin, ns);
            raft::linalg::map(
              handle,
              minClusterAndDistanceView,
              [cIdx] __device__(KeyValueT current, KeyValueT batch) {
                batch.key += cIdx;
                return batch.value < current.value ? batch : current;
              },
              raft::make_const_mdspan(minClusterAndDistanceView),
              batchMinView);
          }
          cIdx += nc;
        }
        dIdx += ns;
      }
    }
  } else {
    auto dataBatchSize      = getDataBatchSize(batch_samples, n_samples);
    auto centroidsBatchSize = getCentroidsBatchSize(batch_centroids, n_clusters);

    // TODO: Unless pool allocator is used, passing in a workspace for this
    // isn't really increasing performance because this needs to do a re-allocation
    // anyways. ref https://github.com/rapidsai/raft/issues/930
    L2NormBuf_OR_DistBuf.resize(dataBatchSize * centroidsBatchSize, stream);

    // pairwiseDistance[ns x nc] - tensor wrapper around the distance buffer
    auto pairwiseDistance = raft::make_device_matrix_view<DataT, IndexT>(
      L2NormBuf_OR_DistBuf.data(), dataBatchSize, centroidsBatchSize);

    raft::KeyValuePair<IndexT, DataT> initial_value(0, std::numeric_limits<DataT>::max());
    raft::matrix::fill(handle, minClusterAndDistance, initial_value);

    // tile over the input dataset
    for (IndexT dIdx = 0; dIdx < n_samples; dIdx += dataBatchSize) {
      // # of samples for the current batch
      auto ns = std::min((IndexT)dataBatchSize, n_samples - dIdx);

      // datasetView [ns x n_features] - view representing the current batch of
      // input dataset
      auto datasetView = raft::make_device_matrix_view<const DataT, IndexT>(
        X.data_handle() + (dIdx * n_features), ns, n_features);

      // minClusterAndDistanceView [ns x n_clusters]
      auto minClusterAndDistanceView =
        raft::make_device_vector_view<raft::KeyValuePair<IndexT, DataT>, IndexT>(
          minClusterAndDistance.data_handle() + dIdx, ns);

      // tile over the centroids
      for (IndexT cIdx = 0; cIdx < n_clusters; cIdx += centroidsBatchSize) {
        // # of centroids for the current batch
        auto nc = std::min((IndexT)centroidsBatchSize, n_clusters - cIdx);

        // centroidsView [nc x n_features] - view representing the current batch
        // of centroids
        auto centroidsView = raft::make_device_matrix_view<const DataT, IndexT>(
          centroids.data_handle() + (cIdx * n_features), nc, n_features);

        // pairwiseDistanceView [ns x nc] - view representing the pairwise
        // distance for current batch
        auto pairwiseDistanceView =
          raft::make_device_matrix_view<DataT, IndexT>(pairwiseDistance.data_handle(), ns, nc);

        // calculate pairwise distance between current tile of cluster centroids
        // and input dataset
        pairwise_distance_kmeans<DataT, IndexT>(
          handle, datasetView, centroidsView, pairwiseDistanceView, metric);

        // argmin reduction returning <index, value> pair
        // calculates the closest centroid and the distance to the closest
        // centroid
        raft::linalg::coalescedReduction(
          minClusterAndDistanceView.data_handle(),
          pairwiseDistanceView.data_handle(),
          pairwiseDistanceView.extent(1),
          pairwiseDistanceView.extent(0),
          initial_value,
          stream,
          true,
          [=] __device__(const DataT val, const IndexT i) {
            raft::KeyValuePair<IndexT, DataT> pair;
            pair.key   = cIdx + i;
            pair.value = val;
            return pair;
          },
          raft::argmin_op{},
          raft::identity_op{});
      }
    }
  }
}

#define INSTANTIATE_MIN_CLUSTER_AND_DISTANCE(DataT, IndexT)                                    \
  template void minClusterAndDistanceCompute<DataT, IndexT>(                                   \
    raft::resources const& handle,                                                             \
    raft::device_matrix_view<const DataT, IndexT> X,                                           \
    raft::device_matrix_view<const DataT, IndexT> centroids,                                   \
    raft::device_vector_view<raft::KeyValuePair<IndexT, DataT>, IndexT> minClusterAndDistance, \
    raft::device_vector_view<const DataT, IndexT> L2NormX,                                     \
    rmm::device_uvector<DataT>& L2NormBuf_OR_DistBuf,                                          \
    cuvs::distance::DistanceType metric,                                                       \
    int batch_samples,                                                                         \
    int batch_centroids,                                                                       \
    rmm::device_uvector<char>& workspace);

INSTANTIATE_MIN_CLUSTER_AND_DISTANCE(float, int64_t)
INSTANTIATE_MIN_CLUSTER_AND_DISTANCE(double, int64_t)
INSTANTIATE_MIN_CLUSTER_AND_DISTANCE(float, int)
INSTANTIATE_MIN_CLUSTER_AND_DISTANCE(double, int)

#undef INSTANTIATE_MIN_CLUSTER_AND_DISTANCE

template <typename DataT, typename IndexT>
void minClusterDistanceCompute(raft::resources const& handle,
                               raft::device_matrix_view<const DataT, IndexT> X,
                               raft::device_matrix_view<DataT, IndexT> centroids,
                               raft::device_vector_view<DataT, IndexT> minClusterDistance,
                               raft::device_vector_view<DataT, IndexT> L2NormX,
                               rmm::device_uvector<DataT>& L2NormBuf_OR_DistBuf,
                               cuvs::distance::DistanceType metric,
                               int batch_samples,
                               int batch_centroids,
                               rmm::device_uvector<char>& workspace)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  auto n_clusters     = centroids.extent(0);

  bool is_fused = metric == cuvs::distance::DistanceType::L2Expanded ||
                  metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                  metric == cuvs::distance::DistanceType::CosineExpanded;

  raft::matrix::fill(handle, minClusterDistance, std::numeric_limits<DataT>::max());

  if (is_fused) {
    L2NormBuf_OR_DistBuf.resize(n_clusters, stream);
    auto centroidsNorm =
      raft::make_device_vector_view<DataT, IndexT>(L2NormBuf_OR_DistBuf.data(), n_clusters);

    if (metric == cuvs::distance::DistanceType::CosineExpanded) {
      raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
        handle,
        raft::make_device_matrix_view<const DataT, IndexT>(
          centroids.data_handle(), centroids.extent(0), centroids.extent(1)),
        centroidsNorm,
        raft::sqrt_op{});
    } else {
      raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
        handle,
        raft::make_device_matrix_view<const DataT, IndexT>(
          centroids.data_handle(), centroids.extent(0), centroids.extent(1)),
        centroidsNorm);
    }

    workspace.resize(sizeof(int) * n_samples, stream);

    cuvs::distance::fusedDistanceNNMinReduce<DataT, DataT, IndexT>(
      minClusterDistance.data_handle(),
      X.data_handle(),
      centroids.data_handle(),
      L2NormX.data_handle(),
      centroidsNorm.data_handle(),
      n_samples,
      n_clusters,
      n_features,
      (void*)workspace.data(),
      metric != cuvs::distance::DistanceType::L2Expanded,
      false,
      true,
      metric,
      0.0f,
      stream);
  } else {
    auto dataBatchSize      = getDataBatchSize(batch_samples, n_samples);
    auto centroidsBatchSize = getCentroidsBatchSize(batch_centroids, n_clusters);

    L2NormBuf_OR_DistBuf.resize(dataBatchSize * centroidsBatchSize, stream);

    auto pairwiseDistance = raft::make_device_matrix_view<DataT, IndexT>(
      L2NormBuf_OR_DistBuf.data(), dataBatchSize, centroidsBatchSize);

    // tile over the input data and calculate distance matrix [n_samples x
    // n_clusters]
    for (IndexT dIdx = 0; dIdx < n_samples; dIdx += dataBatchSize) {
      auto ns = std::min((IndexT)dataBatchSize, n_samples - dIdx);

      auto datasetView = raft::make_device_matrix_view<const DataT, IndexT>(
        X.data_handle() + dIdx * n_features, ns, n_features);

      auto minClusterDistanceView =
        raft::make_device_vector_view<DataT, IndexT>(minClusterDistance.data_handle() + dIdx, ns);

      // tile over the centroids
      for (IndexT cIdx = 0; cIdx < n_clusters; cIdx += centroidsBatchSize) {
        auto nc = std::min((IndexT)centroidsBatchSize, n_clusters - cIdx);

        auto centroidsView = raft::make_device_matrix_view<DataT, IndexT>(
          centroids.data_handle() + cIdx * n_features, nc, n_features);

        auto pairwiseDistanceView =
          raft::make_device_matrix_view<DataT, IndexT>(pairwiseDistance.data_handle(), ns, nc);

        pairwise_distance_kmeans<DataT, IndexT>(
          handle, datasetView, centroidsView, pairwiseDistanceView, metric);

        raft::linalg::coalescedReduction(minClusterDistanceView.data_handle(),
                                         pairwiseDistanceView.data_handle(),
                                         pairwiseDistanceView.extent(1),
                                         pairwiseDistanceView.extent(0),
                                         std::numeric_limits<DataT>::max(),
                                         stream,
                                         true,
                                         raft::identity_op{},
                                         raft::min_op{},
                                         raft::identity_op{});
      }
    }
  }
}

#define INSTANTIATE_MIN_CLUSTER_DISTANCE(DataT, IndexT)         \
  template void minClusterDistanceCompute<DataT, IndexT>(       \
    raft::resources const& handle,                              \
    raft::device_matrix_view<const DataT, IndexT> X,            \
    raft::device_matrix_view<DataT, IndexT> centroids,          \
    raft::device_vector_view<DataT, IndexT> minClusterDistance, \
    raft::device_vector_view<DataT, IndexT> L2NormX,            \
    rmm::device_uvector<DataT>& L2NormBuf_OR_DistBuf,           \
    cuvs::distance::DistanceType metric,                        \
    int batch_samples,                                          \
    int batch_centroids,                                        \
    rmm::device_uvector<char>& workspace);

INSTANTIATE_MIN_CLUSTER_DISTANCE(float, int64_t)
INSTANTIATE_MIN_CLUSTER_DISTANCE(double, int64_t)
INSTANTIATE_MIN_CLUSTER_DISTANCE(float, int)
INSTANTIATE_MIN_CLUSTER_DISTANCE(double, int)

#undef INSTANTIATE_MIN_CLUSTER_DISTANCE

}  // namespace cuvs::cluster::kmeans::detail
