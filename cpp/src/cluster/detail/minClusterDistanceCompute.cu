/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../distance/fused_distance_nn.cuh"
#include "../../distance/unfused_distance_nn.cuh"
#include "kmeans_common.cuh"

#include <raft/matrix/init.cuh>

namespace cuvs::cluster::kmeans::detail {

namespace {

template <typename IndexT, typename DataT>
__global__ void unpack_kvp_to_soa(IndexT* nearest_idx,
                                  DataT* nearest_dist,
                                  const raft::KeyValuePair<IndexT, DataT>* kvp,
                                  IndexT n)
{
  IndexT i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    if (nearest_idx != nullptr) { nearest_idx[i] = kvp[i].key; }
    if (nearest_dist != nullptr) { nearest_dist[i] = kvp[i].value; }
  }
}

template <typename IndexT, typename DataT>
void unpack_kvp(raft::resources const& handle,
                raft::device_vector_view<IndexT, IndexT> nearest_idx,
                raft::device_vector_view<DataT, IndexT> nearest_dist,
                raft::device_vector_view<const raft::KeyValuePair<IndexT, DataT>, IndexT> kvp)
{
  auto stream = raft::resource::get_cuda_stream(handle);
  auto n      = static_cast<IndexT>(kvp.extent(0));
  int blks    = static_cast<int>((n + 255) / 256);
  unpack_kvp_to_soa<<<blks, 256, 0, stream>>>(
    nearest_idx.data_handle(), nearest_dist.data_handle(), kvp.data_handle(), n);
  RAFT_CUDA_TRY(cudaGetLastError());
}

}  // namespace

template <typename DataT, typename IndexT>
void minClusterAndDistanceCompute(raft::resources const& handle,
                                  raft::device_matrix_view<const DataT, IndexT> X,
                                  raft::device_matrix_view<const DataT, IndexT> centroids,
                                  raft::device_vector_view<IndexT, IndexT> nearest_idx,
                                  raft::device_vector_view<DataT, IndexT> nearest_dist,
                                  raft::device_vector_view<const DataT, IndexT> L2NormX,
                                  rmm::device_uvector<DataT>& L2NormBuf_OR_DistBuf,
                                  cuvs::distance::DistanceType metric,
                                  int batch_samples,
                                  int batch_centroids,
                                  rmm::device_uvector<char>& workspace)
{
  cudaStream_t stream  = raft::resource::get_cuda_stream(handle);
  auto n_samples       = X.extent(0);
  auto n_features      = X.extent(1);
  auto n_clusters      = centroids.extent(0);
  const bool is_l2_cos = metric == cuvs::distance::DistanceType::L2Expanded ||
                         metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                         metric == cuvs::distance::DistanceType::CosineExpanded;
  const FusedDistancePath fused_path =
    use_fused<DataT, IndexT, IndexT>(handle, n_samples, n_clusters, n_features, metric);

  if (uses_fused_distance_nn(fused_path)) {
    L2NormBuf_OR_DistBuf.resize(n_clusters, stream);
    auto centroidsNorm =
      raft::make_device_vector_view<DataT, IndexT>(L2NormBuf_OR_DistBuf.data(), n_clusters);

    if (is_l2_cos) {
      if (metric == cuvs::distance::DistanceType::CosineExpanded) {
        raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
          handle, centroids, centroidsNorm, raft::sqrt_op{});
      } else {
        raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
          handle, centroids, centroidsNorm);
      }
    }

    auto centroidsNormConst =
      raft::make_device_vector_view<const DataT, IndexT>(L2NormBuf_OR_DistBuf.data(), n_clusters);

    raft::KeyValuePair<IndexT, DataT>* cutlass_kvp_scratch = nullptr;
    rmm::device_uvector<raft::KeyValuePair<IndexT, DataT>> temp_kvp(0, stream);
    if (needs_cutlass_kvp_scratch(fused_path)) {
      temp_kvp.resize(n_samples, stream);
      cutlass_kvp_scratch = temp_kvp.data();
      workspace.resize(sizeof(int) * n_samples, stream);
    } else if constexpr (std::is_same_v<IndexT, int64_t>) {
      // The cuTile kernel uses i32 internally and widens labels after the launch.
      workspace.resize(sizeof(int) * static_cast<size_t>(n_samples), stream);
    }

    cuvs::distance::fusedDistanceNNMinReduce<DataT, IndexT>(
      nearest_idx.data_handle(),
      nearest_dist.data_handle(),
      X.data_handle(),
      centroids.data_handle(),
      L2NormX.data_handle(),
      centroidsNormConst.data_handle(),
      n_samples,
      n_clusters,
      n_features,
      needs_fused_mutex_workspace(fused_path) || std::is_same_v<IndexT, int64_t>
        ? (void*)workspace.data()
        : nullptr,
      metric != cuvs::distance::DistanceType::L2Expanded,
      true,
      true,
      metric,
      0.0f,
      cutlass_kvp_scratch,
      stream);
  } else if (is_l2_cos) {
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

    auto centroidsNormConst =
      raft::make_device_vector_view<const DataT, IndexT>(L2NormBuf_OR_DistBuf.data(), n_clusters);

    workspace.resize(sizeof(DataT) * n_samples * n_clusters, stream);
    auto temp_kvp =
      raft::make_device_vector<raft::KeyValuePair<IndexT, DataT>, IndexT>(handle, n_samples);
    raft::KeyValuePair<IndexT, DataT> initial_value(0, std::numeric_limits<DataT>::max());
    raft::matrix::fill(handle, temp_kvp.view(), initial_value);

    cuvs::distance::
      unfusedDistanceNNMinReduce<DataT, DataT, raft::KeyValuePair<IndexT, DataT>, IndexT>(
        handle,
        temp_kvp.data_handle(),
        X.data_handle(),
        centroids.data_handle(),
        L2NormX.data_handle(),
        centroidsNormConst.data_handle(),
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
    unpack_kvp(handle, nearest_idx, nearest_dist, raft::make_const_mdspan(temp_kvp.view()));
  } else {
    auto dataBatchSize      = getDataBatchSize(batch_samples, n_samples);
    auto centroidsBatchSize = getCentroidsBatchSize(batch_centroids, n_clusters);

    L2NormBuf_OR_DistBuf.resize(dataBatchSize * centroidsBatchSize, stream);

    auto pairwiseDistance = raft::make_device_matrix_view<DataT, IndexT>(
      L2NormBuf_OR_DistBuf.data(), dataBatchSize, centroidsBatchSize);

    auto temp_kvp =
      raft::make_device_vector<raft::KeyValuePair<IndexT, DataT>, IndexT>(handle, n_samples);
    raft::KeyValuePair<IndexT, DataT> initial_value(0, std::numeric_limits<DataT>::max());
    raft::matrix::fill(handle, temp_kvp.view(), initial_value);

    for (IndexT dIdx = 0; dIdx < n_samples; dIdx += dataBatchSize) {
      auto ns = std::min((IndexT)dataBatchSize, n_samples - dIdx);

      auto datasetView = raft::make_device_matrix_view<const DataT, IndexT>(
        X.data_handle() + (dIdx * n_features), ns, n_features);

      auto temp_kvp_view = raft::make_device_vector_view<raft::KeyValuePair<IndexT, DataT>, IndexT>(
        temp_kvp.data_handle() + dIdx, ns);

      for (IndexT cIdx = 0; cIdx < n_clusters; cIdx += centroidsBatchSize) {
        auto nc = std::min((IndexT)centroidsBatchSize, n_clusters - cIdx);

        auto centroidsView = raft::make_device_matrix_view<const DataT, IndexT>(
          centroids.data_handle() + (cIdx * n_features), nc, n_features);

        auto pairwiseDistanceView =
          raft::make_device_matrix_view<DataT, IndexT>(pairwiseDistance.data_handle(), ns, nc);

        pairwise_distance_kmeans<DataT, IndexT>(
          handle, datasetView, centroidsView, pairwiseDistanceView, metric);

        raft::linalg::coalescedReduction(
          temp_kvp_view.data_handle(),
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

    unpack_kvp(handle, nearest_idx, nearest_dist, raft::make_const_mdspan(temp_kvp.view()));
  }
}

#define INSTANTIATE_MIN_CLUSTER_AND_DISTANCE(DataT, IndexT)  \
  template void minClusterAndDistanceCompute<DataT, IndexT>( \
    raft::resources const& handle,                           \
    raft::device_matrix_view<const DataT, IndexT> X,         \
    raft::device_matrix_view<const DataT, IndexT> centroids, \
    raft::device_vector_view<IndexT, IndexT> nearest_idx,    \
    raft::device_vector_view<DataT, IndexT> nearest_dist,    \
    raft::device_vector_view<const DataT, IndexT> L2NormX,   \
    rmm::device_uvector<DataT>& L2NormBuf_OR_DistBuf,        \
    cuvs::distance::DistanceType metric,                     \
    int batch_samples,                                       \
    int batch_centroids,                                     \
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

  const bool is_l2_cos = metric == cuvs::distance::DistanceType::L2Expanded ||
                         metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                         metric == cuvs::distance::DistanceType::CosineExpanded;

  raft::matrix::fill(handle, minClusterDistance, std::numeric_limits<DataT>::max());

  const FusedDistancePath fused_path =
    is_l2_cos ? use_fused<DataT, IndexT, IndexT>(handle, n_samples, n_clusters, n_features, metric)
              : FusedDistancePath::Unfused;

  if (uses_fused_distance_nn(fused_path)) {
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

    raft::KeyValuePair<IndexT, DataT>* cutlass_kvp_scratch = nullptr;
    rmm::device_uvector<raft::KeyValuePair<IndexT, DataT>> temp_kvp(0, stream);
    if (needs_cutlass_kvp_scratch(fused_path)) {
      temp_kvp.resize(n_samples, stream);
      cutlass_kvp_scratch = temp_kvp.data();
      workspace.resize(sizeof(int) * n_samples, stream);
    }

    cuvs::distance::fusedDistanceNNMinReduce<DataT, IndexT>(
      nullptr,
      minClusterDistance.data_handle(),
      X.data_handle(),
      centroids.data_handle(),
      L2NormX.data_handle(),
      centroidsNorm.data_handle(),
      n_samples,
      n_clusters,
      n_features,
      needs_fused_mutex_workspace(fused_path) ? (void*)workspace.data() : nullptr,
      metric != cuvs::distance::DistanceType::L2Expanded,
      true,
      true,
      metric,
      0.0f,
      cutlass_kvp_scratch,
      stream);
  } else {
    auto dataBatchSize      = getDataBatchSize(batch_samples, n_samples);
    auto centroidsBatchSize = getCentroidsBatchSize(batch_centroids, n_clusters);

    L2NormBuf_OR_DistBuf.resize(dataBatchSize * centroidsBatchSize, stream);

    auto pairwiseDistance = raft::make_device_matrix_view<DataT, IndexT>(
      L2NormBuf_OR_DistBuf.data(), dataBatchSize, centroidsBatchSize);

    for (IndexT dIdx = 0; dIdx < n_samples; dIdx += dataBatchSize) {
      auto ns = std::min((IndexT)dataBatchSize, n_samples - dIdx);

      auto datasetView = raft::make_device_matrix_view<const DataT, IndexT>(
        X.data_handle() + dIdx * n_features, ns, n_features);

      auto minClusterDistanceView =
        raft::make_device_vector_view<DataT, IndexT>(minClusterDistance.data_handle() + dIdx, ns);

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
