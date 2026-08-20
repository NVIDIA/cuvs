/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../distance/fused_distance_nn.cuh"
#include "../../distance/unfused_distance_nn.cuh"
#include "kmeans_common.cuh"

#include <raft/linalg/coalesced_reduction.cuh>
#include <raft/matrix/init.cuh>

#include <mma.h>
#include <type_traits>

namespace cuvs::cluster::kmeans::detail {

namespace {

__device__ __forceinline__ float round_to_tf32(float value)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return nvcuda::wmma::__float_to_tf32(value);
#else
  return value;
#endif
}

struct tf32_square_op {
  template <typename IndexT>
  __device__ float operator()(float value, IndexT) const
  {
    const float rounded = round_to_tf32(value);
    return rounded * rounded;
  }
};

template <typename IndexT>
void compute_tf32_row_norms(raft::resources const& handle,
                            const float* matrix,
                            float* norms,
                            IndexT n_rows,
                            IndexT n_cols,
                            bool take_sqrt)
{
  if (n_rows == 0) { return; }
  auto matrix_view = raft::make_device_matrix_view<const float, IndexT>(matrix, n_rows, n_cols);
  auto norms_view  = raft::make_device_vector_view<float, IndexT>(norms, n_rows);
  if (take_sqrt) {
    raft::linalg::coalesced_reduction(handle,
                                      matrix_view,
                                      norms_view,
                                      0.0f,
                                      false,
                                      tf32_square_op{},
                                      raft::add_op{},
                                      raft::sqrt_op{});
  } else {
    raft::linalg::coalesced_reduction(handle,
                                      matrix_view,
                                      norms_view,
                                      0.0f,
                                      false,
                                      tf32_square_op{},
                                      raft::add_op{},
                                      raft::identity_op{});
  }
}

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
    const DataT* x_norm_ptr = L2NormX.data_handle();
    const DataT* centroids_norm_ptr;
    if constexpr (std::is_same_v<DataT, float>) {
      if (fused_path == FusedDistancePath::FusedCutile && is_l2_cos) {
        constexpr size_t norm_alignment = 16 / sizeof(float);
        const size_t x_norm_storage = raft::alignTo(static_cast<size_t>(n_samples), norm_alignment);
        L2NormBuf_OR_DistBuf.resize(x_norm_storage + static_cast<size_t>(n_clusters), stream);
        auto* tf32_x_norms        = L2NormBuf_OR_DistBuf.data();
        auto* tf32_centroid_norms = tf32_x_norms + x_norm_storage;
        const bool take_sqrt      = metric == cuvs::distance::DistanceType::CosineExpanded;
        compute_tf32_row_norms(
          handle, X.data_handle(), tf32_x_norms, n_samples, n_features, take_sqrt);
        compute_tf32_row_norms(
          handle, centroids.data_handle(), tf32_centroid_norms, n_clusters, n_features, take_sqrt);
        x_norm_ptr         = tf32_x_norms;
        centroids_norm_ptr = tf32_centroid_norms;
      } else {
        L2NormBuf_OR_DistBuf.resize(n_clusters, stream);
        centroids_norm_ptr = L2NormBuf_OR_DistBuf.data();
      }
    } else {
      L2NormBuf_OR_DistBuf.resize(n_clusters, stream);
      centroids_norm_ptr = L2NormBuf_OR_DistBuf.data();
    }

    if (!(fused_path == FusedDistancePath::FusedCutile && is_l2_cos &&
          std::is_same_v<DataT, float>) &&
        is_l2_cos) {
      auto centroids_norm =
        raft::make_device_vector_view<DataT, IndexT>(L2NormBuf_OR_DistBuf.data(), n_clusters);
      if (metric == cuvs::distance::DistanceType::CosineExpanded) {
        raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
          handle, centroids, centroids_norm, raft::sqrt_op{});
      } else {
        raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
          handle, centroids, centroids_norm);
      }
    }

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
      x_norm_ptr,
      centroids_norm_ptr,
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

    auto dataBatchSize      = getDataBatchSize(batch_samples, n_samples);
    auto centroidsBatchSize = getCentroidsBatchSize(batch_centroids, n_clusters);

    // The unfused reduction indexes its distance matrix with IndexT.
    dataBatchSize =
      std::min(dataBatchSize, std::numeric_limits<IndexT>::max() / centroidsBatchSize);
    workspace.resize(sizeof(DataT) * dataBatchSize * centroidsBatchSize, stream);

    using KeyValueT = raft::KeyValuePair<IndexT, DataT>;
    auto temp_kvp   = raft::make_device_vector<KeyValueT, IndexT>(handle, n_samples);
    KeyValueT initial_value(0, std::numeric_limits<DataT>::max());
    raft::matrix::fill(handle, temp_kvp.view(), initial_value);

    const bool tileCentroids = centroidsBatchSize < n_clusters;
    rmm::device_uvector<KeyValueT> batchMinClusterAndDistance(tileCentroids ? dataBatchSize : 0,
                                                              stream);

    for (IndexT dIdx = 0; dIdx < n_samples;) {
      auto ns = std::min(dataBatchSize, n_samples - dIdx);
      auto minClusterAndDistanceView =
        raft::make_device_vector_view<KeyValueT, IndexT>(temp_kvp.data_handle() + dIdx, ns);

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
          centroidsNormConst.data_handle() + cIdx,
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
          auto batchMinView = raft::make_device_vector_view<const KeyValueT, IndexT>(batchMin, ns);
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
    const DataT* x_norm_ptr = L2NormX.data_handle();
    const DataT* centroids_norm_ptr;
    if constexpr (std::is_same_v<DataT, float>) {
      if (fused_path == FusedDistancePath::FusedCutile && is_l2_cos) {
        constexpr size_t norm_alignment = 16 / sizeof(float);
        const size_t x_norm_storage = raft::alignTo(static_cast<size_t>(n_samples), norm_alignment);
        L2NormBuf_OR_DistBuf.resize(x_norm_storage + static_cast<size_t>(n_clusters), stream);
        auto* tf32_x_norms        = L2NormBuf_OR_DistBuf.data();
        auto* tf32_centroid_norms = tf32_x_norms + x_norm_storage;
        const bool take_sqrt      = metric == cuvs::distance::DistanceType::CosineExpanded;
        compute_tf32_row_norms(
          handle, X.data_handle(), tf32_x_norms, n_samples, n_features, take_sqrt);
        compute_tf32_row_norms(
          handle, centroids.data_handle(), tf32_centroid_norms, n_clusters, n_features, take_sqrt);
        x_norm_ptr         = tf32_x_norms;
        centroids_norm_ptr = tf32_centroid_norms;
      } else {
        L2NormBuf_OR_DistBuf.resize(n_clusters, stream);
        centroids_norm_ptr = L2NormBuf_OR_DistBuf.data();
      }
    } else {
      L2NormBuf_OR_DistBuf.resize(n_clusters, stream);
      centroids_norm_ptr = L2NormBuf_OR_DistBuf.data();
    }

    if (!(fused_path == FusedDistancePath::FusedCutile && is_l2_cos &&
          std::is_same_v<DataT, float>)) {
      auto centroids_norm =
        raft::make_device_vector_view<DataT, IndexT>(L2NormBuf_OR_DistBuf.data(), n_clusters);
      if (metric == cuvs::distance::DistanceType::CosineExpanded) {
        raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
          handle,
          raft::make_device_matrix_view<const DataT, IndexT>(
            centroids.data_handle(), centroids.extent(0), centroids.extent(1)),
          centroids_norm,
          raft::sqrt_op{});
      } else {
        raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
          handle,
          raft::make_device_matrix_view<const DataT, IndexT>(
            centroids.data_handle(), centroids.extent(0), centroids.extent(1)),
          centroids_norm);
      }
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
      x_norm_ptr,
      centroids_norm_ptr,
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
