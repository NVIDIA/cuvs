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

}  // namespace

template <typename IndexT>
void computeCutileRowNorms(raft::resources const& handle,
                           const float* matrix,
                           float* norms,
                           IndexT n_rows,
                           IndexT n_cols,
                           bool take_sqrt)
{
  compute_tf32_row_norms(handle, matrix, norms, n_rows, n_cols, take_sqrt);
}

template <typename DataT, typename IndexT>
Fused1nnRequirements<IndexT> get_fused_1nn_requirements(
  raft::resources const& handle,
  raft::device_matrix_view<const DataT, IndexT> X,
  raft::device_matrix_view<const DataT, IndexT> centroids,
  cuvs::distance::DistanceType metric,
  int batch_samples,
  int batch_centroids)
{
  const auto path = cuvs::distance::detail::resolve_fused_1nn_backend(handle,
                                                                      X.data_handle(),
                                                                      centroids.data_handle(),
                                                                      X.extent(0),
                                                                      centroids.extent(0),
                                                                      X.extent(1),
                                                                      metric);

  Fused1nnRequirements<IndexT> requirements{};
  requirements.path = path;
  const cuvs::distance::detail::Top1nnTuning default_tuning{};
  requirements.sample_tile = std::min(
    getDataBatchSize(batch_samples, X.extent(0)),
    static_cast<IndexT>(std::min(default_tuning.unfused.row_tile,
                                 static_cast<size_t>(std::numeric_limits<IndexT>::max()))));
  requirements.centroid_tile = std::min(
    getCentroidsBatchSize(batch_centroids, centroids.extent(0)),
    static_cast<IndexT>(std::min(default_tuning.unfused.candidate_tile,
                                 static_cast<size_t>(std::numeric_limits<IndexT>::max()))));
  requirements.workspace_alignment = alignof(int);

  if (path == FusedDistancePath::Cutile) {
    requirements.output_layout = Fused1nnOutputLayout::Soa;
    requirements.norm_policy =
      std::is_same_v<DataT, float> ? Fused1nnNormPolicy::Tf32 : Fused1nnNormPolicy::Default;
    requirements.result_alignment = 16;
    requirements.distance_offset =
      raft::alignTo(sizeof(IndexT) * static_cast<size_t>(X.extent(0)), size_t{16});
    requirements.result_bytes =
      requirements.distance_offset + sizeof(DataT) * static_cast<size_t>(X.extent(0));
    if constexpr (std::is_same_v<IndexT, int64_t>) {
      requirements.workspace_bytes =
        sizeof(int) *
        cuvs::distance::detail::fused_1nn_cutile_index_workspace_rows<DataT>(X.extent(0));
    }
  } else {
    requirements.output_layout    = Fused1nnOutputLayout::Kvp;
    requirements.norm_policy      = Fused1nnNormPolicy::Default;
    requirements.result_alignment = alignof(raft::KeyValuePair<IndexT, DataT>);
    requirements.result_bytes =
      sizeof(raft::KeyValuePair<IndexT, DataT>) * static_cast<size_t>(X.extent(0));
    if (path == FusedDistancePath::Cutlass) {
      requirements.workspace_bytes = sizeof(int) * static_cast<size_t>(X.extent(0));
    } else if (path == FusedDistancePath::Unfused &&
               (metric == cuvs::distance::DistanceType::L2Expanded ||
                metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                metric == cuvs::distance::DistanceType::CosineExpanded)) {
      auto sample_tile         = requirements.sample_tile;
      const auto centroid_tile = requirements.centroid_tile;
      sample_tile = std::min(sample_tile, std::numeric_limits<IndexT>::max() / centroid_tile);
      const size_t distance_bytes =
        sizeof(DataT) * static_cast<size_t>(sample_tile) * static_cast<size_t>(centroid_tile);
      requirements.workspace_alignment =
        std::max(alignof(DataT), alignof(raft::KeyValuePair<IndexT, DataT>));
      requirements.workspace_bytes =
        raft::alignTo(distance_bytes, alignof(raft::KeyValuePair<IndexT, DataT>));
      if (centroid_tile < centroids.extent(0)) {
        requirements.workspace_bytes +=
          sizeof(raft::KeyValuePair<IndexT, DataT>) * static_cast<size_t>(sample_tile);
      }
    }
  }
  return requirements;
}

template <typename DataT, typename IndexT>
void min_cluster_and_distance_compute_impl(raft::resources const& handle,
                                           raft::device_matrix_view<const DataT, IndexT> X,
                                           raft::device_matrix_view<const DataT, IndexT> centroids,
                                           IndexT* nearest_idx,
                                           DataT* nearest_dist,
                                           raft::KeyValuePair<IndexT, DataT>* native_kvp,
                                           raft::device_vector_view<const DataT, IndexT> L2NormX,
                                           rmm::device_uvector<DataT>& L2NormBuf_OR_DistBuf,
                                           cuvs::distance::DistanceType metric,
                                           int batch_samples,
                                           int batch_centroids,
                                           rmm::device_uvector<char>& workspace,
                                           const Fused1nnRequirements<IndexT>& requirements,
                                           const DataT* cutile_x_norm)
{
  cudaStream_t stream  = raft::resource::get_cuda_stream(handle);
  auto n_samples       = X.extent(0);
  auto n_features      = X.extent(1);
  auto n_clusters      = centroids.extent(0);
  const bool is_l2_cos = metric == cuvs::distance::DistanceType::L2Expanded ||
                         metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                         metric == cuvs::distance::DistanceType::CosineExpanded;
  const auto fused_path   = requirements.path;
  const bool cutile_ready = fused_path == FusedDistancePath::Cutile;
  if (workspace.size() < requirements.workspace_bytes) {
    workspace.resize(requirements.workspace_bytes, stream);
  }
  if (cutile_ready) {
    RAFT_EXPECTS(native_kvp == nullptr && nearest_idx != nullptr && nearest_dist != nullptr,
                 "cuTile fused 1-NN requires native separate index and distance outputs");
  } else {
    RAFT_EXPECTS(native_kvp != nullptr && nearest_idx == nullptr && nearest_dist == nullptr,
                 "CUTLASS and unfused 1-NN require their native KVP output");
  }

  if (is_l2_cos || cutile_ready) {
    const DataT* x_norm_ptr         = nullptr;
    const DataT* centroids_norm_ptr = nullptr;
    if (is_l2_cos) {
      x_norm_ptr = L2NormX.data_handle();
      if constexpr (std::is_same_v<DataT, float>) {
        if (cutile_ready) {
          constexpr size_t norm_alignment = 16 / sizeof(float);
          const bool take_sqrt            = metric == cuvs::distance::DistanceType::CosineExpanded;
          size_t centroid_offset          = 0;
          if (cutile_x_norm == nullptr) {
            centroid_offset = raft::alignTo(static_cast<size_t>(n_samples), norm_alignment);
          }
          L2NormBuf_OR_DistBuf.resize(centroid_offset + static_cast<size_t>(n_clusters), stream);
          auto* tf32_x_norms        = cutile_x_norm == nullptr ? L2NormBuf_OR_DistBuf.data()
                                                               : const_cast<DataT*>(cutile_x_norm);
          auto* tf32_centroid_norms = L2NormBuf_OR_DistBuf.data() + centroid_offset;
          if (cutile_x_norm == nullptr) {
            compute_tf32_row_norms(
              handle, X.data_handle(), tf32_x_norms, n_samples, n_features, take_sqrt);
          }
          {
            compute_tf32_row_norms(handle,
                                   centroids.data_handle(),
                                   tf32_centroid_norms,
                                   n_clusters,
                                   n_features,
                                   take_sqrt);
          }
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

      if (!cutile_ready) {
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
    }

    const bool needs_index_workspace = cutile_ready && std::is_same_v<IndexT, int64_t>;
    cuvs::distance::detail::Top1nnTuning tuning{};
    tuning.unfused.row_tile       = static_cast<size_t>(requirements.sample_tile);
    tuning.unfused.candidate_tile = static_cast<size_t>(requirements.centroid_tile);
    cuvs::distance::top_1_nn<DataT, IndexT>(
      handle,
      cutile_ready ? nearest_idx : nullptr,
      cutile_ready ? nearest_dist : nullptr,
      X.data_handle(),
      centroids.data_handle(),
      x_norm_ptr,
      centroids_norm_ptr,
      n_samples,
      n_clusters,
      n_features,
      tuning,
      !cutile_ready || needs_index_workspace ? (void*)workspace.data() : nullptr,
      workspace.size(),
      metric != cuvs::distance::DistanceType::L2Expanded,
      false,
      true,
      metric,
      0.0f,
      fused_path,
      native_kvp,
      stream);
  } else {
    auto dataBatchSize      = getDataBatchSize(batch_samples, n_samples);
    auto centroidsBatchSize = getCentroidsBatchSize(batch_centroids, n_clusters);

    L2NormBuf_OR_DistBuf.resize(dataBatchSize * centroidsBatchSize, stream);

    auto pairwiseDistance = raft::make_device_matrix_view<DataT, IndexT>(
      L2NormBuf_OR_DistBuf.data(), dataBatchSize, centroidsBatchSize);

    using KeyValueT      = raft::KeyValuePair<IndexT, DataT>;
    auto* kvp_output     = native_kvp;
    auto kvp_output_view = raft::make_device_vector_view<KeyValueT, IndexT>(kvp_output, n_samples);
    KeyValueT initial_value(0, std::numeric_limits<DataT>::max());
    raft::matrix::fill(handle, kvp_output_view, initial_value);

    for (IndexT dIdx = 0; dIdx < n_samples; dIdx += dataBatchSize) {
      auto ns = std::min((IndexT)dataBatchSize, n_samples - dIdx);

      auto datasetView = raft::make_device_matrix_view<const DataT, IndexT>(
        X.data_handle() + (dIdx * n_features), ns, n_features);

      auto temp_kvp_view = raft::make_device_vector_view<KeyValueT, IndexT>(kvp_output + dIdx, ns);

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
            KeyValueT pair;
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
                                  rmm::device_uvector<char>& workspace,
                                  const Fused1nnRequirements<IndexT>& requirements,
                                  const DataT* cutile_x_norm)
{
  RAFT_EXPECTS(requirements.output_layout == Fused1nnOutputLayout::Soa,
               "resolved fused 1-NN plan requires separate output arrays");
  if constexpr (is_cutile_fused_data_type_v<DataT>) {
    if (requirements.path == FusedDistancePath::Cutile) {
      RAFT_EXPECTS(cuvs::distance::detail::can_launch_fused_1nn_tile(nearest_idx.data_handle(),
                                                                     nearest_dist.data_handle(),
                                                                     X.data_handle(),
                                                                     centroids.data_handle(),
                                                                     X.extent(0),
                                                                     centroids.extent(0),
                                                                     X.extent(1),
                                                                     metric),
                   "resolved cuTile plan has incompatible output storage");
    }
  }

  min_cluster_and_distance_compute_impl<DataT, IndexT>(handle,
                                                       X,
                                                       centroids,
                                                       nearest_idx.data_handle(),
                                                       nearest_dist.data_handle(),
                                                       nullptr,
                                                       L2NormX,
                                                       L2NormBuf_OR_DistBuf,
                                                       metric,
                                                       batch_samples,
                                                       batch_centroids,
                                                       workspace,
                                                       requirements,
                                                       cutile_x_norm);
}

template <typename DataT, typename IndexT>
void minClusterAndDistanceComputeKvp(
  raft::resources const& handle,
  raft::device_matrix_view<const DataT, IndexT> X,
  raft::device_matrix_view<const DataT, IndexT> centroids,
  raft::device_vector_view<raft::KeyValuePair<IndexT, DataT>, IndexT> nearest,
  raft::device_vector_view<const DataT, IndexT> L2NormX,
  rmm::device_uvector<DataT>& L2NormBuf_OR_DistBuf,
  cuvs::distance::DistanceType metric,
  int batch_samples,
  int batch_centroids,
  rmm::device_uvector<char>& workspace,
  const Fused1nnRequirements<IndexT>& requirements)
{
  RAFT_EXPECTS(requirements.output_layout == Fused1nnOutputLayout::Kvp,
               "resolved fused 1-NN plan requires KVP output");
  min_cluster_and_distance_compute_impl<DataT, IndexT>(handle,
                                                       X,
                                                       centroids,
                                                       nullptr,
                                                       nullptr,
                                                       nearest.data_handle(),
                                                       L2NormX,
                                                       L2NormBuf_OR_DistBuf,
                                                       metric,
                                                       batch_samples,
                                                       batch_centroids,
                                                       workspace,
                                                       requirements,
                                                       nullptr);
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
    rmm::device_uvector<char>& workspace,                    \
    const Fused1nnRequirements<IndexT>& requirements,        \
    const DataT* cutile_x_norm);

INSTANTIATE_MIN_CLUSTER_AND_DISTANCE(float, int64_t)
INSTANTIATE_MIN_CLUSTER_AND_DISTANCE(double, int64_t)
INSTANTIATE_MIN_CLUSTER_AND_DISTANCE(float, int)
INSTANTIATE_MIN_CLUSTER_AND_DISTANCE(double, int)

#undef INSTANTIATE_MIN_CLUSTER_AND_DISTANCE

template void computeCutileRowNorms<int>(
  raft::resources const&, const float*, float*, int, int, bool);
template void computeCutileRowNorms<int64_t>(
  raft::resources const&, const float*, float*, int64_t, int64_t, bool);

#define INSTANTIATE_FUSED_1NN_REQUIREMENTS(DataT, IndexT)                          \
  template Fused1nnRequirements<IndexT> get_fused_1nn_requirements<DataT, IndexT>( \
    raft::resources const&,                                                        \
    raft::device_matrix_view<const DataT, IndexT>,                                 \
    raft::device_matrix_view<const DataT, IndexT>,                                 \
    cuvs::distance::DistanceType,                                                  \
    int,                                                                           \
    int);

INSTANTIATE_FUSED_1NN_REQUIREMENTS(float, int64_t)
INSTANTIATE_FUSED_1NN_REQUIREMENTS(double, int64_t)
INSTANTIATE_FUSED_1NN_REQUIREMENTS(float, int)
INSTANTIATE_FUSED_1NN_REQUIREMENTS(double, int)

#undef INSTANTIATE_FUSED_1NN_REQUIREMENTS

#define INSTANTIATE_MIN_CLUSTER_AND_DISTANCE_KVP(DataT, IndexT)          \
  template void minClusterAndDistanceComputeKvp<DataT, IndexT>(          \
    raft::resources const&,                                              \
    raft::device_matrix_view<const DataT, IndexT>,                       \
    raft::device_matrix_view<const DataT, IndexT>,                       \
    raft::device_vector_view<raft::KeyValuePair<IndexT, DataT>, IndexT>, \
    raft::device_vector_view<const DataT, IndexT>,                       \
    rmm::device_uvector<DataT>&,                                         \
    cuvs::distance::DistanceType,                                        \
    int,                                                                 \
    int,                                                                 \
    rmm::device_uvector<char>&,                                          \
    const Fused1nnRequirements<IndexT>&);

INSTANTIATE_MIN_CLUSTER_AND_DISTANCE_KVP(float, int64_t)
INSTANTIATE_MIN_CLUSTER_AND_DISTANCE_KVP(double, int64_t)
INSTANTIATE_MIN_CLUSTER_AND_DISTANCE_KVP(float, int)
INSTANTIATE_MIN_CLUSTER_AND_DISTANCE_KVP(double, int)

#undef INSTANTIATE_MIN_CLUSTER_AND_DISTANCE_KVP

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

  FusedDistancePath fused_path =
    is_l2_cos ? use_fused<DataT, IndexT, IndexT>(handle, n_samples, n_clusters, n_features, metric)
              : FusedDistancePath::Unfused;
  bool cutile_ready = false;
  if constexpr (is_cutile_fused_data_type_v<DataT>) {
    if (fused_path == FusedDistancePath::Cutile) {
      cutile_ready =
        cuvs::distance::detail::can_launch_fused_1nn_tile(static_cast<IndexT*>(nullptr),
                                                          minClusterDistance.data_handle(),
                                                          X.data_handle(),
                                                          centroids.data_handle(),
                                                          n_samples,
                                                          n_clusters,
                                                          n_features,
                                                          metric);
      if (!cutile_ready) { fused_path = use_legacy_fused(handle, n_samples, n_clusters, metric); }
    }
  }

  if (uses_fused_distance_nn(fused_path)) {
    const DataT* x_norm_ptr = L2NormX.data_handle();
    const DataT* centroids_norm_ptr;
    if constexpr (std::is_same_v<DataT, float>) {
      if (cutile_ready) {
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

    if (!cutile_ready) {
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

    if (!cutile_ready) { workspace.resize(sizeof(int) * static_cast<size_t>(n_samples), stream); }

    cuvs::distance::detail::Top1nnTuning tuning{};
    if (cutile_ready) {
      cuvs::distance::top_1_nn<DataT, IndexT>(handle,
                                              nullptr,
                                              minClusterDistance.data_handle(),
                                              X.data_handle(),
                                              centroids.data_handle(),
                                              x_norm_ptr,
                                              centroids_norm_ptr,
                                              n_samples,
                                              n_clusters,
                                              n_features,
                                              tuning,
                                              nullptr,
                                              0,
                                              metric != cuvs::distance::DistanceType::L2Expanded,
                                              false,
                                              true,
                                              metric,
                                              0.0f,
                                              FusedDistancePath::Cutile,
                                              nullptr,
                                              stream);
    } else {
      cuvs::distance::top_1_nn<DataT, IndexT>(nullptr,
                                              minClusterDistance.data_handle(),
                                              X.data_handle(),
                                              centroids.data_handle(),
                                              x_norm_ptr,
                                              centroids_norm_ptr,
                                              n_samples,
                                              n_clusters,
                                              n_features,
                                              tuning,
                                              (void*)workspace.data(),
                                              workspace.size(),
                                              metric != cuvs::distance::DistanceType::L2Expanded,
                                              false,
                                              true,
                                              metric,
                                              0.0f,
                                              FusedDistancePath::Cutlass,
                                              nullptr,
                                              stream);
    }
  } else {
    raft::matrix::fill(handle, minClusterDistance, std::numeric_limits<DataT>::max());
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
