/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../distance/top_1_nn.cuh"
#include "kmeans_common.cuh"

#include <raft/matrix/init.cuh>
#include <raft/util/integer_utils.hpp>
#include <raft/util/kernel_launch.hpp>

#include <rmm/device_uvector.hpp>

#include <mma.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>

namespace cuvs::cluster::kmeans::detail {

namespace {

void* reserve_aligned_workspace(rmm::device_uvector<char>& workspace,
                                std::size_t prefix_bytes,
                                std::size_t workspace_bytes,
                                std::size_t alignment,
                                cuda::stream_ref stream)
{
  RAFT_EXPECTS(alignment > 0, "top_1_nn workspace alignment must be positive");
  const auto required_bytes =
    prefix_bytes + (workspace_bytes == 0 ? std::size_t{0} : workspace_bytes + alignment - 1);
  if (workspace.size() < required_bytes) { workspace.resize(required_bytes, stream); }
  if (workspace_bytes == 0) { return nullptr; }
  const auto candidate = reinterpret_cast<std::uintptr_t>(workspace.data() + prefix_bytes);
  const auto aligned   = (candidate + alignment - 1) / alignment * alignment;
  return reinterpret_cast<void*>(aligned);
}

/**
 * TF32-rounded (and with `Fp32`, also FP32) squared norms of each row in one pass over the input.
 *
 * `LanesPerRow` lanes reduce one row, so short rows do not leave most of a warp idle, and each
 * lane group handles kRows rows at a time to keep more loads in flight.
 */
template <typename IndexT, int LanesPerRow, bool Fp32>
__global__ void fp32_tf32_norms_kernel(const float* input,
                                       float* fp32_norms,
                                       float* tf32_norms,
                                       IndexT rows,
                                       IndexT cols,
                                       bool take_sqrt)
{
  constexpr int kRows           = 2;
  constexpr int groups_per_warp = raft::WarpSize / LanesPerRow;
  const int lane                = threadIdx.x % raft::WarpSize;
  const int sub                 = lane % LanesPerRow;
  // Rows are counted in 64 bits: with a 32-bit IndexT, base + stride can pass INT_MAX.
  const auto warp =
    (static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / raft::WarpSize;
  const auto stride =
    static_cast<std::int64_t>(gridDim.x) * (blockDim.x / raft::WarpSize) * groups_per_warp * kRows;

  for (std::int64_t base = warp * groups_per_warp * kRows; base < rows; base += stride) {
    float fp32_sum[kRows]{};
    float tf32_sum[kRows]{};
    const std::int64_t first_row = base + lane / LanesPerRow;
#pragma unroll
    for (int r = 0; r < kRows; r++) {
      const std::int64_t row = first_row + r * groups_per_warp;
      if (row >= rows) { continue; }
      auto add = [&](float value) {
        if constexpr (Fp32) { fp32_sum[r] += value * value; }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        value = nvcuda::wmma::__float_to_tf32(value);
#endif
        tf32_sum[r] += value * value;
      };
      const float* row_ptr = input + row * static_cast<std::int64_t>(cols);
      for (IndexT col = sub; col < cols; col += LanesPerRow) {
        add(row_ptr[col]);
      }
    }
#pragma unroll
    for (int r = 0; r < kRows; r++) {
      for (int offset = LanesPerRow / 2; offset > 0; offset /= 2) {
        if constexpr (Fp32) { fp32_sum[r] += __shfl_xor_sync(0xffffffff, fp32_sum[r], offset); }
        tf32_sum[r] += __shfl_xor_sync(0xffffffff, tf32_sum[r], offset);
      }
      const std::int64_t row = first_row + r * groups_per_warp;
      if (sub == 0 && row < rows) {
        if constexpr (Fp32) { fp32_norms[row] = take_sqrt ? sqrtf(fp32_sum[r]) : fp32_sum[r]; }
        tf32_norms[row] = take_sqrt ? sqrtf(tf32_sum[r]) : tf32_sum[r];
      }
    }
  }
}

/** Launches fp32_tf32_norms_kernel; `fp32_norms` is written only when `Fp32`. */
template <bool Fp32, typename IndexT>
void launch_row_norms(raft::resources const& handle,
                      const float* x,
                      float* fp32_norms,
                      float* tf32_norms,
                      IndexT rows,
                      IndexT cols,
                      bool take_sqrt)
{
  // About 8 elements of a row per lane: the largest power-of-two lane group up to a warp.
  int lanes = 1;
  while (lanes < raft::WarpSize && lanes * 2 * 8 <= cols) {
    lanes *= 2;
  }
  constexpr int threads_per_block = 256;
  auto launch                     = [&](auto lanes_per_row) {
    constexpr int kLanes      = decltype(lanes_per_row)::value;
    const auto rows_per_block = std::size_t{threads_per_block / kLanes * 2};
    const auto blocks         = static_cast<unsigned int>(std::min<std::size_t>(
      raft::ceildiv<std::size_t>(static_cast<std::size_t>(rows), rows_per_block), 65535));
    raft::launch_kernel(handle,
                        blocks,
                        threads_per_block,
                        fp32_tf32_norms_kernel<IndexT, kLanes, Fp32>,
                        x,
                        fp32_norms,
                        tf32_norms,
                        rows,
                        cols,
                        take_sqrt);
  };
  switch (lanes) {
    case 1: launch(std::integral_constant<int, 1>{}); break;
    case 2: launch(std::integral_constant<int, 2>{}); break;
    case 4: launch(std::integral_constant<int, 4>{}); break;
    case 8: launch(std::integral_constant<int, 8>{}); break;
    case 16: launch(std::integral_constant<int, 16>{}); break;
    default: launch(std::integral_constant<int, 32>{}); break;
  }
}

template <typename IndexT>
void compute_tf32_norms(raft::resources const& handle,
                        const float* matrix,
                        float* norms,
                        IndexT rows,
                        IndexT cols,
                        bool take_sqrt)
{
  launch_row_norms<false>(handle, matrix, nullptr, norms, rows, cols, take_sqrt);
}

template <typename DataT, typename IndexT>
MinClusterAndDistanceResult<DataT, IndexT> make_native_result(
  const cuvs::distance::detail::Top1nnPlan<IndexT>& plan,
  IndexT size,
  rmm::device_uvector<char>& output_storage,
  cuda::stream_ref stream)
{
  auto* storage =
    reserve_aligned_workspace(output_storage, 0, plan.output_bytes, plan.output_alignment, stream);
  return cuvs::distance::bind_top_1_nn_result_view<DataT>(plan, size, storage, plan.output_bytes);
}
}  // namespace

template <typename IndexT>
void compute_fp32_tf32_norms(raft::resources const& handle,
                             raft::device_matrix_view<const float, IndexT> input,
                             raft::device_vector_view<float, IndexT> fp32_norms,
                             raft::device_vector_view<float, IndexT> tf32_norms,
                             bool take_sqrt)
{
  RAFT_EXPECTS(fp32_norms.extent(0) == input.extent(0),
               "FP32 norm output must have one element per input row");
  RAFT_EXPECTS(tf32_norms.extent(0) == input.extent(0),
               "TF32 norm output must have one element per input row");
  if (input.extent(0) == 0) { return; }

  launch_row_norms<true>(handle,
                         input.data_handle(),
                         fp32_norms.data_handle(),
                         tf32_norms.data_handle(),
                         input.extent(0),
                         input.extent(1),
                         take_sqrt);
}

template void compute_fp32_tf32_norms<int>(raft::resources const&,
                                           raft::device_matrix_view<const float, int>,
                                           raft::device_vector_view<float, int>,
                                           raft::device_vector_view<float, int>,
                                           bool);
template void compute_fp32_tf32_norms<int64_t>(raft::resources const&,
                                               raft::device_matrix_view<const float, int64_t>,
                                               raft::device_vector_view<float, int64_t>,
                                               raft::device_vector_view<float, int64_t>,
                                               bool);

// Calculates the nearest centroid and distance for every sample using the requested backend.
template <typename DataT, typename IndexT>
MinClusterAndDistanceResult<DataT, IndexT> minClusterAndDistanceCompute(
  raft::resources const& handle,
  raft::device_matrix_view<const DataT, IndexT> X,
  raft::device_matrix_view<const DataT, IndexT> centroids,
  rmm::device_uvector<char>& output_storage,
  raft::device_vector_view<const DataT, IndexT> L2NormX,
  rmm::device_uvector<DataT>& L2NormBuf_OR_DistBuf,
  cuvs::distance::DistanceType metric,
  int batch_samples,
  int batch_centroids,
  rmm::device_uvector<char>& workspace,
  std::optional<raft::device_vector_view<const DataT, IndexT>> Tf32NormX,
  cuvs::distance::detail::Top1nnBackend backend,
  const cuvs::distance::detail::top_1_nn_cagra_index* ann_index,
  bool need_compute_distance)
{
  auto stream       = raft::resource::get_cuda_stream(handle);
  auto n_samples    = X.extent(0);
  auto n_features   = X.extent(1);
  auto n_clusters   = centroids.extent(0);
  const bool is_1nn = metric == cuvs::distance::DistanceType::L2Expanded ||
                      metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                      metric == cuvs::distance::DistanceType::CosineExpanded;
  cuvs::distance::detail::Top1nnPlan<IndexT> plan{};

  if (is_1nn) {
    cuvs::distance::detail::Top1nnTuning tuning{};
    tuning.unfused.row_tile = static_cast<std::size_t>(std::min(
      getDataBatchSize(batch_samples, n_samples), static_cast<IndexT>(tuning.unfused.row_tile)));
    tuning.unfused.candidate_tile =
      static_cast<std::size_t>(std::min(getCentroidsBatchSize(batch_centroids, n_clusters),
                                        static_cast<IndexT>(tuning.unfused.candidate_tile)));
    tuning.cagra.index = ann_index;  // only used by Top1nnBackend::Cagra
    plan               = cuvs::distance::probe_top_1_nn(handle,
                                          X.data_handle(),
                                          centroids.data_handle(),
                                          n_samples,
                                          n_clusters,
                                          n_features,
                                          tuning,
                                          metric,
                                          backend);
    RAFT_EXPECTS(plan.available, "Requested top_1_nn backend is unavailable for KMeans assignment");

    // The plan's norm policy decides the norms of x and the centroids: TF32 (x from the cached
    // `Tf32NormX` or computed here), Native (x from the caller's `L2NormX`, or computed here when
    // it is empty) or None. One allocation holds the result and everything the 1-NN needs, so a
    // call allocates once: [output | backend workspace | x norms | centroid norms].
    using cuvs::distance::detail::Top1nnNormPolicy;
    RAFT_EXPECTS(L2NormX.extent(0) == 0 || L2NormX.extent(0) == n_samples,
                 "FP32 norms must be empty or have one element per input row");
    const auto norm_policy          = plan.norm_policy;
    const bool take_sqrt            = metric == cuvs::distance::DistanceType::CosineExpanded;
    const bool need_compute_x_norms = norm_policy == Top1nnNormPolicy::Tf32 ? !Tf32NormX.has_value()
                                      : norm_policy == Top1nnNormPolicy::Native
                                        ? L2NormX.extent(0) == 0
                                        : false;
    const bool need_y_norms         = norm_policy != Top1nnNormPolicy::None;
    const auto alignment            = std::max<std::size_t>(
      {std::size_t{256}, plan.output_alignment, plan.workspace_alignment, plan.norm_alignment});
    const auto workspace_offset = raft::round_up_safe(plan.output_bytes, alignment);
    const auto x_norm_offset =
      raft::round_up_safe(workspace_offset + plan.workspace_bytes, alignment);
    const auto y_norm_offset = raft::round_up_safe(
      x_norm_offset +
        (need_compute_x_norms ? sizeof(DataT) * static_cast<std::size_t>(n_samples) : 0),
      alignment);
    const auto storage_bytes =
      y_norm_offset + (need_y_norms ? sizeof(DataT) * static_cast<std::size_t>(n_clusters) : 0);
    auto* storage = static_cast<char*>(
      reserve_aligned_workspace(output_storage, 0, storage_bytes, alignment, stream));
    void* backend_workspace = plan.workspace_bytes == 0 ? nullptr : storage + workspace_offset;

    auto* computed_x_norm = reinterpret_cast<DataT*>(storage + x_norm_offset);
    const DataT* x_norm   = need_compute_x_norms ? computed_x_norm : L2NormX.data_handle();
    DataT* y_norm = need_y_norms ? reinterpret_cast<DataT*>(storage + y_norm_offset) : nullptr;
    switch (norm_policy) {
      case Top1nnNormPolicy::Tf32:
        if constexpr (std::is_same_v<DataT, float>) {
          if (need_compute_x_norms && !need_compute_distance) {
            raft::matrix::fill(
              handle,
              raft::make_device_vector_view<DataT, IndexT>(computed_x_norm, n_samples),
              DataT{0});
          } else if (need_compute_x_norms) {
            compute_tf32_norms(
              handle, X.data_handle(), computed_x_norm, n_samples, n_features, take_sqrt);
          } else {
            RAFT_EXPECTS(Tf32NormX->extent(0) == n_samples,
                         "Cached TF32 norms must have one element per input row");
            x_norm = Tf32NormX->data_handle();
          }
          compute_tf32_norms(
            handle, centroids.data_handle(), y_norm, n_clusters, n_features, take_sqrt);
        }
        break;
      case Top1nnNormPolicy::Native: {
        auto native_norms = [&](raft::device_matrix_view<const DataT, IndexT> matrix, DataT* out) {
          auto rows = raft::make_device_matrix_view<const DataT, int64_t>(
            matrix.data_handle(), matrix.extent(0), matrix.extent(1));
          auto out_view = raft::make_device_vector_view<DataT, int64_t>(out, rows.extent(0));
          if (take_sqrt) {
            raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
              handle, rows, out_view, raft::sqrt_op{});
          } else {
            raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
              handle, rows, out_view);
          }
        };
        if (need_compute_x_norms) { native_norms(X, computed_x_norm); }
        native_norms(centroids, y_norm);
        break;
      }
      case Top1nnNormPolicy::None: break;
    }

    auto result =
      cuvs::distance::bind_top_1_nn_result_view<DataT>(plan, n_samples, storage, plan.output_bytes);

    bool init_out_buffer = true;
    if (plan.output_layout == cuvs::distance::detail::Top1nnOutputLayout::KeyValuePair) {
      auto output = raft::make_device_vector_view<raft::KeyValuePair<IndexT, DataT>, IndexT>(
        result.key_values(), n_samples);
      raft::matrix::fill(
        handle, output, raft::KeyValuePair<IndexT, DataT>{0, std::numeric_limits<DataT>::max()});
      init_out_buffer = false;
    }

    auto launch = [&](auto output) {
      cuvs::distance::top_1_nn<DataT, IndexT>(handle,
                                              output,
                                              X.data_handle(),
                                              centroids.data_handle(),
                                              x_norm,
                                              y_norm,
                                              n_samples,
                                              n_clusters,
                                              n_features,
                                              tuning,
                                              backend_workspace,
                                              plan.workspace_bytes,
                                              metric != cuvs::distance::DistanceType::L2Expanded,
                                              init_out_buffer,
                                              true,
                                              metric,
                                              0.0f,
                                              plan);
    };
    if constexpr (std::is_same_v<DataT, float>) {
      result.visit_native(
        [&](IndexT* indices, DataT* distances) {
          launch(cuvs::distance::Top1nnOutput<IndexT, DataT>{indices, distances});
        },
        [&](raft::KeyValuePair<IndexT, DataT>* key_values) { launch(key_values); });
    } else {
      RAFT_EXPECTS(result.key_values() != nullptr, "KMeans assignment requires KVP output");
      launch(result.key_values());
    }
    return result;
  }

  using KeyValueT = raft::KeyValuePair<IndexT, DataT>;
  RAFT_EXPECTS(n_samples >= 0, "KMeans sample count must not be negative");
  const auto output_size = static_cast<std::size_t>(n_samples);
  RAFT_EXPECTS(output_size <= std::numeric_limits<std::size_t>::max() / sizeof(KeyValueT),
               "KMeans assignment output size overflows size_t");
  plan.m                = n_samples;
  plan.output_layout    = cuvs::distance::detail::Top1nnOutputLayout::KeyValuePair;
  plan.output_alignment = alignof(KeyValueT);
  plan.output_bytes     = output_size * sizeof(KeyValueT);
  auto result        = make_native_result<DataT, IndexT>(plan, n_samples, output_storage, stream);
  auto* kvp_output   = result.key_values();
  auto dataBatchSize = getDataBatchSize(batch_samples, n_samples);
  auto centroidsBatchSize = getCentroidsBatchSize(batch_centroids, n_clusters);
  L2NormBuf_OR_DistBuf.resize(dataBatchSize * centroidsBatchSize, stream);
  auto pairwiseDistance = raft::make_device_matrix_view<DataT, IndexT>(
    L2NormBuf_OR_DistBuf.data(), dataBatchSize, centroidsBatchSize);
  auto output =
    raft::make_device_vector_view<raft::KeyValuePair<IndexT, DataT>, IndexT>(kvp_output, n_samples);
  raft::KeyValuePair<IndexT, DataT> initial_value(0, std::numeric_limits<DataT>::max());
  raft::matrix::fill(handle, output, initial_value);

  for (IndexT dIdx = 0; dIdx < n_samples; dIdx += dataBatchSize) {
    auto ns          = std::min(static_cast<IndexT>(dataBatchSize), n_samples - dIdx);
    auto datasetView = raft::make_device_matrix_view<const DataT, IndexT>(
      X.data_handle() + static_cast<std::size_t>(dIdx) * n_features, ns, n_features);
    auto output_view = raft::make_device_vector_view<raft::KeyValuePair<IndexT, DataT>, IndexT>(
      kvp_output + dIdx, ns);
    for (IndexT cIdx = 0; cIdx < n_clusters; cIdx += centroidsBatchSize) {
      auto nc            = std::min(static_cast<IndexT>(centroidsBatchSize), n_clusters - cIdx);
      auto centroidsView = raft::make_device_matrix_view<const DataT, IndexT>(
        centroids.data_handle() + static_cast<std::size_t>(cIdx) * n_features, nc, n_features);
      auto pairwiseDistanceView =
        raft::make_device_matrix_view<DataT, IndexT>(pairwiseDistance.data_handle(), ns, nc);
      pairwise_distance_kmeans<DataT, IndexT>(
        handle, datasetView, centroidsView, pairwiseDistanceView, metric);
      raft::linalg::coalescedReduction(
        output_view.data_handle(),
        pairwiseDistanceView.data_handle(),
        pairwiseDistanceView.extent(1),
        pairwiseDistanceView.extent(0),
        initial_value,
        stream.get(),
        true,
        [=] __device__(const DataT val, const IndexT i) {
          return raft::KeyValuePair<IndexT, DataT>{cIdx + i, val};
        },
        raft::argmin_op{},
        raft::identity_op{});
    }
  }
  return result;
}

#define INSTANTIATE_MIN_CLUSTER_AND_DISTANCE(DataT, IndexT)                                        \
  template MinClusterAndDistanceResult<DataT, IndexT> minClusterAndDistanceCompute<DataT, IndexT>( \
    raft::resources const&,                                                                        \
    raft::device_matrix_view<const DataT, IndexT>,                                                 \
    raft::device_matrix_view<const DataT, IndexT>,                                                 \
    rmm::device_uvector<char>&,                                                                    \
    raft::device_vector_view<const DataT, IndexT>,                                                 \
    rmm::device_uvector<DataT>&,                                                                   \
    cuvs::distance::DistanceType,                                                                  \
    int,                                                                                           \
    int,                                                                                           \
    rmm::device_uvector<char>&,                                                                    \
    std::optional<raft::device_vector_view<const DataT, IndexT>>,                                  \
    cuvs::distance::detail::Top1nnBackend,                                                         \
    const cuvs::distance::detail::top_1_nn_cagra_index*,                                           \
    bool);

INSTANTIATE_MIN_CLUSTER_AND_DISTANCE(float, int64_t)
INSTANTIATE_MIN_CLUSTER_AND_DISTANCE(double, int64_t)
INSTANTIATE_MIN_CLUSTER_AND_DISTANCE(float, int)
INSTANTIATE_MIN_CLUSTER_AND_DISTANCE(double, int)

#undef INSTANTIATE_MIN_CLUSTER_AND_DISTANCE

template <typename DataT, typename IndexT>
void minClusterDistanceCompute(
  raft::resources const& handle,
  raft::device_matrix_view<const DataT, IndexT> X,
  raft::device_matrix_view<DataT, IndexT> centroids,
  raft::device_vector_view<DataT, IndexT> minClusterDistance,
  raft::device_vector_view<const DataT, IndexT> L2NormX,
  rmm::device_uvector<DataT>& L2NormBuf_OR_DistBuf,
  cuvs::distance::DistanceType metric,
  int batch_samples,
  int batch_centroids,
  rmm::device_uvector<char>& workspace,
  std::optional<raft::device_vector_view<const DataT, IndexT>> Tf32NormX,
  bool accurate_distances)
{
  auto stream       = raft::resource::get_cuda_stream(handle);
  auto n_samples    = X.extent(0);
  auto n_features   = X.extent(1);
  auto n_clusters   = centroids.extent(0);
  const bool is_1nn = metric == cuvs::distance::DistanceType::L2Expanded ||
                      metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                      metric == cuvs::distance::DistanceType::CosineExpanded;

  if (is_1nn) {
    cuvs::distance::detail::Top1nnTuning tuning{};
    tuning.unfused.row_tile = static_cast<std::size_t>(std::min(
      getDataBatchSize(batch_samples, n_samples), static_cast<IndexT>(tuning.unfused.row_tile)));
    tuning.unfused.candidate_tile =
      static_cast<std::size_t>(std::min(getCentroidsBatchSize(batch_centroids, n_clusters),
                                        static_cast<IndexT>(tuning.unfused.candidate_tile)));
    const auto centroids_const = raft::make_device_matrix_view<const DataT, IndexT>(
      centroids.data_handle(), n_clusters, n_features);
    // Accurate distances (cluster cost) are refined in FP32 by top_1_nn after the 1-NN.
    const auto plan = cuvs::distance::probe_top_1_nn(handle,
                                                     X.data_handle(),
                                                     centroids.data_handle(),
                                                     n_samples,
                                                     n_clusters,
                                                     n_features,
                                                     tuning,
                                                     metric,
                                                     cuvs::distance::detail::Top1nnBackend::Auto,
                                                     false,
                                                     accurate_distances);
    RAFT_EXPECTS(plan.available,
                 "Requested top_1_nn backend is unavailable for KMeans distance reduction");

    const DataT* x_norm  = L2NormX.data_handle();
    const DataT* y_norm  = nullptr;
    const bool take_sqrt = metric == cuvs::distance::DistanceType::CosineExpanded;
    if constexpr (std::is_same_v<DataT, float>) {
      if (plan.norm_policy == cuvs::distance::detail::Top1nnNormPolicy::Tf32) {
        if (Tf32NormX.has_value()) {
          RAFT_EXPECTS(Tf32NormX->extent(0) == n_samples,
                       "Cached TF32 norms must have one element per input row");
          x_norm = Tf32NormX->data_handle();
          L2NormBuf_OR_DistBuf.resize(n_clusters, stream);
          auto* y_norm_output = L2NormBuf_OR_DistBuf.data();
          y_norm              = y_norm_output;
          compute_tf32_norms(
            handle, centroids.data_handle(), y_norm_output, n_clusters, n_features, take_sqrt);
        } else {
          const auto x_norm_bytes = sizeof(DataT) * static_cast<std::size_t>(n_samples);
          const auto y_norm_bytes_offset =
            (x_norm_bytes + plan.norm_alignment - 1) / plan.norm_alignment * plan.norm_alignment;
          const auto y_norm_offset = y_norm_bytes_offset / sizeof(DataT);
          L2NormBuf_OR_DistBuf.resize(y_norm_offset + static_cast<std::size_t>(n_clusters), stream);
          x_norm = L2NormBuf_OR_DistBuf.data();
          y_norm = x_norm + y_norm_offset;
          compute_tf32_norms(
            handle, X.data_handle(), L2NormBuf_OR_DistBuf.data(), n_samples, n_features, take_sqrt);
          compute_tf32_norms(handle,
                             centroids.data_handle(),
                             L2NormBuf_OR_DistBuf.data() + y_norm_offset,
                             n_clusters,
                             n_features,
                             take_sqrt);
        }
      }
    }
    if (y_norm == nullptr) {
      L2NormBuf_OR_DistBuf.resize(n_clusters, stream);
      auto centroids_norm =
        raft::make_device_vector_view<DataT, IndexT>(L2NormBuf_OR_DistBuf.data(), n_clusters);
      if (take_sqrt) {
        raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
          handle, centroids_const, centroids_norm, raft::sqrt_op{});
      } else {
        raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
          handle, centroids_const, centroids_norm);
      }
      y_norm = L2NormBuf_OR_DistBuf.data();
    }

    auto* backend_workspace = reserve_aligned_workspace(
      workspace, 0, plan.workspace_bytes, plan.workspace_alignment, stream);

    auto launch = [&](auto output) {
      cuvs::distance::top_1_nn<DataT, IndexT>(handle,
                                              output,
                                              X.data_handle(),
                                              centroids.data_handle(),
                                              x_norm,
                                              y_norm,
                                              n_samples,
                                              n_clusters,
                                              n_features,
                                              tuning,
                                              backend_workspace,
                                              plan.workspace_bytes,
                                              metric != cuvs::distance::DistanceType::L2Expanded,
                                              true,
                                              true,
                                              metric,
                                              0.0f,
                                              plan);
    };
    if constexpr (std::is_same_v<DataT, float>) {
      if (plan.output_layout == cuvs::distance::detail::Top1nnOutputLayout::Separate) {
        launch(
          cuvs::distance::Top1nnOutput<IndexT, DataT>{nullptr, minClusterDistance.data_handle()});
      } else {
        launch(minClusterDistance.data_handle());
      }
    } else {
      launch(minClusterDistance.data_handle());
    }
    return;
  }

  auto dataBatchSize      = getDataBatchSize(batch_samples, n_samples);
  auto centroidsBatchSize = getCentroidsBatchSize(batch_centroids, n_clusters);
  L2NormBuf_OR_DistBuf.resize(dataBatchSize * centroidsBatchSize, stream);
  auto pairwiseDistance = raft::make_device_matrix_view<DataT, IndexT>(
    L2NormBuf_OR_DistBuf.data(), dataBatchSize, centroidsBatchSize);
  raft::matrix::fill(handle, minClusterDistance, std::numeric_limits<DataT>::max());
  for (IndexT dIdx = 0; dIdx < n_samples; dIdx += dataBatchSize) {
    auto ns          = std::min(static_cast<IndexT>(dataBatchSize), n_samples - dIdx);
    auto datasetView = raft::make_device_matrix_view<const DataT, IndexT>(
      X.data_handle() + static_cast<std::size_t>(dIdx) * n_features, ns, n_features);
    auto minDistanceView =
      raft::make_device_vector_view<DataT, IndexT>(minClusterDistance.data_handle() + dIdx, ns);
    for (IndexT cIdx = 0; cIdx < n_clusters; cIdx += centroidsBatchSize) {
      auto nc            = std::min(static_cast<IndexT>(centroidsBatchSize), n_clusters - cIdx);
      auto centroidsView = raft::make_device_matrix_view<DataT, IndexT>(
        centroids.data_handle() + static_cast<std::size_t>(cIdx) * n_features, nc, n_features);
      auto pairwiseDistanceView =
        raft::make_device_matrix_view<DataT, IndexT>(pairwiseDistance.data_handle(), ns, nc);
      pairwise_distance_kmeans<DataT, IndexT>(
        handle, datasetView, centroidsView, pairwiseDistanceView, metric);
      raft::linalg::coalescedReduction(minDistanceView.data_handle(),
                                       pairwiseDistanceView.data_handle(),
                                       pairwiseDistanceView.extent(1),
                                       pairwiseDistanceView.extent(0),
                                       std::numeric_limits<DataT>::max(),
                                       stream.get(),
                                       true,
                                       raft::identity_op{},
                                       raft::min_op{},
                                       raft::identity_op{});
    }
  }
}

#define INSTANTIATE_MIN_CLUSTER_DISTANCE(DataT, IndexT)           \
  template void minClusterDistanceCompute<DataT, IndexT>(         \
    raft::resources const& handle,                                \
    raft::device_matrix_view<const DataT, IndexT> X,              \
    raft::device_matrix_view<DataT, IndexT> centroids,            \
    raft::device_vector_view<DataT, IndexT> minClusterDistance,   \
    raft::device_vector_view<const DataT, IndexT> L2NormX,        \
    rmm::device_uvector<DataT>& L2NormBuf_OR_DistBuf,             \
    cuvs::distance::DistanceType metric,                          \
    int batch_samples,                                            \
    int batch_centroids,                                          \
    rmm::device_uvector<char>& workspace,                         \
    std::optional<raft::device_vector_view<const DataT, IndexT>>, \
    bool accurate_distances);

INSTANTIATE_MIN_CLUSTER_DISTANCE(float, int64_t)
INSTANTIATE_MIN_CLUSTER_DISTANCE(double, int64_t)
INSTANTIATE_MIN_CLUSTER_DISTANCE(float, int)
INSTANTIATE_MIN_CLUSTER_DISTANCE(double, int)

#undef INSTANTIATE_MIN_CLUSTER_DISTANCE

}  // namespace cuvs::cluster::kmeans::detail
