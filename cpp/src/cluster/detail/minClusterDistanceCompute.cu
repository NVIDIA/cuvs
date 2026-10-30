/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../distance/top_1_nn.cuh"
#include "kmeans_common.cuh"

#include <raft/linalg/coalesced_reduction.cuh>
#include <raft/matrix/init.cuh>

#include <mma.h>

#include <cstdint>

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

struct tf32_square_op {
  template <typename IndexT>
  __device__ float operator()(float value, IndexT) const
  {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    value = nvcuda::wmma::__float_to_tf32(value);
#endif
    return value * value;
  }
};

struct compensated_sum {
  float sum;
  float correction;

  __device__ void add(float value)
  {
    const float next = sum + value;
    correction += fabsf(sum) >= fabsf(value) ? (sum - next) + value : (value - next) + sum;
    sum = next;
  }

  __device__ float value() const { return sum + correction; }
};

struct dual_norm_accumulator {
  compensated_sum fp32;
  compensated_sum tf32;
};

struct add_dual_norm_accumulators {
  __device__ dual_norm_accumulator operator()(dual_norm_accumulator lhs,
                                              dual_norm_accumulator rhs) const
  {
    lhs.fp32.add(rhs.fp32.sum);
    lhs.fp32.add(rhs.fp32.correction);
    lhs.tf32.add(rhs.tf32.sum);
    lhs.tf32.add(rhs.tf32.correction);
    return lhs;
  }
};

template <typename IndexT, int ThreadsPerBlock>
__global__ void fp32_tf32_norms_kernel(const float* input,
                                       float* fp32_norms,
                                       float* tf32_norms,
                                       IndexT rows,
                                       IndexT cols,
                                       bool take_sqrt)
{
  constexpr int warp_size       = 32;
  constexpr int warps_per_block = ThreadsPerBlock / warp_size;
  const int lane                = threadIdx.x % warp_size;
  const int warp                = threadIdx.x / warp_size;
  const auto row_stride = static_cast<IndexT>(gridDim.x) * static_cast<IndexT>(warps_per_block);

  for (IndexT row = static_cast<IndexT>(blockIdx.x) * static_cast<IndexT>(warps_per_block) +
                    static_cast<IndexT>(warp);
       row < rows;
       row += row_stride) {
    dual_norm_accumulator thread_sum{};
    const auto row_offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);
    for (IndexT col = static_cast<IndexT>(lane); col < cols;
         col += static_cast<IndexT>(warp_size)) {
      const float value = input[row_offset + static_cast<std::size_t>(col)];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
      const float tf32_value = nvcuda::wmma::__float_to_tf32(value);
#else
      const float tf32_value = value;
#endif
      thread_sum.fp32.add(value * value);
      thread_sum.tf32.add(tf32_value * tf32_value);
    }

    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
      dual_norm_accumulator other{
        {__shfl_down_sync(0xffffffff, thread_sum.fp32.sum, offset),
         __shfl_down_sync(0xffffffff, thread_sum.fp32.correction, offset)},
        {__shfl_down_sync(0xffffffff, thread_sum.tf32.sum, offset),
         __shfl_down_sync(0xffffffff, thread_sum.tf32.correction, offset)}};
      thread_sum = add_dual_norm_accumulators{}(thread_sum, other);
    }
    if (lane == 0) {
      const float fp32_norm = thread_sum.fp32.value();
      const float tf32_norm = thread_sum.tf32.value();
      fp32_norms[row]       = take_sqrt ? sqrtf(fp32_norm) : fp32_norm;
      tf32_norms[row]       = take_sqrt ? sqrtf(tf32_norm) : tf32_norm;
    }
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
  auto input  = raft::make_device_matrix_view<const float, IndexT>(matrix, rows, cols);
  auto output = raft::make_device_vector_view<float, IndexT>(norms, rows);
  if (take_sqrt) {
    raft::linalg::coalesced_reduction(
      handle, input, output, 0.0f, false, tf32_square_op{}, raft::add_op{}, raft::sqrt_op{});
  } else {
    raft::linalg::coalesced_reduction(
      handle, input, output, 0.0f, false, tf32_square_op{}, raft::add_op{}, raft::identity_op{});
  }
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

  constexpr int threads_per_block = 256;
  constexpr int warps_per_block   = threads_per_block / 32;
  const auto blocks               = static_cast<unsigned int>(std::min<std::size_t>(
    raft::ceildiv<std::size_t>(static_cast<std::size_t>(input.extent(0)), warps_per_block), 65535));
  fp32_tf32_norms_kernel<IndexT, threads_per_block>
    <<<blocks, threads_per_block, 0, raft::resource::get_cuda_stream(handle).get()>>>(
      input.data_handle(),
      fp32_norms.data_handle(),
      tf32_norms.data_handle(),
      input.extent(0),
      input.extent(1),
      take_sqrt);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
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
  cuvs::distance::detail::Top1nnBackend backend)
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
    plan = cuvs::distance::probe_top_1_nn(handle,
                                          X.data_handle(),
                                          centroids.data_handle(),
                                          n_samples,
                                          n_clusters,
                                          n_features,
                                          tuning,
                                          metric,
                                          backend);
    RAFT_EXPECTS(plan.available, "Requested top_1_nn backend is unavailable for KMeans assignment");

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
          handle, centroids, centroids_norm, raft::sqrt_op{});
      } else {
        raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
          handle, centroids, centroids_norm);
      }
      y_norm = L2NormBuf_OR_DistBuf.data();
    }

    auto result = make_native_result<DataT, IndexT>(plan, n_samples, output_storage, stream);

    bool init_out_buffer = true;
    if (plan.output_layout == cuvs::distance::detail::Top1nnOutputLayout::KeyValuePair) {
      auto output = raft::make_device_vector_view<raft::KeyValuePair<IndexT, DataT>, IndexT>(
        result.key_values(), n_samples);
      raft::matrix::fill(
        handle, output, raft::KeyValuePair<IndexT, DataT>{0, std::numeric_limits<DataT>::max()});
      init_out_buffer = false;
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
    cuvs::distance::detail::Top1nnBackend);

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
  cuvs::distance::detail::Top1nnBackend backend)
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
    const auto plan = cuvs::distance::probe_top_1_nn(handle,
                                                     X.data_handle(),
                                                     centroids.data_handle(),
                                                     n_samples,
                                                     n_clusters,
                                                     n_features,
                                                     tuning,
                                                     metric,
                                                     backend,
                                                     false);
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

    std::size_t backend_workspace_bytes = plan.workspace_bytes;
    void* backend_workspace             = reserve_aligned_workspace(
      workspace, 0, backend_workspace_bytes, plan.workspace_alignment, stream);

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
                                              backend_workspace_bytes,
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
    cuvs::distance::detail::Top1nnBackend backend);

INSTANTIATE_MIN_CLUSTER_DISTANCE(float, int64_t)
INSTANTIATE_MIN_CLUSTER_DISTANCE(double, int64_t)
INSTANTIATE_MIN_CLUSTER_DISTANCE(float, int)
INSTANTIATE_MIN_CLUSTER_DISTANCE(double, int)

#undef INSTANTIATE_MIN_CLUSTER_DISTANCE

}  // namespace cuvs::cluster::kmeans::detail
