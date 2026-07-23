/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ivf_rabitq/gpu_index/ivf_gpu.cuh"
#include "ivf_rabitq/gpu_index/searcher_gpu.cuh"
#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/neighbors/ivf_rabitq.hpp>

#include "../core/nvtx.hpp"
#include "detail/ann_utils.cuh"

#include <raft/core/operators.hpp>
#include <raft/linalg/map.cuh>
#include <raft/linalg/reduce.cuh>
#include <raft/matrix/sample_rows.cuh>
#include <raft/util/cudart_utils.hpp>

namespace cuvs::neighbors::ivf_rabitq {

namespace detail {

using namespace cuvs::spatial::knn::detail;  // NOLINT

template <typename T, typename IdxT, typename accessor>
auto build(raft::resources const& handle,
           const index_params& params,
           raft::mdspan<const T, raft::matrix_extent<IdxT>, raft::row_major, accessor> dataset)
  -> cuvs::neighbors::ivf_rabitq::index<IdxT>
{
  IdxT n_rows = dataset.extent(0);
  IdxT dim    = dataset.extent(1);
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "ivf_rabitq::build(%zu, %u)", size_t(n_rows), dim);
  static_assert(std::is_same_v<T, float>, "Unsupported data type");

  RAFT_EXPECTS(n_rows > 0 && dim > 0, "empty dataset");
  RAFT_EXPECTS(n_rows >= params.n_lists, "number of rows can't be less than n_lists");
  RAFT_EXPECTS(params.metric == cuvs::distance::DistanceType::L2Expanded ||
                 params.metric == cuvs::distance::DistanceType::InnerProduct,
               "ivf_rabitq only supports L2Expanded and InnerProduct metrics");

  // Calculate dataset size and available workspace once
  size_t dataset_bytes             = sizeof(T) * n_rows * dim;
  size_t available_workspace       = raft::resource::get_workspace_free_bytes(handle);
  constexpr size_t kTolerableRatio = 4;

  rmm::device_async_resource_ref device_memory = raft::resource::get_workspace_resource_ref(handle);
  // If the dataset is small enough to comfortably fit into device memory, put it there.
  // Otherwise, use the managed memory.
  rmm::device_async_resource_ref big_memory_resource =
    raft::resource::get_large_workspace_resource_ref(handle);
  if (dataset_bytes * kTolerableRatio < available_workspace) {
    big_memory_resource = device_memory;
  }

  auto stream = raft::resource::get_cuda_stream(handle);

  // Determine if we should use streaming construction
  bool use_streaming            = false;
  const float* host_dataset_ptr = nullptr;

  // Check if dataset is already on device
  auto dataset_residency = utils::check_pointer_residency(dataset.data_handle());
  bool dataset_on_device = (dataset_residency == utils::pointer_residency::device_only);

  // If dataset is on host, check if we should use streaming construction
  if (!dataset_on_device) {
    // Use streaming if explicitly requested or if dataset doesn't fit comfortably
    if (params.force_streaming || dataset_bytes * kTolerableRatio >= available_workspace) {
      use_streaming    = true;
      host_dataset_ptr = dataset.data_handle();
      if (params.force_streaming) {
        RAFT_LOG_INFO(
          "Using streaming construction: explicitly requested via force_streaming parameter");
      } else {
        RAFT_LOG_INFO(
          "Using streaming construction: dataset size (%.2f GB) exceeds comfortable GPU memory "
          "limit",
          dataset_bytes / (1024.0 * 1024.0 * 1024.0));
      }
    }
  }

  // create device view of dataset (only if not using streaming)
  auto d_dataset_array =
    raft::make_device_mdarray<T>(handle, big_memory_resource, raft::make_extents<int64_t>(0, 0));
  auto d_dataset_view =
    raft::make_mdspan(dataset.data_handle(), raft::make_extents<int64_t>(n_rows, dim));

  if (!use_streaming && !dataset_on_device) {
    try {
      d_dataset_array = raft::make_device_mdarray<T>(
        handle, big_memory_resource, raft::make_extents<int64_t>(n_rows, dim));
    } catch (raft::logic_error& e) {
      RAFT_LOG_ERROR(
        "Insufficient memory for full GPU construction. Please decrease "
        "dataset size, or set large_workspace_resource appropriately.");
      throw;
    }
    raft::copy(d_dataset_array.data_handle(), dataset.data_handle(), n_rows * dim, stream);
    d_dataset_view = d_dataset_array.view();
  }

  auto dataset_const_view = raft::make_const_mdspan(d_dataset_view);
  rmm::device_uvector<float> cluster_centers(params.n_lists * dim, stream, big_memory_resource);
  rmm::device_uvector<uint32_t> labels(n_rows, stream, big_memory_resource);

  // Scope for kmeans training set allocation.
  {
    raft::random::RngState random_state{137};
    size_t n_rows_train =
      std::min(static_cast<size_t>(n_rows),
               static_cast<size_t>(params.max_train_points_per_cluster) * params.n_lists);

    // Besides just sampling, we transform the input dataset into floats to make it easier
    // to use gemm operations from cublas.
    auto trainset =
      raft::make_device_mdarray<T>(handle, big_memory_resource, raft::make_extents<int64_t>(0, 0));
    try {
      trainset = raft::make_device_mdarray<T>(
        handle, big_memory_resource, raft::make_extents<int64_t>(n_rows_train, dim));
    } catch (raft::logic_error& e) {
      RAFT_LOG_ERROR(
        "Insufficient memory for kmeans training set allocation. Please decrease "
        "max_train_points_per_cluster, or set large_workspace_resource appropriately.");
      throw;
    }
    raft::matrix::sample_rows<T, int64_t>(handle, random_state, dataset, trainset.view());

    // perform k-means clustering
    // NB: here cluster_centers is used as if it is [n_clusters, data_dim] not [n_clusters,
    // dim_ext]!
    auto centers_view =
      raft::make_device_matrix_view<float, int64_t>(cluster_centers.data(), params.n_lists, dim);
    cuvs::cluster::kmeans::balanced_params kmeans_params;
    kmeans_params.n_iters = params.kmeans_n_iters;
    kmeans_params.metric  = cuvs::distance::DistanceType::L2Expanded;
    // find cluster labels for dataset vectors
    auto labels_view = raft::make_device_vector_view<uint32_t, int64_t>(labels.data(), n_rows);
    auto centers_const_view = raft::make_device_matrix_view<const float, int64_t>(
      cluster_centers.data(), params.n_lists, dim);
    cuvs::cluster::kmeans::fit(
      handle, kmeans_params, raft::make_const_mdspan(trainset.view()), centers_view);
    if (use_streaming) {
      // The full dataset lives on host (it doesn't fit comfortably in device memory, which is
      // why we stream). `predict` launches device kernels, so it can't read the host pointer
      // directly. Compute cluster labels in device-sized batches: copy each chunk to the GPU and
      // predict into the matching slice of `labels`.
      size_t batch_rows =
        std::min(static_cast<size_t>(n_rows), std::max<size_t>(1, params.streaming_batch_size));
      auto d_batch = raft::make_device_mdarray<T>(
        handle, big_memory_resource, raft::make_extents<int64_t>(batch_rows, dim));
      for (size_t row_start = 0; row_start < static_cast<size_t>(n_rows); row_start += batch_rows) {
        size_t rows = std::min(batch_rows, static_cast<size_t>(n_rows) - row_start);
        raft::copy(d_batch.data_handle(), host_dataset_ptr + row_start * dim, rows * dim, stream);
        auto batch_view =
          raft::make_device_matrix_view<const float, int64_t>(d_batch.data_handle(), rows, dim);
        auto batch_labels =
          raft::make_device_vector_view<uint32_t, int64_t>(labels.data() + row_start, rows);
        cuvs::cluster::kmeans::predict(
          handle, kmeans_params, batch_view, centers_const_view, batch_labels);
      }
    } else {
      cuvs::cluster::kmeans::predict(
        handle, kmeans_params, dataset_const_view, centers_const_view, labels_view);
    }
  }

  index<IdxT> index(handle, n_rows, dim, params.n_lists, params.bits_per_dim, params.metric);

  // Call RaBitQ index construct - use streaming if dataset doesn't fit in GPU memory
  if (use_streaming) {
    index.rabitq_index().construct_on_gpu_streaming(host_dataset_ptr,
                                                    cluster_centers.data(),
                                                    labels.data(),
                                                    params.fast_quantize_flag,
                                                    params.streaming_batch_size);
  } else {
    index.rabitq_index().construct_on_gpu(d_dataset_view.data_handle(),
                                          cluster_centers.data(),
                                          labels.data(),
                                          params.fast_quantize_flag);
  }

  return index;
}

template <typename T, typename IdxT>
void search(raft::resources const& handle,
            const search_params& params,
            index<IdxT>& idx,
            raft::device_matrix_view<const T, IdxT, raft::row_major> queries,
            raft::device_matrix_view<IdxT, IdxT, raft::row_major> neighbors,
            raft::device_matrix_view<float, IdxT, raft::row_major> distances)
{
  auto stream = raft::resource::get_cuda_stream(handle).value();

  size_t NQ = queries.extent(0);
  size_t k  = neighbors.extent(1);
  if (NQ == 0 || k == 0 || params.n_probes == 0) return;
  size_t dim = queries.extent(1);

  RAFT_EXPECTS(neighbors.extent(0) == static_cast<IdxT>(NQ),
               "neighbors rows must equal queries rows");
  RAFT_EXPECTS(distances.extent(0) == static_cast<IdxT>(NQ),
               "distances rows must equal queries rows");
  RAFT_EXPECTS(distances.extent(1) == static_cast<IdxT>(k),
               "distances cols must equal neighbors cols");
  RAFT_EXPECTS(dim == static_cast<size_t>(idx.dim()),
               "query dimensionality (%zu) must match index dimensionality (%u)",
               dim,
               idx.dim());
  RAFT_EXPECTS(params.n_probes <= idx.rabitq_index().get_num_centroids(),
               "n_probes (%u) cannot exceed number of IVF lists (%zu)",
               params.n_probes,
               idx.rabitq_index().get_num_centroids());
  // InnerProduct is currently implemented only on the bitwise (QUANT4/QUANT8) search paths; the
  // LUT paths would silently return squared-L2 results, so reject them explicitly.
  RAFT_EXPECTS(idx.rabitq_index().metric() != cuvs::distance::DistanceType::InnerProduct ||
                 params.mode == search_mode::QUANT4 || params.mode == search_mode::QUANT8,
               "ivf_rabitq: InnerProduct search currently supports only QUANT4 and QUANT8 modes");

  auto padded_dim      = idx.rabitq_index().get_num_padded_dim();
  auto rotated_queries = raft::make_device_matrix<T, int64_t>(handle, NQ, padded_dim);
  if (padded_dim == dim) {
    // TODO: replace RotatorGPU::rotate with cuVS/RAFT primitives
    idx.rabitq_index().rotator().rotate(
      handle, queries.data_handle(), rotated_queries.data_handle(), NQ);
  } else {
    auto padded_queries = raft::make_device_matrix<T, int64_t>(handle, NQ, padded_dim);
    RAFT_CUDA_TRY(
      cudaMemsetAsync(padded_queries.data_handle(), 0, sizeof(T) * NQ * padded_dim, stream));
    RAFT_CUDA_TRY(cudaMemcpy2DAsync(padded_queries.data_handle(),
                                    sizeof(T) * padded_dim,
                                    queries.data_handle(),
                                    sizeof(T) * dim,
                                    sizeof(T) * dim,
                                    NQ,
                                    cudaMemcpyDefault,
                                    stream));
    // TODO: replace RotatorGPU::rotate with cuVS/RAFT primitives
    idx.rabitq_index().rotator().rotate(
      handle, padded_queries.data_handle(), rotated_queries.data_handle(), NQ);
  }

  auto search_mode_to_string = [](search_mode mode) -> std::string {
    switch (mode) {
      case search_mode::LUT16: return std::string("lut16");
      case search_mode::LUT32: return std::string("lut32");
      case search_mode::QUANT4: return std::string("quant4");
      case search_mode::QUANT8: return std::string("quant8");
      default: RAFT_FAIL("Invalid search mode");
    }
  };
  detail::SearcherGPU searcher(handle,
                               rotated_queries.data_handle(),
                               padded_dim,
                               idx.rabitq_index().get_ex_bits(),
                               search_mode_to_string(params.mode),
                               idx.rabitq_index().quantizer().get_query_scaling_factor(),
                               /* rabitq_quantize_flag = */ true);
  searcher.AllocateSearcherSpace(idx.rabitq_index().get_num_centroids(), NQ);

  auto final_ids = raft::make_device_matrix<uint32_t, int64_t>(handle, NQ, k);
  auto queries_view =
    raft::make_device_matrix_view<const T, int64_t>(rotated_queries.data_handle(), NQ, padded_dim);

  if (params.mode == search_mode::LUT32) {
    idx.rabitq_index().BatchClusterSearch(
      queries_view, k, params.n_probes, searcher, NQ, distances, final_ids.view());
  } else if (params.mode == search_mode::LUT16) {
    // test v3 lut using fp16
    idx.rabitq_index().BatchClusterSearchLUT16(
      queries_view, k, params.n_probes, searcher, NQ, distances, final_ids.view());
  } else if (params.mode == search_mode::QUANT8) {
    idx.rabitq_index().BatchClusterSearchQuantizeQuery(
      queries_view, k, params.n_probes, searcher, NQ, distances, final_ids.view(), 8);
  } else if (params.mode == search_mode::QUANT4) {
    idx.rabitq_index().BatchClusterSearchQuantizeQuery(
      queries_view, k, params.n_probes, searcher, NQ, distances, final_ids.view(), 4);
  }

  // cast data in final_ids to array of IdxT in neighbors
  raft::linalg::map(
    handle,
    neighbors,
    raft::cast_op<IdxT>{},
    raft::make_device_vector_view<const uint32_t, IdxT>(final_ids.data_handle(), NQ * k));
}

template <typename IdxT>
void serialize(raft::resources const& handle, const std::string& filename, index<IdxT>& index)
{
  // Save the index to a file.
  index.rabitq_index().save(filename.c_str());
}

template <typename IdxT>
void deserialize(raft::resources const& handle,
                 const std::string& filename,
                 cuvs::neighbors::ivf_rabitq::index<IdxT>* index)
{
  index->rabitq_index().load_transposed(filename.c_str());
}

}  // namespace detail

template <typename IdxT>
index<IdxT>::index(raft::resources const& handle)
  // this constructor is just for a temporary index, for use in the deserialization
  // api. all the parameters here will get replaced with loaded values - that aren't
  // necessarily known ahead of time before deserialization.
  // TODO: do we even need a handle here - could just construct one?
  : rabitq_index_(std::make_unique<detail::IVFGPU>(handle))
{
}

template <typename IdxT>
index<IdxT>::index(raft::resources const& handle,
                   size_t n_rows,
                   uint32_t dim,
                   uint32_t n_lists,
                   uint32_t bits_per_dim,
                   cuvs::distance::DistanceType metric)
{
  RAFT_EXPECTS(bits_per_dim >= 1 && bits_per_dim <= 9, "Unsupported bits_per_dim");
  rabitq_index_ =
    std::make_unique<detail::IVFGPU>(handle, n_rows, dim, n_lists, bits_per_dim, metric);
}

template <typename IdxT>
index<IdxT>::index(index&&) = default;

template <typename IdxT>
auto index<IdxT>::operator=(index&&) -> index<IdxT>& = default;

template <typename IdxT>
index<IdxT>::~index() = default;

template struct index<int64_t>;

template <typename IdxT>
detail::IVFGPU& index<IdxT>::rabitq_index() noexcept
{
  return *rabitq_index_;
}

template <typename IdxT>
uint32_t index<IdxT>::dim() const noexcept
{
  return rabitq_index_->get_num_dimensions();
}

template <typename IdxT>
IdxT index<IdxT>::size() const noexcept
{
  return rabitq_index_->get_num_vectors();
}

template <typename IdxT>
cuvs::distance::DistanceType index<IdxT>::metric() const noexcept
{
  return rabitq_index_->metric();
}

auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_rabitq::index_params& index_params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_rabitq::index<int64_t>
{
  return cuvs::neighbors::ivf_rabitq::detail::build(handle, index_params, dataset);
}

auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_rabitq::index_params& index_params,
           raft::host_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_rabitq::index<int64_t>
{
  return cuvs::neighbors::ivf_rabitq::detail::build(handle, index_params, dataset);
}

void search(raft::resources const& handle,
            const cuvs::neighbors::ivf_rabitq::search_params& search_params,
            cuvs::neighbors::ivf_rabitq::index<int64_t>& index,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances)
{
  cuvs::neighbors::ivf_rabitq::detail::search(
    handle, search_params, index, queries, neighbors, distances);
}

void serialize(raft::resources const& handle,
               const std::string& filename,
               cuvs::neighbors::ivf_rabitq::index<int64_t>& index)
{
  cuvs::neighbors::ivf_rabitq::detail::serialize(handle, filename, index);
}

void deserialize(raft::resources const& handle,
                 const std::string& filename,
                 cuvs::neighbors::ivf_rabitq::index<int64_t>* index)
{
  if (!index) { RAFT_FAIL("Invalid index pointer"); }
  cuvs::neighbors::ivf_rabitq::detail::deserialize(handle, filename, index);
}

}  // namespace cuvs::neighbors::ivf_rabitq
