/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../cluster/kmeans_balanced.cuh"
#include "../../core/nvtx.hpp"
#include "../ivf_common.cuh"
#include "../ivf_flat/ivf_flat_build.cuh"
#include "../ivf_list.cuh"
#include "ivf_binary_index_access.hpp"

#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/neighbors/ivf_binary.hpp>
#include <cuvs/preprocessing/quantize/binary.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/map.cuh>
#include <raft/stats/histogram.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/integer_utils.hpp>

#include <algorithm>
#include <optional>
#include <vector>

namespace cuvs::neighbors::ivf_binary::detail {

using namespace cuvs::spatial::knn::detail;  // NOLINT

// One thread per input byte; list sizes bound every per-bit count by uint32_t.
RAFT_KERNEL accumulate_center_counts(
  const uint8_t* data, const uint32_t* labels, uint32_t* counts, uint64_t n_elements, uint32_t dim)
{
  const uint64_t offset = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (offset >= n_elements) { return; }
  const auto row      = offset / dim;
  const auto col      = offset % dim;
  auto* output        = counts + (uint64_t(labels[row]) * dim + col) * 8;
  const uint8_t value = data[offset];
  for (int bit = 0; bit < 8; ++bit) {
    if ((value >> bit) & 1) { atomicAdd(output + bit, 1u); }
  }
}

/** The rows of the codes must be as long as the dimension of the quantizer requires. */
inline void check_dataset(device_binary_dataset_view<float, int64_t> dataset)
{
  RAFT_EXPECTS(dataset.dim() > 0, "The dimension of the dataset must be positive.");
  RAFT_EXPECTS(
    dataset.data_const_view().extent(1) == raft::div_rounding_up_safe<int64_t>(dataset.dim(), 8),
    "The row length of the codes must be ceil(dim / 8) bytes.");
}

/** See cuvs::neighbors::ivf_binary::extend docs */
inline void extend(raft::resources const& handle,
                   device_binary_dataset_view<float, int64_t> dataset,
                   std::optional<raft::device_vector_view<const int64_t, int64_t>> indices,
                   index<int64_t>* idx)
{
  using LabelT = uint32_t;
  RAFT_EXPECTS(idx != nullptr, "index cannot be empty.");
  check_dataset(dataset);
  RAFT_EXPECTS(dataset.dim() == idx->dim(),
               "The dimension of the vectors must equal the dimension of the index.");
  const int64_t n_rows = dataset.n_rows();
  if (indices.has_value()) {
    RAFT_EXPECTS(n_rows == indices->extent(0),
                 "new_vectors and new_indices have different number of rows");
  }
  const uint8_t* new_vectors = dataset.data_const_view().data_handle();
  const int64_t* new_indices = indices.has_value() ? indices->data_handle() : nullptr;
  RAFT_EXPECTS(new_indices != nullptr || idx->size() == 0,
               "You must pass data indices when the index is non-empty.");
  if (n_rows == 0) { return; }

  auto stream  = raft::resource::get_cuda_stream(handle);
  auto& lists  = index_access::lists(*idx);
  auto n_lists = idx->n_lists();
  auto dim     = idx->storage_dim();
  cuvs::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "ivf_binary::extend(%zu, %u)", size_t(n_rows), dim);

  auto& centers = index_access::centers(*idx);
  auto labels =
    raft::make_device_mdarray<LabelT>(handle,
                                      raft::resource::get_large_workspace_resource_ref(handle),
                                      raft::make_extents<int64_t>(n_rows));
  cuvs::cluster::kmeans::detail::predict_bitwise_hamming(
    handle,
    raft::make_device_matrix_view<const uint8_t, int64_t>(new_vectors, n_rows, dim),
    raft::make_device_matrix_view<const uint8_t, int64_t>(centers.data_handle(), n_lists, dim),
    raft::make_device_vector_view<LabelT, int64_t>(labels.data_handle(), n_rows));

  auto* list_sizes_ptr = lists.list_sizes().data_handle();
  auto old_list_sizes_dev =
    raft::make_device_mdarray<uint32_t>(handle,
                                        raft::resource::get_workspace_resource_ref(handle),
                                        raft::make_extents<int64_t>(n_lists));
  raft::copy(handle,
             old_list_sizes_dev.view(),
             raft::make_device_vector_view<const uint32_t, int64_t>(list_sizes_ptr, n_lists));

  raft::stats::histogram<uint32_t, int64_t>(raft::stats::HistTypeAuto,
                                            reinterpret_cast<int32_t*>(list_sizes_ptr),
                                            int64_t(n_lists),
                                            labels.data_handle(),
                                            n_rows,
                                            1,
                                            stream.get());
  raft::linalg::add(handle,
                    raft::make_device_vector_view<const uint32_t, int64_t>(list_sizes_ptr, n_lists),
                    raft::make_device_vector_view<const uint32_t, int64_t>(
                      old_list_sizes_dev.data_handle(), n_lists),
                    raft::make_device_vector_view<uint32_t, int64_t>(list_sizes_ptr, n_lists));

  // Adaptive centers are the majority bits of everything added so far. The exact per-bit counts are
  // kept, because rounding a previous mean back to a sum could flip an exact majority tie.
  if (idx->adaptive_centers()) {
    auto& counts          = index_access::center_counts(*idx);
    const auto n_elements = uint64_t(n_rows) * dim;
    accumulate_center_counts<<<raft::ceildiv(n_elements, uint64_t{256}), 256, 0, stream.get()>>>(
      new_vectors, labels.data_handle(), counts.data_handle(), n_elements, dim);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    const auto* counts_ptr  = counts.data_handle();
    const auto* centers_ptr = centers.data_handle();
    raft::linalg::map_offset(
      handle,
      centers.view(),
      [counts_ptr, centers_ptr, list_sizes_ptr, dim] __device__(int64_t offset) {
        const auto list_size = list_sizes_ptr[offset / dim];
        if (list_size == 0) { return centers_ptr[offset]; }  // keep the trained center
        uint8_t packed = 0;
        for (int bit = 0; bit < 8; ++bit) {
          if (uint64_t(counts_ptr[offset * 8 + bit]) * 2 > list_size) {
            packed |= uint8_t(1u << bit);
          }
        }
        return packed;
      });
  }

  // Calculate and allocate new list data
  ivf_flat::list_spec<uint32_t, uint8_t, int64_t> list_device_spec{
    dim, idx->conservative_memory_allocation()};
  std::vector<uint32_t> new_list_sizes(n_lists);
  std::vector<uint32_t> old_list_sizes(n_lists);
  raft::copy(handle,
             raft::make_host_vector_view(old_list_sizes.data(), n_lists),
             raft::make_device_vector_view(old_list_sizes_dev.data_handle(), n_lists));
  raft::copy(handle,
             raft::make_host_vector_view(new_list_sizes.data(), n_lists),
             raft::make_device_vector_view(list_sizes_ptr, n_lists));
  raft::resource::sync_stream(handle);
  for (uint32_t label = 0; label < n_lists; label++) {
    ivf::resize_list(
      handle, lists.lists()[label], list_device_spec, new_list_sizes[label], old_list_sizes[label]);
  }
  ivf::detail::recompute_internal_state(handle, lists);
  // Restart from the old sizes: the kernel below uses them as the atomic counters of the lists.
  raft::copy(
    handle,
    raft::make_device_vector_view(list_sizes_ptr, n_lists),
    raft::make_device_vector_view<const uint32_t>(old_list_sizes_dev.data_handle(), n_lists));

  const dim3 block_dim(256);
  const dim3 grid_dim(raft::ceildiv<int64_t>(n_rows, block_dim.x));
  ivf_flat::detail::build_index_kernel<uint8_t, int64_t, LabelT>
    <<<grid_dim, block_dim, 0, stream.get()>>>(labels.data_handle(),
                                               new_vectors,
                                               new_indices,
                                               lists.data_ptrs().data_handle(),
                                               lists.inds_ptrs().data_handle(),
                                               list_sizes_ptr,
                                               n_rows,
                                               dim,
                                               lists.veclen());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/** See cuvs::neighbors::ivf_binary::build docs */
inline auto build(raft::resources const& handle,
                  const index_params& params,
                  device_binary_dataset_view<float, int64_t> dataset) -> index<int64_t>
{
  check_dataset(dataset);
  auto stream          = raft::resource::get_cuda_stream(handle);
  const int64_t n_rows = dataset.n_rows();
  const auto codes     = dataset.data_const_view();
  const auto manifest  = dataset.const_manifest_view();
  cuvs::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "ivf_binary::build(%zu, %u)", size_t(n_rows), dataset.dim());

  RAFT_EXPECTS(params.n_lists > 0, "n_lists must be positive");
  RAFT_EXPECTS(n_rows > 0, "empty dataset");
  RAFT_EXPECTS(n_rows >= params.n_lists, "number of rows can't be less than n_lists");

  index<int64_t> idx(handle, params, dataset.dim());
  const auto dim = idx.storage_dim();
  auto& lists    = index_access::lists(idx);

  // The index starts with the zero-threshold quantizer of the dataset dimension
  if (const auto* threshold = std::get_if<1>(&manifest.value)) {
    auto thresholds = raft::make_device_vector<float, int64_t>(handle, threshold->extent(0));
    raft::copy(handle, thresholds.view(), *threshold);
    index_access::quantizer(idx) =
      cuvs::preprocessing::quantize::binary::quantizer<float>(std::move(thresholds));
  }

  utils::memzero(
    lists.accum_sorted_sizes().data_handle(), lists.accum_sorted_sizes().size(), stream);
  utils::memzero(lists.list_sizes().data_handle(), lists.list_sizes().size(), stream);
  utils::memzero(lists.data_ptrs().data_handle(), lists.data_ptrs().size(), stream);
  utils::memzero(lists.inds_ptrs().data_handle(), lists.inds_ptrs().size(), stream);

  // Train the k-means clustering on the expanded bits and pack the centers again
  {
    auto trainset_ratio = std::max<size_t>(
      1, n_rows / std::max<size_t>(params.kmeans_trainset_fraction * n_rows, params.n_lists));
    auto n_rows_train = n_rows / trainset_ratio;

    rmm::device_uvector<uint8_t> trainset(
      n_rows_train * dim, stream, raft::resource::get_large_workspace_resource_ref(handle));
    // TODO: a proper sampling
    raft::copy_matrix(
      trainset.data(), dim, codes.data_handle(), dim * trainset_ratio, dim, n_rows_train, stream);

    cuvs::cluster::kmeans::balanced_params kmeans_params;
    kmeans_params.n_iters          = params.kmeans_n_iters;
    kmeans_params.metric           = cuvs::distance::DistanceType::L2Expanded;
    kmeans_params.is_packed_binary = true;

    rmm::device_uvector<float> decoded_centers(
      size_t(params.n_lists) * dim * 8, stream, raft::resource::get_workspace_resource_ref(handle));
    auto decoded_centers_view = raft::make_device_matrix_view<float, int64_t>(
      decoded_centers.data(), params.n_lists, int64_t(dim) * 8);
    cuvs::cluster::kmeans_balanced::fit(
      handle,
      kmeans_params,
      raft::make_device_matrix_view<const uint8_t, int64_t>(trainset.data(), n_rows_train, dim),
      decoded_centers_view,
      raft::identity_op{});

    cuvs::preprocessing::quantize::binary::quantizer<float> zero_threshold(idx.dim());
    cuvs::preprocessing::quantize::binary::transform(handle,
                                                     zero_threshold,
                                                     raft::make_const_mdspan(decoded_centers_view),
                                                     index_access::centers(idx).view());
  }

  if (params.add_data_on_build) { detail::extend(handle, dataset, std::nullopt, &idx); }
  return idx;
}

}  // namespace cuvs::neighbors::ivf_binary::detail
