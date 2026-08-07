/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuvs/neighbors/cagra.hpp>

#include "cagra_build.cuh"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/copy.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/refine.hpp>

#include <rmm/resource_ref.hpp>

#include <vector>

namespace cuvs::neighbors::cagra::detail {

template <class T, class IdxT, cuvs::neighbors::ann_dataset_view DatasetViewT>
int64_t merged_dataset_size(
  raft::resources const& handle,
  std::vector<cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>*> const& indices,
  cuvs::neighbors::filtering::base_filter const& row_filter)
{
  int64_t merged_rows = 0;
  for (auto* index : indices) {
    RAFT_EXPECTS(index != nullptr,
                 "Null pointer detected in 'indices'. Ensure all elements are valid before usage.");
    merged_rows += static_cast<int64_t>(index->size());
  }
  if (row_filter.get_filter_type() == cuvs::neighbors::filtering::FilterType::Bitset) {
    auto const& actual_filter =
      dynamic_cast<const cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>&>(row_filter);
    return actual_filter.view().count(handle);
  }
  RAFT_EXPECTS(row_filter.get_filter_type() == cuvs::neighbors::filtering::FilterType::None,
               "Only none and bitset filters are supported inside cagra::merge");
  return merged_rows;
}

template <class T, class IdxT, cuvs::neighbors::ann_dataset_view DatasetViewT>
cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT> merge(
  raft::resources const& handle,
  const cagra::index_params& params,
  std::vector<cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>*>& indices,
  DatasetViewT merged_dataset,
  const cuvs::neighbors::filtering::base_filter& row_filter)
{
  using cagra_index_t = cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>;

  int64_t merged_rows = 0;
  uint32_t dim        = 0;
  int64_t stride      = -1;

  RAFT_EXPECTS(row_filter.get_filter_type() != cuvs::neighbors::filtering::FilterType::Bitmap,
               "Bitmap filter isn't supported inside cagra::merge");
  RAFT_EXPECTS(row_filter.get_filter_type() != cuvs::neighbors::filtering::FilterType::Bloom,
               "Bloom filter isn't supported inside cagra::merge");

  for (cagra_index_t* index : indices) {
    RAFT_EXPECTS(index != nullptr,
                 "Null pointer detected in 'indices'. Ensure all elements are valid before usage.");
    auto const& dataset = index->dataset();
    if constexpr (cuvs::neighbors::is_dense_row_major_dataset_view_v<
                    std::decay_t<decltype(dataset)>>) {
      RAFT_EXPECTS(
        dataset.n_rows() != 0,
        "cagra::merge only supports an index to which the dataset is attached. Please check if "
        "the index has an empty dataset; attach one with update_device_dataset_same_layout "
        "before merge.");
      if (dim == 0) {
        dim    = index->dim();
        stride = static_cast<int64_t>(dataset.stride());
      } else {
        RAFT_EXPECTS(dim == index->dim(), "Dimension of datasets in indices must be equal.");
        RAFT_EXPECTS(stride == static_cast<int64_t>(dataset.stride()),
                     "Row stride of datasets in indices must be equal.");
      }
      merged_rows += static_cast<int64_t>(index->size());
    } else {
      RAFT_FAIL("cagra::merge only supports an uncompressed dense device dataset index");
    }
  }

  bool const bitset_filtered =
    row_filter.get_filter_type() == cuvs::neighbors::filtering::FilterType::Bitset;
  int64_t const final_rows =
    merged_dataset_size<T, IdxT, DatasetViewT>(handle, indices, row_filter);

  RAFT_EXPECTS(merged_dataset.n_rows() == final_rows,
               "merged_dataset rows (%ld) must equal the final merged row count (%ld)",
               long(merged_dataset.n_rows()),
               long(final_rows));
  RAFT_EXPECTS(merged_dataset.dim() == dim,
               "merged_dataset dimension (%u) must equal the input dimension (%u)",
               unsigned(merged_dataset.dim()),
               unsigned(dim));
  RAFT_EXPECTS(merged_dataset.stride() == stride,
               "merged_dataset stride (%u) must equal the input stride (%ld)",
               unsigned(merged_dataset.stride()),
               long(stride));

  auto output_const_view = merged_dataset.view();
  auto output_view       = raft::make_device_matrix_view<T, int64_t>(
    const_cast<T*>(output_const_view.data_handle()), final_rows, stride);

  auto merge_dataset = [&](T* dst, std::size_t dst_ld) {
    IdxT row_offset = 0;
    for (cagra_index_t* index : indices) {
      const T* src_ptr   = nullptr;
      std::size_t n_rows = 0;
      auto const& v      = index->dataset();
      if constexpr (cuvs::neighbors::is_dense_row_major_dataset_view_v<std::decay_t<decltype(v)>>) {
        src_ptr = v.view().data_handle();
        n_rows  = static_cast<std::size_t>(v.n_rows());
      } else {
        RAFT_FAIL("cagra::merge: unexpected dataset type while copying rows");
      }
      raft::copy_matrix(dst + static_cast<std::size_t>(row_offset) * dst_ld,
                        dst_ld,
                        src_ptr,
                        static_cast<std::size_t>(stride),
                        static_cast<std::size_t>(dim),
                        n_rows,
                        raft::resource::get_cuda_stream(handle));

      row_offset += IdxT(index->dataset().n_rows());
    }
  };

  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  if (bitset_filtered) {
    auto staging = raft::make_device_matrix<T, int64_t>(handle, merged_rows, stride);
    RAFT_CUDA_TRY(cudaMemsetAsync(
      staging.data_handle(), 0, static_cast<std::size_t>(staging.size()) * sizeof(T), stream));
    merge_dataset(staging.data_handle(), static_cast<std::size_t>(stride));

    auto actual_filter =
      dynamic_cast<const cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>&>(row_filter);

    auto indices_csr = raft::make_device_csr_matrix<uint32_t, int64_t, int64_t, int64_t>(
      handle, 1, static_cast<std::size_t>(merged_rows));
    indices_csr.initialize_sparsity(final_rows);

    actual_filter.view().to_csr(handle, indices_csr);

    auto csr_indices  = indices_csr.structure_view().get_indices();
    auto indices_view = raft::make_device_vector_view<const int64_t, int64_t>(
      csr_indices.data(), static_cast<int64_t>(csr_indices.size()));

    RAFT_CUDA_TRY(cudaMemsetAsync(
      output_view.data_handle(),
      0,
      static_cast<std::size_t>(final_rows) * static_cast<std::size_t>(stride) * sizeof(T),
      stream));

    raft::matrix::copy_rows(
      handle, raft::make_const_mdspan(staging.view()), output_view, indices_view);

    auto index = ::cuvs::neighbors::cagra::detail::build_from_device_matrix<T, IdxT, DatasetViewT>(
      handle, params, merged_dataset);
    index.update_device_dataset_same_layout(handle, merged_dataset);
    RAFT_LOG_DEBUG("cagra merge: using device memory for merged dataset");
    return index;
  }

  RAFT_CUDA_TRY(cudaMemsetAsync(
    output_view.data_handle(),
    0,
    static_cast<std::size_t>(final_rows) * static_cast<std::size_t>(stride) * sizeof(T),
    stream));
  merge_dataset(output_view.data_handle(), static_cast<std::size_t>(stride));
  auto index = ::cuvs::neighbors::cagra::detail::build_from_device_matrix<T, IdxT, DatasetViewT>(
    handle, params, merged_dataset);
  index.update_device_dataset_same_layout(handle, merged_dataset);
  RAFT_LOG_DEBUG("cagra merge: using device memory for merged dataset");
  return index;
}

}  // namespace cuvs::neighbors::cagra::detail
