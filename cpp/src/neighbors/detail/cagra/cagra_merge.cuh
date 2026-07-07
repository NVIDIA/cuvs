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

template <typename T, cuvs::neighbors::cagra_dataset_view DatasetViewT>
auto make_merged_dense_device_dataset_view(
  raft::device_matrix_view<const T, int64_t, raft::row_major> storage, uint32_t dim) -> DatasetViewT
{
  if constexpr (cuvs::neighbors::is_padded_dataset_view_v<DatasetViewT>) {
    return cuvs::neighbors::device_padded_dataset_view<T, int64_t>(raft::make_const_mdspan(storage),
                                                                   dim);
  } else if constexpr (cuvs::neighbors::is_standard_dataset_view_v<DatasetViewT>) {
    return cuvs::neighbors::device_standard_dataset_view<T, int64_t>(
      raft::make_const_mdspan(storage), dim);
  } else {
    static_assert(sizeof(DatasetViewT) == 0,
                  "cagra::merge supports dense padded or standard dataset views only");
  }
}

template <class T, class IdxT, cuvs::neighbors::cagra_dataset_view DatasetViewT>
merged_dataset compute_merged_dataset_layout(
  raft::resources const& handle,
  std::vector<cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>*> const& indices,
  cuvs::neighbors::filtering::base_filter const& row_filter)
{
  using cagra_index_t = cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>;

  std::size_t dim              = 0;
  std::size_t new_dataset_size = 0;
  int64_t stride               = -1;

  RAFT_EXPECTS(row_filter.get_filter_type() != cuvs::neighbors::filtering::FilterType::Bitmap,
               "Bitmap filter isn't supported inside cagra::merge");

  for (cagra_index_t* index : indices) {
    RAFT_EXPECTS(index != nullptr,
                 "Null pointer detected in 'indices'. Ensure all elements are valid before usage.");
    auto const& v = index->dataset();
    if constexpr (cuvs::neighbors::is_dense_row_major_dataset_view_v<std::decay_t<decltype(v)>>) {
      if (v.n_rows() == 0) {
        RAFT_FAIL(
          "cagra::merge only supports an index to which the dataset is attached. Please check if "
          "the index has an empty dataset; attach one with update_dataset before merge.");
      }
      if (dim == 0) {
        dim    = index->dim();
        stride = static_cast<int64_t>(v.stride());
      } else {
        RAFT_EXPECTS(dim == index->dim(), "Dimension of datasets in indices must be equal.");
        RAFT_EXPECTS(stride == static_cast<int64_t>(v.stride()),
                     "Row stride of datasets in indices must be equal.");
      }
      new_dataset_size += index->size();
    } else {
      RAFT_FAIL("cagra::merge only supports an uncompressed dense device dataset index");
    }
  }

  merged_dataset layout{};
  layout.merged_rows     = static_cast<int64_t>(new_dataset_size);
  layout.stride_elements = stride;
  layout.dim             = static_cast<uint32_t>(dim);
  layout.bitset_filtered =
    (row_filter.get_filter_type() == cuvs::neighbors::filtering::FilterType::Bitset);
  if (layout.bitset_filtered) {
    auto const& actual_filter =
      dynamic_cast<const cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>&>(row_filter);
    layout.filtered_rows = actual_filter.view().count(handle);
  } else {
    layout.filtered_rows = layout.merged_rows;
  }
  return layout;
}

template <class T, class IdxT, cuvs::neighbors::cagra_dataset_view DatasetViewT>
cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT> merge(
  raft::resources const& handle,
  const cagra::index_params& params,
  std::vector<cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>*>& indices,
  merged_dataset_storage<T, IdxT>& storage,
  const cuvs::neighbors::filtering::base_filter& row_filter)
{
  using cagra_index_t = cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>;

  auto const expected =
    compute_merged_dataset_layout<T, IdxT, DatasetViewT>(handle, indices, row_filter);
  RAFT_EXPECTS(expected.merged_rows == storage.layout.merged_rows &&
                 expected.filtered_rows == storage.layout.filtered_rows &&
                 expected.stride_elements == storage.layout.stride_elements &&
                 expected.dim == storage.layout.dim &&
                 expected.bitset_filtered == storage.layout.bitset_filtered,
               "merged_dataset_storage.layout does not match indices and row_filter (use the same "
               "arguments as "
               "make_merged_dataset).");

  auto merged_storage = storage.merged_storage.view();
  RAFT_EXPECTS(merged_storage.extent(0) == storage.layout.merged_rows,
               "merged_storage rows (%ld) must equal layout.merged_rows (%ld)",
               long(merged_storage.extent(0)),
               long(storage.layout.merged_rows));
  RAFT_EXPECTS(merged_storage.extent(1) == storage.layout.stride_elements,
               "merged_storage stride (%ld) must equal layout.stride_elements (%ld)",
               long(merged_storage.extent(1)),
               long(storage.layout.stride_elements));

  std::optional<raft::device_matrix_view<T, int64_t, raft::row_major>> filtered_view{};
  if (storage.layout.bitset_filtered) {
    RAFT_EXPECTS(storage.filtered_storage.has_value(),
                 "Bitset-filtered merge requires merged_dataset_storage.filtered_storage.");
    filtered_view = storage.filtered_storage->view();
    RAFT_EXPECTS(filtered_view->extent(0) == storage.layout.filtered_rows,
                 "filtered_storage rows (%ld) must equal layout.filtered_rows (%ld)",
                 long(filtered_view->extent(0)),
                 long(storage.layout.filtered_rows));
    RAFT_EXPECTS(filtered_view->extent(1) == storage.layout.stride_elements,
                 "filtered_storage stride (%ld) must equal layout.stride_elements (%ld)",
                 long(filtered_view->extent(1)),
                 long(storage.layout.stride_elements));
  } else {
    RAFT_EXPECTS(!storage.filtered_storage.has_value(),
                 "Non-bitset merge requires merged_dataset_storage.filtered_storage be unset.");
  }

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
                        static_cast<std::size_t>(storage.layout.stride_elements),
                        static_cast<std::size_t>(storage.layout.dim),
                        n_rows,
                        raft::resource::get_cuda_stream(handle));

      row_offset += IdxT(index->dataset().n_rows());
    }
  };

  cudaStream_t stream     = raft::resource::get_cuda_stream(handle);
  const auto merged_bytes = static_cast<std::size_t>(merged_storage.size()) * sizeof(T);
  RAFT_CUDA_TRY(cudaMemsetAsync(merged_storage.data_handle(), 0, merged_bytes, stream));

  merge_dataset(merged_storage.data_handle(),
                static_cast<std::size_t>(storage.layout.stride_elements));

  if (storage.layout.bitset_filtered) {
    auto actual_filter =
      dynamic_cast<const cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>&>(row_filter);

    auto indices_csr = raft::make_device_csr_matrix<uint32_t, int64_t, int64_t, int64_t>(
      handle, 1, static_cast<std::size_t>(storage.layout.merged_rows));
    indices_csr.initialize_sparsity(storage.layout.filtered_rows);

    actual_filter.view().to_csr(handle, indices_csr);

    auto csr_indices  = indices_csr.structure_view().get_indices();
    auto indices_view = raft::make_device_vector_view<const int64_t, int64_t>(
      csr_indices.data(), static_cast<int64_t>(csr_indices.size()));

    auto& filtered_storage = *filtered_view;
    RAFT_CUDA_TRY(cudaMemsetAsync(filtered_storage.data_handle(),
                                  0,
                                  static_cast<std::size_t>(filtered_storage.size()) * sizeof(T),
                                  stream));

    raft::matrix::copy_rows(
      handle, raft::make_const_mdspan(merged_storage), filtered_storage, indices_view);

    auto dv = make_merged_dense_device_dataset_view<T, DatasetViewT>(
      raft::make_const_mdspan(filtered_storage), storage.layout.dim);
    auto index = ::cuvs::neighbors::cagra::detail::build_from_device_matrix<T, IdxT, DatasetViewT>(
      handle, params, dv);
    index.update_dataset(handle, dv);
    RAFT_LOG_DEBUG("cagra merge: using device memory for merged dataset");
    return index;
  }

  auto dv = make_merged_dense_device_dataset_view<T, DatasetViewT>(
    raft::make_const_mdspan(merged_storage), storage.layout.dim);
  auto index = ::cuvs::neighbors::cagra::detail::build_from_device_matrix<T, IdxT, DatasetViewT>(
    handle, params, dv);
  index.update_dataset(handle, dv);
  RAFT_LOG_DEBUG("cagra merge: using device memory for merged dataset");
  return index;
}

}  // namespace cuvs::neighbors::cagra::detail

namespace cuvs::neighbors::cagra {

template <class T, class IdxT, cuvs::neighbors::cagra_dataset_view DatasetViewT>
merged_dataset_storage<T, IdxT> make_merged_dataset(
  raft::resources const& res,
  std::vector<cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>*> const& indices,
  cuvs::neighbors::filtering::base_filter const& row_filter)
{
  merged_dataset layout = detail::compute_merged_dataset_layout<T, IdxT>(res, indices, row_filter);
  auto merged_storage =
    raft::make_device_matrix<T, int64_t>(res, layout.merged_rows, layout.stride_elements);
  std::optional<raft::device_matrix<T, int64_t, raft::row_major>> filtered_storage;
  if (layout.bitset_filtered) {
    filtered_storage.emplace(
      raft::make_device_matrix<T, int64_t>(res, layout.filtered_rows, layout.stride_elements));
  }
  return {layout, std::move(merged_storage), std::move(filtered_storage)};
}

}  // namespace cuvs::neighbors::cagra
