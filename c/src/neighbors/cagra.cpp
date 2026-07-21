/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <cstring>
#include <dlpack/dlpack.h>
#include <memory>
#include <type_traits>
#include <variant>
#include <vector>

#include <raft/core/copy.hpp>
#include <raft/core/error.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/numpy_serializer.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/serialize.hpp>

#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.h>
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/common.h>
#include <cuvs/neighbors/cagra.hpp>
#include "../core/exceptions.hpp"
#include "../core/interop.hpp"

#include "cagra.hpp"
#include "c_api_box.hpp"
#include <fstream>

namespace {

/**
 * Heap-allocated bundle for the C API: owns only `cagra::index`.
 * Lives behind `cuvsCagraIndex::addr` via `sg_cagra_c_api_index_box`.
 */
template <typename T, cuvs::neighbors::ann_dataset_view DatasetViewT>
struct cuvs_cagra_c_api_index_lifetime_holder {
  cuvs::neighbors::cagra::index<T, uint32_t, DatasetViewT> idx;
};

template <typename T>
struct cagra_c_api_extended_dataset_holder {
  cuvs::neighbors::cagra::extended_dataset_storage<T, uint32_t> storage;
};

template <typename T>
struct cagra_c_api_merged_dataset_holder {
  cuvs::neighbors::cagra::merged_dataset_storage<T, uint32_t> storage;
};

/** Owns how to delete co-located index storage; `cuvsCagraIndex::addr` points here. */
struct sg_cagra_c_api_index_box {
  void* index_ptr;
  enum class dataset_layout : uint8_t { device_padded, device_standard, host_padded, host_standard } layout;
  cuvs::neighbors::c_api::detail::owner_record owner_rec;
};

template <cuvs::neighbors::ann_dataset_view DatasetViewT>
constexpr auto sg_cagra_index_layout_from_view()
{
  if constexpr (cuvs::neighbors::is_device_standard_dataset_view_v<DatasetViewT>) {
    return sg_cagra_c_api_index_box::dataset_layout::device_standard;
  } else if constexpr (cuvs::neighbors::is_device_padded_dataset_view_v<DatasetViewT>) {
    return sg_cagra_c_api_index_box::dataset_layout::device_padded;
  } else if constexpr (cuvs::neighbors::is_host_standard_dataset_view_v<DatasetViewT>) {
    return sg_cagra_c_api_index_box::dataset_layout::host_standard;
  } else {
    return sg_cagra_c_api_index_box::dataset_layout::host_padded;
  }
}

template <typename T, typename IdxT = uint32_t, bool AllowHost = false, typename Fn>
static void with_index_by_layout(sg_cagra_c_api_index_box* box,
                                 const char* null_handle_err,
                                 const char* host_not_allowed_err,
                                 Fn&& fn)
{
  RAFT_EXPECTS(box != nullptr, "%s", null_handle_err);
  switch (box->layout) {
    case sg_cagra_c_api_index_box::dataset_layout::device_padded: {
      auto* idx =
        reinterpret_cast<cuvs::neighbors::cagra::device_padded_index<T, IdxT>*>(box->index_ptr);
      fn(*idx);
      break;
    }
    case sg_cagra_c_api_index_box::dataset_layout::host_padded: {
      if constexpr (AllowHost) {
        auto* idx =
          reinterpret_cast<cuvs::neighbors::cagra::host_padded_index<T, IdxT>*>(box->index_ptr);
        fn(*idx);
      } else {
        RAFT_FAIL("%s", host_not_allowed_err);
      }
      break;
    }
    case sg_cagra_c_api_index_box::dataset_layout::device_standard: {
      auto* idx =
        reinterpret_cast<cuvs::neighbors::cagra::device_standard_index<T, IdxT>*>(box->index_ptr);
      fn(*idx);
      break;
    }
    case sg_cagra_c_api_index_box::dataset_layout::host_standard: {
      if constexpr (AllowHost) {
        auto* idx =
          reinterpret_cast<cuvs::neighbors::cagra::host_standard_index<T, IdxT>*>(box->index_ptr);
        fn(*idx);
      } else {
        RAFT_FAIL("%s", host_not_allowed_err);
      }
      break;
    }
  }
}

template <typename T, cuvs::neighbors::ann_dataset_view DatasetViewT>
static void merge_indices_for_layout(
  raft::resources* res_ptr,
  cuvs::neighbors::cagra::index_params const& params_cpp,
  std::vector<cuvs::neighbors::cagra::index<T, uint32_t, DatasetViewT>*>& index_ptrs,
  cuvsFilter filter,
  cuvs::neighbors::cagra::merged_dataset_storage<T, uint32_t>& merge_storage,
  cuvsCagraIndex_t output_index)
{
  if (filter.type == NO_FILTER) {
    auto merged_idx    = cuvs::neighbors::cagra::merge(*res_ptr, params_cpp, index_ptrs, merge_storage);
    auto* holder =
      new cuvs_cagra_c_api_index_lifetime_holder<T, DatasetViewT>{std::move(merged_idx)};
    bind_index_lifetime_holder_to_C_index<T, DatasetViewT>(output_index, output_index->dtype, holder);
  } else if (filter.type == BITSET) {
    int64_t merged_row_count = 0;
    for (auto* idx_ptr : index_ptrs) {
      merged_row_count += static_cast<int64_t>(idx_ptr->size());
    }
    using filter_mdspan_type = raft::device_vector_view<std::uint32_t, int64_t, raft::row_major>;
    auto removed_indices_tensor = reinterpret_cast<DLManagedTensor*>(filter.addr);
    auto removed_indices = cuvs::core::from_dlpack<filter_mdspan_type>(removed_indices_tensor);
    cuvs::core::bitset_view<std::uint32_t, int64_t> removed_indices_bitset(
      removed_indices, merged_row_count);
    auto bitset_filter_obj =
      cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>(removed_indices_bitset);
    auto merged_idx = cuvs::neighbors::cagra::merge(
      *res_ptr, params_cpp, index_ptrs, merge_storage, bitset_filter_obj);
    auto* holder =
      new cuvs_cagra_c_api_index_lifetime_holder<T, DatasetViewT>{std::move(merged_idx)};
    bind_index_lifetime_holder_to_C_index<T, DatasetViewT>(output_index, output_index->dtype, holder);
  } else {
    RAFT_FAIL("Unsupported filter type: BITMAP");
  }
}

template <typename T, cuvs::neighbors::ann_dataset_view DatasetViewT>
static auto convert_opaque_indices_to_concrete_types(cuvsCagraIndex_t* indices, size_t num_indices)
  -> std::vector<cuvs::neighbors::cagra::index<T, uint32_t, DatasetViewT>*>
{
  std::vector<cuvs::neighbors::cagra::index<T, uint32_t, DatasetViewT>*> index_ptrs;
  index_ptrs.reserve(num_indices);
  for (size_t i = 0; i < num_indices; ++i) {
    auto* box = reinterpret_cast<sg_cagra_c_api_index_box*>(indices[i]->addr);
    RAFT_EXPECTS(box != nullptr, "cuvsCagraMerge: null index handle");
    index_ptrs.push_back(
      reinterpret_cast<cuvs::neighbors::cagra::index<T, uint32_t, DatasetViewT>*>(box->index_ptr));
  }
  return index_ptrs;
}

template <typename T, bool AllowHost = false, typename Fn>
static void with_dataset_view_for_layout(raft::resources* res_ptr,
                                         DLManagedTensor* dataset_tensor,
                                         sg_cagra_c_api_index_box::dataset_layout layout,
                                         const char* err_prefix,
                                         const char* host_not_allowed_err,
                                         Fn&& fn)
{
  auto dataset = dataset_tensor->dl_tensor;
  if (layout == sg_cagra_c_api_index_box::dataset_layout::device_padded) {
    if (cuvs::core::is_dlpack_device_compatible(dataset)) {
      using mdspan_type = raft::device_matrix_view<T const, int64_t, raft::row_major>;
      auto mds = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
      auto ds_view = cuvs::neighbors::make_device_padded_dataset_view(*res_ptr, mds);
      fn(ds_view);
      return;
    } else if (cuvs::core::is_dlpack_host_compatible(dataset)) {
      if constexpr (!AllowHost) { RAFT_FAIL("%s", host_not_allowed_err); }
      using mdspan_type = raft::host_matrix_view<T const, int64_t, raft::row_major>;
      auto mds = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
      auto ds_view = cuvs::neighbors::make_host_padded_dataset_view(mds);
      fn(ds_view);
      return;
    }
  } else if (layout == sg_cagra_c_api_index_box::dataset_layout::device_standard) {
    if (cuvs::core::is_dlpack_device_compatible(dataset)) {
      using mdspan_type = raft::device_matrix_view<T const, int64_t, raft::row_major>;
      auto mds = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
      auto ds_view = cuvs::neighbors::make_device_standard_dataset_view(mds);
      fn(ds_view);
      return;
    } else if (cuvs::core::is_dlpack_host_compatible(dataset)) {
      if constexpr (!AllowHost) { RAFT_FAIL("%s", host_not_allowed_err); }
      using mdspan_type = raft::host_matrix_view<T const, int64_t, raft::row_major>;
      auto mds = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
      auto ds_view = cuvs::neighbors::make_host_standard_dataset_view(mds);
      fn(ds_view);
      return;
    }
  } else {
    RAFT_FAIL("%s: unsupported index layout for dataset view dispatch", err_prefix);
  }
  RAFT_FAIL("%s: dataset must have host- or device-compatible memory", err_prefix);
}

template <typename T, cuvs::neighbors::ann_dataset_view DatasetViewT>
static void compute_ivfpq_shape_from_indices(cuvsCagraIndex_t* indices,
                                             size_t num_indices,
                                             int64_t* total_size,
                                             int64_t* dim)
{
  auto* first_box = reinterpret_cast<sg_cagra_c_api_index_box*>(indices[0]->addr);
  // Caller validates non-null boxes and uniform layout for all indices.
  auto* first_idx_ptr =
    reinterpret_cast<cuvs::neighbors::cagra::index<T, uint32_t, DatasetViewT>*>(first_box->index_ptr);
  *dim = first_idx_ptr->dim();
  *total_size = 0;
  for (size_t i = 0; i < num_indices; ++i) {
    auto* box = reinterpret_cast<sg_cagra_c_api_index_box*>(indices[i]->addr);
    auto* idx_ptr =
      reinterpret_cast<cuvs::neighbors::cagra::index<T, uint32_t, DatasetViewT>*>(box->index_ptr);
    *total_size += static_cast<int64_t>(idx_ptr->size());
  }
}

template <typename T, cuvs::neighbors::ann_dataset_view DatasetViewT>
static void bind_index_lifetime_holder_to_C_index(
  cuvsCagraIndex_t out,
  DLDataType dtype,
  cuvs_cagra_c_api_index_lifetime_holder<T, DatasetViewT>* holder)
{
  auto* box = new sg_cagra_c_api_index_box{&holder->idx,
                                         sg_cagra_index_layout_from_view<DatasetViewT>(),
                                         cuvs::neighbors::c_api::detail::make_owner_record(holder)};
  out->addr = reinterpret_cast<uintptr_t>(box);
  out->dtype  = dtype;
}

template <typename T, cuvs::neighbors::ann_dataset_view DatasetViewT>
static void wrap_CPP_index_in_lifetime_holder_and_bind_to_C_index(
  cuvsCagraIndex_t out,
  DLDataType dtype,
  cuvs::neighbors::cagra::index<T, uint32_t, DatasetViewT>* raw)
{
  auto* holder = new cuvs_cagra_c_api_index_lifetime_holder<T, DatasetViewT>{std::move(*raw)};
  delete raw;
  bind_index_lifetime_holder_to_C_index<T, DatasetViewT>(out, dtype, holder);
}

static void destroy_sg_cagra_c_api_box(uintptr_t addr)
{
  if (addr == 0) { return; }
  auto* box = reinterpret_cast<sg_cagra_c_api_index_box*>(addr);
  cuvs::neighbors::c_api::detail::destroy_owner_record(box->owner_rec);
  delete box;
}

template <typename T>
static void destroy_typed_addr(void* ptr)
{
  delete reinterpret_cast<T*>(ptr);
}

template <typename T>
static void destroy_extended_dataset_typed(uintptr_t addr)
{
  delete reinterpret_cast<cagra_c_api_extended_dataset_holder<T>*>(addr);
}

template <typename T>
static void destroy_merged_dataset_typed(uintptr_t addr)
{
  delete reinterpret_cast<cagra_c_api_merged_dataset_holder<T>*>(addr);
}

template <typename T>
static void make_device_padded_dataset(raft::resources* res_ptr,
                                       DLManagedTensor* dataset_tensor,
                                       cuvsDatasetPadded_t* output_padded_dataset)
{
  auto dataset = dataset_tensor->dl_tensor;
  RAFT_EXPECTS(cuvs::core::is_dlpack_device_compatible(dataset),
               "cuvsDatasetMakeDevicePadded: dataset must have device-compatible memory");
  using mdspan_type = raft::device_matrix_view<T const, int64_t, raft::row_major>;
  auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
  auto owner = cuvs::neighbors::make_device_padded_dataset(*res_ptr, mds);
  auto* out  = new cuvsDatasetPadded{};
  out->addr  = reinterpret_cast<uintptr_t>(owner.release());
  out->destroy_addr =
    &destroy_typed_addr<cuvs::neighbors::device_padded_dataset<T, int64_t>>;
  out->dtype  = dataset.dtype;
  out->layout = CUVS_DATASET_LAYOUT_PADDED;
  *output_padded_dataset = out;
}

template <typename T>
static void make_host_padded_dataset(raft::resources* res_ptr,
                                     DLManagedTensor* dataset_tensor,
                                     cuvsDatasetPadded_t* output_padded_dataset)
{
  auto dataset = dataset_tensor->dl_tensor;
  RAFT_EXPECTS(cuvs::core::is_dlpack_host_compatible(dataset),
               "cuvsDatasetMakeHostPadded: dataset must have host-compatible memory");
  using mdspan_type = raft::host_matrix_view<T const, int64_t, raft::row_major>;
  auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
  auto owner = cuvs::neighbors::make_host_padded_dataset(*res_ptr, mds);
  auto* out  = new cuvsDatasetPadded{};
  out->addr  = reinterpret_cast<uintptr_t>(owner.release());
  out->destroy_addr =
    &destroy_typed_addr<cuvs::neighbors::host_padded_dataset<T, int64_t>>;
  out->dtype  = dataset.dtype;
  out->layout = CUVS_DATASET_LAYOUT_PADDED;
  *output_padded_dataset = out;
}

template <typename T>
static void make_device_padded_dataset_view(raft::resources* res_ptr,
                                            DLManagedTensor* dataset_tensor,
                                            cuvsDatasetPaddedView_t* output_padded_dataset)
{
  auto dataset = dataset_tensor->dl_tensor;
  auto* out    = new cuvsDatasetPaddedView{};
  if (!cuvs::core::is_dlpack_device_compatible(dataset)) {
    delete out;
    RAFT_FAIL("cuvsDatasetMakeDevicePaddedView: dataset must have device-compatible memory");
  }
  using mdspan_type = raft::device_matrix_view<T const, int64_t, raft::row_major>;
  auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
  auto ds_view      = cuvs::neighbors::make_device_padded_dataset_view(*res_ptr, mds);
  auto* owned_view = new decltype(ds_view){ds_view};
  out->addr        = reinterpret_cast<uintptr_t>(owned_view);
  out->destroy_addr = &destroy_typed_addr<decltype(ds_view)>;
  out->kind         = CUVS_DATASET_VIEW_KIND_DEVICE_PADDED;
  out->dtype        = dataset.dtype;
  out->layout       = CUVS_DATASET_LAYOUT_PADDED;
  *output_padded_dataset = out;
}

template <typename T>
static void make_view_from_owning_padded(cuvsDatasetPadded_t padded_dataset,
                                         cuvsDatasetPaddedView_t* output_padded_view)
{
  RAFT_EXPECTS(padded_dataset != nullptr,
               "cuvsDatasetMakeViewFromOwningPadded: null padded dataset");
  RAFT_EXPECTS(padded_dataset->addr != 0,
               "cuvsDatasetMakeViewFromOwningPadded: null padded dataset storage");

  auto* owner =
    reinterpret_cast<cuvs::neighbors::device_padded_dataset<T, int64_t>*>(padded_dataset->addr);
  auto ds_view = owner->as_dataset_view();

  auto* ds_view_holder = new decltype(ds_view){ds_view};
  auto* out            = new cuvsDatasetPaddedView{reinterpret_cast<uintptr_t>(ds_view_holder),
                                         &destroy_typed_addr<decltype(ds_view)>,
                                         CUVS_DATASET_VIEW_KIND_DEVICE_PADDED,
                                         padded_dataset->dtype,
                                         CUVS_DATASET_LAYOUT_PADDED};
  *output_padded_view = out;
}

template <typename T>
static void make_host_padded_dataset_view(raft::resources*,
                                          DLManagedTensor* dataset_tensor,
                                          cuvsDatasetPaddedView_t* output_padded_dataset)
{
  auto dataset = dataset_tensor->dl_tensor;
  auto* out    = new cuvsDatasetPaddedView{};
  if (!cuvs::core::is_dlpack_host_compatible(dataset)) {
    delete out;
    RAFT_FAIL("cuvsDatasetMakeHostPaddedView: dataset must have host-compatible memory");
  }
  using mdspan_type = raft::host_matrix_view<T const, int64_t, raft::row_major>;
  auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
  auto ds_view      = cuvs::neighbors::make_host_padded_dataset_view(mds);
  auto* owned_view = new decltype(ds_view){ds_view};
  out->addr        = reinterpret_cast<uintptr_t>(owned_view);
  out->destroy_addr = &destroy_typed_addr<decltype(ds_view)>;
  out->kind         = CUVS_DATASET_VIEW_KIND_HOST_PADDED;
  out->dtype        = dataset.dtype;
  out->layout       = CUVS_DATASET_LAYOUT_PADDED;
  *output_padded_dataset = out;
}

template <typename T>
static void make_device_standard_dataset_view(raft::resources*,
                                              DLManagedTensor* dataset_tensor,
                                              cuvsDatasetStandardView_t* output_standard_dataset)
{
  auto dataset = dataset_tensor->dl_tensor;
  auto* out    = new cuvsDatasetStandardView{};
  if (!cuvs::core::is_dlpack_device_compatible(dataset)) {
    delete out;
    RAFT_FAIL("cuvsDatasetMakeDeviceStandardView: dataset must have device-compatible memory");
  }
  using mdspan_type = raft::device_matrix_view<T const, int64_t, raft::row_major>;
  auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
  auto ds_view      = cuvs::neighbors::make_device_standard_dataset_view(mds);
  auto* owned_view = new decltype(ds_view){ds_view};
  out->addr        = reinterpret_cast<uintptr_t>(owned_view);
  out->destroy_addr = &destroy_typed_addr<decltype(ds_view)>;
  out->kind         = CUVS_DATASET_VIEW_KIND_DEVICE_STANDARD;
  out->dtype        = dataset.dtype;
  out->layout       = CUVS_DATASET_LAYOUT_STANDARD;
  *output_standard_dataset = out;
}

template <typename T>
static void make_host_standard_dataset_view(raft::resources*,
                                            DLManagedTensor* dataset_tensor,
                                            cuvsDatasetStandardView_t* output_standard_dataset)
{
  auto dataset = dataset_tensor->dl_tensor;
  auto* out    = new cuvsDatasetStandardView{};
  if (!cuvs::core::is_dlpack_host_compatible(dataset)) {
    delete out;
    RAFT_FAIL("cuvsDatasetMakeHostStandardView: dataset must have host-compatible memory");
  }
  using mdspan_type = raft::host_matrix_view<T const, int64_t, raft::row_major>;
  auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
  auto ds_view      = cuvs::neighbors::make_host_standard_dataset_view(mds);
  auto* owned_view = new decltype(ds_view){ds_view};
  out->addr        = reinterpret_cast<uintptr_t>(owned_view);
  out->destroy_addr = &destroy_typed_addr<decltype(ds_view)>;
  out->kind         = CUVS_DATASET_VIEW_KIND_HOST_STANDARD;
  out->dtype        = dataset.dtype;
  out->layout       = CUVS_DATASET_LAYOUT_STANDARD;
  *output_standard_dataset = out;
}

template <typename T>
static void attach_padded_dataset_for_search(raft::resources* res_ptr,
                                             cuvsDatasetPaddedView_t padded_dataset,
                                             cuvsCagraIndex_t index)
{
  RAFT_EXPECTS(padded_dataset != nullptr, "cuvsCagraAttachPaddedDatasetForSearch: null padded dataset");
  RAFT_EXPECTS(index != nullptr, "cuvsCagraAttachPaddedDatasetForSearch: null index handle");
  RAFT_EXPECTS(index->addr != 0, "cuvsCagraAttachPaddedDatasetForSearch: null index storage");

  auto* box = reinterpret_cast<sg_cagra_c_api_index_box*>(index->addr);
  RAFT_EXPECTS(box->layout == sg_cagra_c_api_index_box::dataset_layout::device_standard,
               "cuvsCagraAttachPaddedDatasetForSearch: index must be standard layout");

  RAFT_EXPECTS(padded_dataset->kind == CUVS_DATASET_VIEW_KIND_DEVICE_PADDED,
               "cuvsCagraAttachPaddedDatasetForSearch: padded dataset view must be device-resident");
  RAFT_EXPECTS(padded_dataset->addr != 0,
               "cuvsCagraAttachPaddedDatasetForSearch: null padded dataset view storage");
  auto const& padded_view =
    *reinterpret_cast<cuvs::neighbors::device_padded_dataset_view<T, int64_t> const*>(
      padded_dataset->addr);

  auto* standard_idx =
    reinterpret_cast<cuvs::neighbors::cagra::device_standard_index<T, uint32_t>*>(box->index_ptr);

  auto padded_idx = cuvs::neighbors::cagra::attach_padded_dataset_for_search(
    *res_ptr, *standard_idx, padded_view);

  auto* holder =
    new cuvs_cagra_c_api_index_lifetime_holder<T, cuvs::neighbors::device_padded_dataset_view<T, int64_t>>{
      std::move(padded_idx)};

  destroy_sg_cagra_c_api_box(index->addr);
  index->addr = 0;
  bind_index_lifetime_holder_to_C_index<T, cuvs::neighbors::device_padded_dataset_view<T, int64_t>>(
    index, index->dtype, holder);
}

template <typename T>
static void attach_device_dataset_on_host_index(raft::resources* res_ptr,
                                                DLManagedTensor* device_dataset_tensor,
                                                cuvsCagraIndex_t index)
{
  RAFT_EXPECTS(index != nullptr, "cuvsCagraAttachDeviceDatasetOnHostIndex: null index handle");
  RAFT_EXPECTS(index->addr != 0, "cuvsCagraAttachDeviceDatasetOnHostIndex: null index storage");
  auto* box = reinterpret_cast<sg_cagra_c_api_index_box*>(index->addr);
  RAFT_EXPECTS(box->layout == sg_cagra_c_api_index_box::dataset_layout::host_padded ||
                 box->layout == sg_cagra_c_api_index_box::dataset_layout::host_standard,
               "cuvsCagraAttachDeviceDatasetOnHostIndex: index must be host layout");
  RAFT_EXPECTS(cuvs::core::is_dlpack_device_compatible(device_dataset_tensor->dl_tensor),
               "cuvsCagraAttachDeviceDatasetOnHostIndex: dataset must be device-compatible");

  if (box->layout == sg_cagra_c_api_index_box::dataset_layout::host_padded) {
    auto* host_idx =
      reinterpret_cast<cuvs::neighbors::cagra::host_padded_index<T, uint32_t>*>(box->index_ptr);
    using mdspan_type = raft::device_matrix_view<T const, int64_t, raft::row_major>;
    auto mds = cuvs::core::from_dlpack<mdspan_type>(device_dataset_tensor);
    auto device_dataset_view = cuvs::neighbors::make_device_padded_dataset_view(*res_ptr, mds);
    auto device_idx = cuvs::neighbors::cagra::attach_device_dataset_on_host_index(
      *res_ptr, *host_idx, device_dataset_view);
    auto* holder =
      new cuvs_cagra_c_api_index_lifetime_holder<T, cuvs::neighbors::device_padded_dataset_view<T, int64_t>>{
        std::move(device_idx)};
    destroy_sg_cagra_c_api_box(index->addr);
    index->addr = 0;
    bind_index_lifetime_holder_to_C_index<T, cuvs::neighbors::device_padded_dataset_view<T, int64_t>>(
      index, index->dtype, holder);
  } else {
    auto* host_idx =
      reinterpret_cast<cuvs::neighbors::cagra::host_standard_index<T, uint32_t>*>(box->index_ptr);
    using mdspan_type = raft::device_matrix_view<T const, int64_t, raft::row_major>;
    auto mds = cuvs::core::from_dlpack<mdspan_type>(device_dataset_tensor);
    auto device_dataset_view = cuvs::neighbors::make_device_standard_dataset_view(mds);
    auto device_idx = cuvs::neighbors::cagra::attach_device_dataset_on_host_index(
      *res_ptr, *host_idx, device_dataset_view);
    auto* holder =
      new cuvs_cagra_c_api_index_lifetime_holder<T, cuvs::neighbors::device_standard_dataset_view<T, int64_t>>{
        std::move(device_idx)};
    destroy_sg_cagra_c_api_box(index->addr);
    index->addr = 0;
    bind_index_lifetime_holder_to_C_index<T, cuvs::neighbors::device_standard_dataset_view<T, int64_t>>(
      index, index->dtype, holder);
  }
}

template <typename T>
static void make_extended_storage(raft::resources* res_ptr,
                                  cuvsDatasetPaddedView_t additional_dataset,
                                  cuvsCagraIndex_t index,
                                  cuvsDatasetStorage_t* output_extended_storage)
{
  auto* out    = new cuvsDatasetStorage{
    0, index->dtype, CUVS_DATASET_STORAGE_KIND_EXTENDED};
  auto* box    = reinterpret_cast<sg_cagra_c_api_index_box*>(index->addr);
  RAFT_EXPECTS(box != nullptr, "cuvsMakeExtendedStorage: null index handle");
  RAFT_EXPECTS(
    box->layout == sg_cagra_c_api_index_box::dataset_layout::device_padded,
    "cuvsMakeExtendedStorage: only device_padded indices are extendable. "
    "For standard indices, explicitly create/attach a padded dataset first.");
  RAFT_EXPECTS(additional_dataset != nullptr, "cuvsMakeExtendedStorage: null additional dataset view");
  RAFT_EXPECTS(additional_dataset->addr != 0,
               "cuvsMakeExtendedStorage: null additional dataset view storage");
  RAFT_EXPECTS(additional_dataset->kind == CUVS_DATASET_VIEW_KIND_DEVICE_PADDED ||
                 additional_dataset->kind == CUVS_DATASET_VIEW_KIND_HOST_PADDED,
               "cuvsMakeExtendedStorage: additional dataset must be a padded dataset view");
  with_index_by_layout<T, uint32_t, false>(
    box,
    "cuvsMakeExtendedStorage: null index handle",
    "cuvsMakeExtendedStorage: host indices are unsupported; attach device dataset to host "
    "index first",
    [&](auto& idx) {
      using index_t = std::decay_t<decltype(idx)>;
      constexpr bool idx_is_padded =
        std::is_same_v<index_t, cuvs::neighbors::cagra::device_padded_index<T, uint32_t>>;
      if constexpr (!idx_is_padded) {
        RAFT_FAIL("cuvsMakeExtendedStorage: only device_padded indices are extendable");
      } else {
        if (additional_dataset->kind == CUVS_DATASET_VIEW_KIND_DEVICE_PADDED) {
          auto* ds_view =
            reinterpret_cast<cuvs::neighbors::device_padded_dataset_view<T, int64_t>*>(
              additional_dataset->addr);
          auto storage      = cuvs::neighbors::cagra::make_extended_storage(*res_ptr, *ds_view, idx);
          auto* storage_holder = new cagra_c_api_extended_dataset_holder<T>{std::move(storage)};
          out->addr         = reinterpret_cast<uintptr_t>(storage_holder);
        } else {
          auto* ds_view =
            reinterpret_cast<cuvs::neighbors::host_padded_dataset_view<T, int64_t>*>(
              additional_dataset->addr);
          auto storage      = cuvs::neighbors::cagra::make_extended_storage(*res_ptr, *ds_view, idx);
          auto* storage_holder = new cagra_c_api_extended_dataset_holder<T>{std::move(storage)};
          out->addr         = reinterpret_cast<uintptr_t>(storage_holder);
        }
      }
    });

  *output_extended_storage = out;
}

static void _set_graph_build_params(
  std::variant<std::monostate,
               cuvs::neighbors::cagra::graph_build_params::ivf_pq_params,
               cuvs::neighbors::cagra::graph_build_params::nn_descent_params,
               cuvs::neighbors::cagra::graph_build_params::ace_params,
               cuvs::neighbors::cagra::graph_build_params::iterative_search_params>& out_params,
  cuvsCagraIndexParams& params,
  cuvsCagraGraphBuildAlgo algo,
  int64_t n_rows,
  int64_t dim)

{
  auto metric = static_cast<cuvs::distance::DistanceType>((int)params.metric);
  switch (algo) {
    case cuvsCagraGraphBuildAlgo::AUTO_SELECT: break;
    case cuvsCagraGraphBuildAlgo::IVF_PQ: {
      auto pq_params = cuvs::neighbors::cagra::graph_build_params::ivf_pq_params(
        raft::matrix_extent<int64_t>(n_rows, dim), metric);
      if (params.graph_build_params) {
        auto ivf_params = static_cast<cuvsIvfPqParams*>(params.graph_build_params);
        if (ivf_params->ivf_pq_build_params) {
          auto bp                                         = ivf_params->ivf_pq_build_params;
          pq_params.build_params.add_data_on_build        = bp->add_data_on_build;
          pq_params.build_params.n_lists                  = bp->n_lists;
          pq_params.build_params.kmeans_n_iters           = bp->kmeans_n_iters;
          pq_params.build_params.kmeans_trainset_fraction = bp->kmeans_trainset_fraction;
          pq_params.build_params.pq_bits                  = bp->pq_bits;
          pq_params.build_params.pq_dim                   = bp->pq_dim;
          pq_params.build_params.codebook_kind =
            static_cast<cuvs::neighbors::ivf_pq::codebook_gen>(bp->codebook_kind);
          pq_params.build_params.force_random_rotation = bp->force_random_rotation;
          pq_params.build_params.conservative_memory_allocation =
            bp->conservative_memory_allocation;
          pq_params.build_params.max_train_points_per_pq_code = bp->max_train_points_per_pq_code;
        }
        if (ivf_params->ivf_pq_search_params) {
          auto sp                                          = ivf_params->ivf_pq_search_params;
          pq_params.search_params.n_probes                 = sp->n_probes;
          pq_params.search_params.lut_dtype                = sp->lut_dtype;
          pq_params.search_params.internal_distance_dtype  = sp->internal_distance_dtype;
          pq_params.search_params.preferred_shmem_carveout = sp->preferred_shmem_carveout;
        }
        if (ivf_params->refinement_rate > 1.0f) {
          pq_params.refinement_rate = ivf_params->refinement_rate;
        }
      }
      out_params = pq_params;
      break;
    }
    case cuvsCagraGraphBuildAlgo::NN_DESCENT: {
      auto nn_params =
        cuvs::neighbors::nn_descent::index_params(params.intermediate_graph_degree, metric);
      nn_params.max_iterations = params.nn_descent_niter;
      out_params               = nn_params;
      break;
    }
    case cuvsCagraGraphBuildAlgo::ACE: {
      cuvs::neighbors::cagra::graph_build_params::ace_params ace_p;
      if (params.graph_build_params) {
        auto ace_params_c             = static_cast<cuvsAceParams*>(params.graph_build_params);
        ace_p.npartitions         = ace_params_c->npartitions;
        ace_p.ef_construction     = ace_params_c->ef_construction;
        ace_p.build_dir           = std::string(ace_params_c->build_dir);
        ace_p.use_disk            = ace_params_c->use_disk;
        ace_p.max_host_memory_gb  = ace_params_c->max_host_memory_gb;
        ace_p.max_gpu_memory_gb   = ace_params_c->max_gpu_memory_gb;
      }
      out_params = ace_p;
      break;
    }
    case cuvsCagraGraphBuildAlgo::ITERATIVE_CAGRA_SEARCH: {
      cuvs::neighbors::cagra::graph_build_params::iterative_search_params p;
      out_params = p;
      break;
    }
  }
}

template <typename T>
void _from_args(cuvsResources_t res,
                cuvsDistanceType _metric,
                DLManagedTensor* graph_tensor,
                DLManagedTensor* dataset_tensor,
                cuvsCagraIndex_t output_index)
{
  auto metric  = static_cast<cuvs::distance::DistanceType>((int)_metric);
  auto dataset = dataset_tensor->dl_tensor;
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  auto update_graph_from_dlpack = [&](auto* idx) {
    if (cuvs::core::is_dlpack_device_compatible(graph_tensor->dl_tensor)) {
      using graph_mdspan_type = raft::device_matrix_view<uint32_t const, int64_t, raft::row_major>;
      auto graph_mds = cuvs::core::from_dlpack<graph_mdspan_type>(graph_tensor);
      idx->update_graph(*res_ptr, graph_mds);
    } else {
      using graph_mdspan_type = raft::host_matrix_view<uint32_t const, int64_t, raft::row_major>;
      auto graph_mds = cuvs::core::from_dlpack<graph_mdspan_type>(graph_tensor);
      idx->update_graph(*res_ptr, graph_mds);
    }
  };

  if (cuvs::core::is_dlpack_device_compatible(dataset)) {
    using mdspan_type = raft::device_matrix_view<T const, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
    if (cuvs::neighbors::matrix_row_width_matches_cagra_required(mds)) {
      auto dataset_view = cuvs::neighbors::make_device_padded_dataset_view(*res_ptr, mds);
      auto* raw         = new cuvs::neighbors::cagra::device_padded_index<T, uint32_t>(
        *res_ptr, metric);
      raw->update_dataset(*res_ptr, dataset_view);
      update_graph_from_dlpack(raw);
      wrap_CPP_index_in_lifetime_holder_and_bind_to_C_index<
        T,
        cuvs::neighbors::device_padded_dataset_view<T, int64_t>>(
        output_index, output_index->dtype, raw);
    } else {
      auto dataset_view = cuvs::neighbors::make_device_standard_dataset_view(mds);
      auto* raw         = new cuvs::neighbors::cagra::device_standard_index<T, uint32_t>(
        *res_ptr, metric);
      raw->update_dataset(*res_ptr, dataset_view);
      update_graph_from_dlpack(raw);
      wrap_CPP_index_in_lifetime_holder_and_bind_to_C_index<
        T,
        cuvs::neighbors::device_standard_dataset_view<T, int64_t>>(
        output_index, output_index->dtype, raw);
    }
  } else if (cuvs::core::is_dlpack_host_compatible(dataset)) {
    RAFT_FAIL("cuvsCagraIndexFromArgs: host dataset is unsupported; use cuvsCagraBuild for host "
              "datasets, then cuvsCagraAttachDeviceDatasetOnHostIndex to get a device index.");
  }
}

template <typename T>
void _extend(cuvsResources_t res,
             cuvsCagraExtendParams params,
             cuvsCagraIndex index,
             cuvsDatasetPaddedView_t additional_dataset,
             cuvsDatasetStorage_t extended_dataset)
{
  auto* box      = reinterpret_cast<sg_cagra_c_api_index_box*>(index.addr);
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  RAFT_EXPECTS(box != nullptr, "cuvsCagraExtend: null index handle");
  RAFT_EXPECTS(
    box->layout == sg_cagra_c_api_index_box::dataset_layout::device_padded,
    "cuvsCagraExtend: only device_padded indices are extendable. "
    "For standard indices, explicitly create/attach a padded dataset first.");
  RAFT_EXPECTS(additional_dataset != nullptr, "cuvsCagraExtend: null additional dataset view");
  RAFT_EXPECTS(additional_dataset->addr != 0,
               "cuvsCagraExtend: null additional dataset view storage");
  RAFT_EXPECTS(additional_dataset->kind == CUVS_DATASET_VIEW_KIND_DEVICE_PADDED ||
                 additional_dataset->kind == CUVS_DATASET_VIEW_KIND_HOST_PADDED,
               "cuvsCagraExtend: additional dataset must be a padded dataset view");
  RAFT_EXPECTS(extended_dataset != nullptr, "cuvsCagraExtend: null extended dataset handle");
  RAFT_EXPECTS(extended_dataset->kind == CUVS_DATASET_STORAGE_KIND_EXTENDED,
               "cuvsCagraExtend: storage handle kind must be EXTENDED");
  auto* storage_holder =
    reinterpret_cast<cagra_c_api_extended_dataset_holder<T>*>(extended_dataset->addr);
  RAFT_EXPECTS(storage_holder != nullptr, "cuvsCagraExtend: null extended dataset storage");

  // TODO: use C struct here (see issue #487)
  auto extend_params           = cuvs::neighbors::cagra::extend_params();
  extend_params.max_chunk_size = params.max_chunk_size;
  with_index_by_layout<T, uint32_t, false>(
    box,
    "cuvsCagraExtend: null index handle",
    "cuvsCagraExtend: host indices are not extendable; attach a device dataset to the host index "
    "first.",
    [&](auto& idx) {
      using index_t = std::decay_t<decltype(idx)>;
      constexpr bool idx_is_padded =
        std::is_same_v<index_t, cuvs::neighbors::cagra::device_padded_index<T, uint32_t>>;
      if constexpr (!idx_is_padded) {
        RAFT_FAIL("cuvsCagraExtend: only device_padded indices are extendable");
      } else {
        if (additional_dataset->kind == CUVS_DATASET_VIEW_KIND_DEVICE_PADDED) {
          auto* ds_view =
            reinterpret_cast<cuvs::neighbors::device_padded_dataset_view<T, int64_t>*>(
              additional_dataset->addr);
          cuvs::neighbors::cagra::extend(
            *res_ptr, extend_params, *ds_view, idx, storage_holder->storage);
        } else {
          auto* ds_view =
            reinterpret_cast<cuvs::neighbors::host_padded_dataset_view<T, int64_t>*>(
              additional_dataset->addr);
          cuvs::neighbors::cagra::extend(
            *res_ptr, extend_params, *ds_view, idx, storage_holder->storage);
        }
      }
    });
}

template <typename T, typename IdxT>
void _search(cuvsResources_t res,
             cuvsCagraSearchParams params,
             cuvsCagraIndex index,
             DLManagedTensor* queries_tensor,
             DLManagedTensor* neighbors_tensor,
             DLManagedTensor* distances_tensor,
             cuvsFilter filter)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  auto* box    = reinterpret_cast<sg_cagra_c_api_index_box*>(index.addr);
  with_index_by_layout<T, uint32_t, false>(
    box,
    "cuvsCagraSearch: null index handle",
    "cuvsCagraSearch: host index must be converted to device first via "
    "cuvsCagraAttachDeviceDatasetOnHostIndex",
    [&](auto& idx) {
      auto search_params = cuvs::neighbors::cagra::search_params();
      convert_c_search_params(params, &search_params);

      using queries_mdspan_type   = raft::device_matrix_view<T const, int64_t, raft::row_major>;
      using neighbors_mdspan_type = raft::device_matrix_view<IdxT, int64_t, raft::row_major>;
      using distances_mdspan_type = raft::device_matrix_view<float, int64_t, raft::row_major>;
      auto queries_mds            = cuvs::core::from_dlpack<queries_mdspan_type>(queries_tensor);
      auto neighbors_mds          = cuvs::core::from_dlpack<neighbors_mdspan_type>(neighbors_tensor);
      auto distances_mds          = cuvs::core::from_dlpack<distances_mdspan_type>(distances_tensor);
      if (filter.type == NO_FILTER) {
        cuvs::neighbors::cagra::search(
          *res_ptr, search_params, idx, queries_mds, neighbors_mds, distances_mds);
      } else if (filter.type == BITSET) {
        using filter_mdspan_type = raft::device_vector_view<std::uint32_t, int64_t, raft::row_major>;
        auto removed_indices_tensor = reinterpret_cast<DLManagedTensor*>(filter.addr);
        auto removed_indices = cuvs::core::from_dlpack<filter_mdspan_type>(removed_indices_tensor);
        cuvs::core::bitset_view<std::uint32_t, int64_t> removed_indices_bitset(
          removed_indices, idx.dataset().n_rows());
        auto bitset_filter_obj = cuvs::neighbors::filtering::bitset_filter(removed_indices_bitset);
        cuvs::neighbors::cagra::search(*res_ptr,
                                       search_params,
                                       idx,
                                       queries_mds,
                                       neighbors_mds,
                                       distances_mds,
                                       bitset_filter_obj);
      } else {
        RAFT_FAIL("Unsupported filter type: BITMAP");
      }
    });
}

template <typename T>
void _search(cuvsResources_t res,
             cuvsCagraSearchParams params,
             cuvsCagraIndex index,
             DLManagedTensor* queries_tensor,
             DLManagedTensor* neighbors_tensor,
             DLManagedTensor* distances_tensor,
             cuvsFilter filter)
{
  if (neighbors_tensor->dl_tensor.dtype.code == kDLUInt &&
      neighbors_tensor->dl_tensor.dtype.bits == 32) {
    _search<T, uint32_t>(
      res, params, index, queries_tensor, neighbors_tensor, distances_tensor, filter);
  } else if (neighbors_tensor->dl_tensor.dtype.code == kDLInt &&
             neighbors_tensor->dl_tensor.dtype.bits == 64) {
    _search<T, int64_t>(
      res, params, index, queries_tensor, neighbors_tensor, distances_tensor, filter);
  } else {
    RAFT_FAIL("neighbors should be of type uint32_t or int64_t");
  }
}

template <typename T>
void _serialize(cuvsResources_t res,
                const char* filename,
                cuvsCagraIndex_t index,
                bool include_dataset)
{
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto* box      = reinterpret_cast<sg_cagra_c_api_index_box*>(index->addr);
  with_index_by_layout<T, uint32_t, false>(
    box,
    "cuvsCagraSerialize: null index handle",
    "cuvsCagraSerialize: host index must be converted to device first via "
    "cuvsCagraAttachDeviceDatasetOnHostIndex",
    [&](auto& idx) { cuvs::neighbors::cagra::serialize(*res_ptr, std::string(filename), idx, include_dataset); });
}

template <typename T>
void _serialize_to_hnswlib(cuvsResources_t res, const char* filename, cuvsCagraIndex_t index)
{
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto* box      = reinterpret_cast<sg_cagra_c_api_index_box*>(index->addr);
  with_index_by_layout<T, uint32_t, false>(
    box,
    "cuvsCagraSerializeToHnswlib: null index handle",
    "cuvsCagraSerializeToHnswlib: host index must be converted to device first via "
    "cuvsCagraAttachDeviceDatasetOnHostIndex",
    [&](auto& idx) { cuvs::neighbors::cagra::serialize_to_hnswlib(*res_ptr, std::string(filename), idx); });
}

template <typename T>
void _deserialize_padded(cuvsResources_t res,
                         const char* filename,
                         cuvsCagraIndex_t output_index,
                         cuvsDatasetPadded_t* out_padded_dataset)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  using view_t          = cuvs::neighbors::device_padded_dataset_view<T, int64_t>;
  using owner_dataset_t = cuvs::neighbors::owning_dataset_for_view_t<view_t>;
  auto holder = std::make_unique<
    cuvs_cagra_c_api_index_lifetime_holder<T, view_t>>(
    cuvs::neighbors::cagra::device_padded_index<T, uint32_t>(*res_ptr));
  std::unique_ptr<owner_dataset_t> out_dataset{};
  cuvs::neighbors::cagra::deserialize(*res_ptr, std::string(filename), &holder->idx, &out_dataset);
  if (out_dataset != nullptr) {
    RAFT_EXPECTS(out_padded_dataset != nullptr,
                 "cuvsCagraDeserializePadded: out_padded_dataset must be non-null when "
                 "deserializing a padded dataset payload");
    auto* out      = new cuvsDatasetPadded{};
    out->addr      = reinterpret_cast<uintptr_t>(out_dataset.release());
    out->destroy_addr = &destroy_typed_addr<owner_dataset_t>;
    out->dtype     = output_index->dtype;
    out->layout    = CUVS_DATASET_LAYOUT_PADDED;
    *out_padded_dataset = out;
  }
  bind_index_lifetime_holder_to_C_index<T, view_t>(output_index, output_index->dtype, holder.release());
}

template <typename T>
void _deserialize_standard(cuvsResources_t res,
                           const char* filename,
                           cuvsCagraIndex_t output_index,
                           cuvsDatasetStandard_t* out_standard_dataset)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  using view_t          = cuvs::neighbors::device_standard_dataset_view<T, int64_t>;
  using owner_dataset_t = cuvs::neighbors::owning_dataset_for_view_t<view_t>;
  auto holder = std::make_unique<
    cuvs_cagra_c_api_index_lifetime_holder<T, view_t>>(
    cuvs::neighbors::cagra::device_standard_index<T, uint32_t>(*res_ptr));
  std::unique_ptr<owner_dataset_t> out_dataset{};
  cuvs::neighbors::cagra::deserialize(*res_ptr, std::string(filename), &holder->idx, &out_dataset);
  if (out_dataset != nullptr) {
    RAFT_EXPECTS(out_standard_dataset != nullptr,
                 "cuvsCagraDeserializeStandard: out_standard_dataset must be non-null when "
                 "deserializing a standard dataset payload");
    auto* out      = new cuvsDatasetStandard{};
    out->addr      = reinterpret_cast<uintptr_t>(out_dataset.release());
    out->destroy_addr = &destroy_typed_addr<owner_dataset_t>;
    out->dtype     = output_index->dtype;
    out->layout    = CUVS_DATASET_LAYOUT_STANDARD;
    *out_standard_dataset = out;
  }
  bind_index_lifetime_holder_to_C_index<T, view_t>(output_index, output_index->dtype, holder.release());
}

template <typename T>
void _merge(cuvsResources_t res,
            cuvsCagraIndexParams params,
            cuvsCagraIndex_t* indices,
            size_t num_indices,
            cuvsFilter filter,
            cuvsDatasetStorage_t merged_dataset,
            cuvsCagraIndex_t output_index)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  auto* first_box = reinterpret_cast<sg_cagra_c_api_index_box*>(indices[0]->addr);
  RAFT_EXPECTS(first_box != nullptr, "cuvsCagraMerge: null index handle");
  auto layout = first_box->layout;
  RAFT_EXPECTS(layout == sg_cagra_c_api_index_box::dataset_layout::device_padded ||
                 layout == sg_cagra_c_api_index_box::dataset_layout::device_standard,
               "cuvsCagraMerge: host indices are not mergeable; attach a device dataset to each "
               "host index first.");
  cuvs::neighbors::cagra::index_params params_cpp;

  params_cpp.metric =
    static_cast<cuvs::distance::DistanceType>((int)params.metric);
  params_cpp.intermediate_graph_degree =
    params.intermediate_graph_degree;
  params_cpp.graph_degree = params.graph_degree;

  int64_t total_size = 0;
  int64_t dim        = 0;
  for (size_t i = 0; i < num_indices; ++i) {
    auto* box = reinterpret_cast<sg_cagra_c_api_index_box*>(indices[i]->addr);
    RAFT_EXPECTS(box != nullptr, "cuvsCagraMerge: null index handle");
    RAFT_EXPECTS(box->layout == layout,
                 "cuvsCagraMerge: all input indices must share the same dataset layout");
  }
  if (params.build_algo == cuvsCagraGraphBuildAlgo::IVF_PQ) {
    if (layout == sg_cagra_c_api_index_box::dataset_layout::device_padded) {
      compute_ivfpq_shape_from_indices<T, cuvs::neighbors::device_padded_dataset_view<T, int64_t>>(
        indices, num_indices, &total_size, &dim);
    } else {
      compute_ivfpq_shape_from_indices<T, cuvs::neighbors::device_standard_dataset_view<T, int64_t>>(
        indices, num_indices, &total_size, &dim);
    }
  }

  _set_graph_build_params(params_cpp.graph_build_params,
                          params,
                          params.build_algo,
                          total_size,
                          dim);
  auto* merge_holder =
    reinterpret_cast<cagra_c_api_merged_dataset_holder<T>*>(merged_dataset->addr);
  RAFT_EXPECTS(merged_dataset->kind == CUVS_DATASET_STORAGE_KIND_MERGED,
               "cuvsCagraMerge: storage handle kind must be MERGED");
  RAFT_EXPECTS(merge_holder != nullptr, "cuvsCagraMerge: null merged dataset storage");

  if (layout == sg_cagra_c_api_index_box::dataset_layout::device_padded) {
    auto index_ptrs =
      convert_opaque_indices_to_concrete_types<T, cuvs::neighbors::device_padded_dataset_view<T, int64_t>>(
        indices, num_indices);
    merge_indices_for_layout<T, cuvs::neighbors::device_padded_dataset_view<T, int64_t>>(
      res_ptr, params_cpp, index_ptrs, filter, merge_holder->storage, output_index);
  } else {
    auto index_ptrs =
      convert_opaque_indices_to_concrete_types<T, cuvs::neighbors::device_standard_dataset_view<T, int64_t>>(
        indices, num_indices);
    merge_indices_for_layout<T, cuvs::neighbors::device_standard_dataset_view<T, int64_t>>(
      res_ptr, params_cpp, index_ptrs, filter, merge_holder->storage, output_index);
  }
}

template <typename T>
void _make_merged_storage(cuvsResources_t res,
                          cuvsCagraIndex_t* indices,
                          size_t num_indices,
                          cuvsFilter filter,
                          cuvsDatasetStorage_t* output_merged_storage)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  auto* first_box = reinterpret_cast<sg_cagra_c_api_index_box*>(indices[0]->addr);
  RAFT_EXPECTS(first_box != nullptr, "cuvsMakeMergedStorage: null index handle");
  auto layout = first_box->layout;
  RAFT_EXPECTS(layout == sg_cagra_c_api_index_box::dataset_layout::device_padded ||
                 layout == sg_cagra_c_api_index_box::dataset_layout::device_standard,
               "cuvsMakeMergedStorage: host indices are unsupported; attach device dataset to "
               "host indices first.");
  for (size_t i = 0; i < num_indices; ++i) {
    auto* box = reinterpret_cast<sg_cagra_c_api_index_box*>(indices[i]->addr);
    RAFT_EXPECTS(box != nullptr, "cuvsMakeMergedStorage: null index handle");
    RAFT_EXPECTS(box->layout == layout,
                 "cuvsMakeMergedStorage: all input indices must share the same dataset "
                 "layout");
  }

  auto* out = new cuvsDatasetStorage{
    0, indices[0]->dtype, CUVS_DATASET_STORAGE_KIND_MERGED};
  cuvs::neighbors::cagra::merged_dataset_storage<T, uint32_t> storage =
    [&]() -> cuvs::neighbors::cagra::merged_dataset_storage<T, uint32_t> {
    if (layout == sg_cagra_c_api_index_box::dataset_layout::device_padded) {
      auto index_ptrs =
        convert_opaque_indices_to_concrete_types<T, cuvs::neighbors::device_padded_dataset_view<T, int64_t>>(
          indices, num_indices);
      if (filter.type == NO_FILTER) { return cuvs::neighbors::cagra::make_merged_storage(*res_ptr, index_ptrs); }
      if (filter.type == BITSET) {
        int64_t merged_row_count = 0;
        for (auto* idx_ptr : index_ptrs) {
          merged_row_count += static_cast<int64_t>(idx_ptr->size());
        }
        using filter_mdspan_type = raft::device_vector_view<std::uint32_t, int64_t, raft::row_major>;
        auto removed_indices_tensor = reinterpret_cast<DLManagedTensor*>(filter.addr);
        auto removed_indices = cuvs::core::from_dlpack<filter_mdspan_type>(removed_indices_tensor);
        cuvs::core::bitset_view<std::uint32_t, int64_t> removed_indices_bitset(
          removed_indices, merged_row_count);
        auto bitset_filter_obj =
          cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>(removed_indices_bitset);
        return cuvs::neighbors::cagra::make_merged_storage(*res_ptr, index_ptrs, bitset_filter_obj);
      }
      RAFT_FAIL("Unsupported filter type: BITMAP");
    } else if (layout == sg_cagra_c_api_index_box::dataset_layout::device_standard) {
      auto index_ptrs =
      convert_opaque_indices_to_concrete_types<T, cuvs::neighbors::device_standard_dataset_view<T, int64_t>>(
        indices, num_indices);
      if (filter.type == NO_FILTER) { return cuvs::neighbors::cagra::make_merged_storage(*res_ptr, index_ptrs); }
      if (filter.type == BITSET) {
        int64_t merged_row_count = 0;
        for (auto* idx_ptr : index_ptrs) {
          merged_row_count += static_cast<int64_t>(idx_ptr->size());
        }
        using filter_mdspan_type = raft::device_vector_view<std::uint32_t, int64_t, raft::row_major>;
        auto removed_indices_tensor = reinterpret_cast<DLManagedTensor*>(filter.addr);
        auto removed_indices = cuvs::core::from_dlpack<filter_mdspan_type>(removed_indices_tensor);
        cuvs::core::bitset_view<std::uint32_t, int64_t> removed_indices_bitset(
          removed_indices, merged_row_count);
        auto bitset_filter_obj =
          cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>(removed_indices_bitset);
        return cuvs::neighbors::cagra::make_merged_storage(*res_ptr, index_ptrs, bitset_filter_obj);
      }
      RAFT_FAIL("Unsupported filter type: BITMAP");
    } else {
      RAFT_FAIL("cuvsMakeMergedStorage: unsupported index layout");
    }
  }();
  auto* storage_holder = new cagra_c_api_merged_dataset_holder<T>{std::move(storage)};
  out->addr            = reinterpret_cast<uintptr_t>(storage_holder);
  *output_merged_storage = out;
}

template <typename T, typename IdxT>
void get_dataset_view(cuvsCagraIndex_t index, DLManagedTensor* dataset)
{
  auto* box = reinterpret_cast<sg_cagra_c_api_index_box*>(index->addr);
  with_index_by_layout<T, IdxT, true>(
    box,
    "cuvsCagraIndexGetDataset: null index handle",
    "cuvsCagraIndexGetDataset: host indices are allowed",
    [&](auto& idx) { cuvs::core::to_dlpack(idx.dataset().view(), dataset); });
}

template <typename T, typename IdxT>
void get_graph_view(cuvsCagraIndex_t index, DLManagedTensor* graph)
{
  auto* box = reinterpret_cast<sg_cagra_c_api_index_box*>(index->addr);
  with_index_by_layout<T, IdxT, true>(
    box,
    "cuvsCagraIndexGetGraph: null index handle",
    "cuvsCagraIndexGetGraph: host indices are allowed",
    [&](auto& idx) { cuvs::core::to_dlpack(idx.graph(), graph); });
}

// Helper function to populate C IVF-PQ params from C++ params
static void _populate_c_ivf_pq_params(cuvsIvfPqParams* c_ivf_pq,
                                    const cuvs::neighbors::cagra::graph_build_params::ivf_pq_params& cpp_ivf_pq)
{
  // Populate the IVF-PQ build params
  auto& bp = cpp_ivf_pq.build_params;
  c_ivf_pq->ivf_pq_build_params->metric = static_cast<cuvsDistanceType>(bp.metric);
  c_ivf_pq->ivf_pq_build_params->metric_arg = bp.metric_arg;
  c_ivf_pq->ivf_pq_build_params->add_data_on_build = bp.add_data_on_build;
  c_ivf_pq->ivf_pq_build_params->n_lists = bp.n_lists;
  c_ivf_pq->ivf_pq_build_params->kmeans_n_iters = bp.kmeans_n_iters;
  c_ivf_pq->ivf_pq_build_params->kmeans_trainset_fraction = bp.kmeans_trainset_fraction;
  c_ivf_pq->ivf_pq_build_params->pq_bits = bp.pq_bits;
  c_ivf_pq->ivf_pq_build_params->pq_dim = bp.pq_dim;
  c_ivf_pq->ivf_pq_build_params->codebook_kind = static_cast<cuvsIvfPqCodebookGen>(bp.codebook_kind);
  c_ivf_pq->ivf_pq_build_params->force_random_rotation = bp.force_random_rotation;
  c_ivf_pq->ivf_pq_build_params->conservative_memory_allocation = bp.conservative_memory_allocation;
  c_ivf_pq->ivf_pq_build_params->max_train_points_per_pq_code = bp.max_train_points_per_pq_code;

  // Populate the IVF-PQ search params
  auto& sp = cpp_ivf_pq.search_params;
  c_ivf_pq->ivf_pq_search_params->n_probes = sp.n_probes;
  c_ivf_pq->ivf_pq_search_params->lut_dtype = sp.lut_dtype;
  c_ivf_pq->ivf_pq_search_params->internal_distance_dtype = sp.internal_distance_dtype;
  c_ivf_pq->ivf_pq_search_params->preferred_shmem_carveout = sp.preferred_shmem_carveout;

  c_ivf_pq->refinement_rate = cpp_ivf_pq.refinement_rate;
}

// Helper function to populate C struct from C++ index_params
static void _populate_cagra_index_params_from_cpp(cuvsCagraIndexParams_t c_params,
                                                 const cuvs::neighbors::cagra::index_params& cpp_params)
{
  c_params->metric = static_cast<cuvsDistanceType>(cpp_params.metric);
  c_params->intermediate_graph_degree = cpp_params.intermediate_graph_degree;
  c_params->graph_degree = cpp_params.graph_degree;

  // Set build algo and parameters based on the variant
  if (std::holds_alternative<cuvs::neighbors::cagra::graph_build_params::nn_descent_params>(
        cpp_params.graph_build_params)) {
    c_params->build_algo = NN_DESCENT;
    auto nn_params =
      std::get<cuvs::neighbors::cagra::graph_build_params::nn_descent_params>(
        cpp_params.graph_build_params);
    c_params->nn_descent_niter = nn_params.max_iterations;
  } else if (std::holds_alternative<cuvs::neighbors::cagra::graph_build_params::ivf_pq_params>(
               cpp_params.graph_build_params)) {
    c_params->build_algo = IVF_PQ;
    auto ivf_pq_params =
      std::get<cuvs::neighbors::cagra::graph_build_params::ivf_pq_params>(
        cpp_params.graph_build_params);

    _populate_c_ivf_pq_params(static_cast<cuvsIvfPqParams*>(c_params->graph_build_params), ivf_pq_params);
  } else if (std::holds_alternative<cuvs::neighbors::cagra::graph_build_params::ace_params>(
               cpp_params.graph_build_params)) {
    c_params->build_algo = ACE;
    auto ace_params =
      std::get<cuvs::neighbors::cagra::graph_build_params::ace_params>(
        cpp_params.graph_build_params);
    cuvsAceParams* c_ace_params = new cuvsAceParams;
    c_ace_params->npartitions = ace_params.npartitions;
    c_ace_params->ef_construction = ace_params.ef_construction;
    c_ace_params->build_dir = ace_params.build_dir.empty() ? nullptr : strdup(ace_params.build_dir.c_str());
    c_ace_params->use_disk = ace_params.use_disk;
    c_params->graph_build_params = c_ace_params;
  }
}

}  // namespace

namespace cuvs::neighbors::cagra {
void convert_c_index_params(cuvsCagraIndexParams params,
                            int64_t n_rows,
                            int64_t dim,
                            cuvs::neighbors::cagra::index_params* out)
{
  out->metric                    = static_cast<cuvs::distance::DistanceType>((int)params.metric);
  out->intermediate_graph_degree = params.intermediate_graph_degree;
  out->graph_degree              = params.graph_degree;
  _set_graph_build_params(out->graph_build_params, params, params.build_algo, n_rows, dim);

}
void convert_c_search_params(cuvsCagraSearchParams params,
                             cuvs::neighbors::cagra::search_params* out)
{
  out->max_queries           = params.max_queries;
  out->itopk_size            = params.itopk_size;
  out->max_iterations        = params.max_iterations;
  out->algo                  = static_cast<cuvs::neighbors::cagra::search_algo>(params.algo);
  out->team_size             = params.team_size;
  out->search_width          = params.search_width;
  out->min_iterations        = params.min_iterations;
  out->thread_block_size     = params.thread_block_size;
  out->hashmap_mode          = static_cast<cuvs::neighbors::cagra::hash_mode>(params.hashmap_mode);
  out->hashmap_min_bitlen    = params.hashmap_min_bitlen;
  out->hashmap_max_fill_rate = params.hashmap_max_fill_rate;
  out->num_random_samplings  = params.num_random_samplings;
  out->rand_xor_mask         = params.rand_xor_mask;
  out->persistent            = params.persistent;
  out->persistent_lifetime   = params.persistent_lifetime;
  out->persistent_device_usage = params.persistent_device_usage;
}

void* cagra_c_api_index_ptr(cuvsCagraIndex const* idx)
{
  // Matches `sg_cagra_c_api_index_box::index_ptr` (first member); keep in sync with that layout.
  if (idx == nullptr || idx->addr == 0) { return nullptr; }
  return *reinterpret_cast<void**>(idx->addr);
}
}  // namespace cuvs::neighbors::cagra

extern "C" cuvsError_t cuvsCagraIndexCreate(cuvsCagraIndex_t* index)
{
  return cuvs::core::translate_exceptions([=] {
    *index = new cuvsCagraIndex{0, {}};
  });
}

extern "C" cuvsError_t cuvsCagraIndexDestroy(cuvsCagraIndex_t index_c_ptr)
{
  return cuvs::core::translate_exceptions([=] {
    destroy_sg_cagra_c_api_box(index_c_ptr->addr);
    delete index_c_ptr;
  });
}

extern "C" cuvsError_t cuvsCagraIndexGetDims(cuvsCagraIndex_t index, int64_t* dim)
{
  return cuvs::core::translate_exceptions([=] {
    auto* box = reinterpret_cast<sg_cagra_c_api_index_box*>(index->addr);
    with_index_by_layout<float, uint32_t, true>(
      box,
      "cuvsCagraIndexGetDims: null index handle",
      "cuvsCagraIndexGetDims: host indices are allowed",
      [&](auto& idx) { *dim = idx.dim(); });
  });
}

extern "C" cuvsError_t cuvsCagraIndexGetSize(cuvsCagraIndex_t index, int64_t* size)
{
  return cuvs::core::translate_exceptions([=] {
    auto* box = reinterpret_cast<sg_cagra_c_api_index_box*>(index->addr);
    with_index_by_layout<float, uint32_t, true>(
      box,
      "cuvsCagraIndexGetSize: null index handle",
      "cuvsCagraIndexGetSize: host indices are allowed",
      [&](auto& idx) { *size = idx.size(); });
  });
}

extern "C" cuvsError_t cuvsCagraIndexGetGraphDegree(cuvsCagraIndex_t index, int64_t* graph_degree)
{
  return cuvs::core::translate_exceptions([=] {
    auto* box = reinterpret_cast<sg_cagra_c_api_index_box*>(index->addr);
    with_index_by_layout<float, uint32_t, true>(
      box,
      "cuvsCagraIndexGetGraphDegree: null index handle",
      "cuvsCagraIndexGetGraphDegree: host indices are allowed",
      [&](auto& idx) { *graph_degree = idx.graph_degree(); });
  });
}

extern "C" cuvsError_t cuvsCagraIndexGetDataset(cuvsCagraIndex_t index, DLManagedTensor* dataset)
{
  return cuvs::core::translate_exceptions([=] {
    if (index->dtype.code == kDLFloat && index->dtype.bits == 32) {
      get_dataset_view<float, uint32_t>(index, dataset);
    } else if (index->dtype.code == kDLFloat && index->dtype.bits == 16) {
      get_dataset_view<half, uint32_t>(index, dataset);
    } else if (index->dtype.code == kDLInt && index->dtype.bits == 8) {
      get_dataset_view<int8_t, uint32_t>(index, dataset);
    } else if (index->dtype.code == kDLUInt && index->dtype.bits == 8) {
      get_dataset_view<uint8_t, uint32_t>(index, dataset);
    } else {
      RAFT_FAIL("Unsupported index dtype: %d and bits: %d", index->dtype.code, index->dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsCagraIndexGetGraph(cuvsCagraIndex_t index, DLManagedTensor* graph)
{
  return cuvs::core::translate_exceptions([=] {
    if (index->dtype.code == kDLFloat && index->dtype.bits == 32) {
      get_graph_view<float, uint32_t>(index, graph);
    } else if (index->dtype.code == kDLFloat && index->dtype.bits == 16) {
      get_graph_view<half, uint32_t>(index, graph);
    } else if (index->dtype.code == kDLInt && index->dtype.bits == 8) {
      get_graph_view<int8_t, uint32_t>(index, graph);
    } else if (index->dtype.code == kDLUInt && index->dtype.bits == 8) {
      get_graph_view<uint8_t, uint32_t>(index, graph);
    } else {
      RAFT_FAIL("Unsupported index dtype: %d and bits: %d", index->dtype.code, index->dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsDatasetMakeDevicePadded(cuvsResources_t res,
                                                   DLManagedTensor* dataset_tensor,
                                                   cuvsDatasetPadded_t* padded_dataset)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset  = dataset_tensor->dl_tensor;
    auto* res_ptr = reinterpret_cast<raft::resources*>(res);
    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      make_device_padded_dataset<float>(res_ptr, dataset_tensor, padded_dataset);
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 16) {
      make_device_padded_dataset<half>(res_ptr, dataset_tensor, padded_dataset);
    } else if (dataset.dtype.code == kDLInt && dataset.dtype.bits == 8) {
      make_device_padded_dataset<int8_t>(res_ptr, dataset_tensor, padded_dataset);
    } else if (dataset.dtype.code == kDLUInt && dataset.dtype.bits == 8) {
      make_device_padded_dataset<uint8_t>(res_ptr, dataset_tensor, padded_dataset);
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsDatasetMakeHostPadded(cuvsResources_t res,
                                                 DLManagedTensor* dataset_tensor,
                                                 cuvsDatasetPadded_t* padded_dataset)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset  = dataset_tensor->dl_tensor;
    auto* res_ptr = reinterpret_cast<raft::resources*>(res);
    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      make_host_padded_dataset<float>(res_ptr, dataset_tensor, padded_dataset);
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 16) {
      make_host_padded_dataset<half>(res_ptr, dataset_tensor, padded_dataset);
    } else if (dataset.dtype.code == kDLInt && dataset.dtype.bits == 8) {
      make_host_padded_dataset<int8_t>(res_ptr, dataset_tensor, padded_dataset);
    } else if (dataset.dtype.code == kDLUInt && dataset.dtype.bits == 8) {
      make_host_padded_dataset<uint8_t>(res_ptr, dataset_tensor, padded_dataset);
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsDatasetMakeDevicePaddedView(cuvsResources_t res,
                                                       DLManagedTensor* dataset_tensor,
                                                       cuvsDatasetPaddedView_t* padded_dataset)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset  = dataset_tensor->dl_tensor;
    auto* res_ptr = reinterpret_cast<raft::resources*>(res);
    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      make_device_padded_dataset_view<float>(res_ptr, dataset_tensor, padded_dataset);
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 16) {
      make_device_padded_dataset_view<half>(res_ptr, dataset_tensor, padded_dataset);
    } else if (dataset.dtype.code == kDLInt && dataset.dtype.bits == 8) {
      make_device_padded_dataset_view<int8_t>(res_ptr, dataset_tensor, padded_dataset);
    } else if (dataset.dtype.code == kDLUInt && dataset.dtype.bits == 8) {
      make_device_padded_dataset_view<uint8_t>(res_ptr, dataset_tensor, padded_dataset);
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsDatasetMakeHostPaddedView(cuvsResources_t res,
                                                     DLManagedTensor* dataset_tensor,
                                                     cuvsDatasetPaddedView_t* padded_dataset)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset  = dataset_tensor->dl_tensor;
    auto* res_ptr = reinterpret_cast<raft::resources*>(res);
    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      make_host_padded_dataset_view<float>(res_ptr, dataset_tensor, padded_dataset);
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 16) {
      make_host_padded_dataset_view<half>(res_ptr, dataset_tensor, padded_dataset);
    } else if (dataset.dtype.code == kDLInt && dataset.dtype.bits == 8) {
      make_host_padded_dataset_view<int8_t>(res_ptr, dataset_tensor, padded_dataset);
    } else if (dataset.dtype.code == kDLUInt && dataset.dtype.bits == 8) {
      make_host_padded_dataset_view<uint8_t>(res_ptr, dataset_tensor, padded_dataset);
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsDatasetMakeViewFromOwningPadded(
  cuvsDatasetPadded_t padded_dataset, cuvsDatasetPaddedView_t* padded_view)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(padded_dataset != nullptr,
                 "cuvsDatasetMakeViewFromOwningPadded: null padded dataset");
    RAFT_EXPECTS(padded_view != nullptr,
                 "cuvsDatasetMakeViewFromOwningPadded: null output padded view");
    RAFT_EXPECTS(padded_dataset->layout == CUVS_DATASET_LAYOUT_PADDED,
                 "cuvsDatasetMakeViewFromOwningPadded: input dataset must be padded");
    auto dtype = padded_dataset->dtype;
    if (dtype.code == kDLFloat && dtype.bits == 32) {
      make_view_from_owning_padded<float>(padded_dataset, padded_view);
    } else if (dtype.code == kDLFloat && dtype.bits == 16) {
      make_view_from_owning_padded<half>(padded_dataset, padded_view);
    } else if (dtype.code == kDLInt && dtype.bits == 8) {
      make_view_from_owning_padded<int8_t>(padded_dataset, padded_view);
    } else if (dtype.code == kDLUInt && dtype.bits == 8) {
      make_view_from_owning_padded<uint8_t>(padded_dataset, padded_view);
    } else {
      RAFT_FAIL("Unsupported dataset dtype: code=%d, bits=%d", dtype.code, dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsDatasetPaddedDestroy(cuvsDatasetPadded_t padded_dataset)
{
  return cuvs::core::translate_exceptions([=] {
    if (padded_dataset == nullptr) { return; }
    RAFT_EXPECTS(padded_dataset->layout == CUVS_DATASET_LAYOUT_PADDED,
                 "cuvsDatasetPaddedDestroy: dataset handle layout must be PADDED");
    if (padded_dataset->destroy_addr != nullptr && padded_dataset->addr != 0) {
      padded_dataset->destroy_addr(reinterpret_cast<void*>(padded_dataset->addr));
    }
    delete padded_dataset;
  });
}

extern "C" cuvsError_t cuvsDatasetStandardDestroy(cuvsDatasetStandard_t standard_dataset)
{
  return cuvs::core::translate_exceptions([=] {
    if (standard_dataset == nullptr) { return; }
    RAFT_EXPECTS(standard_dataset->layout == CUVS_DATASET_LAYOUT_STANDARD,
                 "cuvsDatasetStandardDestroy: dataset handle layout must be STANDARD");
    if (standard_dataset->destroy_addr != nullptr && standard_dataset->addr != 0) {
      standard_dataset->destroy_addr(reinterpret_cast<void*>(standard_dataset->addr));
    }
    delete standard_dataset;
  });
}

extern "C" cuvsError_t cuvsDatasetPaddedViewDestroy(cuvsDatasetPaddedView_t padded_dataset)
{
  return cuvs::core::translate_exceptions([=] {
    if (padded_dataset == nullptr) { return; }
    RAFT_EXPECTS(padded_dataset->layout == CUVS_DATASET_LAYOUT_PADDED,
                 "cuvsDatasetPaddedViewDestroy: dataset handle layout must be PADDED");
    if (padded_dataset->destroy_addr != nullptr && padded_dataset->addr != 0) {
      padded_dataset->destroy_addr(reinterpret_cast<void*>(padded_dataset->addr));
    }
    delete padded_dataset;
  });
}

extern "C" cuvsError_t cuvsDatasetMakeDeviceStandardView(cuvsResources_t res,
                                                         DLManagedTensor* dataset_tensor,
                                                         cuvsDatasetStandardView_t* standard_dataset)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset  = dataset_tensor->dl_tensor;
    auto* res_ptr = reinterpret_cast<raft::resources*>(res);
    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      make_device_standard_dataset_view<float>(res_ptr, dataset_tensor, standard_dataset);
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 16) {
      make_device_standard_dataset_view<half>(res_ptr, dataset_tensor, standard_dataset);
    } else if (dataset.dtype.code == kDLInt && dataset.dtype.bits == 8) {
      make_device_standard_dataset_view<int8_t>(res_ptr, dataset_tensor, standard_dataset);
    } else if (dataset.dtype.code == kDLUInt && dataset.dtype.bits == 8) {
      make_device_standard_dataset_view<uint8_t>(res_ptr, dataset_tensor, standard_dataset);
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsDatasetMakeHostStandardView(cuvsResources_t res,
                                                       DLManagedTensor* dataset_tensor,
                                                       cuvsDatasetStandardView_t* standard_dataset)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset  = dataset_tensor->dl_tensor;
    auto* res_ptr = reinterpret_cast<raft::resources*>(res);
    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      make_host_standard_dataset_view<float>(res_ptr, dataset_tensor, standard_dataset);
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 16) {
      make_host_standard_dataset_view<half>(res_ptr, dataset_tensor, standard_dataset);
    } else if (dataset.dtype.code == kDLInt && dataset.dtype.bits == 8) {
      make_host_standard_dataset_view<int8_t>(res_ptr, dataset_tensor, standard_dataset);
    } else if (dataset.dtype.code == kDLUInt && dataset.dtype.bits == 8) {
      make_host_standard_dataset_view<uint8_t>(res_ptr, dataset_tensor, standard_dataset);
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsDatasetStandardViewDestroy(cuvsDatasetStandardView_t standard_dataset)
{
  return cuvs::core::translate_exceptions([=] {
    if (standard_dataset == nullptr) { return; }
    RAFT_EXPECTS(standard_dataset->layout == CUVS_DATASET_LAYOUT_STANDARD,
                 "cuvsDatasetStandardViewDestroy: dataset handle layout must be STANDARD");
    if (standard_dataset->destroy_addr != nullptr && standard_dataset->addr != 0) {
      standard_dataset->destroy_addr(reinterpret_cast<void*>(standard_dataset->addr));
    }
    delete standard_dataset;
  });
}

extern "C" cuvsError_t cuvsCagraAttachPaddedDatasetForSearch(cuvsResources_t res,
                                                              cuvsDatasetPaddedView_t padded_dataset,
                                                              cuvsCagraIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    auto* res_ptr = reinterpret_cast<raft::resources*>(res);
    RAFT_EXPECTS(padded_dataset != nullptr,
                 "cuvsCagraAttachPaddedDatasetForSearch: null padded dataset");
    RAFT_EXPECTS(index != nullptr, "cuvsCagraAttachPaddedDatasetForSearch: null index handle");
    RAFT_EXPECTS(padded_dataset->layout == CUVS_DATASET_LAYOUT_PADDED,
                 "cuvsCagraAttachPaddedDatasetForSearch: dataset handle layout must be PADDED");
    RAFT_EXPECTS(index->dtype.code == padded_dataset->dtype.code &&
                   index->dtype.bits == padded_dataset->dtype.bits,
                 "cuvsCagraAttachPaddedDatasetForSearch: dtype mismatch between index and padded dataset");
    if (index->dtype.code == kDLFloat && index->dtype.bits == 32) {
      attach_padded_dataset_for_search<float>(res_ptr, padded_dataset, index);
    } else if (index->dtype.code == kDLFloat && index->dtype.bits == 16) {
      attach_padded_dataset_for_search<half>(res_ptr, padded_dataset, index);
    } else if (index->dtype.code == kDLInt && index->dtype.bits == 8) {
      attach_padded_dataset_for_search<int8_t>(res_ptr, padded_dataset, index);
    } else if (index->dtype.code == kDLUInt && index->dtype.bits == 8) {
      attach_padded_dataset_for_search<uint8_t>(res_ptr, padded_dataset, index);
    } else {
      RAFT_FAIL("Unsupported index dtype: %d and bits: %d", index->dtype.code, index->dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsCagraAttachDeviceDatasetOnHostIndex(cuvsResources_t res,
                                                                DLManagedTensor* device_dataset_tensor,
                                                                cuvsCagraIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    auto* res_ptr = reinterpret_cast<raft::resources*>(res);
    RAFT_EXPECTS(index != nullptr, "cuvsCagraAttachDeviceDatasetOnHostIndex: null index handle");
    RAFT_EXPECTS(device_dataset_tensor != nullptr,
                 "cuvsCagraAttachDeviceDatasetOnHostIndex: null dataset tensor");
    if (index->dtype.code == kDLFloat && index->dtype.bits == 32) {
      attach_device_dataset_on_host_index<float>(res_ptr, device_dataset_tensor, index);
    } else if (index->dtype.code == kDLFloat && index->dtype.bits == 16) {
      attach_device_dataset_on_host_index<half>(res_ptr, device_dataset_tensor, index);
    } else if (index->dtype.code == kDLInt && index->dtype.bits == 8) {
      attach_device_dataset_on_host_index<int8_t>(res_ptr, device_dataset_tensor, index);
    } else if (index->dtype.code == kDLUInt && index->dtype.bits == 8) {
      attach_device_dataset_on_host_index<uint8_t>(res_ptr, device_dataset_tensor, index);
    } else {
      RAFT_FAIL("Unsupported index dtype: %d and bits: %d", index->dtype.code, index->dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsMakeExtendedStorage(cuvsResources_t res,
                                               cuvsDatasetPaddedView_t additional_dataset,
                                               cuvsCagraIndex_t index,
                                               cuvsDatasetStorage_t* extended_storage)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(additional_dataset != nullptr,
                 "cuvsMakeExtendedStorage: null additional dataset view");
    RAFT_EXPECTS(index != nullptr, "cuvsMakeExtendedStorage: null index handle");
    RAFT_EXPECTS(additional_dataset->dtype.code == index->dtype.code &&
                   additional_dataset->dtype.bits == index->dtype.bits,
                 "cuvsMakeExtendedStorage: dtype mismatch between index and additional dataset "
                 "view");
    if (index->dtype.code == kDLFloat && index->dtype.bits == 32) {
      make_extended_storage<float>(reinterpret_cast<raft::resources*>(res),
                                   additional_dataset,
                                   index,
                                   extended_storage);
    } else if (index->dtype.code == kDLFloat && index->dtype.bits == 16) {
      make_extended_storage<half>(reinterpret_cast<raft::resources*>(res),
                                  additional_dataset,
                                  index,
                                  extended_storage);
    } else if (index->dtype.code == kDLInt && index->dtype.bits == 8) {
      make_extended_storage<int8_t>(reinterpret_cast<raft::resources*>(res),
                                    additional_dataset,
                                    index,
                                    extended_storage);
    } else if (index->dtype.code == kDLUInt && index->dtype.bits == 8) {
      make_extended_storage<uint8_t>(reinterpret_cast<raft::resources*>(res),
                                     additional_dataset,
                                     index,
                                     extended_storage);
    } else {
      RAFT_FAIL("Unsupported index dtype: %d and bits: %d", index->dtype.code, index->dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsDatasetStorageDestroy(cuvsDatasetStorage_t dataset_storage)
{
  return cuvs::core::translate_exceptions([=] {
    if (dataset_storage == nullptr) { return; }
    RAFT_EXPECTS(dataset_storage->kind == CUVS_DATASET_STORAGE_KIND_EXTENDED ||
                   dataset_storage->kind == CUVS_DATASET_STORAGE_KIND_MERGED,
                 "cuvsDatasetStorageDestroy: unsupported storage handle kind");
    if (dataset_storage->addr != 0) {
      if (dataset_storage->dtype.code == kDLFloat && dataset_storage->dtype.bits == 32) {
        if (dataset_storage->kind == CUVS_DATASET_STORAGE_KIND_EXTENDED) {
          destroy_extended_dataset_typed<float>(dataset_storage->addr);
        } else {
          destroy_merged_dataset_typed<float>(dataset_storage->addr);
        }
      } else if (dataset_storage->dtype.code == kDLFloat && dataset_storage->dtype.bits == 16) {
        if (dataset_storage->kind == CUVS_DATASET_STORAGE_KIND_EXTENDED) {
          destroy_extended_dataset_typed<half>(dataset_storage->addr);
        } else {
          destroy_merged_dataset_typed<half>(dataset_storage->addr);
        }
      } else if (dataset_storage->dtype.code == kDLInt && dataset_storage->dtype.bits == 8) {
        if (dataset_storage->kind == CUVS_DATASET_STORAGE_KIND_EXTENDED) {
          destroy_extended_dataset_typed<int8_t>(dataset_storage->addr);
        } else {
          destroy_merged_dataset_typed<int8_t>(dataset_storage->addr);
        }
      } else if (dataset_storage->dtype.code == kDLUInt && dataset_storage->dtype.bits == 8) {
        if (dataset_storage->kind == CUVS_DATASET_STORAGE_KIND_EXTENDED) {
          destroy_extended_dataset_typed<uint8_t>(dataset_storage->addr);
        } else {
          destroy_merged_dataset_typed<uint8_t>(dataset_storage->addr);
        }
      } else {
        RAFT_FAIL("Unsupported dataset storage dtype: %d and bits: %d",
                  dataset_storage->dtype.code,
                  dataset_storage->dtype.bits);
      }
    }
    delete dataset_storage;
  });
}

extern "C" cuvsError_t cuvsMakeMergedStorage(cuvsResources_t res,
                                             cuvsCagraIndex_t* indices,
                                             size_t num_indices,
                                             cuvsFilter filter,
                                             cuvsDatasetStorage_t* merged_storage)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(indices != nullptr && num_indices > 0,
                 "cuvsMakeMergedStorage: indices array cannot be null or empty");
    auto dtype = (*indices[0]).dtype;
    for (size_t i = 1; i < num_indices; ++i) {
      RAFT_EXPECTS((*indices[i]).dtype.code == dtype.code && (*indices[i]).dtype.bits == dtype.bits,
                   "All input indices must have the same data type");
    }
    if (dtype.code == kDLFloat && dtype.bits == 32) {
      _make_merged_storage<float>(res, indices, num_indices, filter, merged_storage);
    } else if (dtype.code == kDLFloat && dtype.bits == 16) {
      _make_merged_storage<half>(res, indices, num_indices, filter, merged_storage);
    } else if (dtype.code == kDLInt && dtype.bits == 8) {
      _make_merged_storage<int8_t>(res, indices, num_indices, filter, merged_storage);
    } else if (dtype.code == kDLUInt && dtype.bits == 8) {
      _make_merged_storage<uint8_t>(res, indices, num_indices, filter, merged_storage);
    } else {
      RAFT_FAIL("Unsupported index data type: code=%d, bits=%d", dtype.code, dtype.bits);
    }
  });
}

template <typename T, typename DatasetViewT, typename IndexT>
static void build_index_from_dataset_view(cuvsResources_t res,
                                          cuvsCagraIndexParams_t params,
                                          uintptr_t dataset_view_addr,
                                          cuvsCagraIndex_t index)
{
  auto* res_ptr       = reinterpret_cast<raft::resources*>(res);
  auto const& ds_view = *reinterpret_cast<DatasetViewT const*>(dataset_view_addr);
  auto index_params   = cuvs::neighbors::cagra::index_params();
  convert_c_index_params(*params,
                         static_cast<int64_t>(ds_view.n_rows()),
                         static_cast<int64_t>(ds_view.dim()),
                         &index_params);
  auto cpp_index = cuvs::neighbors::cagra::build(*res_ptr, index_params, ds_view);
  auto* raw      = new IndexT(std::move(cpp_index));
  wrap_CPP_index_in_lifetime_holder_and_bind_to_C_index<T, DatasetViewT>(index, index->dtype, raw);
}

template <typename T>
static cuvsDatasetViewKind_t get_dataset_view_kind_for_t(DLManagedTensor* dataset)
{
  if (cuvs::core::is_dlpack_device_compatible(dataset->dl_tensor)) {
    using mdspan_type = raft::device_matrix_view<T const, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset);
    return cuvs::neighbors::matrix_row_width_matches_cagra_required(mds)
             ? CUVS_DATASET_VIEW_KIND_DEVICE_PADDED
             : CUVS_DATASET_VIEW_KIND_DEVICE_STANDARD;
  } else if (cuvs::core::is_dlpack_host_compatible(dataset->dl_tensor)) {
    using mdspan_type = raft::host_matrix_view<T const, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset);
    return cuvs::neighbors::matrix_row_width_matches_cagra_required(mds)
             ? CUVS_DATASET_VIEW_KIND_HOST_PADDED
             : CUVS_DATASET_VIEW_KIND_HOST_STANDARD;
  }
  RAFT_FAIL("cuvsCagraGetDatasetViewKind: unsupported dataset device type: %d",
            static_cast<int>(dataset->dl_tensor.device.device_type));
}

extern "C" cuvsError_t cuvsCagraGetDatasetViewKind(DLManagedTensor* dataset,
                                                   cuvsDatasetViewKind_t* kind)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(dataset != nullptr, "cuvsCagraGetDatasetViewKind: null dataset tensor");
    RAFT_EXPECTS(kind != nullptr, "cuvsCagraGetDatasetViewKind: null output kind");
    auto const dtype = dataset->dl_tensor.dtype;
    if (dtype.code == kDLFloat && dtype.bits == 32) {
      *kind = get_dataset_view_kind_for_t<float>(dataset);
    } else if (dtype.code == kDLFloat && dtype.bits == 16) {
      *kind = get_dataset_view_kind_for_t<half>(dataset);
    } else if (dtype.code == kDLInt && dtype.bits == 8) {
      *kind = get_dataset_view_kind_for_t<int8_t>(dataset);
    } else if (dtype.code == kDLUInt && dtype.bits == 8) {
      *kind = get_dataset_view_kind_for_t<uint8_t>(dataset);
    } else {
      RAFT_FAIL("cuvsCagraGetDatasetViewKind: unsupported dataset dtype: code=%d, bits=%d",
                dtype.code,
                dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsCagraBuildDevicePadded(cuvsResources_t res,
                                                  cuvsCagraIndexParams_t params,
                                                  cuvsDatasetPaddedView_t dataset_view,
                                                  cuvsCagraIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(dataset_view != nullptr, "cuvsCagraBuildDevicePadded: null dataset view handle");
    RAFT_EXPECTS(dataset_view->addr != 0, "cuvsCagraBuildDevicePadded: null dataset view storage");
    RAFT_EXPECTS(dataset_view->kind == CUVS_DATASET_VIEW_KIND_DEVICE_PADDED,
                 "cuvsCagraBuildDevicePadded: dataset view must be device-padded");
    destroy_sg_cagra_c_api_box(index->addr);
    index->addr  = 0;
    index->dtype = dataset_view->dtype;
    if (dataset_view->dtype.code == kDLFloat && dataset_view->dtype.bits == 32) {
      build_index_from_dataset_view<float,
                                    cuvs::neighbors::device_padded_dataset_view<float, int64_t>,
                                    cuvs::neighbors::cagra::device_padded_index<float, uint32_t>>(
        res, params, dataset_view->addr, index);
    } else if (dataset_view->dtype.code == kDLFloat && dataset_view->dtype.bits == 16) {
      build_index_from_dataset_view<half,
                                    cuvs::neighbors::device_padded_dataset_view<half, int64_t>,
                                    cuvs::neighbors::cagra::device_padded_index<half, uint32_t>>(
        res, params, dataset_view->addr, index);
    } else if (dataset_view->dtype.code == kDLInt && dataset_view->dtype.bits == 8) {
      build_index_from_dataset_view<int8_t,
                                    cuvs::neighbors::device_padded_dataset_view<int8_t, int64_t>,
                                    cuvs::neighbors::cagra::device_padded_index<int8_t, uint32_t>>(
        res, params, dataset_view->addr, index);
    } else if (dataset_view->dtype.code == kDLUInt && dataset_view->dtype.bits == 8) {
      build_index_from_dataset_view<uint8_t,
                                    cuvs::neighbors::device_padded_dataset_view<uint8_t, int64_t>,
                                    cuvs::neighbors::cagra::device_padded_index<uint8_t, uint32_t>>(
        res, params, dataset_view->addr, index);
    } else {
      RAFT_FAIL("Unsupported dataset dtype: code=%d, bits=%d",
                dataset_view->dtype.code,
                dataset_view->dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsCagraBuildDeviceStandard(cuvsResources_t res,
                                                    cuvsCagraIndexParams_t params,
                                                    cuvsDatasetStandardView_t dataset_view,
                                                    cuvsCagraIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(dataset_view != nullptr, "cuvsCagraBuildDeviceStandard: null dataset view handle");
    RAFT_EXPECTS(dataset_view->addr != 0, "cuvsCagraBuildDeviceStandard: null dataset view storage");
    RAFT_EXPECTS(dataset_view->kind == CUVS_DATASET_VIEW_KIND_DEVICE_STANDARD,
                 "cuvsCagraBuildDeviceStandard: dataset view must be device-standard");
    destroy_sg_cagra_c_api_box(index->addr);
    index->addr  = 0;
    index->dtype = dataset_view->dtype;
    if (dataset_view->dtype.code == kDLFloat && dataset_view->dtype.bits == 32) {
      build_index_from_dataset_view<float,
                                    cuvs::neighbors::device_standard_dataset_view<float, int64_t>,
                                    cuvs::neighbors::cagra::device_standard_index<float, uint32_t>>(
        res, params, dataset_view->addr, index);
    } else if (dataset_view->dtype.code == kDLFloat && dataset_view->dtype.bits == 16) {
      build_index_from_dataset_view<half,
                                    cuvs::neighbors::device_standard_dataset_view<half, int64_t>,
                                    cuvs::neighbors::cagra::device_standard_index<half, uint32_t>>(
        res, params, dataset_view->addr, index);
    } else if (dataset_view->dtype.code == kDLInt && dataset_view->dtype.bits == 8) {
      build_index_from_dataset_view<int8_t,
                                    cuvs::neighbors::device_standard_dataset_view<int8_t, int64_t>,
                                    cuvs::neighbors::cagra::device_standard_index<int8_t, uint32_t>>(
        res, params, dataset_view->addr, index);
    } else if (dataset_view->dtype.code == kDLUInt && dataset_view->dtype.bits == 8) {
      build_index_from_dataset_view<uint8_t,
                                    cuvs::neighbors::device_standard_dataset_view<uint8_t, int64_t>,
                                    cuvs::neighbors::cagra::device_standard_index<uint8_t, uint32_t>>(
        res, params, dataset_view->addr, index);
    } else {
      RAFT_FAIL("Unsupported dataset dtype: code=%d, bits=%d",
                dataset_view->dtype.code,
                dataset_view->dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsCagraBuildHostPadded(cuvsResources_t res,
                                                cuvsCagraIndexParams_t params,
                                                cuvsDatasetPaddedView_t dataset_view,
                                                cuvsCagraIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(dataset_view != nullptr, "cuvsCagraBuildHostPadded: null dataset view handle");
    RAFT_EXPECTS(dataset_view->addr != 0, "cuvsCagraBuildHostPadded: null dataset view storage");
    RAFT_EXPECTS(dataset_view->kind == CUVS_DATASET_VIEW_KIND_HOST_PADDED,
                 "cuvsCagraBuildHostPadded: dataset view must be host-padded");
    destroy_sg_cagra_c_api_box(index->addr);
    index->addr  = 0;
    index->dtype = dataset_view->dtype;
    if (dataset_view->dtype.code == kDLFloat && dataset_view->dtype.bits == 32) {
      build_index_from_dataset_view<float,
                                    cuvs::neighbors::host_padded_dataset_view<float, int64_t>,
                                    cuvs::neighbors::cagra::host_padded_index<float, uint32_t>>(
        res, params, dataset_view->addr, index);
    } else if (dataset_view->dtype.code == kDLFloat && dataset_view->dtype.bits == 16) {
      build_index_from_dataset_view<half,
                                    cuvs::neighbors::host_padded_dataset_view<half, int64_t>,
                                    cuvs::neighbors::cagra::host_padded_index<half, uint32_t>>(
        res, params, dataset_view->addr, index);
    } else if (dataset_view->dtype.code == kDLInt && dataset_view->dtype.bits == 8) {
      build_index_from_dataset_view<int8_t,
                                    cuvs::neighbors::host_padded_dataset_view<int8_t, int64_t>,
                                    cuvs::neighbors::cagra::host_padded_index<int8_t, uint32_t>>(
        res, params, dataset_view->addr, index);
    } else if (dataset_view->dtype.code == kDLUInt && dataset_view->dtype.bits == 8) {
      build_index_from_dataset_view<uint8_t,
                                    cuvs::neighbors::host_padded_dataset_view<uint8_t, int64_t>,
                                    cuvs::neighbors::cagra::host_padded_index<uint8_t, uint32_t>>(
        res, params, dataset_view->addr, index);
    } else {
      RAFT_FAIL("Unsupported dataset dtype: code=%d, bits=%d",
                dataset_view->dtype.code,
                dataset_view->dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsCagraBuildHostStandard(cuvsResources_t res,
                                                  cuvsCagraIndexParams_t params,
                                                  cuvsDatasetStandardView_t dataset_view,
                                                  cuvsCagraIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(dataset_view != nullptr, "cuvsCagraBuildHostStandard: null dataset view handle");
    RAFT_EXPECTS(dataset_view->addr != 0, "cuvsCagraBuildHostStandard: null dataset view storage");
    RAFT_EXPECTS(dataset_view->kind == CUVS_DATASET_VIEW_KIND_HOST_STANDARD,
                 "cuvsCagraBuildHostStandard: dataset view must be host-standard");
    destroy_sg_cagra_c_api_box(index->addr);
    index->addr  = 0;
    index->dtype = dataset_view->dtype;
    if (dataset_view->dtype.code == kDLFloat && dataset_view->dtype.bits == 32) {
      build_index_from_dataset_view<float,
                                    cuvs::neighbors::host_standard_dataset_view<float, int64_t>,
                                    cuvs::neighbors::cagra::host_standard_index<float, uint32_t>>(
        res, params, dataset_view->addr, index);
    } else if (dataset_view->dtype.code == kDLFloat && dataset_view->dtype.bits == 16) {
      build_index_from_dataset_view<half,
                                    cuvs::neighbors::host_standard_dataset_view<half, int64_t>,
                                    cuvs::neighbors::cagra::host_standard_index<half, uint32_t>>(
        res, params, dataset_view->addr, index);
    } else if (dataset_view->dtype.code == kDLInt && dataset_view->dtype.bits == 8) {
      build_index_from_dataset_view<int8_t,
                                    cuvs::neighbors::host_standard_dataset_view<int8_t, int64_t>,
                                    cuvs::neighbors::cagra::host_standard_index<int8_t, uint32_t>>(
        res, params, dataset_view->addr, index);
    } else if (dataset_view->dtype.code == kDLUInt && dataset_view->dtype.bits == 8) {
      build_index_from_dataset_view<uint8_t,
                                    cuvs::neighbors::host_standard_dataset_view<uint8_t, int64_t>,
                                    cuvs::neighbors::cagra::host_standard_index<uint8_t, uint32_t>>(
        res, params, dataset_view->addr, index);
    } else {
      RAFT_FAIL("Unsupported dataset dtype: code=%d, bits=%d",
                dataset_view->dtype.code,
                dataset_view->dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsCagraIndexFromArgs(cuvsResources_t res,
                                              cuvsDistanceType metric,
                                              DLManagedTensor* graph_tensor,
                                              DLManagedTensor* dataset_tensor,
                                              cuvsCagraIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset = dataset_tensor->dl_tensor;
    destroy_sg_cagra_c_api_box(index->addr);
    index->addr         = 0;
    index->dtype        = dataset.dtype;
    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      _from_args<float>(res, metric, graph_tensor, dataset_tensor, index);
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 16) {
      _from_args<half>(res, metric, graph_tensor, dataset_tensor, index);
    } else if (dataset.dtype.code == kDLInt && dataset.dtype.bits == 8) {
      _from_args<int8_t>(res, metric, graph_tensor, dataset_tensor, index);
    } else if (dataset.dtype.code == kDLUInt && dataset.dtype.bits == 8) {
      _from_args<uint8_t>(res, metric, graph_tensor, dataset_tensor, index);
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsCagraExtend(cuvsResources_t res,
                                       cuvsCagraExtendParams_t params,
                                       cuvsDatasetPaddedView_t additional_dataset,
                                       cuvsDatasetStorage_t extended_dataset,
                                       cuvsCagraIndex_t index_c_ptr)
{
  return cuvs::core::translate_exceptions([=] {
    auto index   = *index_c_ptr;
    RAFT_EXPECTS(additional_dataset != nullptr, "cuvsCagraExtend: null additional dataset view");
    RAFT_EXPECTS(additional_dataset->dtype.code == index.dtype.code &&
                   additional_dataset->dtype.bits == index.dtype.bits,
                 "cuvsCagraExtend: dtype mismatch between index and additional dataset view");
    RAFT_EXPECTS(extended_dataset != nullptr, "cuvsCagraExtend: null extended dataset handle");
    RAFT_EXPECTS(extended_dataset->dtype.code == index.dtype.code &&
                   extended_dataset->dtype.bits == index.dtype.bits,
                 "cuvsCagraExtend: dtype mismatch between index and extended dataset");

    if ((index.dtype.code == kDLFloat) && (index.dtype.bits == 32)) {
      _extend<float>(res, *params, index, additional_dataset, extended_dataset);
    } else if (index.dtype.code == kDLFloat && index.dtype.bits == 16) {
      _extend<half>(res, *params, index, additional_dataset, extended_dataset);
    } else if (index.dtype.code == kDLInt && index.dtype.bits == 8) {
      _extend<int8_t>(res, *params, index, additional_dataset, extended_dataset);
    } else if (index.dtype.code == kDLUInt && index.dtype.bits == 8) {
      _extend<uint8_t>(res, *params, index, additional_dataset, extended_dataset);
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                index.dtype.code,
                index.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsCagraSearch(cuvsResources_t res,
                                       cuvsCagraSearchParams_t params,
                                       cuvsCagraIndex_t index_c_ptr,
                                       DLManagedTensor* queries_tensor,
                                       DLManagedTensor* neighbors_tensor,
                                       DLManagedTensor* distances_tensor,
                                       cuvsFilter filter)
{
  return cuvs::core::translate_exceptions([=] {
    auto queries   = queries_tensor->dl_tensor;
    auto neighbors = neighbors_tensor->dl_tensor;
    auto distances = distances_tensor->dl_tensor;

    RAFT_EXPECTS(cuvs::core::is_dlpack_device_compatible(queries),
                 "queries should have device compatible memory");
    RAFT_EXPECTS(cuvs::core::is_dlpack_device_compatible(neighbors),
                 "neighbors should have device compatible memory");
    RAFT_EXPECTS(cuvs::core::is_dlpack_device_compatible(distances),
                 "distances should have device compatible memory");

    // NB: the dtype of neighbors is checked later in _search function
    RAFT_EXPECTS(distances.dtype.code == kDLFloat && distances.dtype.bits == 32,
                 "distances should be of type float32");

    auto index = *index_c_ptr;
    auto* box  = reinterpret_cast<sg_cagra_c_api_index_box*>(index.addr);
    RAFT_EXPECTS(box != nullptr, "cuvsCagraSearch: null index handle");
    RAFT_EXPECTS(box->layout == sg_cagra_c_api_index_box::dataset_layout::device_padded,
                 "cuvsCagraSearch: index must be device-padded. For standard indices, call "
                 "cuvsCagraAttachPaddedDatasetForSearch first.");
    RAFT_EXPECTS(queries.dtype.code == index.dtype.code, "type mismatch between index and queries");

    if (queries.dtype.code == kDLFloat && queries.dtype.bits == 32) {
      _search<float>(
        res, *params, index, queries_tensor, neighbors_tensor, distances_tensor, filter);
    } else if (queries.dtype.code == kDLFloat && queries.dtype.bits == 16) {
      _search<half>(
        res, *params, index, queries_tensor, neighbors_tensor, distances_tensor, filter);
    } else if (queries.dtype.code == kDLInt && queries.dtype.bits == 8) {
      _search<int8_t>(
        res, *params, index, queries_tensor, neighbors_tensor, distances_tensor, filter);
    } else if (queries.dtype.code == kDLUInt && queries.dtype.bits == 8) {
      _search<uint8_t>(
        res, *params, index, queries_tensor, neighbors_tensor, distances_tensor, filter);
    } else {
      RAFT_FAIL("Unsupported queries DLtensor dtype: %d and bits: %d",
                queries.dtype.code,
                queries.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsCagraMerge(cuvsResources_t res,
                                      cuvsCagraIndexParams_t params,
                                      cuvsCagraIndex_t* indices,
                                      size_t num_indices,
                                      cuvsFilter filter,
                                      cuvsDatasetStorage_t merged_dataset,
                                      cuvsCagraIndex_t output_index)
{
  return cuvs::core::translate_exceptions([=] {
    // Basic checks on inputs
    RAFT_EXPECTS(indices != nullptr && num_indices > 0, "indices array cannot be null or empty");
    RAFT_EXPECTS(params != nullptr, "params cannot be null");

    // Use first index dtype as reference
    auto dtype = (*indices[0]).dtype;
    for (size_t i = 1; i < num_indices; ++i) {
      RAFT_EXPECTS((*indices[i]).dtype.code == dtype.code && (*indices[i]).dtype.bits == dtype.bits,
                   "All input indices must have the same data type");
      RAFT_EXPECTS((*indices[i]).addr != 0, "All input indices must be built (non-empty)");
    }
    RAFT_EXPECTS(output_index != nullptr, "Output index pointer must not be null");
    RAFT_EXPECTS(merged_dataset != nullptr, "Merged dataset storage must not be null");
    RAFT_EXPECTS(merged_dataset->dtype.code == dtype.code && merged_dataset->dtype.bits == dtype.bits,
                 "dtype mismatch between merged dataset storage and indices");
    output_index->dtype = dtype;  // output index type matches inputs
    destroy_sg_cagra_c_api_box(output_index->addr);
    output_index->addr = 0;
    // Dispatch based on data type
    if (dtype.code == kDLFloat && dtype.bits == 32) {
      _merge<float>(res, *params, indices, num_indices, filter, merged_dataset, output_index);
    } else if (dtype.code == kDLFloat && dtype.bits == 16) {
      _merge<half>(res, *params, indices, num_indices, filter, merged_dataset, output_index);
    } else if (dtype.code == kDLInt && dtype.bits == 8) {
      _merge<int8_t>(res, *params, indices, num_indices, filter, merged_dataset, output_index);
    } else if (dtype.code == kDLUInt && dtype.bits == 8) {
      _merge<uint8_t>(res, *params, indices, num_indices, filter, merged_dataset, output_index);
    } else {
      RAFT_FAIL("Unsupported index data type: code=%d, bits=%d", dtype.code, dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsCagraIndexParamsCreate(cuvsCagraIndexParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    *params                       = new cuvsCagraIndexParams{.metric                    = L2Expanded,
                                                             .intermediate_graph_degree = 128,
                                                             .graph_degree              = 64,
                                                             .build_algo                = IVF_PQ,
                                                             .nn_descent_niter          = 20};
    (*params)->graph_build_params = new cuvsIvfPqParams{nullptr, nullptr, 1};
  });
}

extern "C" cuvsError_t cuvsCagraIndexParamsDestroy(cuvsCagraIndexParams_t params)
{
  return cuvs::core::translate_exceptions([=] {
    // Delete graph_build_params based on the build algorithm type
    if (params->graph_build_params != nullptr) {
      switch (params->build_algo) {
      case cuvsCagraGraphBuildAlgo::IVF_PQ:
        delete static_cast<cuvsIvfPqParams *>(params->graph_build_params);
        break;
      case cuvsCagraGraphBuildAlgo::ACE: {
        auto ace_params = static_cast<cuvsAceParams *>(params->graph_build_params);
        // Free the allocated build directory string
        if (ace_params->build_dir) { free(const_cast<char*>(ace_params->build_dir)); }
        delete ace_params;
        break;
      }
      case cuvsCagraGraphBuildAlgo::AUTO_SELECT:
      case cuvsCagraGraphBuildAlgo::NN_DESCENT:
      case cuvsCagraGraphBuildAlgo::ITERATIVE_CAGRA_SEARCH:
        // These algorithms don't have separate parameter structs
        break;
      }
    }
    delete params;
  });
}

extern "C" cuvsError_t cuvsCagraCompressionParamsCreate(cuvsCagraCompressionParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    auto ps = cuvs::neighbors::vpq_params();
    *params =
      new cuvsCagraCompressionParams{.pq_bits                     = ps.pq_bits,
                                     .pq_dim                      = ps.pq_dim,
                                     .vq_n_centers                = ps.vq_n_centers,
                                     .kmeans_n_iters              = ps.kmeans_n_iters,
                                     .vq_kmeans_trainset_fraction = ps.vq_kmeans_trainset_fraction,
                                     .pq_kmeans_trainset_fraction = ps.pq_kmeans_trainset_fraction};
  });
}

extern "C" cuvsError_t cuvsCagraCompressionParamsDestroy(cuvsCagraCompressionParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsAceParamsCreate(cuvsAceParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    auto ps = cuvs::neighbors::cagra::graph_build_params::ace_params();

    // Allocate and copy the build directory string
    const char* build_dir = strdup(ps.build_dir.c_str());

    *params = new cuvsAceParams{.npartitions         = ps.npartitions,
                                .ef_construction     = ps.ef_construction,
                                .build_dir           = build_dir,
                                .use_disk            = ps.use_disk,
                                .max_host_memory_gb  = ps.max_host_memory_gb,
                                .max_gpu_memory_gb   = ps.max_gpu_memory_gb};
  });
}

extern "C" cuvsError_t cuvsAceParamsDestroy(cuvsAceParams_t params)
{
  return cuvs::core::translate_exceptions([=] {
    if (params) {
      // Free the allocated build directory string
      if (params->build_dir) { free(const_cast<char*>(params->build_dir)); }
      delete params;
    }
  });
}

extern "C" cuvsError_t cuvsCagraIndexParamsFromHnswParams(cuvsCagraIndexParams_t params,
                                                           int64_t n_rows,
                                                           int64_t dim,
                                                           int M,
                                                           int ef_construction,
                                                           enum cuvsCagraHnswHeuristicType heuristic,
                                                           cuvsDistanceType metric)
{
  return cuvs::core::translate_exceptions([=] {
    auto cpp_metric = static_cast<cuvs::distance::DistanceType>((int)metric);
    auto cpp_heuristic = static_cast<cuvs::neighbors::cagra::hnsw_heuristic_type>((int)heuristic);
    auto cpp_params = cuvs::neighbors::cagra::index_params::from_hnsw_params(
      raft::matrix_extent<int64_t>(n_rows, dim), M, ef_construction, cpp_heuristic, cpp_metric);

    _populate_cagra_index_params_from_cpp(params, cpp_params);
  });
}

extern "C" cuvsError_t cuvsCagraExtendParamsCreate(cuvsCagraExtendParams_t* params)
{
  return cuvs::core::translate_exceptions(
    [=] { *params = new cuvsCagraExtendParams{.max_chunk_size = 0}; });
}

extern "C" cuvsError_t cuvsCagraExtendParamsDestroy(cuvsCagraExtendParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsCagraSearchParamsCreate(cuvsCagraSearchParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    *params = new cuvsCagraSearchParams{
      .itopk_size              = 64,
      .search_width            = 1,
      .hashmap_max_fill_rate   = 0.5,
      .num_random_samplings    = 1,
      .rand_xor_mask           = 0x128394,
      .persistent              = false,
      .persistent_lifetime     = 2,
      .persistent_device_usage = 1.0,
    };
  });
}

extern "C" cuvsError_t cuvsCagraSearchParamsDestroy(cuvsCagraSearchParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsCagraDeserializePadded(cuvsResources_t res,
                                                  const char* filename,
                                                  cuvsCagraIndex_t index,
                                                  cuvsDatasetPadded_t* out_padded_dataset)
{
  return cuvs::core::translate_exceptions([=] {
    if (out_padded_dataset != nullptr) { *out_padded_dataset = nullptr; }
    destroy_sg_cagra_c_api_box(index->addr);
    index->addr = 0;

    // read the numpy dtype from the beginning of the file
    std::ifstream is(filename, std::ios::in | std::ios::binary);
    if (!is) { RAFT_FAIL("Cannot open file %s", filename); }
    char dtype_string[4]{};
    if (!is.read(dtype_string, sizeof(dtype_string))) {
      RAFT_FAIL("Invalid or truncated index header in file %s", filename);
    }
    auto dtype =
      raft::numpy_serializer::parse_descr(std::string(dtype_string, sizeof(dtype_string)));
    is.close();

    index->dtype.bits = dtype.itemsize * 8;
    if (dtype.kind == 'f' && dtype.itemsize == 4) {
      _deserialize_padded<float>(res, filename, index, out_padded_dataset);
      index->dtype.code = kDLFloat;
    } else if (dtype.kind == 'e' && dtype.itemsize == 2) {
      _deserialize_padded<half>(res, filename, index, out_padded_dataset);
      index->dtype.code = kDLFloat;
    } else if (dtype.kind == 'i' && dtype.itemsize == 1) {
      _deserialize_padded<int8_t>(res, filename, index, out_padded_dataset);
      index->dtype.code = kDLInt;
    } else if (dtype.kind == 'u' && dtype.itemsize == 1) {
      _deserialize_padded<uint8_t>(res, filename, index, out_padded_dataset);
      index->dtype.code = kDLUInt;
    } else {
      RAFT_FAIL("Unsupported dtype in file %s", filename);
    }
  });
}

extern "C" cuvsError_t cuvsCagraDeserializeStandard(cuvsResources_t res,
                                                    const char* filename,
                                                    cuvsCagraIndex_t index,
                                                    cuvsDatasetStandard_t* out_standard_dataset)
{
  return cuvs::core::translate_exceptions([=] {
    if (out_standard_dataset != nullptr) { *out_standard_dataset = nullptr; }
    destroy_sg_cagra_c_api_box(index->addr);
    index->addr = 0;

    std::ifstream is(filename, std::ios::in | std::ios::binary);
    if (!is) { RAFT_FAIL("Cannot open file %s", filename); }
    char dtype_string[4]{};
    if (!is.read(dtype_string, sizeof(dtype_string))) {
      RAFT_FAIL("Invalid or truncated index header in file %s", filename);
    }
    auto dtype =
      raft::numpy_serializer::parse_descr(std::string(dtype_string, sizeof(dtype_string)));
    is.close();

    index->dtype.bits = dtype.itemsize * 8;
    if (dtype.kind == 'f' && dtype.itemsize == 4) {
      _deserialize_standard<float>(res, filename, index, out_standard_dataset);
      index->dtype.code = kDLFloat;
    } else if (dtype.kind == 'e' && dtype.itemsize == 2) {
      _deserialize_standard<half>(res, filename, index, out_standard_dataset);
      index->dtype.code = kDLFloat;
    } else if (dtype.kind == 'i' && dtype.itemsize == 1) {
      _deserialize_standard<int8_t>(res, filename, index, out_standard_dataset);
      index->dtype.code = kDLInt;
    } else if (dtype.kind == 'u' && dtype.itemsize == 1) {
      _deserialize_standard<uint8_t>(res, filename, index, out_standard_dataset);
      index->dtype.code = kDLUInt;
    } else {
      RAFT_FAIL("Unsupported dtype in file %s", filename);
    }
  });
}

extern "C" cuvsError_t cuvsCagraSerialize(cuvsResources_t res,
                                          const char* filename,
                                          cuvsCagraIndex_t index,
                                          bool include_dataset)
{
  return cuvs::core::translate_exceptions([=] {
    if (index->dtype.code == kDLFloat && index->dtype.bits == 32) {
      _serialize<float>(res, filename, index, include_dataset);
    } else if (index->dtype.code == kDLFloat && index->dtype.bits == 16) {
      _serialize<half>(res, filename, index, include_dataset);
    } else if (index->dtype.code == kDLInt && index->dtype.bits == 8) {
      _serialize<int8_t>(res, filename, index, include_dataset);
    } else if (index->dtype.code == kDLUInt && index->dtype.bits == 8) {
      _serialize<uint8_t>(res, filename, index, include_dataset);
    } else {
      RAFT_FAIL("Unsupported index dtype: %d and bits: %d", index->dtype.code, index->dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsCagraSerializeToHnswlib(cuvsResources_t res,
                                                   const char* filename,
                                                   cuvsCagraIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    if (index->dtype.code == kDLFloat && index->dtype.bits == 32) {
      _serialize_to_hnswlib<float>(res, filename, index);
    } else if (index->dtype.code == kDLFloat && index->dtype.bits == 16) {
      _serialize_to_hnswlib<half>(res, filename, index);
    } else if (index->dtype.code == kDLInt && index->dtype.bits == 8) {
      _serialize_to_hnswlib<int8_t>(res, filename, index);
    } else if (index->dtype.code == kDLUInt && index->dtype.bits == 8) {
      _serialize_to_hnswlib<uint8_t>(res, filename, index);
    } else {
      RAFT_FAIL("Unsupported index dtype: %d and bits: %d", index->dtype.code, index->dtype.bits);
    }
  });
}
