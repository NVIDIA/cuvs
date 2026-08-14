/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
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
#include <raft/core/numpy_serializer.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/numpy_serializer.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/serialize.hpp>

#include "../core/exceptions.hpp"
#include "../core/interop.hpp"
#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.h>
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/common.h>
#include <cuvs/neighbors/cagra.hpp>
#include "../core/exceptions.hpp"
#include "../core/interop.hpp"

#include "c_api_box.hpp"
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

template <typename T>
static void destroy_typed_addr(void* ptr);

template <typename T, cuvs::neighbors::ann_dataset_view DatasetViewT>
static void merge_indices_for_layout(
  raft::resources* res_ptr,
  cuvs::neighbors::cagra::index_params const& params_cpp,
  std::vector<cuvs::neighbors::cagra::index<T, uint32_t, DatasetViewT>*>& index_ptrs,
  cuvsFilter filter,
  cuvsDataset_t merged_dataset,
  cuvsCagraIndex_t output_index)
{
  RAFT_EXPECTS(merged_dataset != nullptr, "cuvsCagraMerge: null merged dataset handle");
  RAFT_EXPECTS(merged_dataset->addr == 0,
               "cuvsCagraMerge: merged dataset handle must be empty");

  constexpr auto output_layout =
    cuvs::neighbors::is_padded_dataset_view_v<DatasetViewT> ? CUVS_DATASET_LAYOUT_PADDED
                                                            : CUVS_DATASET_LAYOUT_STANDARD;

  int64_t merged_row_count = 0;
  for (auto* idx_ptr : index_ptrs) {
    merged_row_count += static_cast<int64_t>(idx_ptr->size());
  }

  auto merge_into_dataset = [&](auto const& row_filter) {
    auto const final_row_count =
      cuvs::neighbors::cagra::detail::merged_dataset_size<T, uint32_t, DatasetViewT>(
        *res_ptr, index_ptrs, row_filter);
    auto const dim    = static_cast<uint32_t>(index_ptrs.front()->dim());
    auto const stride = static_cast<int64_t>(index_ptrs.front()->dataset().stride());

    try {
      auto matrix = raft::make_device_matrix<T, int64_t>(*res_ptr, final_row_count, stride);
      using owner_t = cuvs::neighbors::owning_dataset_for_view_t<DatasetViewT>;
      auto owner    = std::make_unique<owner_t>(std::move(matrix), dim);
      auto view     = owner->as_dataset_view();
      auto merged_idx =
        cuvs::neighbors::cagra::merge(*res_ptr, params_cpp, index_ptrs, view, row_filter);
      auto* holder =
        new cuvs_cagra_c_api_index_lifetime_holder<T, DatasetViewT>{std::move(merged_idx)};
      bind_index_lifetime_holder_to_C_index<T, DatasetViewT>(
        output_index, output_index->dtype, holder);

      merged_dataset->addr         = reinterpret_cast<uintptr_t>(owner.release());
      merged_dataset->destroy_addr = &destroy_typed_addr<owner_t>;
      merged_dataset->dtype        = output_index->dtype;
      merged_dataset->mem_type     = CUVS_DATASET_MEM_TYPE_DEVICE;
      merged_dataset->layout       = output_layout;
      merged_dataset->is_owning    = true;
      return;
    } catch (std::bad_alloc const&) {
      // Filtered merge gathers rows with device-only primitives, matching the restriction on the
      // legacy host fallback.
      RAFT_EXPECTS(filter.type == NO_FILTER,
                   "Filtered merge isn't available with the host-memory OOM fallback");
      RAFT_LOG_DEBUG("cagra::merge: device allocation failed; using host memory for merged dataset");
    }

    using host_view_t = std::conditional_t<
      cuvs::neighbors::is_padded_dataset_view_v<DatasetViewT>,
      cuvs::neighbors::host_padded_dataset_view<T, int64_t>,
      cuvs::neighbors::host_standard_dataset_view<T, int64_t>>;
    using host_owner_t = cuvs::neighbors::owning_dataset_for_view_t<host_view_t>;

    auto matrix = raft::make_host_matrix<T, int64_t>(final_row_count, stride);
    std::fill_n(matrix.data_handle(), static_cast<std::size_t>(matrix.size()), T{});

    std::size_t row_offset = 0;
    auto stream            = raft::resource::get_cuda_stream(*res_ptr);
    for (auto* index : index_ptrs) {
      auto const& input = index->dataset();
      raft::copy_matrix(matrix.data_handle() + row_offset * static_cast<std::size_t>(stride),
                        static_cast<std::size_t>(stride),
                        input.view().data_handle(),
                        static_cast<std::size_t>(input.stride()),
                        static_cast<std::size_t>(dim),
                        static_cast<std::size_t>(input.n_rows()),
                        stream);
      row_offset += static_cast<std::size_t>(input.n_rows());
    }
    raft::resource::sync_stream(*res_ptr);

    auto owner      = std::make_unique<host_owner_t>(std::move(matrix), dim);
    auto view       = owner->as_dataset_view();
    auto merged_idx = cuvs::neighbors::cagra::build(*res_ptr, params_cpp, view);
    auto* holder =
      new cuvs_cagra_c_api_index_lifetime_holder<T, host_view_t>{std::move(merged_idx)};
    bind_index_lifetime_holder_to_C_index<T, host_view_t>(
      output_index, output_index->dtype, holder);

    merged_dataset->addr         = reinterpret_cast<uintptr_t>(owner.release());
    merged_dataset->destroy_addr = &destroy_typed_addr<host_owner_t>;
    merged_dataset->dtype        = output_index->dtype;
    merged_dataset->mem_type     = CUVS_DATASET_MEM_TYPE_HOST;
    merged_dataset->layout       = output_layout;
    merged_dataset->is_owning    = true;
  };

  if (filter.type == NO_FILTER) {
    merge_into_dataset(cuvs::neighbors::filtering::none_sample_filter{});
  } else if (filter.type == BITSET) {
    using filter_mdspan_type = raft::device_vector_view<std::uint32_t, int64_t, raft::row_major>;
    auto removed_indices_tensor = reinterpret_cast<DLManagedTensor*>(filter.addr);
    auto removed_indices = cuvs::core::from_dlpack<filter_mdspan_type>(removed_indices_tensor);
    cuvs::core::bitset_view<std::uint32_t, int64_t> removed_indices_bitset(
      removed_indices, merged_row_count);
    auto bitset_filter_obj =
      cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>(removed_indices_bitset);
    merge_into_dataset(bitset_filter_obj);
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
static auto make_sg_cagra_c_api_index_box(
  cuvs_cagra_c_api_index_lifetime_holder<T, DatasetViewT>* holder)
  -> std::unique_ptr<sg_cagra_c_api_index_box>
{
  return std::make_unique<sg_cagra_c_api_index_box>(
    sg_cagra_c_api_index_box{&holder->idx,
                             sg_cagra_index_layout_from_view<DatasetViewT>(),
                             cuvs::neighbors::c_api::detail::make_owner_record(holder)});
}

template <typename T, cuvs::neighbors::ann_dataset_view DatasetViewT>
static void bind_index_lifetime_holder_to_C_index(
  cuvsCagraIndex_t out,
  DLDataType dtype,
  cuvs_cagra_c_api_index_lifetime_holder<T, DatasetViewT>* holder)
{
  auto box   = make_sg_cagra_c_api_index_box<T, DatasetViewT>(holder);
  out->addr  = reinterpret_cast<uintptr_t>(box.release());
  out->dtype = dtype;
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

template <typename OwnerT, typename ViewT, typename Fn>
static void with_dataset_view(cuvsDataset_t dataset, Fn&& fn)
{
  RAFT_EXPECTS(dataset != nullptr, "null dataset handle");
  RAFT_EXPECTS(dataset->addr != 0, "null dataset storage");
  if (dataset->is_owning) {
    auto* owner = reinterpret_cast<OwnerT*>(dataset->addr);
    auto view   = owner->as_dataset_view();
    fn(view);
  } else {
    fn(*reinterpret_cast<ViewT*>(dataset->addr));
  }
}

template <typename T>
static void make_device_padded_dataset(raft::resources* res_ptr,
                                       DLManagedTensor* dataset_tensor,
                                       cuvsDataset_t* output_padded_dataset)
{
  auto dataset = dataset_tensor->dl_tensor;
  using owner_type = cuvs::neighbors::device_padded_dataset<T, int64_t>;
  std::unique_ptr<owner_type> owner;
  if (cuvs::core::is_dlpack_device_compatible(dataset)) {
    using mdspan_type = raft::device_matrix_view<T const, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
    owner             = cuvs::neighbors::make_device_padded_dataset(*res_ptr, mds);
  } else if (cuvs::core::is_dlpack_host_compatible(dataset)) {
    using mdspan_type = raft::host_matrix_view<T const, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
    owner             = cuvs::neighbors::make_device_padded_dataset(*res_ptr, mds);
  } else {
    RAFT_FAIL("cuvsDatasetMakePadded: unsupported source tensor memory type");
  }
  auto* out  = new cuvsDataset{};
  out->addr  = reinterpret_cast<uintptr_t>(owner.release());
  out->destroy_addr = &destroy_typed_addr<owner_type>;
  out->dtype    = dataset.dtype;
  out->mem_type = CUVS_DATASET_MEM_TYPE_DEVICE;
  out->layout   = CUVS_DATASET_LAYOUT_PADDED;
  out->is_owning = true;
  *output_padded_dataset = out;
}

template <typename T>
static void make_host_padded_dataset(raft::resources* res_ptr,
                                     DLManagedTensor* dataset_tensor,
                                     cuvsDataset_t* output_padded_dataset)
{
  auto dataset = dataset_tensor->dl_tensor;
  using owner_type = cuvs::neighbors::host_padded_dataset<T, int64_t>;
  std::unique_ptr<owner_type> owner;
  if (cuvs::core::is_dlpack_host_compatible(dataset)) {
    using mdspan_type = raft::host_matrix_view<T const, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
    owner             = cuvs::neighbors::make_host_padded_dataset(*res_ptr, mds);
  } else if (cuvs::core::is_dlpack_device_compatible(dataset)) {
    using mdspan_type = raft::device_matrix_view<T const, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
    owner             = cuvs::neighbors::make_host_padded_dataset(*res_ptr, mds);
  } else {
    RAFT_FAIL("cuvsDatasetMakePadded: unsupported source tensor memory type");
  }
  auto* out  = new cuvsDataset{};
  out->addr  = reinterpret_cast<uintptr_t>(owner.release());
  out->destroy_addr = &destroy_typed_addr<owner_type>;
  out->dtype    = dataset.dtype;
  out->mem_type = CUVS_DATASET_MEM_TYPE_HOST;
  out->layout   = CUVS_DATASET_LAYOUT_PADDED;
  out->is_owning = true;
  *output_padded_dataset = out;
}

template <typename T>
static void make_device_padded_dataset_view(raft::resources* res_ptr,
                                            DLManagedTensor* dataset_tensor,
                                            cuvsDataset_t* output_padded_dataset)
{
  auto dataset = dataset_tensor->dl_tensor;
  auto* out    = new cuvsDataset{};
  if (!cuvs::core::is_dlpack_device_compatible(dataset)) {
    delete out;
    RAFT_FAIL("cuvsDatasetMakePaddedView: dataset must have device-compatible memory");
  }
  using mdspan_type = raft::device_matrix_view<T const, int64_t, raft::row_major>;
  auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
  auto ds_view      = cuvs::neighbors::make_device_padded_dataset_view(*res_ptr, mds);
  auto* owned_view = new decltype(ds_view){ds_view};
  out->addr        = reinterpret_cast<uintptr_t>(owned_view);
  out->destroy_addr = &destroy_typed_addr<decltype(ds_view)>;
  out->dtype        = dataset.dtype;
  out->mem_type     = CUVS_DATASET_MEM_TYPE_DEVICE;
  out->layout       = CUVS_DATASET_LAYOUT_PADDED;
  out->is_owning    = false;
  *output_padded_dataset = out;
}

template <typename T>
static void make_host_padded_dataset_view(raft::resources*,
                                          DLManagedTensor* dataset_tensor,
                                          cuvsDataset_t* output_padded_dataset)
{
  auto dataset = dataset_tensor->dl_tensor;
  auto* out    = new cuvsDataset{};
  if (!cuvs::core::is_dlpack_host_compatible(dataset)) {
    delete out;
    RAFT_FAIL("cuvsDatasetMakePaddedView: dataset must have host-compatible memory");
  }
  using mdspan_type = raft::host_matrix_view<T const, int64_t, raft::row_major>;
  auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
  auto ds_view      = cuvs::neighbors::make_host_padded_dataset_view(mds);
  auto* owned_view = new decltype(ds_view){ds_view};
  out->addr        = reinterpret_cast<uintptr_t>(owned_view);
  out->destroy_addr = &destroy_typed_addr<decltype(ds_view)>;
  out->dtype        = dataset.dtype;
  out->mem_type     = CUVS_DATASET_MEM_TYPE_HOST;
  out->layout       = CUVS_DATASET_LAYOUT_PADDED;
  out->is_owning    = false;
  *output_padded_dataset = out;
}

template <typename T>
static void make_device_standard_dataset_view(raft::resources*,
                                              DLManagedTensor* dataset_tensor,
                                              cuvsDataset_t* output_standard_dataset)
{
  auto dataset = dataset_tensor->dl_tensor;
  auto* out    = new cuvsDataset{};
  if (!cuvs::core::is_dlpack_device_compatible(dataset)) {
    delete out;
    RAFT_FAIL("cuvsDatasetMakeStandardView: dataset must have device-compatible memory");
  }
  using mdspan_type = raft::device_matrix_view<T const, int64_t, raft::row_major>;
  auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
  auto ds_view      = cuvs::neighbors::make_device_standard_dataset_view(mds);
  auto* owned_view = new decltype(ds_view){ds_view};
  out->addr        = reinterpret_cast<uintptr_t>(owned_view);
  out->destroy_addr = &destroy_typed_addr<decltype(ds_view)>;
  out->dtype        = dataset.dtype;
  out->mem_type     = CUVS_DATASET_MEM_TYPE_DEVICE;
  out->layout       = CUVS_DATASET_LAYOUT_STANDARD;
  out->is_owning    = false;
  *output_standard_dataset = out;
}

template <typename T>
static void make_host_standard_dataset_view(raft::resources*,
                                            DLManagedTensor* dataset_tensor,
                                            cuvsDataset_t* output_standard_dataset)
{
  auto dataset = dataset_tensor->dl_tensor;
  auto* out    = new cuvsDataset{};
  if (!cuvs::core::is_dlpack_host_compatible(dataset)) {
    delete out;
    RAFT_FAIL("cuvsDatasetMakeStandardView: dataset must have host-compatible memory");
  }
  using mdspan_type = raft::host_matrix_view<T const, int64_t, raft::row_major>;
  auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
  auto ds_view      = cuvs::neighbors::make_host_standard_dataset_view(mds);
  auto* owned_view = new decltype(ds_view){ds_view};
  out->addr        = reinterpret_cast<uintptr_t>(owned_view);
  out->destroy_addr = &destroy_typed_addr<decltype(ds_view)>;
  out->dtype        = dataset.dtype;
  out->mem_type     = CUVS_DATASET_MEM_TYPE_HOST;
  out->layout       = CUVS_DATASET_LAYOUT_STANDARD;
  out->is_owning    = false;
  *output_standard_dataset = out;
}

template <typename T>
static void attach_dataset(raft::resources* res_ptr,
                           cuvsDataset_t device_padded_dataset,
                           cuvsCagraIndex_t index)
{
  RAFT_EXPECTS(device_padded_dataset != nullptr, "cuvsCagraUpdateDataset: null padded dataset");
  RAFT_EXPECTS(index != nullptr, "cuvsCagraUpdateDataset: null index handle");
  RAFT_EXPECTS(index->addr != 0, "cuvsCagraUpdateDataset: null index storage");
  RAFT_EXPECTS(device_padded_dataset->addr != 0,
               "cuvsCagraUpdateDataset: null padded dataset storage");

  auto* box = reinterpret_cast<sg_cagra_c_api_index_box*>(index->addr);
  RAFT_EXPECTS(device_padded_dataset->mem_type == CUVS_DATASET_MEM_TYPE_DEVICE &&
                 device_padded_dataset->layout == CUVS_DATASET_LAYOUT_PADDED,
               "cuvsCagraAttachDataset: dataset must be device padded");

  using owner_t = cuvs::neighbors::device_padded_dataset<T, int64_t>;
  using view_t  = cuvs::neighbors::device_padded_dataset_view<T, int64_t>;
  with_dataset_view<owner_t, view_t>(device_padded_dataset, [&](auto const& padded_view) {
    with_index_by_layout<T, uint32_t, true>(
      box,
      "cuvsCagraUpdateDataset: null index handle",
      "cuvsCagraUpdateDataset: host index layout is allowed for this operation",
      [&](auto& idx) {
        auto padded_idx = cuvs::neighbors::cagra::attach_dataset(*res_ptr, idx, padded_view);
        auto* holder =
          new cuvs_cagra_c_api_index_lifetime_holder<T, view_t>{std::move(padded_idx)};
        destroy_sg_cagra_c_api_box(index->addr);
        index->addr = 0;
        bind_index_lifetime_holder_to_C_index<T, view_t>(index, index->dtype, holder);
      });
  });
}

template <typename T>
static void update_device_dataset_same_layout(raft::resources* res_ptr,
                                              cuvsDataset_t device_dataset,
                                              cuvsCagraIndex_t index)
{
  RAFT_EXPECTS(device_dataset != nullptr, "cuvsCagraUpdateDataset: null dataset");
  RAFT_EXPECTS(index != nullptr, "cuvsCagraUpdateDataset: null index handle");
  RAFT_EXPECTS(index->addr != 0, "cuvsCagraUpdateDataset: null index storage");
  RAFT_EXPECTS(device_dataset->addr != 0, "cuvsCagraUpdateDataset: null dataset storage");

  auto* box = reinterpret_cast<sg_cagra_c_api_index_box*>(index->addr);
  if (box->layout == sg_cagra_c_api_index_box::dataset_layout::device_padded) {
    RAFT_EXPECTS(device_dataset->mem_type == CUVS_DATASET_MEM_TYPE_DEVICE &&
                   device_dataset->layout == CUVS_DATASET_LAYOUT_PADDED,
                 "cuvsCagraUpdateDeviceDatasetSameLayout: device-padded index "
                 "requires a "
                 "device-padded dataset");
    using owner_t = cuvs::neighbors::device_padded_dataset<T, int64_t>;
    using view_t  = cuvs::neighbors::device_padded_dataset_view<T, int64_t>;
    with_dataset_view<owner_t, view_t>(device_dataset, [&](auto const& dataset_view) {
      auto* idx =
        reinterpret_cast<cuvs::neighbors::cagra::device_padded_index<T, uint32_t>*>(box->index_ptr);
      RAFT_EXPECTS(idx != nullptr, "cuvsCagraUpdateDataset: null index handle");
      idx->update_device_dataset_same_layout(*res_ptr, dataset_view);
    });
  } else if (box->layout == sg_cagra_c_api_index_box::dataset_layout::device_standard) {
    RAFT_EXPECTS(device_dataset->mem_type == CUVS_DATASET_MEM_TYPE_DEVICE &&
                   device_dataset->layout == CUVS_DATASET_LAYOUT_STANDARD,
                 "cuvsCagraUpdateDeviceDatasetSameLayout: device-standard "
                 "index requires a "
                 "device-standard dataset");
    using owner_t = cuvs::neighbors::device_standard_dataset<T, int64_t>;
    using view_t  = cuvs::neighbors::device_standard_dataset_view<T, int64_t>;
    with_dataset_view<owner_t, view_t>(device_dataset, [&](auto const& dataset_view) {
      auto* idx =
        reinterpret_cast<cuvs::neighbors::cagra::device_standard_index<T, uint32_t>*>(box->index_ptr);
      RAFT_EXPECTS(idx != nullptr, "cuvsCagraUpdateDataset: null index handle");
      idx->update_device_dataset_same_layout(*res_ptr, dataset_view);
    });
  } else {
    RAFT_FAIL(
      "cuvsCagraUpdateDataset: C++ "
      "update_device_dataset_same_layout "
      "requires a device index and dataset");
  }
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
      raw->update_device_dataset_same_layout(*res_ptr, dataset_view);
      update_graph_from_dlpack(raw);
      wrap_CPP_index_in_lifetime_holder_and_bind_to_C_index<
        T,
        cuvs::neighbors::device_padded_dataset_view<T, int64_t>>(
        output_index, output_index->dtype, raw);
    } else {
      auto dataset_view = cuvs::neighbors::make_device_standard_dataset_view(mds);
      auto* raw         = new cuvs::neighbors::cagra::device_standard_index<T, uint32_t>(
        *res_ptr, metric);
      raw->update_device_dataset_same_layout(*res_ptr, dataset_view);
      update_graph_from_dlpack(raw);
      wrap_CPP_index_in_lifetime_holder_and_bind_to_C_index<
        T,
        cuvs::neighbors::device_standard_dataset_view<T, int64_t>>(
        output_index, output_index->dtype, raw);
    }
  } else if (cuvs::core::is_dlpack_host_compatible(dataset)) {
    RAFT_FAIL("cuvsCagraIndexFromArgs: host dataset is unsupported; use cuvsCagraBuild for host "
              "datasets, then attach a matching device dataset view via "
              "cuvsCagraUpdateDataset with a device padded dataset view.");
  }
}

template <typename T>
void _extend(cuvsResources_t res,
             cuvsCagraExtendParams params,
             cuvsCagraIndex index,
             cuvsDataset_t extended_dataset,
             int64_t new_start_row)
{
  auto* box      = reinterpret_cast<sg_cagra_c_api_index_box*>(index.addr);
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  RAFT_EXPECTS(box != nullptr, "cuvsCagraExtend: null index handle");
  RAFT_EXPECTS(
    box->layout == sg_cagra_c_api_index_box::dataset_layout::device_padded,
    "cuvsCagraExtend: only device_padded indices are extendable. "
    "For standard indices, explicitly create/attach a padded dataset first.");
  RAFT_EXPECTS(extended_dataset != nullptr, "cuvsCagraExtend: null extended dataset view");
  RAFT_EXPECTS(extended_dataset->mem_type == CUVS_DATASET_MEM_TYPE_DEVICE &&
                 extended_dataset->layout == CUVS_DATASET_LAYOUT_PADDED,
               "cuvsCagraExtend: extended dataset must be a device-padded dataset view");
  RAFT_EXPECTS(extended_dataset->addr != 0,
               "cuvsCagraExtend: null extended dataset storage");
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
        using out_owner_t = cuvs::neighbors::device_padded_dataset<T, int64_t>;
        using out_view_t  = cuvs::neighbors::device_padded_dataset_view<T, int64_t>;
        with_dataset_view<out_owner_t, out_view_t>(extended_dataset, [&](auto& out_dataset) {
          cuvs::neighbors::cagra::extend(
            *res_ptr, extend_params, out_dataset, new_start_row, idx);
        });
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
    "cuvsCagraUpdateDataset with a device padded dataset view",
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

template <typename T, typename OutIdxT>
void _search_multi_partition(cuvsResources_t res,
                             cuvsCagraSearchParams params,
                             uint32_t num_partitions,
                             cuvsCagraIndex_t* indices,
                             DLManagedTensor* queries,
                             DLManagedTensor* partition_ids,
                             DLManagedTensor* neighbors,
                             DLManagedTensor* distances,
                             cuvsFilter* filters)
{
  using IdxT      = uint32_t;
  using DistanceT = float;
  using IndexT    = cuvs::neighbors::cagra::index<T, IdxT>;

  auto res_ptr       = reinterpret_cast<raft::resources*>(res);
  auto search_params = cuvs::neighbors::cagra::search_params();
  convert_c_search_params(params, &search_params);

  std::vector<const IndexT*> idx_vec(num_partitions);
  for (uint32_t i = 0; i < num_partitions; i++) {
    RAFT_EXPECTS(indices[i] != nullptr && indices[i]->addr != 0,
                 "cuvsCagraSearchMultiPartition: null index handle (partition %u)",
                 i);
    auto* box = reinterpret_cast<sg_cagra_c_api_index_box*>(indices[i]->addr);
    RAFT_EXPECTS(box->layout == sg_cagra_c_api_index_box::dataset_layout::device_padded,
                 "cuvsCagraSearchMultiPartition: every partition must hold a device padded "
                 "dataset; attach one via cuvsCagraUpdateDataset (partition %u)",
                 i);
    idx_vec[i] = reinterpret_cast<const IndexT*>(box->index_ptr);
  }

  using queries_view_t = raft::device_matrix_view<const T, int64_t, raft::row_major>;
  using pid_view_t     = raft::device_matrix_view<uint32_t, int64_t, raft::row_major>;
  using nbrs_view_t    = raft::device_matrix_view<OutIdxT, int64_t, raft::row_major>;
  using dist_view_t    = raft::device_matrix_view<DistanceT, int64_t, raft::row_major>;

  auto queries_view       = cuvs::core::from_dlpack<queries_view_t>(queries);
  auto partition_ids_view = cuvs::core::from_dlpack<pid_view_t>(partition_ids);
  auto neighbors_view     = cuvs::core::from_dlpack<nbrs_view_t>(neighbors);
  auto distances_view     = cuvs::core::from_dlpack<dist_view_t>(distances);

  // One bitset view per partition; an empty view (null ptr) means "no filter for that partition".
  using bitset_view_t   = cuvs::core::bitset_view<std::uint32_t, int64_t>;
  using bitset_mdspan_t = raft::device_vector_view<std::uint32_t, int64_t, raft::row_major>;
  std::vector<bitset_view_t> partition_bitsets(
    num_partitions, bitset_view_t(static_cast<std::uint32_t*>(nullptr), static_cast<int64_t>(0)));
  if (filters != nullptr) {
    for (uint32_t i = 0; i < num_partitions; i++) {
      if (filters[i].type == NO_FILTER) {
        continue;  // leave the empty (accept-all) view for this partition
      } else if (filters[i].type == BITSET) {
        auto* bitset_tensor = reinterpret_cast<DLManagedTensor*>(filters[i].addr);
        RAFT_EXPECTS(
          bitset_tensor != nullptr, "BITSET filter addr must be non-null (partition %u)", i);
        auto bitset_mds = cuvs::core::from_dlpack<bitset_mdspan_t>(bitset_tensor);
        partition_bitsets[i] =
          bitset_view_t(bitset_mds, static_cast<int64_t>(bitset_mds.size()) * 32);
      } else {
        RAFT_FAIL("Unsupported filter type for multi-partition search (partition %u): %d",
                  i,
                  (int)filters[i].type);
      }
    }
  }

  cuvs::neighbors::cagra::search(*res_ptr,
                                 search_params,
                                 idx_vec,
                                 queries_view,
                                 partition_ids_view,
                                 neighbors_view,
                                 distances_view,
                                 partition_bitsets);
}

template <typename T>
void _search_multi_partition(cuvsResources_t res,
                             cuvsCagraSearchParams params,
                             uint32_t num_partitions,
                             cuvsCagraIndex_t* indices,
                             DLManagedTensor* queries,
                             DLManagedTensor* partition_ids,
                             DLManagedTensor* neighbors,
                             DLManagedTensor* distances,
                             cuvsFilter* filters)
{
  if (neighbors->dl_tensor.dtype.code == kDLUInt && neighbors->dl_tensor.dtype.bits == 32) {
    _search_multi_partition<T, uint32_t>(
      res, params, num_partitions, indices, queries, partition_ids, neighbors, distances, filters);
  } else if (neighbors->dl_tensor.dtype.code == kDLInt && neighbors->dl_tensor.dtype.bits == 64) {
    _search_multi_partition<T, int64_t>(
      res, params, num_partitions, indices, queries, partition_ids, neighbors, distances, filters);
  } else {
    RAFT_FAIL("neighbors should be of type uint32_t or int64_t");
  }
}

template <typename T>
void _serialize(cuvsResources_t res, const char *filename,
                cuvsCagraIndex_t index, bool include_dataset) {
  auto res_ptr = reinterpret_cast<raft::resources *>(res);
  auto *box = reinterpret_cast<sg_cagra_c_api_index_box *>(index->addr);
  auto const *null_handle_err =
      include_dataset ? "cuvsCagraSerializeGraphAndDataset: null index handle"
                      : "cuvsCagraSerializeGraph: null index handle";
  with_index_by_layout<T, uint32_t,
                       true>(box, null_handle_err, "", [&](auto &idx) {
    if (include_dataset) {
      RAFT_EXPECTS(
          idx.dataset().n_rows() > 0,
          "cuvsCagraSerializeGraphAndDataset: index has no attached dataset");
    }
    cuvs::neighbors::cagra::serialize(*res_ptr, std::string(filename), idx,
                                      include_dataset);
  });
}

struct serialized_cagra_header {
  DLDataType dtype;
  cuvs::neighbors::cagra::serialized_dataset_kind dataset_kind;
};

static auto read_serialized_header(cuvsResources_t res, const char *filename)
    -> serialized_cagra_header {
  auto res_ptr = reinterpret_cast<raft::resources *>(res);
  std::ifstream is(filename, std::ios::in | std::ios::binary);
  if (!is) {
    RAFT_FAIL("Cannot open file %s", filename);
  }

  char dtype_string[4]{};
  if (!is.read(dtype_string, sizeof(dtype_string))) {
    RAFT_FAIL("Invalid or truncated index header in file %s", filename);
  }

  auto const dtype = raft::numpy_serializer::parse_descr(
      std::string(dtype_string, sizeof(dtype_string)));
  DLDataType output_dtype{
      .code = 0, .bits = static_cast<uint8_t>(dtype.itemsize * 8), .lanes = 1};
  if (dtype.kind == 'f' && dtype.itemsize == 4) {
    output_dtype.code = kDLFloat;
  } else if (dtype.kind == 'e' && dtype.itemsize == 2) {
    output_dtype.code = kDLFloat;
  } else if (dtype.kind == 'i' && dtype.itemsize == 1) {
    output_dtype.code = kDLInt;
  } else if (dtype.kind == 'u' && dtype.itemsize == 1) {
    output_dtype.code = kDLUInt;
  } else {
    RAFT_FAIL("Unsupported dtype in file %s", filename);
  }

  auto const version = raft::deserialize_scalar<int>(*res_ptr, is);
  auto const dataset_kind_raw =
    raft::deserialize_scalar<std::uint32_t>(*res_ptr, is);
  RAFT_EXPECTS(
      version == cuvs::neighbors::cagra::cagra_serialization_version,
      "serialization version mismatch, expected %d, got %d",
      cuvs::neighbors::cagra::cagra_serialization_version, version);
  using kind = cuvs::neighbors::cagra::serialized_dataset_kind;
  RAFT_EXPECTS(dataset_kind_raw <= static_cast<std::uint32_t>(kind::host_standard),
               "Invalid serialized dataset kind %u in file %s",
               dataset_kind_raw, filename);
  return {output_dtype, static_cast<kind>(dataset_kind_raw)};
}

template <typename Fn>
void dispatch_serialized_dtype(DLDataType dtype, Fn &&fn) {
  if (dtype.code == kDLFloat && dtype.bits == 32) {
    fn.template operator()<float>();
  } else if (dtype.code == kDLFloat && dtype.bits == 16) {
    fn.template operator()<half>();
  } else if (dtype.code == kDLInt && dtype.bits == 8) {
    fn.template operator()<int8_t>();
  } else if (dtype.code == kDLUInt && dtype.bits == 8) {
    fn.template operator()<uint8_t>();
  } else {
    RAFT_FAIL("Unsupported index dtype: %d and bits: %d", dtype.code,
              dtype.bits);
  }
}

template <typename T, typename Fn>
void dispatch_serialized_dataset_kind(
    cuvs::neighbors::cagra::serialized_dataset_kind kind, Fn &&fn) {
  using serialized_kind = cuvs::neighbors::cagra::serialized_dataset_kind;
  switch (kind) {
    case serialized_kind::device_padded:
      fn.template operator()<
          cuvs::neighbors::device_padded_dataset_view<T, int64_t>>();
      break;
    case serialized_kind::device_standard:
      fn.template operator()<
          cuvs::neighbors::device_standard_dataset_view<T, int64_t>>();
      break;
    case serialized_kind::host_padded:
      fn.template operator()<
          cuvs::neighbors::host_padded_dataset_view<T, int64_t>>();
      break;
    case serialized_kind::host_standard:
      fn.template operator()<
          cuvs::neighbors::host_standard_dataset_view<T, int64_t>>();
      break;
    case serialized_kind::none:
      fn.template operator()<
          cuvs::neighbors::device_padded_dataset_view<T, int64_t>>();
      break;
  }
}

template <typename T, cuvs::neighbors::ann_dataset_view ViewT>
void _deserialize(cuvsResources_t res, const char *filename,
                  cuvsCagraIndex_t output_index, DLDataType dtype,
                  bool include_dataset, cuvsDataset_t *out_dataset) {
  auto res_ptr = reinterpret_cast<raft::resources *>(res);
  using view_t = ViewT;
  using owner_dataset_t = cuvs::neighbors::owning_dataset_for_view_t<view_t>;
  using holder_t = cuvs_cagra_c_api_index_lifetime_holder<T, view_t>;

  auto holder = std::make_unique<holder_t>(
      cuvs::neighbors::cagra::index<T, uint32_t, view_t>(*res_ptr));
  std::unique_ptr<owner_dataset_t> dataset_owner{};
  cuvs::neighbors::cagra::deserialize(*res_ptr, std::string(filename),
                                      &holder->idx,
                                      include_dataset ? &dataset_owner : nullptr);

  if (include_dataset) {
    RAFT_EXPECTS(
        dataset_owner != nullptr,
        "cuvsCagraDeserializeGraphAndDataset: serialized index has no dataset");
  }

  std::unique_ptr<cuvsDataset> dataset_handle{};
  if (include_dataset) {
    dataset_handle = std::make_unique<cuvsDataset>();
    dataset_handle->addr = reinterpret_cast<uintptr_t>(dataset_owner.get());
    dataset_handle->destroy_addr = &destroy_typed_addr<owner_dataset_t>;
    dataset_handle->dtype = dtype;
    dataset_handle->mem_type =
        cuvs::neighbors::is_device_dataset_view_v<view_t>
            ? CUVS_DATASET_MEM_TYPE_DEVICE
            : CUVS_DATASET_MEM_TYPE_HOST;
    dataset_handle->layout =
        cuvs::neighbors::is_padded_dataset_view_v<view_t>
            ? CUVS_DATASET_LAYOUT_PADDED
            : CUVS_DATASET_LAYOUT_STANDARD;
    dataset_handle->is_owning = true;
  }

  auto box = make_sg_cagra_c_api_index_box<T, view_t>(holder.get());
  auto old_addr = output_index->addr;
  output_index->addr = reinterpret_cast<uintptr_t>(box.release());
  output_index->dtype = dtype;
  holder.release();

  if (include_dataset) {
    dataset_handle->addr = reinterpret_cast<uintptr_t>(dataset_owner.release());
    *out_dataset = dataset_handle.release();
  }
  destroy_sg_cagra_c_api_box(old_addr);
}

template <typename T>
void _serialize_to_hnswlib(cuvsResources_t res, const char *filename,
                           cuvsCagraIndex_t index) {
  auto res_ptr = reinterpret_cast<raft::resources *>(res);
  auto *box = reinterpret_cast<sg_cagra_c_api_index_box *>(index->addr);
  with_index_by_layout<T, uint32_t, true>(
      box, "cuvsCagraSerializeToHnswlib: null index handle",
      "cuvsCagraSerializeToHnswlib: host indices are allowed",
      [&](auto &idx) {
        cuvs::neighbors::cagra::serialize_to_hnswlib(
            *res_ptr, std::string(filename), idx);
      });
}
template <typename T>
void _merge(cuvsResources_t res,
            cuvsCagraIndexParams params,
            cuvsCagraIndex_t* indices,
            size_t num_indices,
            cuvsFilter filter,
            cuvsDataset_t merged_dataset,
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
  if (layout == sg_cagra_c_api_index_box::dataset_layout::device_padded) {
    auto index_ptrs =
      convert_opaque_indices_to_concrete_types<T, cuvs::neighbors::device_padded_dataset_view<T, int64_t>>(
        indices, num_indices);
    merge_indices_for_layout<T, cuvs::neighbors::device_padded_dataset_view<T, int64_t>>(
      res_ptr, params_cpp, index_ptrs, filter, merged_dataset, output_index);
  } else {
    auto index_ptrs =
      convert_opaque_indices_to_concrete_types<T, cuvs::neighbors::device_standard_dataset_view<T, int64_t>>(
        indices, num_indices);
    merge_indices_for_layout<T, cuvs::neighbors::device_standard_dataset_view<T, int64_t>>(
      res_ptr, params_cpp, index_ptrs, filter, merged_dataset, output_index);
  }
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

extern "C" cuvsError_t cuvsDatasetCreate(cuvsDataset_t* dataset)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(dataset != nullptr, "cuvsDatasetCreate: output handle must not be null");
    *dataset = new cuvsDataset{0,
                               nullptr,
                               DLDataType{},
                               CUVS_DATASET_MEM_TYPE_DEVICE,
                               CUVS_DATASET_LAYOUT_STANDARD,
                               true};
  });
}

extern "C" cuvsError_t cuvsDatasetMakePadded(cuvsResources_t res,
                                             DLManagedTensor* dataset_tensor,
                                             cuvsDatasetMemType_t target_mem_type,
                                             cuvsDataset_t* padded_dataset)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(dataset_tensor != nullptr, "cuvsDatasetMakePadded: null input tensor");
    RAFT_EXPECTS(padded_dataset != nullptr, "cuvsDatasetMakePadded: null output dataset");
    *padded_dataset = nullptr;
    auto dataset  = dataset_tensor->dl_tensor;
    auto* res_ptr = reinterpret_cast<raft::resources*>(res);
    auto make_typed = [&]<typename T>() {
      switch (target_mem_type) {
        case CUVS_DATASET_MEM_TYPE_DEVICE:
          make_device_padded_dataset<T>(res_ptr, dataset_tensor, padded_dataset);
          break;
        case CUVS_DATASET_MEM_TYPE_HOST:
          make_host_padded_dataset<T>(res_ptr, dataset_tensor, padded_dataset);
          break;
        default: RAFT_FAIL("cuvsDatasetMakePadded: invalid target memory type");
      }
    };

    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      make_typed.template operator()<float>();
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 16) {
      make_typed.template operator()<half>();
    } else if (dataset.dtype.code == kDLInt && dataset.dtype.bits == 8) {
      make_typed.template operator()<int8_t>();
    } else if (dataset.dtype.code == kDLUInt && dataset.dtype.bits == 8) {
      make_typed.template operator()<uint8_t>();
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsDatasetMakePaddedView(cuvsResources_t res,
                                                 DLManagedTensor* dataset_tensor,
                                                 cuvsDataset_t* padded_dataset)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(dataset_tensor != nullptr, "cuvsDatasetMakePaddedView: null input tensor");
    RAFT_EXPECTS(padded_dataset != nullptr, "cuvsDatasetMakePaddedView: null output view");
    *padded_dataset = nullptr;
    auto dataset  = dataset_tensor->dl_tensor;
    auto* res_ptr = reinterpret_cast<raft::resources*>(res);
    auto make_typed = [&]<typename T>() {
      if (cuvs::core::is_dlpack_device_compatible(dataset)) {
        make_device_padded_dataset_view<T>(res_ptr, dataset_tensor, padded_dataset);
      } else if (cuvs::core::is_dlpack_host_compatible(dataset)) {
        make_host_padded_dataset_view<T>(res_ptr, dataset_tensor, padded_dataset);
      } else {
        RAFT_FAIL("cuvsDatasetMakePaddedView: unsupported tensor memory type");
      }
    };

    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      make_typed.template operator()<float>();
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 16) {
      make_typed.template operator()<half>();
    } else if (dataset.dtype.code == kDLInt && dataset.dtype.bits == 8) {
      make_typed.template operator()<int8_t>();
    } else if (dataset.dtype.code == kDLUInt && dataset.dtype.bits == 8) {
      make_typed.template operator()<uint8_t>();
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsDatasetDestroy(cuvsDataset_t dataset)
{
  return cuvs::core::translate_exceptions([=] {
    if (dataset == nullptr) { return; }
    if (dataset->destroy_addr != nullptr && dataset->addr != 0) {
      dataset->destroy_addr(reinterpret_cast<void*>(dataset->addr));
    }
    delete dataset;
  });
}

extern "C" cuvsError_t cuvsDatasetGetMemType(cuvsDataset_t dataset, cuvsDatasetMemType_t* mem_type)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(dataset != nullptr, "cuvsDatasetGetMemType: null dataset");
    RAFT_EXPECTS(mem_type != nullptr, "cuvsDatasetGetMemType: null output");
    *mem_type = dataset->mem_type;
  });
}

extern "C" cuvsError_t cuvsDatasetGetLayout(cuvsDataset_t dataset, cuvsDatasetLayout_t* layout)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(dataset != nullptr, "cuvsDatasetGetLayout: null dataset");
    RAFT_EXPECTS(layout != nullptr, "cuvsDatasetGetLayout: null output");
    *layout = dataset->layout;
  });
}

extern "C" cuvsError_t cuvsDatasetGetIsOwning(cuvsDataset_t dataset, bool* is_owning)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(dataset != nullptr, "cuvsDatasetGetIsOwning: null dataset");
    RAFT_EXPECTS(is_owning != nullptr, "cuvsDatasetGetIsOwning: null output");
    *is_owning = dataset->is_owning;
  });
}

extern "C" cuvsError_t cuvsDatasetGetDtype(cuvsDataset_t dataset, DLDataType* dtype)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(dataset != nullptr, "cuvsDatasetGetDtype: null dataset");
    RAFT_EXPECTS(dtype != nullptr, "cuvsDatasetGetDtype: null output");
    *dtype = dataset->dtype;
  });
}

extern "C" cuvsError_t cuvsDatasetMakeStandardView(cuvsResources_t res,
                                                   DLManagedTensor* dataset_tensor,
                                                   cuvsDataset_t* standard_dataset)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(dataset_tensor != nullptr, "cuvsDatasetMakeStandardView: null input tensor");
    RAFT_EXPECTS(standard_dataset != nullptr, "cuvsDatasetMakeStandardView: null output view");
    *standard_dataset = nullptr;
    auto dataset  = dataset_tensor->dl_tensor;
    auto* res_ptr = reinterpret_cast<raft::resources*>(res);
    auto make_typed = [&]<typename T>() {
      if (cuvs::core::is_dlpack_device_compatible(dataset)) {
        make_device_standard_dataset_view<T>(res_ptr, dataset_tensor, standard_dataset);
      } else if (cuvs::core::is_dlpack_host_compatible(dataset)) {
        make_host_standard_dataset_view<T>(res_ptr, dataset_tensor, standard_dataset);
      } else {
        RAFT_FAIL("cuvsDatasetMakeStandardView: unsupported tensor memory type");
      }
    };

    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      make_typed.template operator()<float>();
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 16) {
      make_typed.template operator()<half>();
    } else if (dataset.dtype.code == kDLInt && dataset.dtype.bits == 8) {
      make_typed.template operator()<int8_t>();
    } else if (dataset.dtype.code == kDLUInt && dataset.dtype.bits == 8) {
      make_typed.template operator()<uint8_t>();
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}

static cuvsError_t dispatch_attach_dataset(cuvsResources_t res,
                                           cuvsDataset_t device_padded_dataset,
                                           cuvsCagraIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    auto* res_ptr = reinterpret_cast<raft::resources*>(res);
    RAFT_EXPECTS(index != nullptr, "cuvsCagraUpdateDataset: null index handle");
    RAFT_EXPECTS(device_padded_dataset != nullptr, "cuvsCagraUpdateDataset: null dataset view");
    RAFT_EXPECTS(device_padded_dataset->layout == CUVS_DATASET_LAYOUT_PADDED,
                 "cuvsCagraUpdateDataset: dataset handle layout must be PADDED");
    RAFT_EXPECTS(index->dtype.code == device_padded_dataset->dtype.code &&
                   index->dtype.bits == device_padded_dataset->dtype.bits,
                 "cuvsCagraUpdateDataset: dtype mismatch between index and dataset");
    if (index->dtype.code == kDLFloat && index->dtype.bits == 32) {
      attach_dataset<float>(res_ptr, device_padded_dataset, index);
    } else if (index->dtype.code == kDLFloat && index->dtype.bits == 16) {
      attach_dataset<half>(res_ptr, device_padded_dataset, index);
    } else if (index->dtype.code == kDLInt && index->dtype.bits == 8) {
      attach_dataset<int8_t>(res_ptr, device_padded_dataset, index);
    } else if (index->dtype.code == kDLUInt && index->dtype.bits == 8) {
      attach_dataset<uint8_t>(res_ptr, device_padded_dataset, index);
    } else {
      RAFT_FAIL("Unsupported index dtype: %d and bits: %d", index->dtype.code, index->dtype.bits);
    }
  });
}

static cuvsError_t dispatch_update_device_dataset_same_layout(cuvsResources_t res,
                                                              cuvsDataset_t device_dataset,
                                                              cuvsCagraIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    auto* res_ptr = reinterpret_cast<raft::resources*>(res);
    RAFT_EXPECTS(index != nullptr, "cuvsCagraUpdateDataset: null index handle");
    RAFT_EXPECTS(device_dataset != nullptr,
                 "cuvsCagraUpdateDataset: null dataset view");
    RAFT_EXPECTS(index->dtype.code == device_dataset->dtype.code &&
                   index->dtype.bits == device_dataset->dtype.bits,
                 "cuvsCagraUpdateDataset: dtype mismatch "
                 "between index and dataset");
    if (index->dtype.code == kDLFloat && index->dtype.bits == 32) {
      update_device_dataset_same_layout<float>(res_ptr, device_dataset, index);
    } else if (index->dtype.code == kDLFloat && index->dtype.bits == 16) {
      update_device_dataset_same_layout<half>(res_ptr, device_dataset, index);
    } else if (index->dtype.code == kDLInt && index->dtype.bits == 8) {
      update_device_dataset_same_layout<int8_t>(res_ptr, device_dataset, index);
    } else if (index->dtype.code == kDLUInt && index->dtype.bits == 8) {
      update_device_dataset_same_layout<uint8_t>(res_ptr, device_dataset, index);
    } else {
      RAFT_FAIL("Unsupported index dtype: %d and bits: %d", index->dtype.code, index->dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsCagraUpdateDataset(cuvsResources_t res,
                                              cuvsDataset_t device_padded_dataset,
                                              cuvsCagraIndex_t index)
{
  auto status = cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(index != nullptr, "cuvsCagraUpdateDataset: null index handle");
    RAFT_EXPECTS(index->addr != 0, "cuvsCagraUpdateDataset: null index storage");
    RAFT_EXPECTS(device_padded_dataset != nullptr, "cuvsCagraUpdateDataset: null dataset view");
    RAFT_EXPECTS(device_padded_dataset->addr != 0,
                 "cuvsCagraUpdateDataset: null dataset view storage");
    RAFT_EXPECTS(device_padded_dataset->mem_type == CUVS_DATASET_MEM_TYPE_DEVICE &&
                   device_padded_dataset->layout == CUVS_DATASET_LAYOUT_PADDED,
                 "cuvsCagraUpdateDataset: dataset view must be device padded");
    RAFT_EXPECTS(index->dtype.code == device_padded_dataset->dtype.code &&
                   index->dtype.bits == device_padded_dataset->dtype.bits,
                 "cuvsCagraUpdateDataset: dtype mismatch between index and dataset");
  });
  if (status != CUVS_SUCCESS) { return status; }

  auto* box = reinterpret_cast<sg_cagra_c_api_index_box*>(index->addr);
  if (box->layout == sg_cagra_c_api_index_box::dataset_layout::device_padded) {
    return dispatch_update_device_dataset_same_layout(res, device_padded_dataset, index);
  }
  return dispatch_attach_dataset(res, device_padded_dataset, index);
}

/**
 * Build from an already-constructed C++ dataset view. `DatasetViewT` selects the
 * `cuvs::neighbors::cagra::build` overload, and therefore the resulting index type.
 */
template <typename T, cuvs::neighbors::ann_dataset_view DatasetViewT>
static void build_index_from_dataset_view(raft::resources* res_ptr,
                                          cuvsCagraIndexParams_t params,
                                          DatasetViewT const& ds_view,
                                          cuvsCagraIndex_t index)
{
  auto index_params = cuvs::neighbors::cagra::index_params();
  convert_c_index_params(*params,
                         static_cast<int64_t>(ds_view.n_rows()),
                         static_cast<int64_t>(ds_view.dim()),
                         &index_params);
  auto cpp_index = cuvs::neighbors::cagra::build(*res_ptr, index_params, ds_view);
  auto* raw      = new decltype(cpp_index)(std::move(cpp_index));
  wrap_CPP_index_in_lifetime_holder_and_bind_to_C_index<T, DatasetViewT>(index, index->dtype, raw);
}

/**
 * Build through the C++ overload matching the memory space and layout the caller's dataset view
 * handle was constructed with.
 */
template <typename T>
static void build_dispatch_on_mem_type_and_layout(raft::resources* res_ptr,
                                                  cuvsCagraIndexParams_t params,
                                                  cuvsDataset_t dataset,
                                                  cuvsCagraIndex_t index)
{
  auto const is_padded = dataset->layout == CUVS_DATASET_LAYOUT_PADDED;

  if (dataset->mem_type == CUVS_DATASET_MEM_TYPE_DEVICE) {
    if (is_padded) {
      using owner_t = cuvs::neighbors::device_padded_dataset<T, int64_t>;
      using view_t = cuvs::neighbors::device_padded_dataset_view<T, int64_t>;
      with_dataset_view<owner_t, view_t>(dataset, [&](auto const& view) {
        build_index_from_dataset_view<T>(res_ptr, params, view, index);
      });
    } else {
      using owner_t = cuvs::neighbors::device_standard_dataset<T, int64_t>;
      using view_t = cuvs::neighbors::device_standard_dataset_view<T, int64_t>;
      with_dataset_view<owner_t, view_t>(dataset, [&](auto const& view) {
        build_index_from_dataset_view<T>(res_ptr, params, view, index);
      });
    }
  } else if (dataset->mem_type == CUVS_DATASET_MEM_TYPE_HOST) {
    if (is_padded) {
      using owner_t = cuvs::neighbors::host_padded_dataset<T, int64_t>;
      using view_t = cuvs::neighbors::host_padded_dataset_view<T, int64_t>;
      with_dataset_view<owner_t, view_t>(dataset, [&](auto const& view) {
        build_index_from_dataset_view<T>(res_ptr, params, view, index);
      });
    } else {
      using owner_t = cuvs::neighbors::host_standard_dataset<T, int64_t>;
      using view_t = cuvs::neighbors::host_standard_dataset_view<T, int64_t>;
      with_dataset_view<owner_t, view_t>(dataset, [&](auto const& view) {
        build_index_from_dataset_view<T>(res_ptr, params, view, index);
      });
    }
  } else {
    RAFT_FAIL("cuvsCagraBuild: invalid dataset memory type");
  }
}

extern "C" cuvsError_t cuvsCagraBuild(cuvsResources_t res,
                                      cuvsCagraIndexParams_t params,
                                      cuvsDataset_t dataset,
                                      cuvsCagraIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(dataset != nullptr, "cuvsCagraBuild: null dataset view");
    RAFT_EXPECTS(dataset->addr != 0, "cuvsCagraBuild: null dataset view storage");
    RAFT_EXPECTS(params != nullptr, "cuvsCagraBuild: null index params");
    RAFT_EXPECTS(index != nullptr, "cuvsCagraBuild: null index handle");

    auto* res_ptr    = reinterpret_cast<raft::resources*>(res);
    auto const dtype = dataset->dtype;

    destroy_sg_cagra_c_api_box(index->addr);
    index->addr  = 0;
    index->dtype = dtype;

    if (dtype.code == kDLFloat && dtype.bits == 32) {
      build_dispatch_on_mem_type_and_layout<float>(res_ptr, params, dataset, index);
    } else if (dtype.code == kDLFloat && dtype.bits == 16) {
      build_dispatch_on_mem_type_and_layout<half>(res_ptr, params, dataset, index);
    } else if (dtype.code == kDLInt && dtype.bits == 8) {
      build_dispatch_on_mem_type_and_layout<int8_t>(res_ptr, params, dataset, index);
    } else if (dtype.code == kDLUInt && dtype.bits == 8) {
      build_dispatch_on_mem_type_and_layout<uint8_t>(res_ptr, params, dataset, index);
    } else {
      RAFT_FAIL(
        "cuvsCagraBuild: unsupported dataset dtype: code=%d, bits=%d", dtype.code, dtype.bits);
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
                                       cuvsDataset_t extended_dataset,
                                       int64_t new_start_row,
                                       cuvsCagraIndex_t index_c_ptr)
{
  return cuvs::core::translate_exceptions([=] {
    auto index   = *index_c_ptr;
    RAFT_EXPECTS(extended_dataset != nullptr, "cuvsCagraExtend: null extended dataset view");
    RAFT_EXPECTS(extended_dataset->dtype.code == index.dtype.code &&
                   extended_dataset->dtype.bits == index.dtype.bits,
                 "cuvsCagraExtend: dtype mismatch between index and extended dataset");

    if ((index.dtype.code == kDLFloat) && (index.dtype.bits == 32)) {
      _extend<float>(res, *params, index, extended_dataset, new_start_row);
    } else if (index.dtype.code == kDLFloat && index.dtype.bits == 16) {
      _extend<half>(res, *params, index, extended_dataset, new_start_row);
    } else if (index.dtype.code == kDLInt && index.dtype.bits == 8) {
      _extend<int8_t>(res, *params, index, extended_dataset, new_start_row);
    } else if (index.dtype.code == kDLUInt && index.dtype.bits == 8) {
      _extend<uint8_t>(res, *params, index, extended_dataset, new_start_row);
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
                 "cuvsCagraUpdateDataset first.");
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

extern "C" cuvsError_t cuvsCagraSearchMultiPartition(cuvsResources_t res,
                                                     cuvsCagraSearchParams_t params,
                                                     uint32_t num_partitions,
                                                     cuvsCagraIndex_t* indices,
                                                     DLManagedTensor* queries,
                                                     DLManagedTensor* partition_ids,
                                                     DLManagedTensor* neighbors,
                                                     DLManagedTensor* distances,
                                                     cuvsFilter* filters)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(num_partitions > 0, "num_partitions must be > 0");
    RAFT_EXPECTS(indices != nullptr && queries != nullptr && partition_ids != nullptr &&
                   neighbors != nullptr && distances != nullptr,
                 "All pointer arguments must be non-null");

    // Every partition index must be present, built, and share one dtype; the search dispatches on
    // that common dtype. The queries dtype is validated against T inside from_dlpack.
    RAFT_EXPECTS(indices[0] != nullptr, "Index at position 0 is null");
    auto index_dtype = indices[0]->dtype;
    for (uint32_t i = 0; i < num_partitions; i++) {
      RAFT_EXPECTS(indices[i] != nullptr && indices[i]->addr != 0,
                   "Index at position %u is null or not built", i);
      RAFT_EXPECTS(
        indices[i]->dtype.code == index_dtype.code && indices[i]->dtype.bits == index_dtype.bits,
        "All partition indices must share the same dtype; position %u differs from position 0", i);
    }

    if (index_dtype.code == kDLFloat && index_dtype.bits == 32) {
      _search_multi_partition<float>(
        res, *params, num_partitions, indices, queries, partition_ids, neighbors, distances, filters);
    } else if (index_dtype.code == kDLFloat && index_dtype.bits == 16) {
      _search_multi_partition<half>(
        res, *params, num_partitions, indices, queries, partition_ids, neighbors, distances, filters);
    } else if (index_dtype.code == kDLInt && index_dtype.bits == 8) {
      _search_multi_partition<int8_t>(
        res, *params, num_partitions, indices, queries, partition_ids, neighbors, distances, filters);
    } else if (index_dtype.code == kDLUInt && index_dtype.bits == 8) {
      _search_multi_partition<uint8_t>(
        res, *params, num_partitions, indices, queries, partition_ids, neighbors, distances, filters);
    } else {
      RAFT_FAIL("Unsupported multi-partition index DLtensor dtype: %d and bits: %d",
                index_dtype.code,
                index_dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsCagraMerge(cuvsResources_t res,
                                      cuvsCagraIndexParams_t params,
                                      cuvsCagraIndex_t* indices,
                                      size_t num_indices,
                                      cuvsFilter filter,
                                      cuvsDataset_t merged_dataset,
                                      cuvsCagraIndex_t output_index)
{
  return cuvs::core::translate_exceptions([=] {
    // Basic checks on inputs
    RAFT_EXPECTS(indices != nullptr && num_indices > 0, "indices array cannot be null or empty");
    RAFT_EXPECTS(params != nullptr, "params cannot be null");
    RAFT_EXPECTS(indices[0] != nullptr && indices[0]->addr != 0,
                 "All input indices must be built (non-empty)");

    // Use first index dtype as reference
    auto dtype = (*indices[0]).dtype;
    for (size_t i = 1; i < num_indices; ++i) {
      RAFT_EXPECTS(indices[i] != nullptr && indices[i]->addr != 0,
                   "All input indices must be built (non-empty)");
      RAFT_EXPECTS((*indices[i]).dtype.code == dtype.code && (*indices[i]).dtype.bits == dtype.bits,
                   "All input indices must have the same data type");
    }
    RAFT_EXPECTS(output_index != nullptr, "Output index pointer must not be null");
    RAFT_EXPECTS(merged_dataset != nullptr, "Merged dataset handle must not be null");
    RAFT_EXPECTS(merged_dataset->addr == 0, "Merged dataset handle must be empty");
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

extern "C" cuvsError_t cuvsCagraIndexParamsFromDataset(cuvsCagraIndexParams_t params,
                                                       int64_t n_rows,
                                                       int64_t dim,
                                                       size_t graph_degree,
                                                       cuvsDistanceType metric,
                                                       size_t build_quality)
{
  return cuvs::core::translate_exceptions([=] {
    auto cpp_metric = static_cast<cuvs::distance::DistanceType>((int)metric);
    auto cpp_params = cuvs::neighbors::cagra::index_params::from_dataset(
      raft::matrix_extent<int64_t>(n_rows, dim), graph_degree, cpp_metric, build_quality);

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

extern "C" cuvsError_t cuvsCagraDeserializeGraph(cuvsResources_t res,
                                                 const char *filename,
                                                 cuvsCagraIndex_t index) {
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(res != 0, "cuvsCagraDeserializeGraph: null resources handle");
    RAFT_EXPECTS(filename != nullptr,
                 "cuvsCagraDeserializeGraph: null filename");
    RAFT_EXPECTS(index != nullptr,
                 "cuvsCagraDeserializeGraph: null index handle");
    auto const header = read_serialized_header(res, filename);
    dispatch_serialized_dtype(header.dtype, [&]<typename T>() {
      using view_t = cuvs::neighbors::device_padded_dataset_view<T, int64_t>;
      _deserialize<T, view_t>(
          res, filename, index, header.dtype, false, nullptr);
    });
  });
}

extern "C" cuvsError_t
cuvsCagraDeserializeGraphAndDataset(cuvsResources_t res, const char *filename,
                                    cuvsCagraIndex_t index,
                                    cuvsDataset_t *out_dataset) {
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(res != 0,
                 "cuvsCagraDeserializeGraphAndDataset: null resources handle");
    RAFT_EXPECTS(filename != nullptr,
                 "cuvsCagraDeserializeGraphAndDataset: null filename");
    RAFT_EXPECTS(index != nullptr,
                 "cuvsCagraDeserializeGraphAndDataset: null index handle");
    RAFT_EXPECTS(
        out_dataset != nullptr,
        "cuvsCagraDeserializeGraphAndDataset: null output dataset pointer");
    RAFT_EXPECTS(*out_dataset == nullptr,
                 "cuvsCagraDeserializeGraphAndDataset: output dataset handle "
                 "must be null");
    auto const header = read_serialized_header(res, filename);
    dispatch_serialized_dtype(header.dtype, [&]<typename T>() {
      dispatch_serialized_dataset_kind<T>(header.dataset_kind, [&]<typename ViewT>() {
        _deserialize<T, ViewT>(
            res, filename, index, header.dtype, true, out_dataset);
      });
    });
  });
}

extern "C" cuvsError_t cuvsCagraSerializeGraph(cuvsResources_t res,
                                               const char *filename,
                                               cuvsCagraIndex_t index) {
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(res != 0, "cuvsCagraSerializeGraph: null resources handle");
    RAFT_EXPECTS(filename != nullptr, "cuvsCagraSerializeGraph: null filename");
    RAFT_EXPECTS(index != nullptr,
                 "cuvsCagraSerializeGraph: null index handle");
    dispatch_serialized_dtype(index->dtype, [&]<typename T>() {
      _serialize<T>(res, filename, index, false);
    });
  });
}

extern "C" cuvsError_t
cuvsCagraSerializeGraphAndDataset(cuvsResources_t res, const char *filename,
                                  cuvsCagraIndex_t index) {
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(res != 0,
                 "cuvsCagraSerializeGraphAndDataset: null resources handle");
    RAFT_EXPECTS(filename != nullptr,
                 "cuvsCagraSerializeGraphAndDataset: null filename");
    RAFT_EXPECTS(index != nullptr,
                 "cuvsCagraSerializeGraphAndDataset: null index handle");
    dispatch_serialized_dtype(index->dtype, [&]<typename T>() {
      _serialize<T>(res, filename, index, true);
    });
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
