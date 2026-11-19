/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <dlpack/dlpack.h>
#include <memory>
#include <utility>

#include <raft/core/error.hpp>
#include <raft/core/resources.hpp>

#include "dataset_handle.hpp"
#include "exceptions.hpp"
#include "interop.hpp"
#include <cuvs/core/c_api.h>
#include <cuvs/core/dataset.h>
#include <cuvs/core/dataset/factories.hpp>
#include <cuvs/distance/distance.hpp>
#include <cuvs/preprocessing/quantize/bbq.h>
#include <cuvs/preprocessing/quantize/bbq.hpp>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/preprocessing/quantize/pq.hpp>

using cuvs::core::c_api::destroy_typed_addr;
using cuvs::core::c_api::device_vpq_owner_t;
using cuvs::core::c_api::device_vpq_view_t;
using cuvs::core::c_api::with_dataset_view;

template <typename T>
void validate_bbq_layouts(cuvs::neighbors::device_bbq_dataset_view<T, int64_t> const& dataset)
{
  using layout_t = cuvs::preprocessing::quantize::bbq::bbq_code_layout;
  if (dataset.quantizers.size() == 1) {
    auto layout = dataset.quantizers.front().layout;
    RAFT_EXPECTS(layout == layout_t::packed_1b || layout == layout_t::transposed_2b ||
                   layout == layout_t::packed_4b || layout == layout_t::packed_7b ||
                   layout == layout_t::packed_8b,
                 "cuvsDatasetMakeBbqView: transposed_4b requires an asymmetric layout pair");
    return;
  }

  const bool has_1b  = dataset.has_layout(layout_t::packed_1b);
  const bool has_2bt = dataset.has_layout(layout_t::transposed_2b);
  const bool has_4b  = dataset.has_layout(layout_t::packed_4b);
  const bool has_4bt = dataset.has_layout(layout_t::transposed_4b);
  RAFT_EXPECTS((has_1b && (has_4b || has_2bt || has_4bt)) || (has_2bt && has_4bt),
               "cuvsDatasetMakeBbqView: unsupported asymmetric BBQ layout pair");
}

template <typename T>
auto get_cpp_bbq_quantizer_view(cuvsBbqQuantizer_t quantizer)
  -> cuvs::preprocessing::quantize::bbq::quantizer_view<T, int64_t>
{
  using owner_t = cuvs::preprocessing::quantize::bbq::quantizer<T, int64_t>;
  using view_t  = cuvs::preprocessing::quantize::bbq::quantizer_view<T, int64_t>;
  RAFT_EXPECTS(quantizer != nullptr && quantizer->addr != 0,
               "cuvsDatasetMakeBbqView: null quantizer storage");
  if (quantizer->is_owning) {
    return reinterpret_cast<owner_t*>(quantizer->addr)->view();
  }
  return *reinterpret_cast<view_t*>(quantizer->addr);
}

template <typename T>
auto make_bbq_dataset_view(cuvsBbqQuantizer_t* quantizers, std::size_t num_quantizers)
  -> std::unique_ptr<cuvs::neighbors::device_bbq_dataset_view<T, int64_t>>
{
  using dataset_view_t = cuvs::neighbors::device_bbq_dataset_view<T, int64_t>;
  auto dataset         = std::make_unique<dataset_view_t>();
  int64_t expected_rows{-1};
  uint32_t expected_dim{};
  cuvs::distance::DistanceType expected_metric{};

  for (std::size_t i = 0; i < num_quantizers; ++i) {
    RAFT_EXPECTS(quantizers[i] != nullptr,
                 "cuvsDatasetMakeBbqView: quantizer %zu is null",
                 i);
    auto quantizer = get_cpp_bbq_quantizer_view<T>(quantizers[i]);
    if (i == 0) {
      expected_rows   = quantizer.n_rows();
      expected_dim    = quantizer.dim();
      expected_metric = quantizer.metric;
    } else {
      RAFT_EXPECTS(quantizer.n_rows() == expected_rows && quantizer.dim() == expected_dim,
                   "cuvsDatasetMakeBbqView: all quantizers must have the same shape");
      RAFT_EXPECTS(quantizer.metric == expected_metric,
                   "cuvsDatasetMakeBbqView: all quantizers must use the same metric");
    }
    dataset->add_quantizer(quantizer);
  }

  return dataset;
}

template <typename T>
void make_and_bind_bbq_dataset(cuvsBbqQuantizer_t* quantizers,
                               std::size_t num_quantizers,
                               DLDataType dtype,
                               cuvsDataset_t* output)
{
  using dataset_view_t = cuvs::neighbors::device_bbq_dataset_view<T, int64_t>;
  auto view            = make_bbq_dataset_view<T>(quantizers, num_quantizers);
  validate_bbq_layouts(*view);

  auto handle          = std::make_unique<cuvsDataset>();
  handle->addr         = reinterpret_cast<uintptr_t>(view.release());
  handle->destroy_addr = &destroy_typed_addr<dataset_view_t>;
  handle->dtype        = dtype;
  handle->mem_type     = CUVS_DATASET_MEM_TYPE_DEVICE;
  handle->layout       = CUVS_DATASET_LAYOUT_BBQ;
  handle->is_owning    = false;
  *output              = handle.release();
}

static void bind_vpq_owner_to_dataset(std::unique_ptr<device_vpq_owner_t> owner,
                                      cuvsDataset_t* output)
{
  RAFT_EXPECTS(output != nullptr, "VPQ output dataset pointer must not be null");
  auto* out         = new cuvsDataset{};
  out->addr         = reinterpret_cast<uintptr_t>(owner.release());
  out->destroy_addr = &destroy_typed_addr<device_vpq_owner_t>;
  out->dtype.code   = kDLFloat;
  out->dtype.bits   = 32;
  out->dtype.lanes  = 1;
  out->mem_type     = CUVS_DATASET_MEM_TYPE_DEVICE;
  out->layout       = CUVS_DATASET_LAYOUT_PQ;
  out->is_owning    = true;
  *output           = out;
}

static auto make_cpp_vpq_params(cuvsPqParams const& params)
  -> cuvs::preprocessing::quantize::pq::vpq_params
{
  auto out                            = cuvs::preprocessing::quantize::pq::vpq_params{};
  out.pq_bits                         = params.pq_bits;
  out.pq_dim                          = params.pq_dim;
  out.vq_n_centers                    = params.vq_n_centers;
  out.kmeans_n_iters                  = params.kmeans_n_iters;
  out.vq_kmeans_trainset_fraction     = params.vq_kmeans_trainset_fraction;
  out.pq_kmeans_trainset_fraction     = params.pq_kmeans_trainset_fraction;
  return out;
}

template <typename T>
static auto make_device_pq_dataset(raft::resources* res_ptr,
                                   cuvsPqParams const& params,
                                   cuvsDataset_t dataset)
  -> std::unique_ptr<device_vpq_owner_t>
{
  RAFT_EXPECTS(dataset->layout == CUVS_DATASET_LAYOUT_STANDARD ||
                 dataset->layout == CUVS_DATASET_LAYOUT_PADDED,
               "cuvsDatasetMakePQ: source dataset must have STANDARD or PADDED layout");
  auto cpp_params = make_cpp_vpq_params(params);
  std::unique_ptr<device_vpq_owner_t> owner;
  auto make = [&](auto const& view) {
    owner = std::make_unique<device_vpq_owner_t>(
      cuvs::preprocessing::quantize::pq::make_vpq_dataset(*res_ptr, cpp_params, view));
  };

  const bool padded = dataset->layout == CUVS_DATASET_LAYOUT_PADDED;
  if (dataset->mem_type == CUVS_DATASET_MEM_TYPE_DEVICE) {
    if (padded) {
      with_dataset_view<cuvs::device_padded_dataset<T, int64_t>,
                        cuvs::device_padded_dataset_view<T, int64_t>>(dataset, make);
    } else {
      with_dataset_view<cuvs::device_contiguous_dataset<T, int64_t>,
                        cuvs::device_contiguous_dataset_view<T, int64_t>>(dataset, make);
    }
  } else if (dataset->mem_type == CUVS_DATASET_MEM_TYPE_HOST) {
    if (padded) {
      with_dataset_view<cuvs::host_padded_dataset<T, int64_t>,
                        cuvs::host_padded_dataset_view<T, int64_t>>(dataset, make);
    } else {
      with_dataset_view<cuvs::host_contiguous_dataset<T, int64_t>,
                        cuvs::host_contiguous_dataset_view<T, int64_t>>(dataset, make);
    }
  } else {
    RAFT_FAIL("cuvsDatasetMakePQ: invalid source dataset memory type");
  }
  return owner;
}

template <typename T>
static void make_device_padded_dataset(raft::resources* res_ptr,
                                       DLManagedTensor* dataset_tensor,
                                       cuvsDataset_t* output_padded_dataset)
{
  auto dataset = dataset_tensor->dl_tensor;
  using owner_type = cuvs::device_padded_dataset<T, int64_t>;
  std::unique_ptr<owner_type> owner;
  if (cuvs::core::is_dlpack_device_compatible(dataset)) {
    using mdspan_type = raft::device_matrix_view<T const, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
    owner             = cuvs::make_device_padded_dataset(*res_ptr, mds);
  } else if (cuvs::core::is_dlpack_host_compatible(dataset)) {
    using mdspan_type = raft::host_matrix_view<T const, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
    owner             = cuvs::make_device_padded_dataset(*res_ptr, mds);
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
  using owner_type = cuvs::host_padded_dataset<T, int64_t>;
  std::unique_ptr<owner_type> owner;
  if (cuvs::core::is_dlpack_host_compatible(dataset)) {
    using mdspan_type = raft::host_matrix_view<T const, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
    owner             = cuvs::make_host_padded_dataset(*res_ptr, mds);
  } else if (cuvs::core::is_dlpack_device_compatible(dataset)) {
    using mdspan_type = raft::device_matrix_view<T const, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
    owner             = cuvs::make_host_padded_dataset(*res_ptr, mds);
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
  auto ds_view      = cuvs::make_device_padded_dataset_view(*res_ptr, mds);
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
  auto ds_view      = cuvs::make_host_padded_dataset_view(mds);
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
static void make_device_contiguous_dataset_view(raft::resources*,
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
  auto ds_view      = cuvs::make_device_contiguous_dataset_view(mds);
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
static void make_host_contiguous_dataset_view(raft::resources*,
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
  auto ds_view      = cuvs::make_host_contiguous_dataset_view(mds);
  auto* owned_view = new decltype(ds_view){ds_view};
  out->addr        = reinterpret_cast<uintptr_t>(owned_view);
  out->destroy_addr = &destroy_typed_addr<decltype(ds_view)>;
  out->dtype        = dataset.dtype;
  out->mem_type     = CUVS_DATASET_MEM_TYPE_HOST;
  out->layout       = CUVS_DATASET_LAYOUT_STANDARD;
  out->is_owning    = false;
  *output_standard_dataset = out;
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
        make_device_contiguous_dataset_view<T>(res_ptr, dataset_tensor, standard_dataset);
      } else if (cuvs::core::is_dlpack_host_compatible(dataset)) {
        make_host_contiguous_dataset_view<T>(res_ptr, dataset_tensor, standard_dataset);
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

extern "C" cuvsError_t cuvsDatasetMakeBbqView(cuvsResources_t,
                                              cuvsBbqQuantizer_t* quantizers,
                                              size_t num_quantizers,
                                              cuvsDataset_t* dataset)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(dataset != nullptr, "cuvsDatasetMakeBbqView: null output dataset");
    *dataset = nullptr;
    RAFT_EXPECTS(quantizers != nullptr, "cuvsDatasetMakeBbqView: null quantizer array");
    RAFT_EXPECTS(num_quantizers == 1 || num_quantizers == 2,
                 "cuvsDatasetMakeBbqView: expected one or two quantizers");
    RAFT_EXPECTS(quantizers[0] != nullptr && quantizers[0]->addr != 0,
                 "cuvsDatasetMakeBbqView: first quantizer has null storage");

    auto dtype = quantizers[0]->dtype;
    for (size_t i = 1; i < num_quantizers; ++i) {
      RAFT_EXPECTS(quantizers[i] != nullptr && quantizers[i]->addr != 0,
                   "cuvsDatasetMakeBbqView: quantizer %zu has null storage",
                   i);
      RAFT_EXPECTS(quantizers[i]->dtype.code == dtype.code &&
                     quantizers[i]->dtype.bits == dtype.bits &&
                     quantizers[i]->dtype.lanes == dtype.lanes,
                   "cuvsDatasetMakeBbqView: all quantizers must use the same dtype");
    }
    if (dtype.code == kDLFloat && dtype.bits == 32) {
      make_and_bind_bbq_dataset<float>(quantizers, num_quantizers, dtype, dataset);
    } else if (dtype.code == kDLFloat && dtype.bits == 16) {
      make_and_bind_bbq_dataset<half>(quantizers, num_quantizers, dtype, dataset);
    } else if (dtype.code == kDLInt && dtype.bits == 8) {
      make_and_bind_bbq_dataset<int8_t>(quantizers, num_quantizers, dtype, dataset);
    } else if (dtype.code == kDLUInt && dtype.bits == 8) {
      make_and_bind_bbq_dataset<uint8_t>(quantizers, num_quantizers, dtype, dataset);
    } else {
      RAFT_FAIL("cuvsDatasetMakeBbqView: unsupported centroid dtype: code=%d, bits=%d",
                dtype.code,
                dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsDatasetMakePQ(cuvsResources_t res,
                                         cuvsPqParams_t params,
                                         cuvsDataset_t dataset,
                                         cuvsDatasetMemType_t target_mem_type,
                                         cuvsDataset_t* pq_dataset)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(params != nullptr, "cuvsDatasetMakePQ: null compression params");
    RAFT_EXPECTS(dataset != nullptr && dataset->addr != 0,
                 "cuvsDatasetMakePQ: null source dataset");
    RAFT_EXPECTS(pq_dataset != nullptr, "cuvsDatasetMakePQ: null output dataset");
    *pq_dataset = nullptr;
    RAFT_EXPECTS(target_mem_type == CUVS_DATASET_MEM_TYPE_DEVICE,
                 "cuvsDatasetMakePQ currently supports only device output");
    auto* res_ptr = reinterpret_cast<raft::resources*>(res);
    std::unique_ptr<device_vpq_owner_t> owner;
    if (dataset->dtype.code == kDLFloat && dataset->dtype.bits == 32) {
      owner = make_device_pq_dataset<float>(res_ptr, *params, dataset);
    } else if (dataset->dtype.code == kDLFloat && dataset->dtype.bits == 16) {
      owner = make_device_pq_dataset<half>(res_ptr, *params, dataset);
    } else if (dataset->dtype.code == kDLInt && dataset->dtype.bits == 8) {
      owner = make_device_pq_dataset<int8_t>(res_ptr, *params, dataset);
    } else if (dataset->dtype.code == kDLUInt && dataset->dtype.bits == 8) {
      owner = make_device_pq_dataset<uint8_t>(res_ptr, *params, dataset);
    } else {
      RAFT_FAIL("cuvsDatasetMakePQ: unsupported source dtype: code=%d, bits=%d",
                dataset->dtype.code,
                dataset->dtype.bits);
    }
    bind_vpq_owner_to_dataset(std::move(owner), pq_dataset);
  });
}
