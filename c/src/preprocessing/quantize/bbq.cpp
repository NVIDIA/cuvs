/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/preprocessing/quantize/bbq.h>
#include <cuvs/preprocessing/quantize/bbq.hpp>

#include "../../core/exceptions.hpp"
#include "../../core/interop.hpp"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/interruptible.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cuda/stream_ref>

#include <cstdint>
#include <memory>

namespace {

template <typename T>
void destroy_quantizer_view(void* ptr)
{
  using storage_t = cuvs::preprocessing::quantize::bbq::bbq_dataset_view_storage<T, int64_t>;
  std::unique_ptr<storage_t> storage{reinterpret_cast<storage_t*>(ptr)};
  RAFT_CUDA_TRY_NO_THROW(
    cudaFree(const_cast<float*>(storage->quantizer.centroid_norm_sq.data_handle())));
}

auto to_cpp_layout(cuvsBbqCodeLayout_t layout)
  -> cuvs::preprocessing::quantize::bbq::bbq_code_layout
{
  using layout_t = cuvs::preprocessing::quantize::bbq::bbq_code_layout;
  switch (layout) {
    case CUVS_BBQ_CODE_LAYOUT_PACKED_1B: return layout_t::packed_1b;
    case CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_2B: return layout_t::transposed_2b;
    case CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_4B: return layout_t::transposed_4b;
    case CUVS_BBQ_CODE_LAYOUT_PACKED_4B: return layout_t::packed_4b;
    case CUVS_BBQ_CODE_LAYOUT_PACKED_7B: return layout_t::packed_7b;
    case CUVS_BBQ_CODE_LAYOUT_PACKED_8B: return layout_t::packed_8b;
  }
  RAFT_FAIL("cuvsBbqQuantizerCreateView: invalid BBQ code layout");
}

template <typename T>
auto make_cpp_quantizer_view(
  DLManagedTensor* codes_tensor,
  DLManagedTensor* lower_intervals_tensor,
  DLManagedTensor* upper_intervals_tensor,
  DLManagedTensor* additional_corrections_tensor,
  DLManagedTensor* quantized_component_sums_tensor,
  DLManagedTensor* centroid_tensor,
  DLManagedTensor* dequant_delta_tensor,
  DLManagedTensor* dequant_sum_delta_tensor,
  DLManagedTensor* row_norm_tensor,
  cuvsBbqCodeLayout_t c_layout,
  cuvsDistanceType metric,
  raft::device_scalar_view<const float, int64_t> centroid_norm_sq)
  -> cuvs::preprocessing::quantize::bbq::bbq_dataset_view_storage<T, int64_t>
{
  using quantizer_view_t = cuvs::preprocessing::quantize::bbq::quantizer_view<T>;
  using storage_view_t =
    cuvs::preprocessing::quantize::bbq::bbq_dataset_view_storage<T, int64_t>;
  using codes_view_t    = raft::device_matrix_view<const uint8_t, int64_t>;
  using float_view_t    = raft::device_vector_view<const float, int64_t>;
  using int_view_t      = raft::device_vector_view<const int32_t, int64_t>;
  using centroid_view_t = raft::device_vector_view<const T, int64_t>;

  auto codes       = cuvs::core::from_dlpack<codes_view_t>(codes_tensor);
  auto lower       = cuvs::core::from_dlpack<float_view_t>(lower_intervals_tensor);
  auto upper       = cuvs::core::from_dlpack<float_view_t>(upper_intervals_tensor);
  auto corrections = cuvs::core::from_dlpack<float_view_t>(additional_corrections_tensor);
  auto sums        = cuvs::core::from_dlpack<int_view_t>(quantized_component_sums_tensor);
  auto centroid    = cuvs::core::from_dlpack<centroid_view_t>(centroid_tensor);
  auto delta       = cuvs::core::from_dlpack<float_view_t>(dequant_delta_tensor);
  auto sum_delta   = cuvs::core::from_dlpack<float_view_t>(dequant_sum_delta_tensor);
  auto row_norm    = cuvs::core::from_dlpack<float_view_t>(row_norm_tensor);
  auto layout      = to_cpp_layout(c_layout);

  const auto n_rows = codes.extent(0);
  const auto dim    = static_cast<uint32_t>(centroid.extent(0));
  RAFT_EXPECTS(n_rows > 0 && dim > 0, "cuvsBbqQuantizerCreateView: quantizer must not be empty");
  RAFT_EXPECTS(
    codes.extent(1) ==
      static_cast<int64_t>(
        cuvs::preprocessing::quantize::bbq::get_encoded_row_length(dim, layout)),
    "cuvsBbqQuantizerCreateView: code row length does not match the dimension and layout");
  RAFT_EXPECTS(lower.extent(0) == n_rows && upper.extent(0) == n_rows &&
                 corrections.extent(0) == n_rows && sums.extent(0) == n_rows &&
                 delta.extent(0) == n_rows && sum_delta.extent(0) == n_rows &&
                 row_norm.extent(0) == n_rows,
               "cuvsBbqQuantizerCreateView: every correction vector must contain one value per row");

  return storage_view_t{quantizer_view_t{centroid, centroid_norm_sq},
                        codes,
                        lower,
                        upper,
                        corrections,
                        sums,
                        delta,
                        sum_delta,
                        row_norm,
                        layout,
                        static_cast<cuvs::distance::DistanceType>(metric)};
}

template <typename T>
void make_and_bind_quantizer_view(
  cuvsBbqQuantizer_t* output,
  DLDataType dtype,
  DLManagedTensor* codes,
  DLManagedTensor* lower_intervals,
  DLManagedTensor* upper_intervals,
  DLManagedTensor* additional_corrections,
  DLManagedTensor* quantized_component_sums,
  DLManagedTensor* centroid,
  DLManagedTensor* dequant_delta,
  DLManagedTensor* dequant_sum_delta,
  DLManagedTensor* row_norm,
  cuvsBbqCodeLayout_t layout,
  cuvsDistanceType metric,
  float centroid_norm_sq)
{
  using storage_t = cuvs::preprocessing::quantize::bbq::bbq_dataset_view_storage<T, int64_t>;
  // The norm arrives by value, transfer it to device memory
  float* device_norm_sq = nullptr;
  RAFT_CUDA_TRY(cudaMalloc(&device_norm_sq, sizeof(float)));
  try {
    // The C API doesn't take a stream, so construct one to copy the norm to device memory
    // This is a temporary workaround until the C API takes a stream.
    const cuda::stream_ref stream{cudaStream_t{}};
    raft::copy(device_norm_sq, &centroid_norm_sq, 1, stream);
    raft::interruptible::synchronize(stream);

    auto storage = std::make_unique<storage_t>(make_cpp_quantizer_view<T>(
      codes,
      lower_intervals,
      upper_intervals,
      additional_corrections,
      quantized_component_sums,
      centroid,
      dequant_delta,
      dequant_sum_delta,
      row_norm,
      layout,
      metric,
      raft::make_device_scalar_view<const float, int64_t>(device_norm_sq)));

    auto handle          = std::make_unique<cuvsBbqQuantizer>();
    handle->addr         = reinterpret_cast<uintptr_t>(storage.release());
    handle->destroy_addr = &destroy_quantizer_view<T>;
    handle->dtype        = dtype;
    handle->is_owning    = false;
    *output              = handle.release();
  } catch (...) {
    RAFT_CUDA_TRY_NO_THROW(cudaFree(device_norm_sq));
    throw;
  }
}

}  // namespace

extern "C" cuvsError_t cuvsBbqQuantizerCreateView(
  DLManagedTensor* codes,
  DLManagedTensor* lower_intervals,
  DLManagedTensor* upper_intervals,
  DLManagedTensor* additional_corrections,
  DLManagedTensor* quantized_component_sums,
  DLManagedTensor* centroid,
  DLManagedTensor* dequant_delta,
  DLManagedTensor* dequant_sum_delta,
  DLManagedTensor* row_norm,
  cuvsBbqCodeLayout_t layout,
  cuvsDistanceType metric,
  float centroid_norm_sq,
  cuvsBbqQuantizer_t* quantizer)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(quantizer != nullptr, "cuvsBbqQuantizerCreateView: null output");
    *quantizer = nullptr;
    RAFT_EXPECTS(codes != nullptr && lower_intervals != nullptr && upper_intervals != nullptr &&
                   additional_corrections != nullptr && quantized_component_sums != nullptr &&
                   centroid != nullptr && dequant_delta != nullptr &&
                   dequant_sum_delta != nullptr && row_norm != nullptr,
                 "cuvsBbqQuantizerCreateView: null tensor");
    auto dtype = centroid->dl_tensor.dtype;
    if (dtype.code == kDLFloat && dtype.bits == 32) {
      make_and_bind_quantizer_view<float>(quantizer,
                                          dtype,
                                          codes,
                                          lower_intervals,
                                          upper_intervals,
                                          additional_corrections,
                                          quantized_component_sums,
                                          centroid,
                                          dequant_delta,
                                          dequant_sum_delta,
                                          row_norm,
                                          layout,
                                          metric,
                                          centroid_norm_sq);
    } else if (dtype.code == kDLFloat && dtype.bits == 16) {
      make_and_bind_quantizer_view<half>(quantizer,
                                         dtype,
                                         codes,
                                         lower_intervals,
                                         upper_intervals,
                                         additional_corrections,
                                         quantized_component_sums,
                                         centroid,
                                         dequant_delta,
                                         dequant_sum_delta,
                                         row_norm,
                                         layout,
                                         metric,
                                         centroid_norm_sq);
    } else if (dtype.code == kDLInt && dtype.bits == 8) {
      make_and_bind_quantizer_view<int8_t>(quantizer,
                                           dtype,
                                           codes,
                                           lower_intervals,
                                           upper_intervals,
                                           additional_corrections,
                                           quantized_component_sums,
                                           centroid,
                                           dequant_delta,
                                           dequant_sum_delta,
                                           row_norm,
                                           layout,
                                           metric,
                                           centroid_norm_sq);
    } else if (dtype.code == kDLUInt && dtype.bits == 8) {
      make_and_bind_quantizer_view<uint8_t>(quantizer,
                                            dtype,
                                            codes,
                                            lower_intervals,
                                            upper_intervals,
                                            additional_corrections,
                                            quantized_component_sums,
                                            centroid,
                                            dequant_delta,
                                            dequant_sum_delta,
                                            row_norm,
                                            layout,
                                            metric,
                                            centroid_norm_sq);
    } else {
      RAFT_FAIL("cuvsBbqQuantizerCreateView: unsupported centroid dtype: code=%d, bits=%d",
                dtype.code,
                dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsBbqQuantizerDestroy(cuvsBbqQuantizer_t quantizer)
{
  return cuvs::core::translate_exceptions([=] {
    if (quantizer == nullptr) { return; }
    if (quantizer->destroy_addr != nullptr && quantizer->addr != 0) {
      quantizer->destroy_addr(reinterpret_cast<void*>(quantizer->addr));
    }
    delete quantizer;
  });
}
