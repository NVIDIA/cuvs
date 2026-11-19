/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_fp16.h>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/mdspan_types.hpp>

#include <dlpack/dlpack.h>
#include <rmm/device_buffer.hpp>
#include <sys/types.h>

#include <cstdint>
#include <limits>

namespace cuvs::core::detail {

inline void validate_dlpack_tensor(DLManagedTensor const* managed_tensor, int32_t expected_ndim)
{
  RAFT_EXPECTS(managed_tensor != nullptr, "DLManagedTensor must not be NULL");
  auto const& tensor = managed_tensor->dl_tensor;

  RAFT_EXPECTS(tensor.ndim >= 0, "DLTensor ndim must not be negative, got %d", tensor.ndim);
  if (expected_ndim >= 0) {
    RAFT_EXPECTS(tensor.ndim == expected_ndim,
                 "DLTensor ndim mismatch: expected %d, got %d",
                 expected_ndim,
                 tensor.ndim);
  }
  RAFT_EXPECTS(tensor.ndim == 0 || tensor.shape != nullptr, "DLTensor shape must not be NULL");

  int64_t n_elements = 1;
  for (int32_t i = 0; i < tensor.ndim; ++i) {
    auto extent = tensor.shape[i];
    RAFT_EXPECTS(extent >= 0,
                 "DLTensor shape[%d] must not be negative, got %lld",
                 i,
                 static_cast<long long>(extent));
    RAFT_EXPECTS(extent == 0 || n_elements <= std::numeric_limits<int64_t>::max() / extent,
                 "DLTensor element count overflows int64");
    n_elements *= extent;
  }
  auto bytes_per_element = static_cast<int64_t>(tensor.dtype.bits) * tensor.dtype.lanes / 8;
  RAFT_EXPECTS(bytes_per_element == 0 ||
                 n_elements <= std::numeric_limits<int64_t>::max() / bytes_per_element,
               "DLTensor size in bytes overflows int64");
  RAFT_EXPECTS(n_elements == 0 || tensor.data != nullptr,
               "DLTensor data must not be NULL for a non-empty tensor");
}

template <typename AccessorType>
DLDevice accessor_type_to_DLDevice()
{
  if constexpr (AccessorType::is_host_accessible and AccessorType::is_device_accessible) {
    return DLDevice{kDLCUDAManaged};
  } else if constexpr (AccessorType::is_device_accessible) {
    return DLDevice{kDLCUDA};
  } else if constexpr (AccessorType::is_host_accessible) {
    return DLDevice{kDLCPU};
  }
}

template <typename T>
DLDataType data_type_to_DLDataType()
{
  uint8_t const bits{sizeof(T) * 8};
  uint16_t const lanes{1};
  // std::is_floating_point returns false for the half type - handle
  // that here
  if constexpr (std::is_floating_point_v<T> || std::is_same_v<T, half>) {
    return DLDataType{kDLFloat, bits, lanes};
  } else if constexpr (std::is_signed_v<T>) {
    return DLDataType{kDLInt, bits, lanes};
  } else {
    return DLDataType{kDLUInt, bits, lanes};
  }
}

inline bool is_dlpack_device_compatible(DLTensor tensor)
{
  return tensor.device.device_type == kDLCUDAManaged || tensor.device.device_type == kDLCUDAHost ||
         tensor.device.device_type == kDLCUDA;
}

inline bool is_dlpack_host_compatible(DLTensor tensor)
{
  return tensor.device.device_type == kDLCUDAManaged || tensor.device.device_type == kDLCUDAHost ||
         tensor.device.device_type == kDLCPU;
}

inline bool is_f_contiguous(DLManagedTensor* managed_tensor)
{
  auto tensor = managed_tensor->dl_tensor;

  if (tensor.ndim > 0 && !tensor.shape) { return false; }
  if (!tensor.strides) { return false; }
  int64_t expected_stride = 1;
  for (int64_t i = 0; i < tensor.ndim; ++i) {
    if (tensor.strides[i] != expected_stride) { return false; }
    expected_stride *= tensor.shape[i];
  }

  return true;
}

inline bool is_c_contiguous(DLManagedTensor* managed_tensor)
{
  auto tensor = managed_tensor->dl_tensor;

  if (tensor.ndim > 0 && !tensor.shape) { return false; }
  if (!tensor.strides) {
    // no stride information indicates a row-major tensor according to the dlpack spec
    return true;
  }

  int64_t expected_stride = 1;
  for (int64_t i = tensor.ndim - 1; i >= 0; --i) {
    if (tensor.strides[i] != expected_stride) { return false; }
    expected_stride *= tensor.shape[i];
  }

  return true;
}

template <typename MdspanType, typename = raft::is_mdspan_t<MdspanType>>
inline MdspanType from_dlpack(DLManagedTensor* managed_tensor)
{
  validate_dlpack_tensor(managed_tensor, MdspanType::extents_type::rank());
  auto tensor = managed_tensor->dl_tensor;

  if constexpr (std::is_same_v<typename MdspanType::layout_type, raft::row_major>) {
    RAFT_EXPECTS(is_c_contiguous(managed_tensor), "Expected a row-major matrix");
  }

  if constexpr (std::is_same_v<typename MdspanType::layout_type, raft::col_major>) {
    RAFT_EXPECTS(is_f_contiguous(managed_tensor), "Expected a col-major matrix");
  }

  auto to_data_type = data_type_to_DLDataType<typename MdspanType::value_type>();
  RAFT_EXPECTS(to_data_type.code == tensor.dtype.code,
               "code mismatch between return mdspan (%i) and DLTensor (%i)",
               to_data_type.code,
               tensor.dtype.code);
  RAFT_EXPECTS(to_data_type.bits == tensor.dtype.bits,
               "bits mismatch between return mdspan (%i) and DLTensor (%i)",
               to_data_type.bits,
               tensor.dtype.bits);
  RAFT_EXPECTS(to_data_type.lanes == tensor.dtype.lanes,
               "lanes mismatch between return mdspan and DLTensor");
  RAFT_EXPECTS(tensor.dtype.lanes == 1, "More than 1 DLTensor lanes not supported");

  auto to_device = accessor_type_to_DLDevice<typename MdspanType::accessor_type>();
  if (to_device.device_type == kDLCUDA) {
    RAFT_EXPECTS(is_dlpack_device_compatible(tensor),
                 "device_type mismatch between return mdspan and DLTensor");
  } else if (to_device.device_type == kDLCPU) {
    RAFT_EXPECTS(is_dlpack_host_compatible(tensor),
                 "device_type mismatch between return mdspan and DLTensor");
  }

  // auto exts = typename MdspanType::extents_type{tensor.shape};
  cuda::std::array<int64_t, MdspanType::extents_type::rank()> shape{};
  for (int64_t i = 0; i < tensor.ndim; ++i) {
    shape[i] = tensor.shape[i];
  }
  auto exts = typename MdspanType::extents_type{shape};

  return MdspanType{reinterpret_cast<typename MdspanType::data_handle_type>(tensor.data), exts};
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
static void free_dlmanaged_tensor_metadata(DLManagedTensor* tensor)
{
  delete[] tensor->dl_tensor.shape;
  delete[] tensor->dl_tensor.strides;
}
#pragma GCC diagnostic pop

template <typename MdspanType, typename = raft::is_mdspan_t<MdspanType>>
static void to_dlpack(MdspanType src, DLManagedTensor* dst)
{
  auto tensor = &dst->dl_tensor;

  tensor->dtype       = data_type_to_DLDataType<typename MdspanType::value_type>();
  tensor->device      = accessor_type_to_DLDevice<typename MdspanType::accessor_type>();
  tensor->ndim        = MdspanType::extents_type::rank();
  tensor->data        = const_cast<typename MdspanType::value_type*>(src.data_handle());
  tensor->byte_offset = 0;
  tensor->shape       = new int64_t[tensor->ndim];
  for (int64_t i = 0; i < tensor->ndim; ++i) {
    tensor->shape[i] = src.extent(i);
  }

  if constexpr (std::is_same_v<typename MdspanType::layout_type, raft::row_major>) {
    tensor->strides = nullptr;
  } else {
    tensor->strides = new int64_t[tensor->ndim];
    for (int64_t i = 0; i < tensor->ndim; ++i) {
      tensor->strides[i] = src.stride(i);
    }
  }

  dst->deleter = free_dlmanaged_tensor_metadata;
}
}  // namespace cuvs::core::detail
