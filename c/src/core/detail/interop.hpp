/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

namespace cuvs::core::detail {

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

  if (!tensor.strides) { return false; }
  int64_t expected_stride = 1;
  for (int64_t i = 0; i < tensor.ndim; ++i) {
    if (tensor.strides[i] != expected_stride) { return false; }
    expected_stride *= tensor.shape[i];
  }

  return true;
}

/**
 * True for a 2-D tensor stored row by row, where elements within a row are adjacent and rows may
 * be padded: no strides, or a unit element stride and a row stride of at least the row length.
 */
inline bool is_row_major_with_row_stride(DLManagedTensor* managed_tensor)
{
  auto tensor = managed_tensor->dl_tensor;
  if (tensor.ndim != 2) { return false; }
  if (!tensor.strides) { return true; }
  return tensor.strides[1] == 1 && tensor.strides[0] >= tensor.shape[1];
}

inline bool is_c_contiguous(DLManagedTensor* managed_tensor)
{
  auto tensor = managed_tensor->dl_tensor;

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

  RAFT_EXPECTS(MdspanType::extents_type::rank() == tensor.ndim,
               "ndim mismatch between return mdspan and DLTensor");

  // auto exts = typename MdspanType::extents_type{tensor.shape};
  cuda::std::array<int64_t, MdspanType::extents_type::rank()> shape{};
  for (int64_t i = 0; i < tensor.ndim; ++i) {
    shape[i] = tensor.shape[i];
  }
  auto exts = typename MdspanType::extents_type{shape};

  if constexpr (std::is_same_v<typename MdspanType::layout_type, raft::layout_stride>) {
    // Only row-major matrices whose rows may be padded; consumers read the row stride.
    static_assert(MdspanType::extents_type::rank() == 2);
    RAFT_EXPECTS(is_row_major_with_row_stride(managed_tensor),
                 "Expected a row-major matrix (unit element stride, row stride >= row length)");
    using index_type   = typename MdspanType::index_type;
    index_type const n = static_cast<index_type>(tensor.shape[1]);
    cuda::std::array<index_type, 2> strides{tensor.strides ? tensor.strides[0] : n, 1};
    return MdspanType{reinterpret_cast<typename MdspanType::data_handle_type>(tensor.data),
                      typename MdspanType::mapping_type{exts, strides}};
  } else {
    return MdspanType{reinterpret_cast<typename MdspanType::data_handle_type>(tensor.data), exts};
  }
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

  tensor->dtype  = data_type_to_DLDataType<typename MdspanType::value_type>();
  tensor->device = accessor_type_to_DLDevice<typename MdspanType::accessor_type>();
  tensor->ndim   = MdspanType::extents_type::rank();
  tensor->data   = const_cast<typename MdspanType::value_type*>(src.data_handle());
  tensor->shape  = new int64_t[tensor->ndim];
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
