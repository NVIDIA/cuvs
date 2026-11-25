/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "detail/interop.hpp"

namespace cuvs::core {

/**
 * @defgroup interop Interoperability between `mdspan` and `DLManagedTensor`
 * @{
 */

/**
 * @brief Validate the structural fields of a DLManagedTensor
 *
 * Throws raft::logic_error if the tensor is NULL, has a negative or unexpected ndim, a NULL shape,
 * negative extents, an element count or byte size that overflows int64, or a NULL data pointer
 * while holding at least one element. from_dlpack calls this before building an mdspan; call it
 * directly before reading `shape` or `data` of a tensor without from_dlpack.
 *
 * @param[in] tensor DLManagedTensor to validate
 * @param[in] expected_ndim required number of dimensions, or -1 to accept any
 */
inline void validate_dlpack_tensor(DLManagedTensor const* tensor, int32_t expected_ndim = -1)
{
  detail::validate_dlpack_tensor(tensor, expected_ndim);
}

/**
 * @brief Check if DLTensor has device accessible memory.
 *        This function returns true for `DLDeviceType` of values
 *        `kDLCUDA`, `kDLCUDAHost`, or `kDLCUDAManaged`
 *
 * @param[in] tensor DLTensor object to check underlying memory type
 * @return bool
 */
inline bool is_dlpack_device_compatible(DLTensor tensor)
{
  return detail::is_dlpack_device_compatible(tensor);
}

/**
 * @brief Check if DLTensor has host accessible memory.
 *        This function returns true for `DLDeviceType` of values
 *        `kDLCPU`, `kDLCUDAHost`, or `kDLCUDAManaged`
 *
 * @param tensor DLTensor object to check underlying memory type
 * @return bool
 */
inline bool is_dlpack_host_compatible(DLTensor tensor)
{
  return detail::is_dlpack_host_compatible(tensor);
}

/**
 * @brief Check if DLManagedTensor has a row-major (c-contiguous) layout
 *
 * @param tensor DLManagedTensor object to check
 * @return bool
 */
inline bool is_c_contiguous(DLManagedTensor* tensor) { return detail::is_c_contiguous(tensor); }

/**
 * @brief Check if DLManagedTensor has a col-major (f-contiguous) layout
 *
 * @param tensor DLManagedTensor object to check
 * @return bool
 */
inline bool is_f_contiguous(DLManagedTensor* tensor) { return detail::is_f_contiguous(tensor); }

/**
 * @brief Convert a DLManagedTensor to a mdspan
 * NOTE: This function only supports compact row-major and col-major layouts.
 *
 * @code {.cpp}
 * #include <raft/core/device_mdspan.hpp>
 * #include <raft/core/interop.hpp>
 * // We have a `DLManagedTensor` with `DLDeviceType == kDLCUDA`,
 * // `DLDataType.code == kDLFloat` and `DLDataType.bits == 8`
 * DLManagedTensor tensor;
 * // declare the return type
 * using mdpsan_type = raft::device_mdspan<float, int64_t, raft::row_major>;
 * auto mds = raft::core::from_dlpack<mdspan_type>(&tensor);
 * @endcode
 *
 * @tparam MdspanType
 * @tparam typename
 * @param[in] managed_tensor
 * @return MdspanType
 */
template <typename MdspanType, typename = raft::is_mdspan_t<MdspanType>>
inline MdspanType from_dlpack(DLManagedTensor* managed_tensor)
{
  return detail::from_dlpack<MdspanType>(managed_tensor);
}

/**
 * @brief Convert a mdspan to a DLManagedTensor
 *
 * Converts a mdspan to a DLManagedTensor object. This lets us pass non-owning
 * views from C++ to C code without copying.  Note that returned DLManagedTensor
 * is a non-owning view, and doesn't ensure that the underlying memory stays valid.
 */
template <typename MdspanType, typename = raft::is_mdspan_t<MdspanType>>
void to_dlpack(MdspanType src, DLManagedTensor* dst)
{
  return detail::to_dlpack(src, dst);
}

/**
 * @}
 */

}  // namespace cuvs::core
