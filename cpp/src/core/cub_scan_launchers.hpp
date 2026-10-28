/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <thrust/device_ptr.h>

#include <cub/device/device_scan.cuh>

namespace cuvs::detail::scan_launch {
using sum_i32_u32_inclusive_pointer = cub::detail::scan::DeviceScanKernelSource<
  cub::detail::scan::
    policy_selector_from_types<int*, int*, int, unsigned int, cuda::std::plus<void>, false>,
  int*,
  int*,
  cuda::std::plus<void>,
  cub::NullType,
  unsigned int,
  int,
  cub::ForceInclusive::No,
  false>;
using sum_i32_u32_exclusive_pointer = cub::detail::scan::DeviceScanKernelSource<
  cub::detail::scan::
    policy_selector_from_types<int*, int*, int, unsigned int, cuda::std::plus<void>, false>,
  int*,
  int*,
  cuda::std::plus<void>,
  cub::detail::InputValue<int, int*>,
  unsigned int,
  int,
  cub::ForceInclusive::No,
  false>;
using sum_i32_ulong_inclusive_pointer = cub::detail::scan::DeviceScanKernelSource<
  cub::detail::scan::
    policy_selector_from_types<int*, int*, int, unsigned long, cuda::std::plus<void>, false>,
  int*,
  int*,
  cuda::std::plus<void>,
  cub::NullType,
  unsigned long,
  int,
  cub::ForceInclusive::No,
  false>;
using sum_i32_ulong_exclusive_pointer = cub::detail::scan::DeviceScanKernelSource<
  cub::detail::scan::
    policy_selector_from_types<int*, int*, int, unsigned long, cuda::std::plus<void>, false>,
  int*,
  int*,
  cuda::std::plus<void>,
  cub::detail::InputValue<int, int*>,
  unsigned long,
  int,
  cub::ForceInclusive::No,
  false>;
using sum_i64_u32_exclusive_pointer = cub::detail::scan::DeviceScanKernelSource<
  cub::detail::scan::
    policy_selector_from_types<long*, long*, long, unsigned int, cuda::std::plus<void>, false>,
  long*,
  long*,
  cuda::std::plus<void>,
  cub::detail::InputValue<long, long*>,
  unsigned int,
  long,
  cub::ForceInclusive::No,
  false>;
using sum_i64_u64_exclusive_pointer = cub::detail::scan::DeviceScanKernelSource<
  cub::detail::scan::policy_selector_from_types<long*,
                                                long*,
                                                long,
                                                unsigned long long,
                                                cuda::std::plus<void>,
                                                false>,
  long*,
  long*,
  cuda::std::plus<void>,
  cub::detail::InputValue<long, long*>,
  unsigned long long,
  long,
  cub::ForceInclusive::No,
  false>;
using sum_i32_u32_exclusive_device = cub::detail::scan::DeviceScanKernelSource<
  cub::detail::scan::policy_selector_from_types<thrust::device_ptr<int>,
                                                thrust::device_ptr<int>,
                                                int,
                                                unsigned int,
                                                cuda::std::plus<void>,
                                                false>,
  int*,
  int*,
  cuda::std::plus<void>,
  cub::detail::InputValue<int, int*>,
  unsigned int,
  int,
  cub::ForceInclusive::No,
  false>;
using sum_i32_ulong_exclusive_device = cub::detail::scan::DeviceScanKernelSource<
  cub::detail::scan::policy_selector_from_types<thrust::device_ptr<int>,
                                                thrust::device_ptr<int>,
                                                int,
                                                unsigned long,
                                                cuda::std::plus<void>,
                                                false>,
  int*,
  int*,
  cuda::std::plus<void>,
  cub::detail::InputValue<int, int*>,
  unsigned long,
  int,
  cub::ForceInclusive::No,
  false>;
using sum_i64_u32_exclusive_device = cub::detail::scan::DeviceScanKernelSource<
  cub::detail::scan::policy_selector_from_types<thrust::device_ptr<long>,
                                                thrust::device_ptr<long>,
                                                long,
                                                unsigned int,
                                                cuda::std::plus<void>,
                                                false>,
  long*,
  long*,
  cuda::std::plus<void>,
  cub::detail::InputValue<long, long*>,
  unsigned int,
  long,
  cub::ForceInclusive::No,
  false>;
using sum_i64_ulong_exclusive_device = cub::detail::scan::DeviceScanKernelSource<
  cub::detail::scan::policy_selector_from_types<thrust::device_ptr<long>,
                                                thrust::device_ptr<long>,
                                                long,
                                                unsigned long,
                                                cuda::std::plus<void>,
                                                false>,
  long*,
  long*,
  cuda::std::plus<void>,
  cub::detail::InputValue<long, long*>,
  unsigned long,
  long,
  cub::ForceInclusive::No,
  false>;

using sum_i32_u32_inclusive_pointer_InitKernel_pointer =
  void (*)(cub::detail::scan::tile_state_kernel_arg_t<cub::ScanTileState<int, true>, int>, int);
using sum_i32_u32_exclusive_pointer_ScanKernel_pointer =
  void (*)(int*,
           int*,
           cub::detail::scan::tile_state_kernel_arg_t<cub::ScanTileState<int, true>, int>,
           int,
           cuda::std::plus<void>,
           cub::detail::InputValue<int, int*>,
           unsigned int,
           int);
using sum_i32_u32_exclusive_pointer_InitKernel_pointer =
  void (*)(cub::detail::scan::tile_state_kernel_arg_t<cub::ScanTileState<int, true>, int>, int);
using sum_i32_ulong_inclusive_pointer_InitKernel_pointer =
  void (*)(cub::detail::scan::tile_state_kernel_arg_t<cub::ScanTileState<int, true>, int>, int);
using sum_i32_ulong_exclusive_pointer_ScanKernel_pointer =
  void (*)(int*,
           int*,
           cub::detail::scan::tile_state_kernel_arg_t<cub::ScanTileState<int, true>, int>,
           int,
           cuda::std::plus<void>,
           cub::detail::InputValue<int, int*>,
           unsigned long,
           int);
using sum_i32_ulong_exclusive_pointer_InitKernel_pointer =
  void (*)(cub::detail::scan::tile_state_kernel_arg_t<cub::ScanTileState<int, true>, int>, int);
using sum_i64_u32_exclusive_pointer_ScanKernel_pointer =
  void (*)(long*,
           long*,
           cub::detail::scan::tile_state_kernel_arg_t<cub::ScanTileState<long, true>, long>,
           int,
           cuda::std::plus<void>,
           cub::detail::InputValue<long, long*>,
           unsigned int,
           int);
using sum_i64_u32_exclusive_pointer_InitKernel_pointer =
  void (*)(cub::detail::scan::tile_state_kernel_arg_t<cub::ScanTileState<long, true>, long>, int);
using sum_i64_u64_exclusive_pointer_ScanKernel_pointer =
  void (*)(long*,
           long*,
           cub::detail::scan::tile_state_kernel_arg_t<cub::ScanTileState<long, true>, long>,
           int,
           cuda::std::plus<void>,
           cub::detail::InputValue<long, long*>,
           unsigned long long,
           int);
using sum_i64_u64_exclusive_pointer_InitKernel_pointer =
  void (*)(cub::detail::scan::tile_state_kernel_arg_t<cub::ScanTileState<long, true>, long>, int);
using sum_i32_u32_exclusive_device_ScanKernel_pointer =
  void (*)(int*,
           int*,
           cub::detail::scan::tile_state_kernel_arg_t<cub::ScanTileState<int, true>, int>,
           int,
           cuda::std::plus<void>,
           cub::detail::InputValue<int, int*>,
           unsigned int,
           int);
using sum_i32_u32_exclusive_device_InitKernel_pointer =
  void (*)(cub::detail::scan::tile_state_kernel_arg_t<cub::ScanTileState<int, true>, int>, int);
using sum_i32_ulong_exclusive_device_ScanKernel_pointer =
  void (*)(int*,
           int*,
           cub::detail::scan::tile_state_kernel_arg_t<cub::ScanTileState<int, true>, int>,
           int,
           cuda::std::plus<void>,
           cub::detail::InputValue<int, int*>,
           unsigned long,
           int);
using sum_i32_ulong_exclusive_device_InitKernel_pointer =
  void (*)(cub::detail::scan::tile_state_kernel_arg_t<cub::ScanTileState<int, true>, int>, int);
using sum_i64_u32_exclusive_device_ScanKernel_pointer =
  void (*)(long*,
           long*,
           cub::detail::scan::tile_state_kernel_arg_t<cub::ScanTileState<long, true>, long>,
           int,
           cuda::std::plus<void>,
           cub::detail::InputValue<long, long*>,
           unsigned int,
           int);
using sum_i64_u32_exclusive_device_InitKernel_pointer =
  void (*)(cub::detail::scan::tile_state_kernel_arg_t<cub::ScanTileState<long, true>, long>, int);
using sum_i64_ulong_exclusive_device_ScanKernel_pointer =
  void (*)(long*,
           long*,
           cub::detail::scan::tile_state_kernel_arg_t<cub::ScanTileState<long, true>, long>,
           int,
           cuda::std::plus<void>,
           cub::detail::InputValue<long, long*>,
           unsigned long,
           int);
using sum_i64_ulong_exclusive_device_InitKernel_pointer =
  void (*)(cub::detail::scan::tile_state_kernel_arg_t<cub::ScanTileState<long, true>, long>, int);
}  // namespace cuvs::detail::scan_launch

template <>
__host__ __device__ cuvs::detail::scan_launch::sum_i32_u32_inclusive_pointer_InitKernel_pointer
cuvs::detail::scan_launch::sum_i32_u32_inclusive_pointer::InitKernel();

template <>
__host__ __device__ cuvs::detail::scan_launch::sum_i32_u32_exclusive_pointer_ScanKernel_pointer
cuvs::detail::scan_launch::sum_i32_u32_exclusive_pointer::ScanKernel();

template <>
__host__ __device__ cuvs::detail::scan_launch::sum_i32_u32_exclusive_pointer_InitKernel_pointer
cuvs::detail::scan_launch::sum_i32_u32_exclusive_pointer::InitKernel();

template <>
__host__ __device__ cuvs::detail::scan_launch::sum_i32_ulong_inclusive_pointer_InitKernel_pointer
cuvs::detail::scan_launch::sum_i32_ulong_inclusive_pointer::InitKernel();

template <>
__host__ __device__ cuvs::detail::scan_launch::sum_i32_ulong_exclusive_pointer_ScanKernel_pointer
cuvs::detail::scan_launch::sum_i32_ulong_exclusive_pointer::ScanKernel();

template <>
__host__ __device__ cuvs::detail::scan_launch::sum_i32_ulong_exclusive_pointer_InitKernel_pointer
cuvs::detail::scan_launch::sum_i32_ulong_exclusive_pointer::InitKernel();

template <>
__host__ __device__ cuvs::detail::scan_launch::sum_i64_u32_exclusive_pointer_ScanKernel_pointer
cuvs::detail::scan_launch::sum_i64_u32_exclusive_pointer::ScanKernel();

template <>
__host__ __device__ cuvs::detail::scan_launch::sum_i64_u32_exclusive_pointer_InitKernel_pointer
cuvs::detail::scan_launch::sum_i64_u32_exclusive_pointer::InitKernel();

template <>
__host__ __device__ cuvs::detail::scan_launch::sum_i64_u64_exclusive_pointer_ScanKernel_pointer
cuvs::detail::scan_launch::sum_i64_u64_exclusive_pointer::ScanKernel();

template <>
__host__ __device__ cuvs::detail::scan_launch::sum_i64_u64_exclusive_pointer_InitKernel_pointer
cuvs::detail::scan_launch::sum_i64_u64_exclusive_pointer::InitKernel();

template <>
__host__ __device__ cuvs::detail::scan_launch::sum_i32_u32_exclusive_device_ScanKernel_pointer
cuvs::detail::scan_launch::sum_i32_u32_exclusive_device::ScanKernel();

template <>
__host__ __device__ cuvs::detail::scan_launch::sum_i32_u32_exclusive_device_InitKernel_pointer
cuvs::detail::scan_launch::sum_i32_u32_exclusive_device::InitKernel();

template <>
__host__ __device__ cuvs::detail::scan_launch::sum_i32_ulong_exclusive_device_ScanKernel_pointer
cuvs::detail::scan_launch::sum_i32_ulong_exclusive_device::ScanKernel();

template <>
__host__ __device__ cuvs::detail::scan_launch::sum_i32_ulong_exclusive_device_InitKernel_pointer
cuvs::detail::scan_launch::sum_i32_ulong_exclusive_device::InitKernel();

template <>
__host__ __device__ cuvs::detail::scan_launch::sum_i64_u32_exclusive_device_ScanKernel_pointer
cuvs::detail::scan_launch::sum_i64_u32_exclusive_device::ScanKernel();

template <>
__host__ __device__ cuvs::detail::scan_launch::sum_i64_u32_exclusive_device_InitKernel_pointer
cuvs::detail::scan_launch::sum_i64_u32_exclusive_device::InitKernel();

template <>
__host__ __device__ cuvs::detail::scan_launch::sum_i64_ulong_exclusive_device_ScanKernel_pointer
cuvs::detail::scan_launch::sum_i64_ulong_exclusive_device::ScanKernel();

template <>
__host__ __device__ cuvs::detail::scan_launch::sum_i64_ulong_exclusive_device_InitKernel_pointer
cuvs::detail::scan_launch::sum_i64_ulong_exclusive_device::InitKernel();

namespace cuvs::detail::scan_launch {
using sum_i64_ulong_exclusive_pointer = cub::detail::scan::DeviceScanKernelSource<
  cub::detail::scan::
    policy_selector_from_types<long*, long*, long, unsigned long, cuda::std::plus<void>, false>,
  long*,
  long*,
  cuda::std::plus<void>,
  cub::detail::InputValue<long, long*>,
  unsigned long,
  long,
  cub::ForceInclusive::No,
  false>;
using sum_i64_ulong_exclusive_pointer_ScanKernel_pointer =
  void (*)(long*,
           long*,
           cub::detail::scan::tile_state_kernel_arg_t<cub::ScanTileState<long, true>, long>,
           int,
           cuda::std::plus<void>,
           cub::detail::InputValue<long, long*>,
           unsigned long,
           int);
using sum_i64_ulong_exclusive_pointer_InitKernel_pointer =
  void (*)(cub::detail::scan::tile_state_kernel_arg_t<cub::ScanTileState<long, true>, long>, int);
}  // namespace cuvs::detail::scan_launch

template <>
__host__ __device__ cuvs::detail::scan_launch::sum_i64_ulong_exclusive_pointer_ScanKernel_pointer
cuvs::detail::scan_launch::sum_i64_ulong_exclusive_pointer::ScanKernel();

template <>
__host__ __device__ cuvs::detail::scan_launch::sum_i64_ulong_exclusive_pointer_InitKernel_pointer
cuvs::detail::scan_launch::sum_i64_ulong_exclusive_pointer::InitKernel();

namespace cuvs::detail::scan_launch {
using sum_i32_i64_ulong_exclusive_pointer = cub::detail::scan::DeviceScanKernelSource<
  cub::detail::scan::
    policy_selector_from_types<int*, long*, int, unsigned long, cuda::std::plus<void>, false>,
  int*,
  long*,
  cuda::std::plus<void>,
  cub::detail::InputValue<int, int*>,
  unsigned long,
  int,
  cub::ForceInclusive::No,
  false>;
using sum_i32_i64_u32_exclusive_pointer = cub::detail::scan::DeviceScanKernelSource<
  cub::detail::scan::
    policy_selector_from_types<int*, long*, int, unsigned int, cuda::std::plus<void>, false>,
  int*,
  long*,
  cuda::std::plus<void>,
  cub::detail::InputValue<int, int*>,
  unsigned int,
  int,
  cub::ForceInclusive::No,
  false>;
using sum_i32_i64_ulong_exclusive_pointer_ScanKernel_pointer =
  void (*)(int*,
           long*,
           cub::detail::scan::tile_state_kernel_arg_t<cub::ScanTileState<int, true>, int>,
           int,
           cuda::std::plus<void>,
           cub::detail::InputValue<int, int*>,
           unsigned long,
           int);
using sum_i32_i64_ulong_exclusive_pointer_InitKernel_pointer =
  void (*)(cub::detail::scan::tile_state_kernel_arg_t<cub::ScanTileState<int, true>, int>, int);
using sum_i32_i64_u32_exclusive_pointer_ScanKernel_pointer =
  void (*)(int*,
           long*,
           cub::detail::scan::tile_state_kernel_arg_t<cub::ScanTileState<int, true>, int>,
           int,
           cuda::std::plus<void>,
           cub::detail::InputValue<int, int*>,
           unsigned int,
           int);
using sum_i32_i64_u32_exclusive_pointer_InitKernel_pointer =
  void (*)(cub::detail::scan::tile_state_kernel_arg_t<cub::ScanTileState<int, true>, int>, int);
}  // namespace cuvs::detail::scan_launch

template <>
__host__ __device__
  cuvs::detail::scan_launch::sum_i32_i64_ulong_exclusive_pointer_ScanKernel_pointer
  cuvs::detail::scan_launch::sum_i32_i64_ulong_exclusive_pointer::ScanKernel();

template <>
__host__ __device__
  cuvs::detail::scan_launch::sum_i32_i64_ulong_exclusive_pointer_InitKernel_pointer
  cuvs::detail::scan_launch::sum_i32_i64_ulong_exclusive_pointer::InitKernel();

template <>
__host__ __device__ cuvs::detail::scan_launch::sum_i32_i64_u32_exclusive_pointer_ScanKernel_pointer
cuvs::detail::scan_launch::sum_i32_i64_u32_exclusive_pointer::ScanKernel();

template <>
__host__ __device__ cuvs::detail::scan_launch::sum_i32_i64_u32_exclusive_pointer_InitKernel_pointer
cuvs::detail::scan_launch::sum_i32_i64_u32_exclusive_pointer::InitKernel();
