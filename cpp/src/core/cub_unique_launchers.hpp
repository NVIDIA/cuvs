/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <thrust/device_ptr.h>

#include <thrust/unique.h>

#include <cub/device/dispatch/dispatch_unique_by_key.cuh>

namespace cuvs::detail::unique_launch {
using unique_0 = cub::detail::unique_by_key::DeviceUniqueByKeyKernelSource<
  cub::detail::unique_by_key::policy_selector_from_types<unsigned int, int>,
  thrust::device_ptr<unsigned int>,
  thrust::device_ptr<int>,
  thrust::device_ptr<unsigned int>,
  thrust::device_ptr<int>,
  unsigned int*,
  cub::ScanTileState<unsigned int, true>,
  thrust::detail::wrapped_function<cuda::std::equal_to<unsigned int>, bool>,
  unsigned int>;
using unique_1 = cub::detail::unique_by_key::DeviceUniqueByKeyKernelSource<
  cub::detail::unique_by_key::policy_selector_from_types<unsigned int, int>,
  thrust::device_ptr<unsigned int>,
  thrust::device_ptr<int>,
  thrust::device_ptr<unsigned int>,
  thrust::device_ptr<int>,
  unsigned long*,
  cub::ScanTileState<unsigned long long, true>,
  thrust::detail::wrapped_function<cuda::std::equal_to<unsigned int>, bool>,
  unsigned long long>;

using unique_0_UniqueByKeySweepKernel_pointer =
  void (*)(thrust::device_ptr<unsigned int>,
           thrust::device_ptr<int>,
           thrust::device_ptr<unsigned int>,
           thrust::device_ptr<int>,
           unsigned int*,
           cub::ScanTileState<unsigned int, true>,
           thrust::detail::wrapped_function<cuda::std::equal_to<unsigned int>, bool>,
           unsigned int,
           int,
           cub::detail::vsmem_t);
using unique_1_UniqueByKeySweepKernel_pointer =
  void (*)(thrust::device_ptr<unsigned int>,
           thrust::device_ptr<int>,
           thrust::device_ptr<unsigned int>,
           thrust::device_ptr<int>,
           unsigned long*,
           cub::ScanTileState<unsigned long long, true>,
           thrust::detail::wrapped_function<cuda::std::equal_to<unsigned int>, bool>,
           unsigned long long,
           int,
           cub::detail::vsmem_t);
}  // namespace cuvs::detail::unique_launch

template <>
__host__ __device__ cuvs::detail::unique_launch::unique_0_UniqueByKeySweepKernel_pointer
cuvs::detail::unique_launch::unique_0::UniqueByKeySweepKernel();

template <>
__host__ __device__ cuvs::detail::unique_launch::unique_1_UniqueByKeySweepKernel_pointer
cuvs::detail::unique_launch::unique_1::UniqueByKeySweepKernel();
