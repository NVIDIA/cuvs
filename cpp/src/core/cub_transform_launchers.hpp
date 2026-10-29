/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <thrust/gather.h>
#include <thrust/sequence.h>

#include <cub/device/device_transform.cuh>

namespace cuvs::detail::transform_launch {
using transform_0 = cub::detail::transform::TransformKernelSource<
  cub::detail::transform::policy_selector_from_types<false, true, cuda::std::tuple<>, double*>,
  long,
  cuda::std::tuple<>,
  double*,
  cuda::always_true,
  cub::detail::__return_constant<double>>;
using transform_1 = cub::detail::transform::TransformKernelSource<
  cub::detail::transform::policy_selector_from_types<false, true, cuda::std::tuple<>, float*>,
  long,
  cuda::std::tuple<>,
  float*,
  cuda::always_true,
  cub::detail::__return_constant<float>>;
using transform_2 = cub::detail::transform::TransformKernelSource<
  cub::detail::transform::policy_selector_from_types<false, true, cuda::std::tuple<>, int*>,
  long,
  cuda::std::tuple<>,
  int*,
  cuda::always_true,
  cub::detail::__return_constant<int>>;
using transform_3 = cub::detail::transform::TransformKernelSource<
  cub::detail::transform::
    policy_selector_from_types<false, true, cuda::std::tuple<>, unsigned int*>,
  long,
  cuda::std::tuple<>,
  unsigned int*,
  cuda::always_true,
  cub::detail::__return_constant<unsigned int>>;
using transform_4 = cub::detail::transform::TransformKernelSource<
  cub::detail::transform::policy_selector_from_types<
    false,
    true,
    cuda::std::tuple<cuda::counting_iterator<long, __int128>>,
    int*>,
  long,
  cuda::std::tuple<cuda::counting_iterator<long, __int128>>,
  int*,
  cuda::always_true,
  cuda::__callable_permitting_copied_arguments<
    thrust::system::detail::generic::detail::compute_sequence_value<int>>>;
using transform_5 = cub::detail::transform::TransformKernelSource<
  cub::detail::transform::policy_selector_from_types<
    true,
    true,
    cuda::std::tuple<thrust::permutation_iterator<float const*, long*>>,
    float*>,
  long,
  cuda::std::tuple<thrust::permutation_iterator<float const*, long*>>,
  float*,
  cuda::always_true,
  cuda::std::identity>;
using transform_6 = cub::detail::transform::TransformKernelSource<
  cub::detail::transform::policy_selector_from_types<
    true,
    true,
    cuda::std::tuple<thrust::permutation_iterator<long*, long*>>,
    long*>,
  long,
  cuda::std::tuple<thrust::permutation_iterator<long*, long*>>,
  long*,
  cuda::always_true,
  cuda::std::identity>;

using transform_0_TransformKernel_pointer =
  void (*)(long, int, bool, cuda::always_true, cub::detail::__return_constant<double>, double*);
using transform_1_TransformKernel_pointer =
  void (*)(long, int, bool, cuda::always_true, cub::detail::__return_constant<float>, float*);
using transform_2_TransformKernel_pointer =
  void (*)(long, int, bool, cuda::always_true, cub::detail::__return_constant<int>, int*);
using transform_3_TransformKernel_pointer = void (*)(
  long, int, bool, cuda::always_true, cub::detail::__return_constant<unsigned int>, unsigned int*);
using transform_4_TransformKernel_pointer =
  void (*)(long,
           int,
           bool,
           cuda::always_true,
           cuda::__callable_permitting_copied_arguments<
             thrust::system::detail::generic::detail::compute_sequence_value<int>>,
           int*,
           cub::detail::transform::kernel_arg<cuda::counting_iterator<long, __int128>>);
using transform_5_TransformKernel_pointer =
  void (*)(long,
           int,
           bool,
           cuda::always_true,
           cuda::std::identity,
           float*,
           cub::detail::transform::kernel_arg<thrust::permutation_iterator<float const*, long*>>);
using transform_6_TransformKernel_pointer =
  void (*)(long,
           int,
           bool,
           cuda::always_true,
           cuda::std::identity,
           long*,
           cub::detail::transform::kernel_arg<thrust::permutation_iterator<long*, long*>>);
}  // namespace cuvs::detail::transform_launch

template <>
__host__ __device__ cuvs::detail::transform_launch::transform_0_TransformKernel_pointer
cuvs::detail::transform_launch::transform_0::TransformKernel();

template <>
__host__ __device__ cuvs::detail::transform_launch::transform_1_TransformKernel_pointer
cuvs::detail::transform_launch::transform_1::TransformKernel();

template <>
__host__ __device__ cuvs::detail::transform_launch::transform_2_TransformKernel_pointer
cuvs::detail::transform_launch::transform_2::TransformKernel();

template <>
__host__ __device__ cuvs::detail::transform_launch::transform_3_TransformKernel_pointer
cuvs::detail::transform_launch::transform_3::TransformKernel();

template <>
__host__ __device__ cuvs::detail::transform_launch::transform_4_TransformKernel_pointer
cuvs::detail::transform_launch::transform_4::TransformKernel();

template <>
__host__ __device__ cuvs::detail::transform_launch::transform_5_TransformKernel_pointer
cuvs::detail::transform_launch::transform_5::TransformKernel();

template <>
__host__ __device__ cuvs::detail::transform_launch::transform_6_TransformKernel_pointer
cuvs::detail::transform_launch::transform_6::TransformKernel();
