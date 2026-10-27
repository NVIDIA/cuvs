/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <thrust/device_ptr.h>

#include <cub/device/device_reduce.cuh>
#include <thrust/reduce.h>

namespace cuvs::detail::reduce_launch {
using sum_second_0_0 = cub::detail::reduce::DeviceReduceKernelSource<
  cub::detail::reduce::policy_selector_from_types<long,
                                                  unsigned int,
                                                  cuda::std::plus<long>,
                                                  (cuda::execution::determinism::__determinism_t)1>,
  thrust::device_ptr<long>,
  long*,
  unsigned int,
  unsigned int,
  cuda::std::plus<long>,
  long,
  long,
  cuda::std::identity,
  true>;
using sum_second_0_1 = cub::detail::reduce::DeviceReduceKernelSource<
  cub::detail::reduce::policy_selector_from_types<long,
                                                  unsigned int,
                                                  cuda::std::plus<long>,
                                                  (cuda::execution::determinism::__determinism_t)1>,
  long*,
  long*,
  unsigned int,
  unsigned int,
  cuda::std::plus<long>,
  long,
  long,
  cuda::std::identity,
  true>;
using sum_second_1_0 = cub::detail::reduce::DeviceReduceKernelSource<
  cub::detail::reduce::policy_selector_from_types<long,
                                                  unsigned long long,
                                                  cuda::std::plus<long>,
                                                  (cuda::execution::determinism::__determinism_t)1>,
  thrust::device_ptr<long>,
  long*,
  unsigned long long,
  unsigned long long,
  cuda::std::plus<long>,
  long,
  long,
  cuda::std::identity,
  true>;
using sum_second_1_1 = cub::detail::reduce::DeviceReduceKernelSource<
  cub::detail::reduce::policy_selector_from_types<long,
                                                  unsigned long long,
                                                  cuda::std::plus<long>,
                                                  (cuda::execution::determinism::__determinism_t)1>,
  long*,
  long*,
  unsigned long long,
  unsigned long long,
  cuda::std::plus<long>,
  long,
  long,
  cuda::std::identity,
  true>;

using sum_second_0_0_SingleTileSecondKernel_pointer =
  void (*)(long*, long*, int, cuda::std::plus<long>, long, cuda::std::identity);
using sum_second_0_1_SingleTileSecondKernel_pointer =
  void (*)(long*, long*, int, cuda::std::plus<long>, long, cuda::std::identity);
using sum_second_1_0_SingleTileSecondKernel_pointer =
  void (*)(long*, long*, int, cuda::std::plus<long>, long, cuda::std::identity);
using sum_second_1_1_SingleTileSecondKernel_pointer =
  void (*)(long*, long*, int, cuda::std::plus<long>, long, cuda::std::identity);
}  // namespace cuvs::detail::reduce_launch

template <>
__host__ __device__ cuvs::detail::reduce_launch::sum_second_0_0_SingleTileSecondKernel_pointer
cuvs::detail::reduce_launch::sum_second_0_0::SingleTileSecondKernel();

template <>
__host__ __device__ cuvs::detail::reduce_launch::sum_second_0_1_SingleTileSecondKernel_pointer
cuvs::detail::reduce_launch::sum_second_0_1::SingleTileSecondKernel();

template <>
__host__ __device__ cuvs::detail::reduce_launch::sum_second_1_0_SingleTileSecondKernel_pointer
cuvs::detail::reduce_launch::sum_second_1_0::SingleTileSecondKernel();

template <>
__host__ __device__ cuvs::detail::reduce_launch::sum_second_1_1_SingleTileSecondKernel_pointer
cuvs::detail::reduce_launch::sum_second_1_1::SingleTileSecondKernel();

namespace cuvs::detail::reduce_launch {
using argmin_f32_i32 = cub::detail::reduce::DeviceReduceKernelSource<
  cub::detail::reduce::policy_selector_from_types<cub::KeyValuePair<int, float>,
                                                  int,
                                                  cub::detail::arg_reduce_op<cuda::std::less<void>>,
                                                  (cuda::execution::determinism::__determinism_t)1>,
  cub::ArgIndexInputIterator<float*, int, float>,
  cub::KeyValuePair<int, float>*,
  int,
  int,
  cub::detail::arg_reduce_op<cuda::std::less<void>>,
  cub::detail::reduce::empty_problem_init_t<cub::KeyValuePair<int, float>>,
  cub::KeyValuePair<int, float>,
  cuda::std::identity,
  true>;
using argmin_f64_i32 = cub::detail::reduce::DeviceReduceKernelSource<
  cub::detail::reduce::policy_selector_from_types<cub::KeyValuePair<int, double>,
                                                  int,
                                                  cub::detail::arg_reduce_op<cuda::std::less<void>>,
                                                  (cuda::execution::determinism::__determinism_t)1>,
  cub::ArgIndexInputIterator<double*, int, double>,
  cub::KeyValuePair<int, double>*,
  int,
  int,
  cub::detail::arg_reduce_op<cuda::std::less<void>>,
  cub::detail::reduce::empty_problem_init_t<cub::KeyValuePair<int, double>>,
  cub::KeyValuePair<int, double>,
  cuda::std::identity,
  true>;
using argmin_f32_i32_SingleTileSecondKernel_pointer =
  void (*)(cub::KeyValuePair<int, float>*,
           cub::KeyValuePair<int, float>*,
           int,
           cub::detail::arg_reduce_op<cuda::std::less<void>>,
           cub::detail::reduce::empty_problem_init_t<cub::KeyValuePair<int, float>>,
           cuda::std::identity);
using argmin_f32_i32_ReductionKernel_pointer =
  void (*)(cub::ArgIndexInputIterator<float*, int, float>,
           cub::KeyValuePair<int, float>*,
           int,
           cub::GridEvenShare<int>,
           cub::detail::arg_reduce_op<cuda::std::less<void>>,
           cub::detail::reduce::empty_problem_init_t<cub::KeyValuePair<int, float>>,
           cuda::std::identity);
using argmin_f32_i32_SingleTileKernel_pointer =
  void (*)(cub::ArgIndexInputIterator<float*, int, float>,
           cub::KeyValuePair<int, float>*,
           int,
           cub::detail::arg_reduce_op<cuda::std::less<void>>,
           cub::detail::reduce::empty_problem_init_t<cub::KeyValuePair<int, float>>,
           cuda::std::identity);
using argmin_f64_i32_SingleTileSecondKernel_pointer =
  void (*)(cub::KeyValuePair<int, double>*,
           cub::KeyValuePair<int, double>*,
           int,
           cub::detail::arg_reduce_op<cuda::std::less<void>>,
           cub::detail::reduce::empty_problem_init_t<cub::KeyValuePair<int, double>>,
           cuda::std::identity);
using argmin_f64_i32_ReductionKernel_pointer =
  void (*)(cub::ArgIndexInputIterator<double*, int, double>,
           cub::KeyValuePair<int, double>*,
           int,
           cub::GridEvenShare<int>,
           cub::detail::arg_reduce_op<cuda::std::less<void>>,
           cub::detail::reduce::empty_problem_init_t<cub::KeyValuePair<int, double>>,
           cuda::std::identity);
using argmin_f64_i32_SingleTileKernel_pointer =
  void (*)(cub::ArgIndexInputIterator<double*, int, double>,
           cub::KeyValuePair<int, double>*,
           int,
           cub::detail::arg_reduce_op<cuda::std::less<void>>,
           cub::detail::reduce::empty_problem_init_t<cub::KeyValuePair<int, double>>,
           cuda::std::identity);
}  // namespace cuvs::detail::reduce_launch

template <>
__host__ __device__ cuvs::detail::reduce_launch::argmin_f32_i32_SingleTileSecondKernel_pointer
cuvs::detail::reduce_launch::argmin_f32_i32::SingleTileSecondKernel();

template <>
__host__ __device__ cuvs::detail::reduce_launch::argmin_f32_i32_ReductionKernel_pointer
cuvs::detail::reduce_launch::argmin_f32_i32::ReductionKernel();

template <>
__host__ __device__ cuvs::detail::reduce_launch::argmin_f32_i32_SingleTileKernel_pointer
cuvs::detail::reduce_launch::argmin_f32_i32::SingleTileKernel();

template <>
__host__ __device__ cuvs::detail::reduce_launch::argmin_f64_i32_SingleTileSecondKernel_pointer
cuvs::detail::reduce_launch::argmin_f64_i32::SingleTileSecondKernel();

template <>
__host__ __device__ cuvs::detail::reduce_launch::argmin_f64_i32_ReductionKernel_pointer
cuvs::detail::reduce_launch::argmin_f64_i32::ReductionKernel();

template <>
__host__ __device__ cuvs::detail::reduce_launch::argmin_f64_i32_SingleTileKernel_pointer
cuvs::detail::reduce_launch::argmin_f64_i32::SingleTileKernel();
