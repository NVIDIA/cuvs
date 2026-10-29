/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cluster/detail/kmeans_common.cuh>
#include <cub/device/device_reduce.cuh>

namespace cuvs::detail::cluster_cost_reduce_launch {
using source_transform_weighteddistanceop_false_f64_int_transform_top1nn_key_value_int_f64_ptr_value_op_key_value_int_f64_ptr_counting_iterator_int_long_f64_identity_op_add_op_int =
  cub::detail::reduce::DeviceReduceKernelSource<
    cub::detail::reduce::policy_selector_from_types<
      double,
      unsigned int,
      raft::add_op,
      (cuda::execution::determinism::__determinism_t)1>,
    ::cuda::transform_iterator<
      raft::identity_op,
      ::cuda::transform_iterator<
        cuvs::cluster::kmeans::detail::WeightedDistanceOp<
          false,
          double,
          int,
          ::cuda::transform_iterator<
            cuvs::distance::detail::Top1nnResultTraits<raft::KeyValuePair<int, double>*>::value_op,
            raft::KeyValuePair<int, double>*>>,
        ::cuda::counting_iterator<int, long>>>,
    double*,
    unsigned int,
    unsigned int,
    raft::add_op,
    double,
    double,
    cuda::std::identity,
    true>;
using source_transform_weighteddistanceop_false_f64_int_f64_ptr_counting_iterator_int_long_f64_identity_op_add_op_int =
  cub::detail::reduce::DeviceReduceKernelSource<
    cub::detail::reduce::policy_selector_from_types<
      double,
      unsigned int,
      raft::add_op,
      (cuda::execution::determinism::__determinism_t)1>,
    ::cuda::transform_iterator<
      raft::identity_op,
      ::cuda::transform_iterator<
        cuvs::cluster::kmeans::detail::WeightedDistanceOp<false, double, int, double*>,
        ::cuda::counting_iterator<int, long>>>,
    double*,
    unsigned int,
    unsigned int,
    raft::add_op,
    double,
    double,
    cuda::std::identity,
    true>;
using source_transform_weighteddistanceop_false_f64_long_transform_top1nn_key_value_long_f64_ptr_value_op_key_value_long_f64_ptr_counting_iterator_long_int128_f64_identity_op_add_op_long =
  cub::detail::reduce::DeviceReduceKernelSource<
    cub::detail::reduce::policy_selector_from_types<
      double,
      unsigned long long,
      raft::add_op,
      (cuda::execution::determinism::__determinism_t)1>,
    ::cuda::transform_iterator<
      raft::identity_op,
      ::cuda::transform_iterator<
        cuvs::cluster::kmeans::detail::WeightedDistanceOp<
          false,
          double,
          long,
          ::cuda::transform_iterator<
            cuvs::distance::detail::Top1nnResultTraits<raft::KeyValuePair<long, double>*>::value_op,
            raft::KeyValuePair<long, double>*>>,
        ::cuda::counting_iterator<long, __int128>>>,
    double*,
    unsigned long long,
    unsigned long long,
    raft::add_op,
    double,
    double,
    cuda::std::identity,
    true>;
using source_transform_weighteddistanceop_false_f64_long_f64_ptr_counting_iterator_long_int128_f64_identity_op_add_op_long =
  cub::detail::reduce::DeviceReduceKernelSource<
    cub::detail::reduce::policy_selector_from_types<
      double,
      unsigned long long,
      raft::add_op,
      (cuda::execution::determinism::__determinism_t)1>,
    ::cuda::transform_iterator<
      raft::identity_op,
      ::cuda::transform_iterator<
        cuvs::cluster::kmeans::detail::WeightedDistanceOp<false, double, long, double*>,
        ::cuda::counting_iterator<long, __int128>>>,
    double*,
    unsigned long long,
    unsigned long long,
    raft::add_op,
    double,
    double,
    cuda::std::identity,
    true>;
using source_transform_weighteddistanceop_false_f32_int_transform_top1nn_key_value_int_f32_ptr_value_op_key_value_int_f32_ptr_counting_iterator_int_long_f32_identity_op_add_op_int =
  cub::detail::reduce::DeviceReduceKernelSource<
    cub::detail::reduce::policy_selector_from_types<
      float,
      unsigned int,
      raft::add_op,
      (cuda::execution::determinism::__determinism_t)1>,
    ::cuda::transform_iterator<
      raft::identity_op,
      ::cuda::transform_iterator<
        cuvs::cluster::kmeans::detail::WeightedDistanceOp<
          false,
          float,
          int,
          ::cuda::transform_iterator<
            cuvs::distance::detail::Top1nnResultTraits<raft::KeyValuePair<int, float>*>::value_op,
            raft::KeyValuePair<int, float>*>>,
        ::cuda::counting_iterator<int, long>>>,
    float*,
    unsigned int,
    unsigned int,
    raft::add_op,
    float,
    float,
    cuda::std::identity,
    true>;
using source_transform_weighteddistanceop_false_f32_int_f32_ptr_counting_iterator_int_long_f32_identity_op_add_op_int =
  cub::detail::reduce::DeviceReduceKernelSource<
    cub::detail::reduce::policy_selector_from_types<
      float,
      unsigned int,
      raft::add_op,
      (cuda::execution::determinism::__determinism_t)1>,
    ::cuda::transform_iterator<
      raft::identity_op,
      ::cuda::transform_iterator<
        cuvs::cluster::kmeans::detail::WeightedDistanceOp<false, float, int, float*>,
        ::cuda::counting_iterator<int, long>>>,
    float*,
    unsigned int,
    unsigned int,
    raft::add_op,
    float,
    float,
    cuda::std::identity,
    true>;
using source_transform_weighteddistanceop_false_f32_long_transform_top1nn_key_value_long_f32_ptr_value_op_key_value_long_f32_ptr_counting_iterator_long_int128_f32_identity_op_add_op_long =
  cub::detail::reduce::DeviceReduceKernelSource<
    cub::detail::reduce::policy_selector_from_types<
      float,
      unsigned long long,
      raft::add_op,
      (cuda::execution::determinism::__determinism_t)1>,
    ::cuda::transform_iterator<
      raft::identity_op,
      ::cuda::transform_iterator<
        cuvs::cluster::kmeans::detail::WeightedDistanceOp<
          false,
          float,
          long,
          ::cuda::transform_iterator<
            cuvs::distance::detail::Top1nnResultTraits<raft::KeyValuePair<long, float>*>::value_op,
            raft::KeyValuePair<long, float>*>>,
        ::cuda::counting_iterator<long, __int128>>>,
    float*,
    unsigned long long,
    unsigned long long,
    raft::add_op,
    float,
    float,
    cuda::std::identity,
    true>;
using source_transform_weighteddistanceop_false_f32_long_f32_ptr_counting_iterator_long_int128_f32_identity_op_add_op_long =
  cub::detail::reduce::DeviceReduceKernelSource<
    cub::detail::reduce::policy_selector_from_types<
      float,
      unsigned long long,
      raft::add_op,
      (cuda::execution::determinism::__determinism_t)1>,
    ::cuda::transform_iterator<
      raft::identity_op,
      ::cuda::transform_iterator<
        cuvs::cluster::kmeans::detail::WeightedDistanceOp<false, float, long, float*>,
        ::cuda::counting_iterator<long, __int128>>>,
    float*,
    unsigned long long,
    unsigned long long,
    raft::add_op,
    float,
    float,
    cuda::std::identity,
    true>;
using source_transform_weighteddistanceop_true_f64_int_transform_top1nn_key_value_int_f64_ptr_value_op_key_value_int_f64_ptr_counting_iterator_int_long_f64_identity_op_add_op_int =
  cub::detail::reduce::DeviceReduceKernelSource<
    cub::detail::reduce::policy_selector_from_types<
      double,
      unsigned int,
      raft::add_op,
      (cuda::execution::determinism::__determinism_t)1>,
    ::cuda::transform_iterator<
      raft::identity_op,
      ::cuda::transform_iterator<
        cuvs::cluster::kmeans::detail::WeightedDistanceOp<
          true,
          double,
          int,
          ::cuda::transform_iterator<
            cuvs::distance::detail::Top1nnResultTraits<raft::KeyValuePair<int, double>*>::value_op,
            raft::KeyValuePair<int, double>*>>,
        ::cuda::counting_iterator<int, long>>>,
    double*,
    unsigned int,
    unsigned int,
    raft::add_op,
    double,
    double,
    cuda::std::identity,
    true>;
using source_transform_weighteddistanceop_true_f64_int_f64_ptr_counting_iterator_int_long_f64_identity_op_add_op_int =
  cub::detail::reduce::DeviceReduceKernelSource<
    cub::detail::reduce::policy_selector_from_types<
      double,
      unsigned int,
      raft::add_op,
      (cuda::execution::determinism::__determinism_t)1>,
    ::cuda::transform_iterator<
      raft::identity_op,
      ::cuda::transform_iterator<
        cuvs::cluster::kmeans::detail::WeightedDistanceOp<true, double, int, double*>,
        ::cuda::counting_iterator<int, long>>>,
    double*,
    unsigned int,
    unsigned int,
    raft::add_op,
    double,
    double,
    cuda::std::identity,
    true>;
using source_transform_weighteddistanceop_true_f64_long_transform_top1nn_key_value_long_f64_ptr_value_op_key_value_long_f64_ptr_counting_iterator_long_int128_f64_identity_op_add_op_long =
  cub::detail::reduce::DeviceReduceKernelSource<
    cub::detail::reduce::policy_selector_from_types<
      double,
      unsigned long long,
      raft::add_op,
      (cuda::execution::determinism::__determinism_t)1>,
    ::cuda::transform_iterator<
      raft::identity_op,
      ::cuda::transform_iterator<
        cuvs::cluster::kmeans::detail::WeightedDistanceOp<
          true,
          double,
          long,
          ::cuda::transform_iterator<
            cuvs::distance::detail::Top1nnResultTraits<raft::KeyValuePair<long, double>*>::value_op,
            raft::KeyValuePair<long, double>*>>,
        ::cuda::counting_iterator<long, __int128>>>,
    double*,
    unsigned long long,
    unsigned long long,
    raft::add_op,
    double,
    double,
    cuda::std::identity,
    true>;
using source_transform_weighteddistanceop_true_f64_long_f64_ptr_counting_iterator_long_int128_f64_identity_op_add_op_long =
  cub::detail::reduce::DeviceReduceKernelSource<
    cub::detail::reduce::policy_selector_from_types<
      double,
      unsigned long long,
      raft::add_op,
      (cuda::execution::determinism::__determinism_t)1>,
    ::cuda::transform_iterator<
      raft::identity_op,
      ::cuda::transform_iterator<
        cuvs::cluster::kmeans::detail::WeightedDistanceOp<true, double, long, double*>,
        ::cuda::counting_iterator<long, __int128>>>,
    double*,
    unsigned long long,
    unsigned long long,
    raft::add_op,
    double,
    double,
    cuda::std::identity,
    true>;
using source_transform_weighteddistanceop_true_f32_int_transform_top1nn_key_value_int_f32_ptr_value_op_key_value_int_f32_ptr_counting_iterator_int_long_f32_identity_op_add_op_int =
  cub::detail::reduce::DeviceReduceKernelSource<
    cub::detail::reduce::policy_selector_from_types<
      float,
      unsigned int,
      raft::add_op,
      (cuda::execution::determinism::__determinism_t)1>,
    ::cuda::transform_iterator<
      raft::identity_op,
      ::cuda::transform_iterator<
        cuvs::cluster::kmeans::detail::WeightedDistanceOp<
          true,
          float,
          int,
          ::cuda::transform_iterator<
            cuvs::distance::detail::Top1nnResultTraits<raft::KeyValuePair<int, float>*>::value_op,
            raft::KeyValuePair<int, float>*>>,
        ::cuda::counting_iterator<int, long>>>,
    float*,
    unsigned int,
    unsigned int,
    raft::add_op,
    float,
    float,
    cuda::std::identity,
    true>;
using source_transform_weighteddistanceop_true_f32_int_f32_ptr_counting_iterator_int_long_f32_identity_op_add_op_int =
  cub::detail::reduce::DeviceReduceKernelSource<
    cub::detail::reduce::policy_selector_from_types<
      float,
      unsigned int,
      raft::add_op,
      (cuda::execution::determinism::__determinism_t)1>,
    ::cuda::transform_iterator<
      raft::identity_op,
      ::cuda::transform_iterator<
        cuvs::cluster::kmeans::detail::WeightedDistanceOp<true, float, int, float*>,
        ::cuda::counting_iterator<int, long>>>,
    float*,
    unsigned int,
    unsigned int,
    raft::add_op,
    float,
    float,
    cuda::std::identity,
    true>;
using source_transform_weighteddistanceop_true_f32_long_transform_top1nn_key_value_long_f32_ptr_value_op_key_value_long_f32_ptr_counting_iterator_long_int128_f32_identity_op_add_op_long =
  cub::detail::reduce::DeviceReduceKernelSource<
    cub::detail::reduce::policy_selector_from_types<
      float,
      unsigned long long,
      raft::add_op,
      (cuda::execution::determinism::__determinism_t)1>,
    ::cuda::transform_iterator<
      raft::identity_op,
      ::cuda::transform_iterator<
        cuvs::cluster::kmeans::detail::WeightedDistanceOp<
          true,
          float,
          long,
          ::cuda::transform_iterator<
            cuvs::distance::detail::Top1nnResultTraits<raft::KeyValuePair<long, float>*>::value_op,
            raft::KeyValuePair<long, float>*>>,
        ::cuda::counting_iterator<long, __int128>>>,
    float*,
    unsigned long long,
    unsigned long long,
    raft::add_op,
    float,
    float,
    cuda::std::identity,
    true>;
using source_transform_weighteddistanceop_true_f32_long_f32_ptr_counting_iterator_long_int128_f32_identity_op_add_op_long =
  cub::detail::reduce::DeviceReduceKernelSource<
    cub::detail::reduce::policy_selector_from_types<
      float,
      unsigned long long,
      raft::add_op,
      (cuda::execution::determinism::__determinism_t)1>,
    ::cuda::transform_iterator<
      raft::identity_op,
      ::cuda::transform_iterator<
        cuvs::cluster::kmeans::detail::WeightedDistanceOp<true, float, long, float*>,
        ::cuda::counting_iterator<long, __int128>>>,
    float*,
    unsigned long long,
    unsigned long long,
    raft::add_op,
    float,
    float,
    cuda::std::identity,
    true>;
using source_f64_ptr_f64_identity_op_add_op_u32 = cub::detail::reduce::DeviceReduceKernelSource<
  cub::detail::reduce::policy_selector_from_types<double,
                                                  unsigned int,
                                                  raft::add_op,
                                                  (cuda::execution::determinism::__determinism_t)1>,
  ::cuda::transform_iterator<raft::identity_op, double*>,
  double*,
  unsigned int,
  unsigned int,
  raft::add_op,
  double,
  double,
  cuda::std::identity,
  true>;
using source_f64_ptr_f64_identity_op_add_op_ulong = cub::detail::reduce::DeviceReduceKernelSource<
  cub::detail::reduce::policy_selector_from_types<double,
                                                  unsigned long long,
                                                  raft::add_op,
                                                  (cuda::execution::determinism::__determinism_t)1>,
  ::cuda::transform_iterator<raft::identity_op, double*>,
  double*,
  unsigned long long,
  unsigned long long,
  raft::add_op,
  double,
  double,
  cuda::std::identity,
  true>;
using source_f32_ptr_f32_identity_op_add_op_u32 = cub::detail::reduce::DeviceReduceKernelSource<
  cub::detail::reduce::policy_selector_from_types<float,
                                                  unsigned int,
                                                  raft::add_op,
                                                  (cuda::execution::determinism::__determinism_t)1>,
  ::cuda::transform_iterator<raft::identity_op, float*>,
  float*,
  unsigned int,
  unsigned int,
  raft::add_op,
  float,
  float,
  cuda::std::identity,
  true>;
using source_f32_ptr_f32_identity_op_add_op_ulong = cub::detail::reduce::DeviceReduceKernelSource<
  cub::detail::reduce::policy_selector_from_types<float,
                                                  unsigned long long,
                                                  raft::add_op,
                                                  (cuda::execution::determinism::__determinism_t)1>,
  ::cuda::transform_iterator<raft::identity_op, float*>,
  float*,
  unsigned long long,
  unsigned long long,
  raft::add_op,
  float,
  float,
  cuda::std::identity,
  true>;
using source_transform_weighteddistanceop_false_f64_int_transform_top1nn_key_value_int_f64_ptr_value_op_key_value_int_f64_ptr_counting_iterator_int_long_f64_identity_op_add_op_int_pointer =
  void (*)(double*, double*, int, raft::add_op, double, cuda::std::identity);
using source_transform_weighteddistanceop_false_f64_int_f64_ptr_counting_iterator_int_long_f64_identity_op_add_op_int_pointer =
  void (*)(double*, double*, int, raft::add_op, double, cuda::std::identity);
using source_transform_weighteddistanceop_false_f64_long_transform_top1nn_key_value_long_f64_ptr_value_op_key_value_long_f64_ptr_counting_iterator_long_int128_f64_identity_op_add_op_long_pointer =
  void (*)(double*, double*, int, raft::add_op, double, cuda::std::identity);
using source_transform_weighteddistanceop_false_f64_long_f64_ptr_counting_iterator_long_int128_f64_identity_op_add_op_long_pointer =
  void (*)(double*, double*, int, raft::add_op, double, cuda::std::identity);
using source_transform_weighteddistanceop_false_f32_int_transform_top1nn_key_value_int_f32_ptr_value_op_key_value_int_f32_ptr_counting_iterator_int_long_f32_identity_op_add_op_int_pointer =
  void (*)(float*, float*, int, raft::add_op, float, cuda::std::identity);
using source_transform_weighteddistanceop_false_f32_int_f32_ptr_counting_iterator_int_long_f32_identity_op_add_op_int_pointer =
  void (*)(float*, float*, int, raft::add_op, float, cuda::std::identity);
using source_transform_weighteddistanceop_false_f32_long_transform_top1nn_key_value_long_f32_ptr_value_op_key_value_long_f32_ptr_counting_iterator_long_int128_f32_identity_op_add_op_long_pointer =
  void (*)(float*, float*, int, raft::add_op, float, cuda::std::identity);
using source_transform_weighteddistanceop_false_f32_long_f32_ptr_counting_iterator_long_int128_f32_identity_op_add_op_long_pointer =
  void (*)(float*, float*, int, raft::add_op, float, cuda::std::identity);
using source_transform_weighteddistanceop_true_f64_int_transform_top1nn_key_value_int_f64_ptr_value_op_key_value_int_f64_ptr_counting_iterator_int_long_f64_identity_op_add_op_int_pointer =
  void (*)(double*, double*, int, raft::add_op, double, cuda::std::identity);
using source_transform_weighteddistanceop_true_f64_int_f64_ptr_counting_iterator_int_long_f64_identity_op_add_op_int_pointer =
  void (*)(double*, double*, int, raft::add_op, double, cuda::std::identity);
using source_transform_weighteddistanceop_true_f64_long_transform_top1nn_key_value_long_f64_ptr_value_op_key_value_long_f64_ptr_counting_iterator_long_int128_f64_identity_op_add_op_long_pointer =
  void (*)(double*, double*, int, raft::add_op, double, cuda::std::identity);
using source_transform_weighteddistanceop_true_f64_long_f64_ptr_counting_iterator_long_int128_f64_identity_op_add_op_long_pointer =
  void (*)(double*, double*, int, raft::add_op, double, cuda::std::identity);
using source_transform_weighteddistanceop_true_f32_int_transform_top1nn_key_value_int_f32_ptr_value_op_key_value_int_f32_ptr_counting_iterator_int_long_f32_identity_op_add_op_int_pointer =
  void (*)(float*, float*, int, raft::add_op, float, cuda::std::identity);
using source_transform_weighteddistanceop_true_f32_int_f32_ptr_counting_iterator_int_long_f32_identity_op_add_op_int_pointer =
  void (*)(float*, float*, int, raft::add_op, float, cuda::std::identity);
using source_transform_weighteddistanceop_true_f32_long_transform_top1nn_key_value_long_f32_ptr_value_op_key_value_long_f32_ptr_counting_iterator_long_int128_f32_identity_op_add_op_long_pointer =
  void (*)(float*, float*, int, raft::add_op, float, cuda::std::identity);
using source_transform_weighteddistanceop_true_f32_long_f32_ptr_counting_iterator_long_int128_f32_identity_op_add_op_long_pointer =
  void (*)(float*, float*, int, raft::add_op, float, cuda::std::identity);
using source_f64_ptr_f64_identity_op_add_op_u32_pointer =
  void (*)(double*, double*, int, raft::add_op, double, cuda::std::identity);
using source_f64_ptr_f64_identity_op_add_op_ulong_pointer =
  void (*)(double*, double*, int, raft::add_op, double, cuda::std::identity);
using source_f32_ptr_f32_identity_op_add_op_u32_pointer =
  void (*)(float*, float*, int, raft::add_op, float, cuda::std::identity);
using source_f32_ptr_f32_identity_op_add_op_ulong_pointer =
  void (*)(float*, float*, int, raft::add_op, float, cuda::std::identity);
}  // namespace cuvs::detail::cluster_cost_reduce_launch

template <>
__host__ __device__ cuvs::detail::cluster_cost_reduce_launch::
  source_transform_weighteddistanceop_false_f64_int_transform_top1nn_key_value_int_f64_ptr_value_op_key_value_int_f64_ptr_counting_iterator_int_long_f64_identity_op_add_op_int_pointer
  cuvs::detail::cluster_cost_reduce_launch::
    source_transform_weighteddistanceop_false_f64_int_transform_top1nn_key_value_int_f64_ptr_value_op_key_value_int_f64_ptr_counting_iterator_int_long_f64_identity_op_add_op_int::
      SingleTileSecondKernel();

template <>
__host__ __device__ cuvs::detail::cluster_cost_reduce_launch::
  source_transform_weighteddistanceop_false_f64_int_f64_ptr_counting_iterator_int_long_f64_identity_op_add_op_int_pointer
  cuvs::detail::cluster_cost_reduce_launch::
    source_transform_weighteddistanceop_false_f64_int_f64_ptr_counting_iterator_int_long_f64_identity_op_add_op_int::
      SingleTileSecondKernel();

template <>
__host__ __device__ cuvs::detail::cluster_cost_reduce_launch::
  source_transform_weighteddistanceop_false_f64_long_transform_top1nn_key_value_long_f64_ptr_value_op_key_value_long_f64_ptr_counting_iterator_long_int128_f64_identity_op_add_op_long_pointer
  cuvs::detail::cluster_cost_reduce_launch::
    source_transform_weighteddistanceop_false_f64_long_transform_top1nn_key_value_long_f64_ptr_value_op_key_value_long_f64_ptr_counting_iterator_long_int128_f64_identity_op_add_op_long::
      SingleTileSecondKernel();

template <>
__host__ __device__ cuvs::detail::cluster_cost_reduce_launch::
  source_transform_weighteddistanceop_false_f64_long_f64_ptr_counting_iterator_long_int128_f64_identity_op_add_op_long_pointer
  cuvs::detail::cluster_cost_reduce_launch::
    source_transform_weighteddistanceop_false_f64_long_f64_ptr_counting_iterator_long_int128_f64_identity_op_add_op_long::
      SingleTileSecondKernel();

template <>
__host__ __device__ cuvs::detail::cluster_cost_reduce_launch::
  source_transform_weighteddistanceop_false_f32_int_transform_top1nn_key_value_int_f32_ptr_value_op_key_value_int_f32_ptr_counting_iterator_int_long_f32_identity_op_add_op_int_pointer
  cuvs::detail::cluster_cost_reduce_launch::
    source_transform_weighteddistanceop_false_f32_int_transform_top1nn_key_value_int_f32_ptr_value_op_key_value_int_f32_ptr_counting_iterator_int_long_f32_identity_op_add_op_int::
      SingleTileSecondKernel();

template <>
__host__ __device__ cuvs::detail::cluster_cost_reduce_launch::
  source_transform_weighteddistanceop_false_f32_int_f32_ptr_counting_iterator_int_long_f32_identity_op_add_op_int_pointer
  cuvs::detail::cluster_cost_reduce_launch::
    source_transform_weighteddistanceop_false_f32_int_f32_ptr_counting_iterator_int_long_f32_identity_op_add_op_int::
      SingleTileSecondKernel();

template <>
__host__ __device__ cuvs::detail::cluster_cost_reduce_launch::
  source_transform_weighteddistanceop_false_f32_long_transform_top1nn_key_value_long_f32_ptr_value_op_key_value_long_f32_ptr_counting_iterator_long_int128_f32_identity_op_add_op_long_pointer
  cuvs::detail::cluster_cost_reduce_launch::
    source_transform_weighteddistanceop_false_f32_long_transform_top1nn_key_value_long_f32_ptr_value_op_key_value_long_f32_ptr_counting_iterator_long_int128_f32_identity_op_add_op_long::
      SingleTileSecondKernel();

template <>
__host__ __device__ cuvs::detail::cluster_cost_reduce_launch::
  source_transform_weighteddistanceop_false_f32_long_f32_ptr_counting_iterator_long_int128_f32_identity_op_add_op_long_pointer
  cuvs::detail::cluster_cost_reduce_launch::
    source_transform_weighteddistanceop_false_f32_long_f32_ptr_counting_iterator_long_int128_f32_identity_op_add_op_long::
      SingleTileSecondKernel();

template <>
__host__ __device__ cuvs::detail::cluster_cost_reduce_launch::
  source_transform_weighteddistanceop_true_f64_int_transform_top1nn_key_value_int_f64_ptr_value_op_key_value_int_f64_ptr_counting_iterator_int_long_f64_identity_op_add_op_int_pointer
  cuvs::detail::cluster_cost_reduce_launch::
    source_transform_weighteddistanceop_true_f64_int_transform_top1nn_key_value_int_f64_ptr_value_op_key_value_int_f64_ptr_counting_iterator_int_long_f64_identity_op_add_op_int::
      SingleTileSecondKernel();

template <>
__host__ __device__ cuvs::detail::cluster_cost_reduce_launch::
  source_transform_weighteddistanceop_true_f64_int_f64_ptr_counting_iterator_int_long_f64_identity_op_add_op_int_pointer
  cuvs::detail::cluster_cost_reduce_launch::
    source_transform_weighteddistanceop_true_f64_int_f64_ptr_counting_iterator_int_long_f64_identity_op_add_op_int::
      SingleTileSecondKernel();

template <>
__host__ __device__ cuvs::detail::cluster_cost_reduce_launch::
  source_transform_weighteddistanceop_true_f64_long_transform_top1nn_key_value_long_f64_ptr_value_op_key_value_long_f64_ptr_counting_iterator_long_int128_f64_identity_op_add_op_long_pointer
  cuvs::detail::cluster_cost_reduce_launch::
    source_transform_weighteddistanceop_true_f64_long_transform_top1nn_key_value_long_f64_ptr_value_op_key_value_long_f64_ptr_counting_iterator_long_int128_f64_identity_op_add_op_long::
      SingleTileSecondKernel();

template <>
__host__ __device__ cuvs::detail::cluster_cost_reduce_launch::
  source_transform_weighteddistanceop_true_f64_long_f64_ptr_counting_iterator_long_int128_f64_identity_op_add_op_long_pointer
  cuvs::detail::cluster_cost_reduce_launch::
    source_transform_weighteddistanceop_true_f64_long_f64_ptr_counting_iterator_long_int128_f64_identity_op_add_op_long::
      SingleTileSecondKernel();

template <>
__host__ __device__ cuvs::detail::cluster_cost_reduce_launch::
  source_transform_weighteddistanceop_true_f32_int_transform_top1nn_key_value_int_f32_ptr_value_op_key_value_int_f32_ptr_counting_iterator_int_long_f32_identity_op_add_op_int_pointer
  cuvs::detail::cluster_cost_reduce_launch::
    source_transform_weighteddistanceop_true_f32_int_transform_top1nn_key_value_int_f32_ptr_value_op_key_value_int_f32_ptr_counting_iterator_int_long_f32_identity_op_add_op_int::
      SingleTileSecondKernel();

template <>
__host__ __device__ cuvs::detail::cluster_cost_reduce_launch::
  source_transform_weighteddistanceop_true_f32_int_f32_ptr_counting_iterator_int_long_f32_identity_op_add_op_int_pointer
  cuvs::detail::cluster_cost_reduce_launch::
    source_transform_weighteddistanceop_true_f32_int_f32_ptr_counting_iterator_int_long_f32_identity_op_add_op_int::
      SingleTileSecondKernel();

template <>
__host__ __device__ cuvs::detail::cluster_cost_reduce_launch::
  source_transform_weighteddistanceop_true_f32_long_transform_top1nn_key_value_long_f32_ptr_value_op_key_value_long_f32_ptr_counting_iterator_long_int128_f32_identity_op_add_op_long_pointer
  cuvs::detail::cluster_cost_reduce_launch::
    source_transform_weighteddistanceop_true_f32_long_transform_top1nn_key_value_long_f32_ptr_value_op_key_value_long_f32_ptr_counting_iterator_long_int128_f32_identity_op_add_op_long::
      SingleTileSecondKernel();

template <>
__host__ __device__ cuvs::detail::cluster_cost_reduce_launch::
  source_transform_weighteddistanceop_true_f32_long_f32_ptr_counting_iterator_long_int128_f32_identity_op_add_op_long_pointer
  cuvs::detail::cluster_cost_reduce_launch::
    source_transform_weighteddistanceop_true_f32_long_f32_ptr_counting_iterator_long_int128_f32_identity_op_add_op_long::
      SingleTileSecondKernel();

template <>
__host__ __device__
  cuvs::detail::cluster_cost_reduce_launch::source_f64_ptr_f64_identity_op_add_op_u32_pointer
  cuvs::detail::cluster_cost_reduce_launch::source_f64_ptr_f64_identity_op_add_op_u32::
    SingleTileSecondKernel();

template <>
__host__ __device__
  cuvs::detail::cluster_cost_reduce_launch::source_f64_ptr_f64_identity_op_add_op_ulong_pointer
  cuvs::detail::cluster_cost_reduce_launch::source_f64_ptr_f64_identity_op_add_op_ulong::
    SingleTileSecondKernel();

template <>
__host__ __device__
  cuvs::detail::cluster_cost_reduce_launch::source_f32_ptr_f32_identity_op_add_op_u32_pointer
  cuvs::detail::cluster_cost_reduce_launch::source_f32_ptr_f32_identity_op_add_op_u32::
    SingleTileSecondKernel();

template <>
__host__ __device__
  cuvs::detail::cluster_cost_reduce_launch::source_f32_ptr_f32_identity_op_add_op_ulong_pointer
  cuvs::detail::cluster_cost_reduce_launch::source_f32_ptr_f32_identity_op_add_op_ulong::
    SingleTileSecondKernel();
