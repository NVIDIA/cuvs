/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cub_select_launchers.hpp"
#include <cub/device/device_select.cuh>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <rmm/exec_policy.hpp>

#include "../distance/detail/sparse/coo_spmv_strategies/hash_strategy.cuh"
#include <raft/sparse/solver/mst.cuh>

#include <cuvs/core/cuda_fp16.hpp>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>

template <>
CUB_RUNTIME_FUNCTION cudaError_t cub::DeviceSelect::FlaggedIf<
  cuda::counting_iterator<int, long>,
  cub::NullType*,
  int*,
  long*,
  cuvs::distance::detail::sparse::hash_strategy<int, double, 1024>::fits_in_hash_table,
  cudaStream_t>(
  void* storage,
  size_t& bytes,
  cuda::counting_iterator<int, long> input,
  cub::NullType* flags,
  int* output,
  long* selected,
  cuda::std::int64_t items,
  cuvs::distance::detail::sparse::hash_strategy<int, double, 1024>::fits_in_hash_table predicate,
  const cudaStream_t& environment);

template <>
CUB_RUNTIME_FUNCTION cudaError_t cub::DeviceSelect::FlaggedIf<
  cuda::counting_iterator<int, long>,
  cub::NullType*,
  int*,
  long*,
  cuvs::distance::detail::sparse::hash_strategy<int, float, 1024>::fits_in_hash_table,
  cudaStream_t>(
  void* storage,
  size_t& bytes,
  cuda::counting_iterator<int, long> input,
  cub::NullType* flags,
  int* output,
  long* selected,
  cuda::std::int64_t items,
  cuvs::distance::detail::sparse::hash_strategy<int, float, 1024>::fits_in_hash_table predicate,
  const cudaStream_t& environment);

template <>
CUB_RUNTIME_FUNCTION cudaError_t
cub::DeviceSelect::FlaggedIf<thrust::counting_iterator<long>,
                             cub::NullType*,
                             long*,
                             long*,
                             cuvs::detail::bitset_row_filter<__half, long, unsigned int, float>,
                             cudaStream_t>(
  void* storage,
  size_t& bytes,
  thrust::counting_iterator<long> input,
  cub::NullType* flags,
  long* output,
  long* selected,
  cuda::std::int64_t items,
  cuvs::detail::bitset_row_filter<__half, long, unsigned int, float> predicate,
  const cudaStream_t& environment);

template <>
CUB_RUNTIME_FUNCTION cudaError_t
cub::DeviceSelect::FlaggedIf<thrust::counting_iterator<long>,
                             cub::NullType*,
                             long*,
                             long*,
                             cuvs::detail::bitset_row_filter<float, long, unsigned int, float>,
                             cudaStream_t>(
  void* storage,
  size_t& bytes,
  thrust::counting_iterator<long> input,
  cub::NullType* flags,
  long* output,
  long* selected,
  cuda::std::int64_t items,
  cuvs::detail::bitset_row_filter<float, long, unsigned int, float> predicate,
  const cudaStream_t& environment);
