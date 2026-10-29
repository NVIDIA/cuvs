/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cub_select_launchers.hpp"
#include <cub/device/device_select.cuh>
#include <cuda/std/functional>

template <>
CUB_RUNTIME_FUNCTION cudaError_t
cub::DeviceSelect::Unique<float*, long*, cuda::std::equal_to<float>, cudaStream_t, 0>(
  void* storage,
  size_t& bytes,
  float* input,
  long* selected,
  cuda::std::int64_t items,
  cuda::std::equal_to<float> equality,
  const cudaStream_t& environment);

template <>
CUB_RUNTIME_FUNCTION cudaError_t
cub::DeviceSelect::Unique<int*, int*, int*, cudaStream_t, 0>(void* storage,
                                                             size_t& bytes,
                                                             int* input,
                                                             int* output,
                                                             int* selected,
                                                             cuda::std::int64_t items,
                                                             const cudaStream_t& environment);

template <>
CUB_RUNTIME_FUNCTION cudaError_t
cub::DeviceSelect::Unique<int*, int*, int*, cuda::std::equal_to<void>, cudaStream_t, 0>(
  void* storage,
  size_t& bytes,
  int* input,
  int* output,
  int* selected,
  cuda::std::int64_t items,
  cuda::std::equal_to<void> equality,
  const cudaStream_t& environment);

template <>
CUB_RUNTIME_FUNCTION cudaError_t
cub::DeviceSelect::Unique<long*, long*, int*, cudaStream_t, 0>(void* storage,
                                                               size_t& bytes,
                                                               long* input,
                                                               long* output,
                                                               int* selected,
                                                               cuda::std::int64_t items,
                                                               const cudaStream_t& environment);

template <>
CUB_RUNTIME_FUNCTION cudaError_t
cub::DeviceSelect::Unique<long*, long*, int*, cuda::std::equal_to<void>, cudaStream_t, 0>(
  void* storage,
  size_t& bytes,
  long* input,
  long* output,
  int* selected,
  cuda::std::int64_t items,
  cuda::std::equal_to<void> equality,
  const cudaStream_t& environment);

template <>
CUB_RUNTIME_FUNCTION cudaError_t
cub::DeviceSelect::Unique<unsigned int const*, unsigned int*, long*, cudaStream_t, 0>(
  void* storage,
  size_t& bytes,
  unsigned int const* input,
  unsigned int* output,
  long* selected,
  cuda::std::int64_t items,
  const cudaStream_t& environment);

template <>
CUB_RUNTIME_FUNCTION cudaError_t cub::DeviceSelect::
  Unique<unsigned int const*, unsigned int*, long*, cuda::std::equal_to<void>, cudaStream_t, 0>(
    void* storage,
    size_t& bytes,
    unsigned int const* input,
    unsigned int* output,
    long* selected,
    cuda::std::int64_t items,
    cuda::std::equal_to<void> equality,
    const cudaStream_t& environment);

template <>
CUB_RUNTIME_FUNCTION cudaError_t
cub::DeviceSelect::Unique<unsigned int*, unsigned int*, int*, cudaStream_t, 0>(
  void* storage,
  size_t& bytes,
  unsigned int* input,
  unsigned int* output,
  int* selected,
  cuda::std::int64_t items,
  const cudaStream_t& environment);

template <>
CUB_RUNTIME_FUNCTION cudaError_t cub::DeviceSelect::
  Unique<unsigned int*, unsigned int*, int*, cuda::std::equal_to<void>, cudaStream_t, 0>(
    void* storage,
    size_t& bytes,
    unsigned int* input,
    unsigned int* output,
    int* selected,
    cuda::std::int64_t items,
    cuda::std::equal_to<void> equality,
    const cudaStream_t& environment);

template <>
CUB_RUNTIME_FUNCTION cudaError_t
cub::DeviceSelect::Unique<unsigned int*, unsigned int*, long*, cudaStream_t, 0>(
  void* storage,
  size_t& bytes,
  unsigned int* input,
  unsigned int* output,
  long* selected,
  cuda::std::int64_t items,
  const cudaStream_t& environment);

template <>
CUB_RUNTIME_FUNCTION cudaError_t cub::DeviceSelect::
  Unique<unsigned int*, unsigned int*, long*, cuda::std::equal_to<void>, cudaStream_t, 0>(
    void* storage,
    size_t& bytes,
    unsigned int* input,
    unsigned int* output,
    long* selected,
    cuda::std::int64_t items,
    cuda::std::equal_to<void> equality,
    const cudaStream_t& environment);
