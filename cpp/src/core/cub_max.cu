/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cub_ptx_query.hpp"

#include "cub_runtime_launchers.hpp"
#include <cub/device/device_reduce.cuh>

namespace cuvs::detail {

template <typename Input, typename Output, typename Count>
cudaError_t cub_max(
  void* storage, size_t& bytes, Input input, Output output, Count count, cudaStream_t stream)
{
  return cub::DeviceReduce::Max(storage, bytes, input, output, count, stream);
}

}  // namespace cuvs::detail

template cudaError cuvs::detail::cub_max(
  void*, unsigned long&, unsigned int*, long*, int, CUstream_st*);
