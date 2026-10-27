/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cub_ptx_query.hpp"
#include "cub_scalar_reduction_launchers.hpp"
#include <cub/device/device_reduce.cuh>
#include <cuvs/core/cuda_fp16.hpp>

namespace cuvs::detail {

template <typename InputIt, typename OutputIt, typename NumItems>
cudaError_t cub_reduce_sum(
  void* storage, size_t& bytes, InputIt input, OutputIt output, NumItems count, cudaStream_t stream)
{
  return cub::DeviceReduce::Sum(storage, bytes, input, output, count, stream);
}

template cudaError_t cuvs::detail::cub_reduce_sum<float*, float*, long>(
  void*, unsigned long&, float*, float*, long, cudaStream_t);

}  // namespace cuvs::detail
