/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cub_ptx_query.hpp"

#include "radix_sorting_launchers.hpp"
#include <cub/device/device_radix_sort.cuh>

namespace cuvs::detail {

template <typename InputIt, typename OutputIt, typename NumItems>
cudaError_t cub_sort_keys_descending(void* storage,
                                     size_t& bytes,
                                     InputIt input,
                                     OutputIt output,
                                     NumItems count,
                                     int begin_bit,
                                     int end_bit,
                                     cudaStream_t stream)
{
  return cub::DeviceRadixSort::SortKeysDescending(
    storage, bytes, input, output, count, begin_bit, end_bit, stream);
}

template cudaError
cuvs::detail::cub_sort_keys_descending<unsigned int*, unsigned int*, unsigned int>(
  void*, unsigned long&, unsigned int*, unsigned int*, unsigned int, int, int, CUstream_st*);

}  // namespace cuvs::detail
