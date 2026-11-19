/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

#include "search_launchers.hpp"
#include <raft/util/cuda_dev_essentials.cuh>

namespace cuvs::neighbors::cagra::detail {

namespace multi_kernel_search {

template <class T>
RAFT_KERNEL set_value_kernel(T* const dev_ptr, const T val)
{
  *dev_ptr = val;
}

template <class T>
void set_value(T* const dev_ptr, const T val, cudaStream_t cuda_stream)
{
  set_value_kernel<T><<<1, 1, 0, cuda_stream>>>(dev_ptr, val);
}

template void set_value<uint32_t>(uint32_t*, uint32_t, cudaStream_t);

}  // namespace multi_kernel_search

}  // namespace cuvs::neighbors::cagra::detail
