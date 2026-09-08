/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file stream_alloc_compat.hpp
 * @brief Stream-ordered allocation with a fallback for devices without memory-pool support.
 *
 * `cudaMallocAsync` / `cudaFreeAsync` are backed by the CUDA stream-ordered memory allocator, which
 * is optional hardware/driver functionality: on devices that report
 * `cudaDevAttrMemoryPoolsSupported == 0` -- Maxwell-generation GPUs among them -- every call fails
 * with `cudaErrorNotSupported`.
 *
 * The few places in cuVS that allocate directly from a stream rather than through RMM therefore go
 * through the helpers below, which fall back to plain `cudaMalloc` / `cudaFree`. The fallback is
 * semantically stronger than what the caller asked for, not weaker:
 *
 *   - `cudaMalloc` is synchronous with respect to the whole device, so the returned memory is
 *     trivially usable by any work subsequently issued into the stream, which is exactly the
 *     guarantee `cudaMallocAsync` gives for that stream;
 *   - `cudaFree` blocks until all previously issued device work has finished, which subsumes
 *     `cudaFreeAsync`'s "free once the preceding work in this stream completes".
 *
 * The price is the loss of pool reuse and the implicit synchronisation, so these helpers are only
 * appropriate for allocations that happen a handful of times per search, not per launch.
 */

#pragma once

#include <raft/util/cuda_rt_essentials.hpp>  // RAFT_CUDA_TRY

#include <cuda_runtime.h>

#include <cstddef>

namespace cuvs::util {

/** Whether the current device supports the CUDA stream-ordered memory allocator. */
inline bool device_supports_memory_pools()
{
  int device = 0;
  RAFT_CUDA_TRY(cudaGetDevice(&device));
  int supported = 0;
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, device));
  return supported != 0;
}

/**
 * @brief `cudaMallocAsync`, or `cudaMalloc` where memory pools are unsupported.
 *
 * @param[out] ptr    allocated device pointer
 * @param[in]  size   allocation size in bytes
 * @param[in]  stream stream the allocation is ordered against
 */
inline cudaError_t malloc_async_compat(void** ptr, std::size_t size, cudaStream_t stream)
{
  if (device_supports_memory_pools()) { return cudaMallocAsync(ptr, size, stream); }
  return cudaMalloc(ptr, size);
}

/**
 * @brief `cudaFreeAsync`, or `cudaFree` where memory pools are unsupported.
 *
 * @param[in] ptr    pointer previously returned by @ref malloc_async_compat
 * @param[in] stream stream the deallocation is ordered against
 */
inline cudaError_t free_async_compat(void* ptr, cudaStream_t stream)
{
  if (device_supports_memory_pools()) { return cudaFreeAsync(ptr, stream); }
  // `cudaFree` is not stream-ordered, so make sure the work that may still be reading this
  // allocation has completed before the memory is handed back.
  auto status = cudaStreamSynchronize(stream);
  if (status != cudaSuccess) { return status; }
  return cudaFree(ptr);
}

}  // namespace cuvs::util
