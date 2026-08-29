/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file jit_kernel_compat.hpp
 * @brief Toolkit-version shim for the CUDA APIs cuVS calls on JIT-LTO kernel handles.
 *
 * The JIT-LTO path hands out `cudaKernel_t` handles for kernels that were linked at run time and
 * loaded from a library image rather than compiled into the binary. cuVS then queries and sets
 * attributes on those handles.
 *
 * The CUDA *runtime* only accepts a `cudaKernel_t` where a `const void* func` is expected from
 * CUDA 12.8 onwards -- "If the specified function does not exist, then it is assumed to be a
 * cudaKernel_t and used as is", a sentence that appears in the 12.8 documentation of
 * `cudaFuncGetAttributes`, `cudaFuncSetAttribute` and
 * `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, and in no earlier version.
 * `cudaKernelSetAttributeForDevice` does not exist at all before 12.8 -- it is absent from both the
 * headers and `libcudart.so`. On an older toolkit the calls therefore fail with
 * `cudaErrorInvalidDeviceFunction`, or do not compile.
 *
 * The *driver* API has always taken the equivalent `CUkernel` / `CUfunction`, and
 * `cudaKernel_t` is `struct CUkern_st*`, i.e. exactly `CUkernel`, so the handles carry over
 * unchanged. This header forwards to the runtime API when it is available and to the driver API
 * otherwise; on 12.8+ every wrapper compiles down to the original runtime call and no driver
 * dependency is added.
 */

#pragma once

#include <cuda_runtime.h>

#include <cstddef>

/** 1 when the CUDA runtime accepts `cudaKernel_t` in the function-attribute APIs, 0 when not. */
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 12080)
#define CUVS_RUNTIME_ACCEPTS_KERNEL_HANDLES 1
#else
#define CUVS_RUNTIME_ACCEPTS_KERNEL_HANDLES 0
#endif

#if !CUVS_RUNTIME_ACCEPTS_KERNEL_HANDLES
#include <cuda.h>
#endif

namespace cuvs::util {

namespace detail {

#if !CUVS_RUNTIME_ACCEPTS_KERNEL_HANDLES

/** Translate a driver-API status into the closest runtime-API status. */
inline cudaError_t from_driver(CUresult status)
{
  switch (status) {
    case CUDA_SUCCESS: return cudaSuccess;
    case CUDA_ERROR_OUT_OF_MEMORY: return cudaErrorMemoryAllocation;
    case CUDA_ERROR_NOT_INITIALIZED: return cudaErrorInitializationError;
    case CUDA_ERROR_DEINITIALIZED: return cudaErrorCudartUnloading;
    case CUDA_ERROR_NO_DEVICE: return cudaErrorNoDevice;
    case CUDA_ERROR_INVALID_DEVICE: return cudaErrorInvalidDevice;
    case CUDA_ERROR_INVALID_VALUE: return cudaErrorInvalidValue;
    case CUDA_ERROR_INVALID_HANDLE: return cudaErrorInvalidResourceHandle;
    case CUDA_ERROR_NOT_FOUND: return cudaErrorSymbolNotFound;
    case CUDA_ERROR_INVALID_IMAGE: return cudaErrorInvalidKernelImage;
    case CUDA_ERROR_NO_BINARY_FOR_GPU: return cudaErrorNoKernelImageForDevice;
    case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: return cudaErrorLaunchOutOfResources;
    case CUDA_ERROR_INVALID_CONTEXT: return cudaErrorDeviceUninitialized;
    default: return cudaErrorUnknown;
  }
}

/**
 * Make sure the runtime's primary context exists and is current.
 *
 * Driver-API calls such as `cuKernelGetFunction` operate on the current context. The CUDA runtime
 * creates and binds its primary context lazily, so force that to have happened.
 * `cudaFree(nullptr)` is the documented, cheap, idempotent way to do it.
 */
inline cudaError_t ensure_context() { return cudaFree(nullptr); }

/** Map a runtime function attribute onto the driver-API enumerator. */
inline CUfunction_attribute to_driver_attribute(cudaFuncAttribute attr)
{
  switch (attr) {
    case cudaFuncAttributeMaxDynamicSharedMemorySize:
      return CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
    case cudaFuncAttributePreferredSharedMemoryCarveout:
      return CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT;
    case cudaFuncAttributeRequiredClusterWidth:
      return CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH;
    case cudaFuncAttributeRequiredClusterHeight:
      return CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT;
    case cudaFuncAttributeRequiredClusterDepth:
      return CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH;
    case cudaFuncAttributeNonPortableClusterSizeAllowed:
      return CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED;
    case cudaFuncAttributeClusterSchedulingPolicyPreference:
      return CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
    default: return CU_FUNC_ATTRIBUTE_MAX;
  }
}

/** Resolve a kernel handle to a `CUfunction` in the current context. */
inline cudaError_t to_function(CUfunction* function, cudaKernel_t kernel)
{
  auto status = ensure_context();
  if (status != cudaSuccess) { return status; }
  return from_driver(cuKernelGetFunction(function, reinterpret_cast<CUkernel>(kernel)));
}

#endif  // !CUVS_RUNTIME_ACCEPTS_KERNEL_HANDLES

}  // namespace detail

/** Set an attribute of a library kernel for one device (`cudaKernelSetAttributeForDevice`). */
inline cudaError_t kernel_set_attribute(cudaKernel_t kernel,
                                        cudaFuncAttribute attr,
                                        int value,
                                        int device)
{
#if CUVS_RUNTIME_ACCEPTS_KERNEL_HANDLES
  return cudaKernelSetAttributeForDevice(kernel, attr, value, device);
#else
  auto status = detail::ensure_context();
  if (status != cudaSuccess) { return status; }
  auto driver_attr = detail::to_driver_attribute(attr);
  if (driver_attr == CU_FUNC_ATTRIBUTE_MAX) { return cudaErrorInvalidValue; }
  return detail::from_driver(cuKernelSetAttribute(
    driver_attr, value, reinterpret_cast<CUkernel>(kernel), static_cast<CUdevice>(device)));
#endif
}

/** Set an attribute of a library kernel on the current device (`cudaFuncSetAttribute`). */
inline cudaError_t kernel_set_attribute(cudaKernel_t kernel, cudaFuncAttribute attr, int value)
{
#if CUVS_RUNTIME_ACCEPTS_KERNEL_HANDLES
  return cudaFuncSetAttribute(kernel, attr, value);
#else
  int device  = 0;
  auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) { return status; }
  return kernel_set_attribute(kernel, attr, value, device);
#endif
}

/**
 * Read the function attributes of a library kernel (`cudaFuncGetAttributes`).
 *
 * On the driver path the struct is filled field by field from `cuFuncGetAttribute`; the fields it
 * cannot provide keep their zero-initialised values.
 */
inline cudaError_t kernel_get_attributes(cudaFuncAttributes* attrs, cudaKernel_t kernel)
{
#if CUVS_RUNTIME_ACCEPTS_KERNEL_HANDLES
  return cudaFuncGetAttributes(attrs, kernel);
#else
  CUfunction function{};
  auto status = detail::to_function(&function, kernel);
  if (status != cudaSuccess) { return status; }

  *attrs      = cudaFuncAttributes{};
  auto result = CUDA_SUCCESS;
  int value{};
  const auto get = [&](CUfunction_attribute attr, auto& dst) {
    if (result != CUDA_SUCCESS) { return; }
    result = cuFuncGetAttribute(&value, attr, function);
    if (result == CUDA_SUCCESS) { dst = value; }
  };
  get(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, attrs->maxThreadsPerBlock);
  get(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, attrs->sharedSizeBytes);
  get(CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, attrs->constSizeBytes);
  get(CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, attrs->localSizeBytes);
  get(CU_FUNC_ATTRIBUTE_NUM_REGS, attrs->numRegs);
  get(CU_FUNC_ATTRIBUTE_PTX_VERSION, attrs->ptxVersion);
  get(CU_FUNC_ATTRIBUTE_BINARY_VERSION, attrs->binaryVersion);
  get(CU_FUNC_ATTRIBUTE_CACHE_MODE_CA, attrs->cacheModeCA);
  get(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, attrs->maxDynamicSharedSizeBytes);
  get(CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, attrs->preferredShmemCarveout);
  return detail::from_driver(result);
#endif
}

/** Occupancy query for a library kernel (`cudaOccupancyMaxActiveBlocksPerMultiprocessor`). */
inline cudaError_t kernel_max_active_blocks_per_multiprocessor(int* num_blocks,
                                                               cudaKernel_t kernel,
                                                               int block_size,
                                                               std::size_t dynamic_smem_bytes)
{
#if CUVS_RUNTIME_ACCEPTS_KERNEL_HANDLES
  return cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    num_blocks, kernel, block_size, dynamic_smem_bytes);
#else
  CUfunction function{};
  auto status = detail::to_function(&function, kernel);
  if (status != cudaSuccess) { return status; }
  return detail::from_driver(cuOccupancyMaxActiveBlocksPerMultiprocessor(
    num_blocks, function, block_size, dynamic_smem_bytes));
#endif
}

}  // namespace cuvs::util
