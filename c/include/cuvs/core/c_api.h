/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>
#include <dlpack/dlpack.h>
#include <stdbool.h>
#include <stdint.h>

#include <cuvs/core/export.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup error_c cuVS Error Messages
 * @{
 */
/**
 * @brief An enum denoting error statuses for function calls
 *
 */
typedef enum { CUVS_ERROR = 0, CUVS_SUCCESS = 1 } cuvsError_t;

/** @brief Returns a string describing the last seen error on this thread, or
 *         NULL if the last function succeeded.
 */
CUVS_EXPORT const char* cuvsGetLastErrorText();

/**
 * @brief Sets a string describing an error seen on the thread. Passing NULL
 *        clears any previously seen error message.
 */
CUVS_EXPORT void cuvsSetLastErrorText(const char* error);

/** @} */

/**
 * @defgroup log_c cuVS Logging
 * @{
 */

/**
 * @brief An enum denoting log levels
 *
 */
typedef enum {
  CUVS_LOG_LEVEL_TRACE    = 0,
  CUVS_LOG_LEVEL_DEBUG    = 1,
  CUVS_LOG_LEVEL_INFO     = 2,
  CUVS_LOG_LEVEL_WARN     = 3,
  CUVS_LOG_LEVEL_ERROR    = 4,
  CUVS_LOG_LEVEL_CRITICAL = 5,
  CUVS_LOG_LEVEL_OFF      = 6
} cuvsLogLevel_t;

/** @brief Returns the current log level
 */
CUVS_EXPORT cuvsLogLevel_t cuvsGetLogLevel();

/** @brief Sets the log level
 */
CUVS_EXPORT void cuvsSetLogLevel(cuvsLogLevel_t);

/** @} */

/**
 * @defgroup resources_c cuVS Resources Handle
 * @{
 */

/**
 * @brief An opaque C handle for C++ type `raft::resources`
 *
 */
typedef uintptr_t cuvsResources_t;

/**
 * @brief Create an Initialized opaque C handle for C++ type `raft::resources`
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsResourcesCreate(cuvsResources_t* res);

/**
 * @brief Create an opaque C handle for C++ type `raft::resources` whose memory
 *        allocations are tracked and written as CSV samples from a background
 *        thread.
 *
 * The returned handle wraps all reachable memory resources (host, pinned,
 * managed, device, workspace, large_workspace) with allocation-tracking
 * adaptors and replaces the global host and device memory resources for the
 * lifetime of the handle. It is otherwise indistinguishable from a handle
 * created by ::cuvsResourcesCreate and can be used wherever a
 * ::cuvsResources_t is accepted. The CSV reporter is stopped and the global
 * memory resources are restored when the handle is destroyed via
 * ::cuvsResourcesDestroy.
 *
 * @param[out] res                 cuvsResources_t opaque C handle
 * @param[in]  csv_path            Path to the output CSV file
 *                                 (created/truncated). Must be a non-empty,
 *                                 null-terminated UTF-8 string.
 * @param[in]  sample_interval_ms  Minimum time in milliseconds between
 *                                 successive CSV samples. Pass 10 to match the
 *                                 C++ default.
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsResourcesCreateWithMemoryTracking(cuvsResources_t* res,
                                                              const char* csv_path,
                                                              int64_t sample_interval_ms);

/**
 * @brief Destroy and de-allocate opaque C handle for C++ type `raft::resources`
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsResourcesDestroy(cuvsResources_t res);

/**
 * @brief Set cudaStream_t on cuvsResources_t to queue CUDA kernels on APIs
 *        that accept a cuvsResources_t handle
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] stream cudaStream_t stream to queue CUDA kernels
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsStreamSet(cuvsResources_t res, cudaStream_t stream);

/**
 * @brief Get the cudaStream_t from a cuvsResources_t
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[out] stream cudaStream_t stream to queue CUDA kernels
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsStreamGet(cuvsResources_t res, cudaStream_t* stream);

/**
 * @brief Syncs the current CUDA stream on the resources object
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsStreamSync(cuvsResources_t res);

/**
 * @brief Get the id of the device associated with this cuvsResources_t
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[out] device_id int the id of the device associated with res
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsDeviceIdGet(cuvsResources_t res, int* device_id);

/**
 * @brief Create an Initialized opaque C handle for C++ type `raft::device_resources_snmg`
 *        for multi-GPU operations
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsMultiGpuResourcesCreate(cuvsResources_t* res);

/**
 * @brief Create an Initialized opaque C handle for C++ type `raft::device_resources_snmg`
 *        for multi-GPU operations with specific device IDs
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] device_ids DLManagedTensor* containing device IDs to use
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsMultiGpuResourcesCreateWithDeviceIds(cuvsResources_t* res,
                                                     DLManagedTensor* device_ids);

/**
 * @brief Destroy and de-allocate opaque C handle for C++ type `raft::device_resources_snmg`
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsMultiGpuResourcesDestroy(cuvsResources_t res);

/**
 * @brief Set a memory pool on all devices managed by the multi-GPU resources
 *
 * @param[in] res cuvsResources_t opaque C handle for multi-GPU resources
 * @param[in] percent_of_free_memory Percent of free memory to allocate for the pool
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsMultiGpuResourcesSetMemoryPool(cuvsResources_t res, int percent_of_free_memory);
/** @} */

/**
 * @defgroup memory_c cuVS Memory Allocation
 * @{
 */

/**
 * @brief Allocates device memory using RMM
 *
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[out] ptr Pointer to allocated device memory
 * @param[in] bytes Size in bytes to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsRMMAlloc(cuvsResources_t res, void** ptr, size_t bytes);

/**
 * @brief Deallocates device memory using RMM
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] ptr Pointer to allocated device memory to free
 * @param[in] bytes Size in bytes to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsRMMFree(cuvsResources_t res, void* ptr, size_t bytes);

/**
 * @brief Switches the working memory resource to use the RMM pool memory resource, which will
 * bypass unnecessary synchronizations by allocating a chunk of device memory up front and carving
 * that up for temporary memory allocations within algorithms. Be aware that this function will
 * change the memory resource for the whole process and the new memory resource will be used until
 * explicitly changed.
 *
 * @param[in] initial_pool_size_percent The initial pool size as a percentage of the total
 * available memory
 * @param[in] max_pool_size_percent The maximum pool size as a percentage of the total
 * available memory
 * @param[in] managed Whether to use a managed memory resource as upstream resource or not
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsRMMPoolMemoryResourceEnable(int initial_pool_size_percent,
                                            int max_pool_size_percent,
                                            bool managed);
/**
 * @brief Resets the memory resource to use the default memory resource (cuda_memory_resource)
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsRMMMemoryResourceReset();

/**
 * @brief Allocates pinned memory on the host using RMM
 * @param[out] ptr Pointer to allocated host memory
 * @param[in] bytes Size in bytes to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsRMMHostAlloc(void** ptr, size_t bytes);

/**
 * @brief Deallocates pinned memory on the host using RMM
 * @param[in] ptr Pointer to allocated host memory to free
 * @param[in] bytes Size in bytes to deallocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsRMMHostFree(void* ptr, size_t bytes);

/**
 * @brief Get the version of the cuVS library
 * @param[out] major Major version
 * @param[out] minor Minor version
 * @param[out] patch Patch version
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsVersionGet(uint16_t* major, uint16_t* minor, uint16_t* patch);

/**
 * @brief Bogus function used to verify end-to-end build propagation from the cuVS C
 *        layer, through the cuvs-java bindings, to cuvs-lucene. Returns the sum of its
 *        two arguments.
 * @param[in] a First addend
 * @param[in] b Second addend
 * @return The sum @p a + @p b
 */
CUVS_EXPORT int cuvsBogusAdd(int a, int b);

/**
 * @brief Copy a matrix
 *
 * This function copies a matrix from dst to src. This lets you copy a matrix
 * from device memory to host memory (or vice versa), while accounting for
 * differences in strides.
 *
 * Both src and dst must have the same shape and dtype, but can have different
 * strides and device type. The memory for the output dst tensor must already be
 * allocated and the tensor initialized.
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] src Pointer to DLManagedTensor to copy
 * @param[out] dst Pointer to DLManagedTensor to receive copy of data
 */
CUVS_EXPORT cuvsError_t cuvsMatrixCopy(cuvsResources_t res, DLManagedTensor* src, DLManagedTensor* dst);

/**
 * @brief Slices rows from a matrix
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] src Pointer to DLManagedTensor to copy
 * @param[in] start First row index to include in the output
 * @param[in] end Last row index to include in the output
 * @param[out] dst Pointer to DLManagedTensor to receive slice from matrix
 */
CUVS_EXPORT cuvsError_t cuvsMatrixSliceRows(
  cuvsResources_t res, DLManagedTensor* src, int64_t start, int64_t end, DLManagedTensor* dst);
/** @} */

#ifdef __cplusplus
}
#endif
