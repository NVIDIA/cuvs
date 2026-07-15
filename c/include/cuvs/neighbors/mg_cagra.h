/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/mg_common.h>
#include <dlpack/dlpack.h>
#include <stdint.h>

#include <cuvs/core/export.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup mg_cagra_c_index_params Multi-GPU CAGRA index build parameters
 * @{
 */

/**
 * @brief Multi-GPU parameters to build CAGRA Index
 *
 * This structure extends the base CAGRA index parameters with multi-GPU specific settings.
 */
struct cuvsMultiGpuCagraIndexParams {
  /** Base CAGRA index parameters */
  cuvsCagraIndexParams_t base_params;
  /** Distribution mode for multi-GPU setup */
  cuvsMultiGpuDistributionMode mode;
};

typedef struct cuvsMultiGpuCagraIndexParams* cuvsMultiGpuCagraIndexParams_t;

/**
 * @brief Allocate Multi-GPU CAGRA Index params, and populate with default values
 *
 * @param[in] index_params cuvsMultiGpuCagraIndexParams_t to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsMultiGpuCagraIndexParamsCreate(cuvsMultiGpuCagraIndexParams_t* index_params);

/**
 * @brief De-allocate Multi-GPU CAGRA Index params
 *
 * @param[in] index_params
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsMultiGpuCagraIndexParamsDestroy(cuvsMultiGpuCagraIndexParams_t index_params);

/**
 * @}
 */

/**
 * @defgroup mg_cagra_c_search_params Multi-GPU CAGRA index search parameters
 * @{
 */

/**
 * @brief Multi-GPU parameters to search CAGRA index
 *
 * This structure extends the base CAGRA search parameters with multi-GPU specific settings.
 */
struct cuvsMultiGpuCagraSearchParams {
  /** Base CAGRA search parameters */
  cuvsCagraSearchParams_t base_params;
  /** Replicated search mode */
  cuvsMultiGpuReplicatedSearchMode search_mode;
  /** Sharded merge mode */
  cuvsMultiGpuShardedMergeMode merge_mode;
  /** Number of rows per batch */
  int64_t n_rows_per_batch;
};

typedef struct cuvsMultiGpuCagraSearchParams* cuvsMultiGpuCagraSearchParams_t;

/**
 * @brief Allocate Multi-GPU CAGRA search params, and populate with default values
 *
 * @param[in] params cuvsMultiGpuCagraSearchParams_t to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsMultiGpuCagraSearchParamsCreate(cuvsMultiGpuCagraSearchParams_t* params);

/**
 * @brief De-allocate Multi-GPU CAGRA search params
 *
 * @param[in] params
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsMultiGpuCagraSearchParamsDestroy(cuvsMultiGpuCagraSearchParams_t params);

/**
 * @}
 */

/**
 * @defgroup mg_cagra_c_index Multi-GPU CAGRA index
 * @{
 */

/**
 * @brief Struct to hold address of cuvs::neighbors::mg_index<cagra::index> and its active trained
 * dtype
 */
typedef struct {
  uintptr_t addr;
  DLDataType dtype;
} cuvsMultiGpuCagraIndex;

typedef cuvsMultiGpuCagraIndex* cuvsMultiGpuCagraIndex_t;

/**
 * @brief Allocate Multi-GPU CAGRA index
 *
 * @param[in] index cuvsMultiGpuCagraIndex_t to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsMultiGpuCagraIndexCreate(cuvsMultiGpuCagraIndex_t* index);

/**
 * @brief De-allocate Multi-GPU CAGRA index
 *
 * @param[in] index cuvsMultiGpuCagraIndex_t to de-allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsMultiGpuCagraIndexDestroy(cuvsMultiGpuCagraIndex_t index);

/**
 * @}
 */

/**
 * @defgroup mg_cagra_c_index_build Multi-GPU CAGRA index build
 * @{
 */

/**
 * @brief Build a Multi-GPU CAGRA index
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] params Multi-GPU CAGRA index parameters
 * @param[in] dataset_tensor DLManagedTensor* training dataset
 * @param[out] index Multi-GPU CAGRA index
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsMultiGpuCagraBuild(cuvsResources_t res,
                                   cuvsMultiGpuCagraIndexParams_t params,
                                   DLManagedTensor* dataset_tensor,
                                   cuvsMultiGpuCagraIndex_t index);

/**
 * @}
 */

/**
 * @defgroup mg_cagra_c_index_search Multi-GPU CAGRA index search
 * @{
 */

/**
 * @brief Search a Multi-GPU CAGRA index
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] params Multi-GPU CAGRA search parameters
 * @param[in] index Multi-GPU CAGRA index
 * @param[in] queries_tensor DLManagedTensor* queries dataset
 * @param[out] neighbors_tensor DLManagedTensor* output neighbors
 * @param[out] distances_tensor DLManagedTensor* output distances
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsMultiGpuCagraSearch(cuvsResources_t res,
                                    cuvsMultiGpuCagraSearchParams_t params,
                                    cuvsMultiGpuCagraIndex_t index,
                                    DLManagedTensor* queries_tensor,
                                    DLManagedTensor* neighbors_tensor,
                                    DLManagedTensor* distances_tensor);

/**
 * @}
 */

/**
 * @defgroup mg_cagra_c_index_extend Multi-GPU CAGRA index extend
 * @{
 */

/**
 * @brief Extend a Multi-GPU CAGRA index
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in,out] index Multi-GPU CAGRA index to extend
 * @param[in] new_vectors_tensor DLManagedTensor* new vectors to add
 * @param[in] new_indices_tensor DLManagedTensor* new indices (optional, can be NULL)
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsMultiGpuCagraExtend(cuvsResources_t res,
                                    cuvsMultiGpuCagraIndex_t index,
                                    DLManagedTensor* new_vectors_tensor,
                                    DLManagedTensor* new_indices_tensor);

/**
 * @}
 */

/**
 * @defgroup mg_cagra_c_index_serialize Multi-GPU CAGRA index serialize
 * @{
 */

/**
 * @brief Serialize a Multi-GPU CAGRA index to file
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] index Multi-GPU CAGRA index to serialize
 * @param[in] filename Path to the output file
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsMultiGpuCagraSerialize(cuvsResources_t res,
                                       cuvsMultiGpuCagraIndex_t index,
                                       const char* filename);

/**
 * @}
 */

/**
 * @defgroup mg_cagra_c_index_deserialize Multi-GPU CAGRA index deserialize
 * @{
 */

/**
 * @brief Deserialize a Multi-GPU CAGRA index from file
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] filename Path to the input file
 * @param[out] index Multi-GPU CAGRA index
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsMultiGpuCagraDeserialize(cuvsResources_t res,
                                         const char* filename,
                                         cuvsMultiGpuCagraIndex_t index);

/**
 * @}
 */

/**
 * @defgroup mg_cagra_c_index_distribute Multi-GPU CAGRA index distribute
 * @{
 */

/**
 * @brief Distribute a local CAGRA index to create a Multi-GPU index
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] filename Path to the local index file
 * @param[out] index Multi-GPU CAGRA index
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsMultiGpuCagraDistribute(cuvsResources_t res,
                                        const char* filename,
                                        cuvsMultiGpuCagraIndex_t index);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif
