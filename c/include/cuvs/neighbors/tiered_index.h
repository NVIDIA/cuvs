/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.h>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/common.h>
#include <cuvs/neighbors/ivf_flat.h>
#include <cuvs/neighbors/ivf_pq.h>
#include <dlpack/dlpack.h>
#include <stdint.h>

#include <cuvs/core/export.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Enum to hold which ANN algorithm is being used in the tiered index
 */
typedef enum {
  CUVS_TIERED_INDEX_ALGO_CAGRA    = 0,
  CUVS_TIERED_INDEX_ALGO_IVF_FLAT = 1,
  CUVS_TIERED_INDEX_ALGO_IVF_PQ   = 2
} cuvsTieredIndexANNAlgo;

/**
 * @defgroup tiered_index_c_index Tiered Index
 * @{
 */
/**
 * @brief Struct to hold address of cuvs::neighbors::tiered_index::index and its active trained
 * dtype
 *
 */
typedef struct {
  uintptr_t addr;
  DLDataType dtype;
  cuvsTieredIndexANNAlgo algo;
} cuvsTieredIndex;

typedef cuvsTieredIndex* cuvsTieredIndex_t;

/**
 * @brief Allocate Tiered Index
 *
 * @param[in] index cuvsTieredIndex_t to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsTieredIndexCreate(cuvsTieredIndex_t* index);

/**
 * @brief De-allocate Tiered index
 *
 * @param[in] index cuvsTieredIndex_t to de-allocate
 */
CUVS_EXPORT cuvsError_t cuvsTieredIndexDestroy(cuvsTieredIndex_t index);
/**
 * @}
 */

/**
 * @defgroup tiered_c_index_params Tiered Index build parameters
 * @{
 */
/**
 * @brief Supplemental parameters to build a TieredIndex
 */
struct cuvsTieredIndexParams {
  /** Distance type. */
  cuvsDistanceType metric;

  /** The type of ANN algorithm we are using */
  cuvsTieredIndexANNAlgo algo;

  /** The minimum number of rows necessary in the index to create an
 ann index */
  int64_t min_ann_rows;

  /** Whether or not to create a new ann index on extend, if the number
  of rows in the incremental (bfknn) portion is above min_ann_rows */
  bool create_ann_index_on_extend;

  /** Optional parameters for building a cagra index */
  cuvsCagraIndexParams_t cagra_params;

  /** Optional parameters for building a ivf_flat index */
  cuvsIvfFlatIndexParams_t ivf_flat_params;

  /** Optional parameters for building a ivf-pq index */
  cuvsIvfPqIndexParams_t ivf_pq_params;
};

typedef struct cuvsTieredIndexParams* cuvsTieredIndexParams_t;

/**
 * @brief Allocate Tiered Index Params and populate with default values
 *
 * @param[in] index_params cuvsTieredIndexParams_t to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsTieredIndexParamsCreate(cuvsTieredIndexParams_t* index_params);

/**
 * @brief De-allocate Tiered Index params
 *
 * @param[in] index_params
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsTieredIndexParamsDestroy(cuvsTieredIndexParams_t index_params);
/**
 * @}
 */

/**
 * @defgroup tieredindex_c_index_build Tiered index build
 * @{
 */
/**
 * @brief Build a TieredIndex index with a `DLManagedTensor` which has underlying
 *        `DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`,
 *        or `kDLCPU`. Also, acceptable underlying types are:
 *        1. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
 *        2. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 16`
 *
 * @code {.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/neighbors/tiered_index.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // Assume a populated `DLManagedTensor` type here
 * DLManagedTensor dataset;
 *
 * // Create TieredIndex index
 * cuvsTieredIndex_t index;
 * cuvsError_t index_create_status = cuvsTieredIndexCreate(&index);
 *
 * // Create default index params
 * cuvsTieredIndexParams_t index_params;
 * cuvsError_t params_create_status = cuvsTieredIndexParamsCreate(&index_params);
 *
 * // Build the TieredIndex Index
 * cuvsError_t build_status = cuvsTieredIndexBuild(res, index_params, &dataset_tensor, index);
 *
 * // de-allocate `index` and `res`
 * cuvsError_t index_destroy_status = cuvsTieredIndexDestroy(index);
 * cuvsError_t res_destroy_status = cuvsResourcesDestroy(res);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] dataset DLManagedTensor* training dataset
 * @param[in] index_params Index parameters to use when building the index
 * @param[out] index cuvsTieredIndex_t Newly built TieredIndex index
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsTieredIndexBuild(cuvsResources_t res,
                                 cuvsTieredIndexParams_t index_params,
                                 DLManagedTensor* dataset,
                                 cuvsTieredIndex_t index);
/**
 * @}
 */

/**
 * @defgroup tieredindex_c_index_search Tiered index search
 * @{
 */
/**
 * @brief Search a TieredIndex index with a `DLManagedTensor`
 *
 * @code {.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/neighbors/tiered_index.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // Assume a populated `DLManagedTensor` type here
 * DLManagedTensor dataset;
 * DLManagedTensor queries;
 * DLManagedTensor neighbors;
 * DLManagedTensor bitmap;
 *
 * cuvsFilter prefilter{(uintptr_t)&bitmap, BITMAP};
 *
 * // Search the `index` built using `cuvsTieredIndexBuild`
 * cuvsError_t search_status = cuvsTieredIndexSearch(res, index, &queries, &neighbors, &distances,
 * prefilter);
 *
 * // de-allocate `res`
 * cuvsError_t res_destroy_status = cuvsResourcesDestroy(res);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] search_params params used to the ANN index, should be one of
 * cuvsCagraSearchParams_t, cuvsIvfFlatSearchParams_t, cuvsIvfPqSearchParams_t
 * depending on the type of the tiered index used
 * @param[in] index cuvsTieredIndex which has been returned by `cuvsTieredIndexBuild`
 * @param[in] queries DLManagedTensor* queries dataset to search
 * @param[out] neighbors DLManagedTensor* output `k` neighbors for queries
 * @param[out] distances DLManagedTensor* output `k` distances for queries
 * @param[in] prefilter cuvsFilter input prefilter that can be used
              to filter queries and neighbors based on the given bitmap.
 */
CUVS_EXPORT cuvsError_t cuvsTieredIndexSearch(cuvsResources_t res,
                                  void* search_params,
                                  cuvsTieredIndex_t index,
                                  DLManagedTensor* queries,
                                  DLManagedTensor* neighbors,
                                  DLManagedTensor* distances,
                                  cuvsFilter prefilter);

/**
 * @}
 */
/**
 * @defgroup tiered_c_index_extend Tiered index extend
 * @{
 */
/**
 * @brief Extend the index with the new data.
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] new_vectors DLManagedTensor* the new vectors to add to the index
 * @param[inout] index Tiered index to be extended
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsTieredIndexExtend(cuvsResources_t res,
                                  DLManagedTensor* new_vectors,
                                  cuvsTieredIndex_t index);
/**
 * @}
 */

/**
 * @defgroup tiered_c_index_merge Tiered index merge
 * @{
 */
/**
 * @brief Merge multiple indices together into a single index
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] index_params Index parameters to use when merging
 * @param[in] indices pointers to indices to merge together
 * @param[in] num_indices the number of indices to merge
 * @param[out] output_index the merged index
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsTieredIndexMerge(cuvsResources_t res,
                                 cuvsTieredIndexParams_t index_params,
                                 cuvsTieredIndex_t* indices,
                                 size_t num_indices,
                                 cuvsTieredIndex_t output_index);

/**
 * @}
 */
#ifdef __cplusplus
}
#endif
