/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.h>
#include <cuvs/neighbors/common.h>
#include <dlpack/dlpack.h>
#include <stdbool.h>
#include <stdint.h>

#include <cuvs/core/export.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup ivf_sq_c_index_params IVF-SQ index build parameters
 * @{
 */
/**
 * @brief Supplemental parameters to build IVF-SQ Index
 *
 */
struct cuvsIvfSqIndexParams {
  /** Distance type. */
  cuvsDistanceType metric;
  /** The argument used by some distance metrics. */
  float metric_arg;
  /**
   * Whether to add the dataset content to the index, i.e.:
   *
   *  - `true` means the index is filled with the dataset vectors and ready to search after calling
   * `build`.
   *  - `false` means `build` only trains the underlying model (e.g. quantizer or clustering), but
   * the index is left empty; you'd need to call `extend` on the index afterwards to populate it.
   */
  bool add_data_on_build;
  /** The number of inverted lists (clusters) */
  uint32_t n_lists;
  /** The number of iterations searching for kmeans centers (index building). */
  uint32_t kmeans_n_iters;
  /**
   * The number of data vectors per cluster to use during iterative kmeans building.
   * The index uses at most `n_lists * max_train_points_per_cluster` rows for training.
   */
  uint32_t max_train_points_per_cluster;
  /**
   * By default, the algorithm allocates more space than necessary for individual clusters
   * (`list_data`). This allows to amortize the cost of memory allocation and reduce the number of
   * data copies during repeated calls to `extend` (extending the database).
   *
   * The alternative is the conservative allocation behavior; when enabled, the algorithm always
   * allocates the minimum amount of memory required to store the given number of records. Set this
   * flag to `true` if you prefer to use as little GPU memory for the database as possible.
   */
  bool conservative_memory_allocation;
};

typedef struct cuvsIvfSqIndexParams* cuvsIvfSqIndexParams_t;

/**
 * @brief Allocate IVF-SQ Index params, and populate with default values
 *
 * @param[in] index_params cuvsIvfSqIndexParams_t to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsIvfSqIndexParamsCreate(cuvsIvfSqIndexParams_t* index_params);

/**
 * @brief De-allocate IVF-SQ Index params
 *
 * @param[in] index_params
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsIvfSqIndexParamsDestroy(cuvsIvfSqIndexParams_t index_params);
/**
 * @}
 */

/**
 * @defgroup ivf_sq_c_search_params IVF-SQ index search parameters
 * @{
 */
/**
 * @brief Supplemental parameters to search IVF-SQ index
 *
 */
struct cuvsIvfSqSearchParams {
  /** The number of clusters to search. */
  uint32_t n_probes;
};

typedef struct cuvsIvfSqSearchParams* cuvsIvfSqSearchParams_t;

/**
 * @brief Allocate IVF-SQ search params, and populate with default values
 *
 * @param[in] params cuvsIvfSqSearchParams_t to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsIvfSqSearchParamsCreate(cuvsIvfSqSearchParams_t* params);

/**
 * @brief De-allocate IVF-SQ search params
 *
 * @param[in] params
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsIvfSqSearchParamsDestroy(cuvsIvfSqSearchParams_t params);
/**
 * @}
 */

/**
 * @defgroup ivf_sq_c_index IVF-SQ index
 * @{
 */
/**
 * @brief Struct to hold address of cuvs::neighbors::ivf_sq::index and its active trained dtype
 *
 */
typedef struct {
  uintptr_t addr;
  DLDataType dtype;
} cuvsIvfSqIndex;

typedef cuvsIvfSqIndex* cuvsIvfSqIndex_t;

/**
 * @brief Allocate IVF-SQ index
 *
 * @param[in] index cuvsIvfSqIndex_t to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsIvfSqIndexCreate(cuvsIvfSqIndex_t* index);

/**
 * @brief De-allocate IVF-SQ index
 *
 * @param[in] index cuvsIvfSqIndex_t to de-allocate
 */
CUVS_EXPORT cuvsError_t cuvsIvfSqIndexDestroy(cuvsIvfSqIndex_t index);

/** Get the number of clusters/inverted lists */
CUVS_EXPORT cuvsError_t cuvsIvfSqIndexGetNLists(cuvsIvfSqIndex_t index, int64_t* n_lists);

/** Get the dimensionality of the data */
CUVS_EXPORT cuvsError_t cuvsIvfSqIndexGetDim(cuvsIvfSqIndex_t index, int64_t* dim);

/** Get the size of the index */
CUVS_EXPORT cuvsError_t cuvsIvfSqIndexGetSize(cuvsIvfSqIndex_t index, int64_t* size);

/**
 * @brief Get the cluster centers corresponding to the lists [n_lists, dim]
 *
 * @param[in] index cuvsIvfSqIndex_t Built Ivf-SQ Index
 * @param[out] centers Preallocated array on host or device memory to store output, [n_lists, dim]
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsIvfSqIndexGetCenters(cuvsIvfSqIndex_t index, DLManagedTensor* centers);

/**
 * @}
 */

/**
 * @defgroup ivf_sq_c_index_build IVF-SQ index build
 * @{
 */
/**
 * @brief Build an IVF-SQ index with a `DLManagedTensor` which has underlying
 *        `DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`,
 *        or `kDLCPU`. Also, acceptable underlying types are:
 *        1. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
 *        2. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 16`
 *
 * @code {.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/neighbors/ivf_sq.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // Assume a populated `DLManagedTensor` type here
 * DLManagedTensor dataset;
 *
 * // Create default index params
 * cuvsIvfSqIndexParams_t index_params;
 * cuvsError_t params_create_status = cuvsIvfSqIndexParamsCreate(&index_params);
 *
 * // Create IVF-SQ index
 * cuvsIvfSqIndex_t index;
 * cuvsError_t index_create_status = cuvsIvfSqIndexCreate(&index);
 *
 * // Build the IVF-SQ Index
 * cuvsError_t build_status = cuvsIvfSqBuild(res, index_params, &dataset, index);
 *
 * // de-allocate `index_params`, `index` and `res`
 * cuvsError_t params_destroy_status = cuvsIvfSqIndexParamsDestroy(index_params);
 * cuvsError_t index_destroy_status = cuvsIvfSqIndexDestroy(index);
 * cuvsError_t res_destroy_status = cuvsResourcesDestroy(res);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] index_params cuvsIvfSqIndexParams_t used to build IVF-SQ index
 * @param[in] dataset DLManagedTensor* training dataset
 * @param[out] index cuvsIvfSqIndex_t Newly built IVF-SQ index
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsIvfSqBuild(cuvsResources_t res,
                           cuvsIvfSqIndexParams_t index_params,
                           DLManagedTensor* dataset,
                           cuvsIvfSqIndex_t index);
/**
 * @}
 */

/**
 * @defgroup ivf_sq_c_index_search IVF-SQ index search
 * @{
 */
/**
 * @brief Search an IVF-SQ index with a `DLManagedTensor` which has underlying
 *        `DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`.
 *        Types for input are:
 *        1. `queries`: `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32` or 16
 *        2. `neighbors`: `kDLDataType.code == kDLInt` and `kDLDataType.bits = 64`
 *        3. `distances`: `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
 *
 * @code {.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/neighbors/ivf_sq.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // Assume a populated `DLManagedTensor` type here
 * DLManagedTensor queries;
 * DLManagedTensor neighbors;
 * DLManagedTensor distances;
 *
 * // Create default search params
 * cuvsIvfSqSearchParams_t search_params;
 * cuvsError_t params_create_status = cuvsIvfSqSearchParamsCreate(&search_params);
 *
 * // Search the `index` built using `cuvsIvfSqBuild`
 * cuvsError_t search_status = cuvsIvfSqSearch(
 *   res, search_params, index, &queries, &neighbors, &distances, (cuvsFilter){});
 *
 * // de-allocate `search_params` and `res`
 * cuvsError_t params_destroy_status = cuvsIvfSqSearchParamsDestroy(search_params);
 * cuvsError_t res_destroy_status = cuvsResourcesDestroy(res);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] search_params cuvsIvfSqSearchParams_t used to search IVF-SQ index
 * @param[in] index ivfSqIndex which has been returned by `cuvsIvfSqBuild`
 * @param[in] queries DLManagedTensor* queries dataset to search
 * @param[out] neighbors DLManagedTensor* output `k` neighbors for queries
 * @param[out] distances DLManagedTensor* output `k` distances for queries
 * @param[in] filter cuvsFilter input filter that can be used
 *            to filter queries and neighbors based on the given bitset.
 */
CUVS_EXPORT cuvsError_t cuvsIvfSqSearch(cuvsResources_t res,
                            cuvsIvfSqSearchParams_t search_params,
                            cuvsIvfSqIndex_t index,
                            DLManagedTensor* queries,
                            DLManagedTensor* neighbors,
                            DLManagedTensor* distances,
                            cuvsFilter filter);

/**
 * @}
 */

/**
 * @defgroup ivf_sq_c_index_serialize IVF-SQ C-API serialize functions
 * @{
 */
/**
 * Save the index to file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.c}
 * #include <cuvs/neighbors/ivf_sq.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // create an index with `cuvsIvfSqBuild`
 * cuvsIvfSqSerialize(res, "/path/to/index", index);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] filename the file name for saving the index
 * @param[in] index IVF-SQ index
 */
CUVS_EXPORT cuvsError_t cuvsIvfSqSerialize(cuvsResources_t res, const char* filename, cuvsIvfSqIndex_t index);

/**
 * Load index from file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] filename the name of the file that stores the index
 * @param[out] index IVF-SQ index loaded from disk
 */
CUVS_EXPORT cuvsError_t cuvsIvfSqDeserialize(cuvsResources_t res,
                                 const char* filename,
                                 cuvsIvfSqIndex_t index);
/**
 * @}
 */

/**
 * @defgroup ivf_sq_c_index_extend IVF-SQ index extend
 * @{
 */
/**
 * @brief Extend the index with the new data.
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] new_vectors DLManagedTensor* the new vectors to add to the index
 * @param[in] new_indices DLManagedTensor* vector of new indices for the new vectors. If the index
 *            is empty, this can be NULL to imply a continuous range `[0...n_rows)`.
 * @param[inout] index IVF-SQ index to be extended
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsIvfSqExtend(cuvsResources_t res,
                            DLManagedTensor* new_vectors,
                            DLManagedTensor* new_indices,
                            cuvsIvfSqIndex_t index);
/**
 * @}
 */
#ifdef __cplusplus
}
#endif
