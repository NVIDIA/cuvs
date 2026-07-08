/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cagra.h"

#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.h>
#include <dlpack/dlpack.h>
#include <stdbool.h>
#include <stdint.h>

#include <cuvs/core/export.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup hnsw_c_index_params C API for HNSW index params
 * @{
 */

/**
 * @brief Hierarchy for HNSW index when converting from CAGRA index
 *
 * NOTE: When the value is `NONE`, the HNSW index is built as a base-layer-only
 * index.
 */
enum cuvsHnswHierarchy {
  /* Flat hierarchy, search is base-layer only */
  NONE = 0,
  /* Full hierarchy is built using the CPU */
  CPU = 1,
  /* Full hierarchy is built using the GPU */
  GPU = 2
};

/**
 * Parameters for ACE (Augmented Core Extraction) graph build for HNSW.
 * ACE enables building indexes for datasets too large to fit in GPU memory by:
 * 1. Partitioning the dataset in core and augmented partitions using balanced
 * k-means
 * 2. Building sub-indexes for each partition independently
 * 3. Concatenating sub-graphs into a final unified index
 */
struct cuvsHnswAceParams {
  /**
   * Number of partitions for ACE partitioned build.
   *
   * When set to 0 (default), the number of partitions is automatically derived
   * based on available host and GPU memory to maximize partition size while
   * ensuring the build fits in memory.
   *
   * Small values might improve recall but potentially degrade performance and
   * increase memory usage. The partition size is on average 2 * (n_rows /
   * npartitions) * dim * sizeof(T). 2 is because of the core and augmented
   * vectors. Please account for imbalance in the partition sizes (up to 3x in
   * our tests).
   *
   * If the specified number of partitions results in partitions that exceed
   * available memory, the value will be automatically increased to fit memory
   * constraints and a warning will be issued.
   */
  size_t npartitions;
  /**
   * Directory to store ACE build artifacts (e.g., KNN graph, optimized graph).
   * Used when `use_disk` is true or when the graph does not fit in memory.
   */
  const char *build_dir;
  /**
   * Whether to use disk-based storage for ACE build.
   * When true, enables disk-based operations for memory-efficient graph
   * construction.
   */
  bool use_disk;
  /**
   * Maximum host memory to use for ACE build in GiB.
   * When set to 0 (default), uses available host memory.
   * Useful for testing or when running alongside other memory-intensive
   * processes.
   */
  double max_host_memory_gb;
  /**
   * Maximum GPU memory to use for ACE build in GiB.
   * When set to 0 (default), uses available GPU memory.
   * Useful for testing or when running alongside other memory-intensive
   * processes.
   */
  double max_gpu_memory_gb;
};

typedef struct cuvsHnswAceParams *cuvsHnswAceParams_t;

/**
 * @brief Allocate HNSW ACE params, and populate with default values
 *
 * @param[in] params cuvsHnswAceParams_t to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsHnswAceParamsCreate(cuvsHnswAceParams_t *params);

/**
 * @brief De-allocate HNSW ACE params
 *
 * @param[in] params
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsHnswAceParamsDestroy(cuvsHnswAceParams_t params);

struct cuvsHnswIndexParams {
  /* hierarchy of the hnsw index */
  enum cuvsHnswHierarchy hierarchy;
  /** Size of the candidate list during index construction. */
  int ef_construction;
  /** Number of host threads to use to construct hierarchy when hierarchy is
     `CPU` or `GPU`. When the value is 0, the number of threads is automatically
     determined to the maximum number of threads available. NOTE: When hierarchy
     is `GPU`, while the majority of the work is done on the GPU, initialization
     of the HNSW index itself and some other work is parallelized with the help
     of CPU threads.
  */
  int num_threads;
  /** HNSW M parameter: number of bi-directional links per node used to derive
   * the internal GPU graph degree.
   */
  size_t M;
  /** Distance type for the index. */
  cuvsDistanceType metric;
  /**
   * Optional: specify ACE parameters for partitioned or disk-backed GPU graph
   * building. Set to nullptr to build with HNSW parameters and let cuVS choose
   * CAGRA graph build settings internally.
   */
  cuvsHnswAceParams_t ace_params;
};

typedef struct cuvsHnswIndexParams *cuvsHnswIndexParams_t;

/**
 * @brief Allocate HNSW Index params, and populate with default values
 *
 * @param[in] params cuvsHnswIndexParams_t to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t
cuvsHnswIndexParamsCreate(cuvsHnswIndexParams_t *params);

/**
 * @brief De-allocate HNSW Index params
 *
 * @param[in] params
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t
cuvsHnswIndexParamsDestroy(cuvsHnswIndexParams_t params);

/**
 * @}
 */

/**
 * @defgroup hnsw_c_index C API for hnswlib wrapper index
 * @{
 */

/**
 * @brief Struct to hold address of cuvs::neighbors::Hnsw::index and its active
 * trained dtype
 *
 */
typedef struct {
  uintptr_t addr;
  DLDataType dtype;

} cuvsHnswIndex;

typedef cuvsHnswIndex *cuvsHnswIndex_t;

/**
 * @brief Allocate HNSW index
 *
 * @param[in] index cuvsHnswIndex_t to allocate
 * @return HnswError_t
 */
CUVS_EXPORT cuvsError_t cuvsHnswIndexCreate(cuvsHnswIndex_t *index);

/**
 * @brief De-allocate HNSW index
 *
 * @param[in] index cuvsHnswIndex_t to de-allocate
 */
CUVS_EXPORT cuvsError_t cuvsHnswIndexDestroy(cuvsHnswIndex_t index);

/**
 * @}
 */

/**
 * @defgroup hnsw_c_extend_params Parameters for extending HNSW index
 * @{
 */

struct cuvsHnswExtendParams {
  /** Number of CPU threads used to extend additional vectors */
  int num_threads;
};

typedef struct cuvsHnswExtendParams *cuvsHnswExtendParams_t;

/**
 * @brief Allocate HNSW extend params, and populate with default values
 *
 * @param[in] params cuvsHnswExtendParams_t to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t
cuvsHnswExtendParamsCreate(cuvsHnswExtendParams_t *params);

/**
 * @brief De-allocate HNSW extend params
 *
 * @param[in] params cuvsHnswExtendParams_t to de-allocate
 * @return cuvsError_t
 */

CUVS_EXPORT cuvsError_t
cuvsHnswExtendParamsDestroy(cuvsHnswExtendParams_t params);

/**
 * @}
 */

/**
 * @defgroup hnsw_c_index_load Load CAGRA index as hnswlib index
 * @{
 */

/**
 * @brief Convert a CAGRA Index to an HNSW index.
 * NOTE: When hierarchy is:
 *       1. `NONE`: This method uses the filesystem to write the CAGRA index in
 * `/tmp/<random_number>.bin` before reading it as an hnswlib index, then
 * deleting the temporary file. The returned index is immutable and can only be
 * searched by the hnswlib wrapper in cuVS, as the format is not compatible with
 * the original hnswlib.
 *       2. `CPU`: The returned index is mutable and can be extended with
 * additional vectors. The serialized index is also compatible with the original
 * hnswlib library.
 *       3. `GPU`: The returned index is mutable, and its hierarchy is built on
 * the GPU. The serialized index is also compatible with the original hnswlib
 * library.
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] params cuvsHnswIndexParams_t used to load Hnsw index
 * @param[in] cagra_index cuvsCagraIndex_t to convert to HNSW index
 * @param[out] hnsw_index cuvsHnswIndex_t to return the HNSW index
 *
 * @return cuvsError_t
 *
 * @code{.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/neighbors/cagra.h>
 * #include <cuvs/neighbors/hnsw.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // create a CAGRA index with `cuvsCagraBuild`
 *
 * // Convert the CAGRA index to an HNSW index
 * cuvsHnswIndex_t hnsw_index;
 * cuvsHnswIndexCreate(&hnsw_index);
 * cuvsHnswIndexParams_t hnsw_params;
 * cuvsHnswIndexParamsCreate(&hnsw_params);
 * cuvsHnswFromCagra(res, hnsw_params, cagra_index, hnsw_index);
 *
 * // de-allocate `hnsw_params`, `hnsw_index` and `res`
 * cuvsError_t hnsw_params_destroy_status =
 * cuvsHnswIndexParamsDestroy(hnsw_params); cuvsError_t
 * hnsw_index_destroy_status = cuvsHnswIndexDestroy(hnsw_index); cuvsError_t
 * res_destroy_status = cuvsResourcesDestroy(res);
 * @endcode
 */
CUVS_EXPORT cuvsError_t cuvsHnswFromCagra(cuvsResources_t res,
                                          cuvsHnswIndexParams_t params,
                                          cuvsCagraIndex_t cagra_index,
                                          cuvsHnswIndex_t hnsw_index);

CUVS_EXPORT cuvsError_t cuvsHnswFromCagraWithDataset(
    cuvsResources_t res, cuvsHnswIndexParams_t params,
    cuvsCagraIndex_t cagra_index, cuvsHnswIndex_t hnsw_index,
    DLManagedTensor *dataset_tensor);

/**
 * @}
 */

/**
 * @defgroup hnsw_c_index_build Build HNSW index on the GPU
 * @{
 */

/**
 * @brief Build an HNSW index on the GPU and search it on the CPU.
 *
 * cuVS accepts HNSW parameters (`M`, `ef_construction`, hierarchy, and metric),
 * builds a compatible graph on the GPU, and returns an HNSW index for CPU
 * search. Internally, cuVS may use CAGRA graph construction and convert the
 * graph to HNSW format. Users can leave `params->ace_params` as nullptr for
 * the default in-memory path, or set ACE parameters to request partitioned or
 * disk-backed graph construction.
 *
 * NOTE: This function requires CUDA to be available at runtime.
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] params cuvsHnswIndexParams_t with HNSW build parameters
 * @param[in] dataset DLManagedTensor* host dataset to build index from
 * @param[out] index cuvsHnswIndex_t to return the built HNSW index
 *
 * @return cuvsError_t
 *
 * @code{.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/neighbors/hnsw.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsResourcesCreate(&res);
 *
 * // Create index parameters
 * cuvsHnswIndexParams_t params;
 * cuvsHnswIndexParamsCreate(&params);
 * params->hierarchy = GPU;
 * params->M = 32;
 * params->ef_construction = 120;
 * params->metric = L2Expanded;
 *
 * // Create HNSW index
 * cuvsHnswIndex_t hnsw_index;
 * cuvsHnswIndexCreate(&hnsw_index);
 *
 * // Assume dataset is a populated DLManagedTensor with host data
 * DLManagedTensor dataset;
 *
 * // Build the index
 * cuvsHnswBuild(res, params, &dataset, hnsw_index);
 *
 * // Clean up
 * cuvsHnswIndexParamsDestroy(params);
 * cuvsHnswIndexDestroy(hnsw_index);
 * cuvsResourcesDestroy(res);
 * @endcode
 */
CUVS_EXPORT cuvsError_t cuvsHnswBuild(cuvsResources_t res,
                                      cuvsHnswIndexParams_t params,
                                      DLManagedTensor *dataset,
                                      cuvsHnswIndex_t index);

/**
 * @}
 */

/**
 * @defgroup hnsw_c_index_extend Extend HNSW index with additional vectors
 * @{
 */

/**
 * @brief Add new vectors to an HNSW index
 * NOTE: A base-layer-only index with hierarchy `NONE` cannot be extended.
 Indexes with hierarchy
 * `CPU` or `GPU` can be extended.

 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] params cuvsHnswExtendParams_t used to extend Hnsw index
 * @param[in] additional_dataset DLManagedTensor* additional dataset to extend
 the index
 * @param[inout] index cuvsHnswIndex_t to extend
  *
  * @return cuvsError_t
  *
  * @code{.c}
  * #include <cuvs/core/c_api.h>
  * #include <cuvs/neighbors/cagra.h>
  * #include <cuvs/neighbors/hnsw.h>
  *
  * // Create cuvsResources_t
  * cuvsResources_t res;
  * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
  *
  * // create an index with `cuvsCagraBuild`
  *
  * // Convert the CAGRA index to an HNSW index
  * cuvsHnswIndex_t hnsw_index;
  * cuvsHnswIndexCreate(&hnsw_index);
  * cuvsHnswIndexParams_t hnsw_params;
  * cuvsHnswIndexParamsCreate(&hnsw_params);
  * cuvsHnswFromCagra(res, hnsw_params, cagra_index, hnsw_index);
  *
  * // Extend the HNSW index with additional vectors
  * DLManagedTensor additional_dataset;
  * cuvsHnswExtendParams_t extend_params;
  * cuvsHnswExtendParamsCreate(&extend_params);
  * cuvsHnswExtend(res, extend_params, &additional_dataset, hnsw_index);
  *
  * // de-allocate `hnsw_params`, `hnsw_index`, `extend_params` and `res`
  * cuvsError_t hnsw_params_destroy_status =
 cuvsHnswIndexParamsDestroy(hnsw_params);
  * cuvsError_t hnsw_index_destroy_status = cuvsHnswIndexDestroy(hnsw_index);
  * cuvsError_t extend_params_destroy_status =
 cuvsHnswExtendParamsDestroy(extend_params);
  * cuvsError_t res_destroy_status = cuvsResourcesDestroy(res);
  * @endcode
  */

CUVS_EXPORT cuvsError_t cuvsHnswExtend(cuvsResources_t res,
                                       cuvsHnswExtendParams_t params,
                                       DLManagedTensor *additional_dataset,
                                       cuvsHnswIndex_t index);

/**
 * @}
 */

/**
 * @defgroup hnsw_c_search_params C API for hnswlib wrapper search params
 * @{
 */

struct cuvsHnswSearchParams {
  int32_t ef;
  int32_t num_threads;
};

typedef struct cuvsHnswSearchParams *cuvsHnswSearchParams_t;

/**
 * @brief Allocate HNSW search params, and populate with default values
 *
 * @param[in] params cuvsHnswSearchParams_t to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t
cuvsHnswSearchParamsCreate(cuvsHnswSearchParams_t *params);

/**
 * @brief De-allocate HNSW search params
 *
 * @param[in] params cuvsHnswSearchParams_t to de-allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t
cuvsHnswSearchParamsDestroy(cuvsHnswSearchParams_t params);

/**
 * @}
 */

/**
 * @defgroup hnsw_c_index_search C API for CUDA ANN Graph-based nearest neighbor
 * search
 * @{
 */
/**
 * @brief Search a HNSW index with a `DLManagedTensor` which has underlying
 *        `DLDeviceType` equal to `kDLCPU`, `kDLCUDAHost`, or `kDLCUDAManaged`.
 *        It is also important to note that the HNSW Index must have been built
 *        with the same type of `queries`, such that `index.dtype.code ==
 *        queries.dl_tensor.dtype.code` and `index.dtype.bits ==
 *        queries.dl_tensor.dtype.bits`
 *        Supported types for input are:
 *        1. `queries`:
 *          a. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
 *          b. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 16`
 *          c. `kDLDataType.code == kDLInt` and `kDLDataType.bits = 8`
 *          d. `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 8`
 *        2. `neighbors`: `kDLDataType.code == kDLUInt` and `kDLDataType.bits =
 * 64`
 *        3. `distances`: `kDLDataType.code == kDLFloat` and `kDLDataType.bits =
 * 32` NOTE: When hierarchy is `NONE`, the HNSW index can only be searched by
 * the hnswlib wrapper in cuVS, as the format is not compatible with the
 * original hnswlib.
 *
 * @code {.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/neighbors/hnsw.h>
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
 * cuvsHnswSearchParams_t params;
 * cuvsError_t params_create_status = cuvsHnswSearchParamsCreate(&params);
 *
 * // Search the HNSW index
 * cuvsError_t search_status = cuvsHnswSearch(res, params, index, &queries,
 * &neighbors, &distances);
 *
 * // de-allocate `params` and `res`
 * cuvsError_t params_destroy_status = cuvsHnswSearchParamsDestroy(params);
 * cuvsError_t res_destroy_status = cuvsResourcesDestroy(res);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] params cuvsHnswSearchParams_t used to search Hnsw index
 * @param[in] index cuvsHnswIndex returned by `cuvsHnswBuild`,
 * `cuvsHnswFromCagra`, or `cuvsHnswDeserialize`
 * @param[in] queries DLManagedTensor* queries dataset to search
 * @param[out] neighbors DLManagedTensor* output `k` neighbors for queries
 * @param[out] distances DLManagedTensor* output `k` distances for queries
 */
CUVS_EXPORT cuvsError_t cuvsHnswSearch(cuvsResources_t res,
                                       cuvsHnswSearchParams_t params,
                                       cuvsHnswIndex_t index,
                                       DLManagedTensor *queries,
                                       DLManagedTensor *neighbors,
                                       DLManagedTensor *distances);

/**
 * @}
 */

/**
 * @defgroup hnsw_c_index_serialize HNSW C-API serialize functions
 * @{
 */

/**
 * @brief Serialize an HNSW index to a file.
 * NOTE: When hierarchy is `NONE`, the saved hnswlib index is immutable and can
 * only be read by the hnswlib wrapper in cuVS, as the serialization format is
 * not compatible with the original hnswlib. However, when hierarchy is `CPU` or
 * `GPU`, the saved hnswlib index is compatible with the original hnswlib
 * library.
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] filename the name of the file to save the index
 * @param[in] index cuvsHnswIndex_t to serialize
 * @return cuvsError_t
 *
 * @code{.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/neighbors/cagra.h>
 * #include <cuvs/neighbors/hnsw.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // Assume a populated `DLManagedTensor` host dataset here
 * DLManagedTensor dataset;
 *
 * // Build an HNSW index
 * cuvsHnswIndexParams_t hnsw_params;
 * cuvsHnswIndexParamsCreate(&hnsw_params);
 * cuvsHnswIndex_t hnsw_index;
 * cuvsHnswIndexCreate(&hnsw_index);
 * cuvsHnswBuild(res, hnsw_params, &dataset, hnsw_index);
 *
 * // Serialize the HNSW index
 * cuvsHnswSerialize(res, "/path/to/index", hnsw_index);
 *
 * // de-allocate `hnsw_params`, `hnsw_index` and `res`
 * cuvsError_t hnsw_params_destroy_status =
 * cuvsHnswIndexParamsDestroy(hnsw_params); cuvsError_t
 * hnsw_index_destroy_status = cuvsHnswIndexDestroy(hnsw_index); cuvsError_t
 * res_destroy_status = cuvsResourcesDestroy(res);
 * @endcode
 */
CUVS_EXPORT cuvsError_t cuvsHnswSerialize(cuvsResources_t res,
                                          const char *filename,
                                          cuvsHnswIndex_t index);

/**
 * Load hnswlib index from file which was serialized from a HNSW index.
 * NOTE: When hierarchy is `NONE`, the loaded hnswlib index is immutable and can
 * only be read by the hnswlib wrapper in cuVS, as the serialization format is
 * not compatible with the original hnswlib. Indexes loaded with hierarchy `CPU`
 * or `GPU` are mutable and compatible with the original hnswlib library.
 * Experimental, both the API and the serialization format are subject to
 * change. NOTE: `params->hierarchy` must match the hierarchy the index was
 * serialized with. Loading a base-layer-only index saved with hierarchy `NONE`
 * requires setting `params->hierarchy = NONE` (the default created by
 * `cuvsHnswIndexParamsCreate` is `GPU`).
 *
 * @code{.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/neighbors/cagra.h>
 * #include <cuvs/neighbors/hnsw.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // Load the serialized HNSW index from file.
 * cuvsHnswIndex_t hnsw_index;
 * cuvsHnswIndexCreate(&hnsw_index);
 * cuvsHnswIndexParams_t hnsw_params;
 * cuvsHnswIndexParamsCreate(&hnsw_params);
 * // hierarchy must match the hierarchy the index was serialized with,
 * // e.g. NONE for a base-layer-only index.
 * hnsw_params->hierarchy = NONE;
 * // The dtype must match the dtype used to build the saved index.
 * hnsw_index->dtype.code = kDLFloat;
 * hnsw_index->dtype.bits = 32;
 * hnsw_index->dtype.lanes = 1;
 * int dim = 128;  // dimension of the vectors in the saved index
 * cuvsHnswDeserialize(res, hnsw_params, "/path/to/index", dim, L2Expanded,
 * hnsw_index);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] params cuvsHnswIndexParams_t used to load Hnsw index;
 * `params->hierarchy` must match the hierarchy the index was serialized with
 * @param[in] filename the name of the file that stores the index
 * @param[in] dim the dimension of the vectors in the index
 * @param[in] metric the distance metric used to build the index; takes
 * precedence over `params->metric`
 * @param[out] index HNSW index loaded disk
 */
CUVS_EXPORT cuvsError_t cuvsHnswDeserialize(cuvsResources_t res,
                                            cuvsHnswIndexParams_t params,
                                            const char *filename, int dim,
                                            cuvsDistanceType metric,
                                            cuvsHnswIndex_t index);
/**
 * @}
 */

#ifdef __cplusplus
}
#endif
