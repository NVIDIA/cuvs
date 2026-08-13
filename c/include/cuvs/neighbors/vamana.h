/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
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
 * @defgroup vamana_c_index_params C API for Vamana index build
 * @{
 */

/**
 * @brief Supplemental parameters to build Vamana Index
 *
 * `graph_degree`: Maximum degree of graph; corresponds to the R parameter of
 * Vamana algorithm in the literature.
 * `visited_size`: Maximum number of visited nodes per search during Vamana algorithm.
 * Loosely corresponds to the L parameter in the literature.
 * `vamana_iters`: The number of times all vectors are inserted into the graph. If > 1,
 * all vectors are re-inserted to improve graph quality.
 * `max_fraction`: The maximum batch size is this fraction of the total dataset size. Larger
 * gives faster build but lower graph quality.
 * `alpha`: Used to determine how aggressive the pruning will be.
 */
struct cuvsVamanaIndexParams {
  /** Distance type. */
  cuvsDistanceType metric;
  /** Maximum degree of output graph corresponds to the R parameter in the original Vamana
   * literature. */
  uint32_t graph_degree;
  /** Maximum number of visited nodes per search corresponds to the L parameter in the Vamana
   * literature **/
  uint32_t visited_size;
  /** Number of Vamana vector insertion iterations (each iteration inserts all vectors). */
  float vamana_iters;
  /** Alpha for pruning parameter */
  float alpha;
  /** Maximum fraction of dataset inserted per batch.              *
   * Larger max batch decreases graph quality, but improves speed */
  float max_fraction;
  /** Base of growth rate of batch sizes **/
  float batch_base;
  /** Size of candidate queue structure - should be (2^x)-1 */
  uint32_t queue_size;
  /** Max batchsize of reverse edge processing (reduces memory footprint) */
  uint32_t reverse_batchsize;
};

typedef struct cuvsVamanaIndexParams* cuvsVamanaIndexParams_t;

/**
 * @brief Allocate Vamana Index params, and populate with default values
 *
 * @param[in] params cuvsVamanaIndexParams_t to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsVamanaIndexParamsCreate(cuvsVamanaIndexParams_t* params);

/**
 * @brief De-allocate Vamana Index params
 *
 * @param[in] params cuvsVamanaIndexParams_t to de-allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsVamanaIndexParamsDestroy(cuvsVamanaIndexParams_t params);

/**
 * @}
 */

/**
 * @defgroup vamana_c_index Vamana index
 * @{
 */

/**
 * @brief Struct to hold address of cuvs::neighbors::vamana::index and its active trained dtype
 *
 */
typedef struct {
  uintptr_t addr;
  DLDataType dtype;

} cuvsVamanaIndex;

typedef cuvsVamanaIndex* cuvsVamanaIndex_t;

/**
 * @brief Allocate Vamana index
 *
 * @param[in] index cuvsVamanaIndex_t to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsVamanaIndexCreate(cuvsVamanaIndex_t* index);

/**
 * @brief De-allocate Vamana index
 *
 * @param[in] index cuvsVamanaIndex_t to de-allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsVamanaIndexDestroy(cuvsVamanaIndex_t index);

/**
 * @brief Get the dimension of the index
 *
 * @param[in] index cuvsVamanaIndex_t to get dimension of
 * @param[out] dim pointer to dimension to set
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsVamanaIndexGetDims(cuvsVamanaIndex_t index, int* dim);

/**
 * @}
 */

/**
 * @defgroup vamana_c_index_build Vamana index build
 * @{
 */

/**
 * @brief Build Vamana index
 *
 * Build the index from the dataset for efficient DiskANN search.
 *
 * The build uses the Vamana insertion-based algorithm to create the graph. The algorithm
 * starts with an empty graph and iteratively inserts batches of nodes. Each batch involves
 * performing a greedy search for each vector to be inserted, and inserting it with edges to
 * all nodes traversed during the search. Reverse edges are also inserted and robustPrune is applied
 * to improve graph quality. The index_params struct controls the degree of the final graph.
 *
 * The following distance metrics are supported:
 * - L2
 *
 * Usage example:
 * @code{.c}
 *   // Create cuvsResources_t
 *   cuvsResources_t res;
 *   cuvsResourcesCreate(&res);
 *
 *   // Assume a row-major dataset [n_rows, n_cols] is defined as `float* dataset`
 *   cuvsVamanaIndexParams_t index_params;
 *   cuvsVamanaIndexParamsCreate(&index_params);
 *   index_params->metric = L2Expanded; // set distance metric
 *   cuvsVamanaIndex_t index;
 *   cuvsVamanaIndexCreate(&index);
 *   cuvsVamanaBuild(res, index_params, dataset, index);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] params cuvsVamanaIndexParams_t used to build Vamana index
 * @param[in] dataset DLManagedTensor* training dataset
 * @param[out] index cuvsVamanaIndex_t Vamana index
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsVamanaBuild(cuvsResources_t res,
                            cuvsVamanaIndexParams_t params,
                            DLManagedTensor* dataset,
                            cuvsVamanaIndex_t index);

/**
 * @}
 */

/**
 * @defgroup vamana_c_index_serialize Vamana index serialize
 * @{
 */

/**
 * @brief Save Vamana index to file
 *
 * Matches the file format used by the DiskANN open-source repository, allowing cross-compatibility.
 *
 * Serialized Index is to be used by the DiskANN open-source repository for graph search.
 *
 * @code{.c}
 *   // Create cuvsResources_t
 *   cuvsResources_t res;
 *   cuvsResourcesCreate(&res);
 *
 *   // create an index with `cuvsVamanaBuild`
 *   cuvsVamanaSerialize(res, "/path/to/index", index, true);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] filename the file prefix for where the index is saved
 * @param[in] index cuvsVamanaIndex_t to serialize
 * @param[in] include_dataset whether to include the dataset in the serialized index
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsVamanaSerialize(cuvsResources_t res,
                                const char* filename,
                                cuvsVamanaIndex_t index,
                                bool include_dataset);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif
