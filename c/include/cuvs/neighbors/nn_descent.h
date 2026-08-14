/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

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
 * @brief Dtype to use for distance computation
 * - `NND_DIST_COMP_AUTO`: Automatically determine the best dtype for distance computation based on the dataset dimensions.
 * - `NND_DIST_COMP_FP32`: Use fp32 distance computation for better precision at the cost of performance and memory usage.
 * - `NND_DIST_COMP_FP16`: Use fp16 distance computation.
 */
typedef enum {
  NND_DIST_COMP_AUTO = 0,
  NND_DIST_COMP_FP32 = 1,
  NND_DIST_COMP_FP16 = 2
} cuvsNNDescentDistCompDtype;

/**
 * @defgroup nn_descent_c_index_params The nn-descent algorithm parameters.
 * @{
 */
/**
 * @brief Parameters used to build an nn-descent index
 *
 * `metric`: The distance metric to use
 * `metric_arg`: The argument used by distance metrics like Minkowskidistance
 * `graph_degree`: For an input dataset of dimensions (N, D),
 * determines the final dimensions of the all-neighbors knn graph
 * which turns out to be of dimensions (N, graph_degree)
 * `intermediate_graph_degree`: Internally, nn-descent builds an
 * all-neighbors knn graph of dimensions (N, intermediate_graph_degree)
 * before selecting the final `graph_degree` neighbors. It's recommended
 * that `intermediate_graph_degree` >= 1.5 * graph_degree
 * `max_iterations`: The number of iterations that nn-descent will refine
 * the graph for. More iterations produce a better quality graph at cost of performance
 * `termination_threshold`: The delta at which nn-descent will terminate its iterations
 * `return_distances`: Boolean to decide whether to return distances array
 * `dist_comp_dtype`: dtype to use for distance computation. Defaults to `NND_DIST_COMP_AUTO` which automatically determines the best dtype for distance computation based on the dataset dimensions. Use `NND_DIST_COMP_FP32` for better precision at the cost of performance and memory usage. This option is only valid when data type is fp32. Use `NND_DIST_COMP_FP16` for better performance and memory usage at the cost of precision.
 */
struct cuvsNNDescentIndexParams {
  cuvsDistanceType metric;
  float metric_arg;
  size_t graph_degree;
  size_t intermediate_graph_degree;
  size_t max_iterations;
  float termination_threshold;
  bool return_distances;
  cuvsNNDescentDistCompDtype dist_comp_dtype;
};

typedef struct cuvsNNDescentIndexParams* cuvsNNDescentIndexParams_t;

/**
 * @brief Allocate NN-Descent Index params, and populate with default values
 *
 * @param[in] index_params cuvsNNDescentIndexParams_t to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsNNDescentIndexParamsCreate(cuvsNNDescentIndexParams_t* index_params);

/**
 * @brief De-allocate NN-Descent Index params
 *
 * @param[in] index_params
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsNNDescentIndexParamsDestroy(cuvsNNDescentIndexParams_t index_params);
/**
 * @}
 */

/**
 * @defgroup nn_descent_c_index NN-Descent index
 * @{
 */
/**
 * @brief Struct to hold address of cuvs::neighbors::nn_descent::index and its active trained dtype
 *
 */
typedef struct {
  uintptr_t addr;
  DLDataType dtype;
} cuvsNNDescentIndex;

typedef cuvsNNDescentIndex* cuvsNNDescentIndex_t;

/**
 * @brief Allocate NN-Descent index
 *
 * @param[in] index cuvsNNDescentIndex_t to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsNNDescentIndexCreate(cuvsNNDescentIndex_t* index);

/**
 * @brief De-allocate NN-Descent index
 *
 * @param[in] index cuvsNNDescentIndex_t to de-allocate
 */
CUVS_EXPORT cuvsError_t cuvsNNDescentIndexDestroy(cuvsNNDescentIndex_t index);
/**
 * @}
 */

/**
 * @defgroup nn_descent_c_index_build NN-Descent index build
 * @{
 */
/**
 * @brief Build a NN-Descent index with a `DLManagedTensor` which has underlying
 *        `DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`,
 *        or `kDLCPU`. Also, acceptable underlying types are:
 *        1. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
 *        2. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 16`
 *        3. `kDLDataType.code == kDLInt` and `kDLDataType.bits = 8`
 *        4. `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 8`
 *
 * @code {.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/neighbors/nn_descent.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // Assume a populated `DLManagedTensor` type here
 * DLManagedTensor dataset;
 *
 * // Create default index params
 * cuvsNNDescentIndexParams_t index_params;
 * cuvsError_t params_create_status = cuvsNNDescentIndexParamsCreate(&index_params);
 *
 * // Create NN-Descent index
 * cuvsNNDescentIndex_t index;
 * cuvsError_t index_create_status = cuvsNNDescentIndexCreate(&index);
 *
 * // Build the NN-Descent Index
 * cuvsError_t build_status = cuvsNNDescentBuild(res, index_params, &dataset, index);
 *
 * // de-allocate `index_params`, `index` and `res`
 * cuvsError_t params_destroy_status = cuvsNNDescentIndexParamsDestroy(index_params);
 * cuvsError_t index_destroy_status = cuvsNNDescentIndexDestroy(index);
 * cuvsError_t res_destroy_status = cuvsResourcesDestroy(res);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] index_params cuvsNNDescentIndexParams_t used to build NN-Descent index
 * @param[in] dataset DLManagedTensor* training dataset on host or device memory
 * @param[inout] graph Optional preallocated graph on host memory to store output
 * @param[out] index cuvsNNDescentIndex_t Newly built NN-Descent index
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsNNDescentBuild(cuvsResources_t res,
                               cuvsNNDescentIndexParams_t index_params,
                               DLManagedTensor* dataset,
                               DLManagedTensor* graph,
                               cuvsNNDescentIndex_t index);
/**
 * @}
 */

/**
 * @brief Get the KNN graph from a built NN-Descent index
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] index cuvsNNDescentIndex_t Built NN-Descent index
 * @param[out] graph Preallocated graph on host memory to store output
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsNNDescentIndexGetGraph(cuvsResources_t res,
                                       cuvsNNDescentIndex_t index,
                                       DLManagedTensor* graph);

/**
 * @brief Get the distances from a build NN_Descent index
 *
 * This requires that the `return_distances` parameter was set when building the
 * graph
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] index cuvsNNDescentIndex_t Built NN-Descent index
 * @param[out] distances Preallocated memory to store the output distances tensor
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsNNDescentIndexGetDistances(cuvsResources_t res,
                                           cuvsNNDescentIndex_t index,
                                           DLManagedTensor* distances);
#ifdef __cplusplus
}
#endif
