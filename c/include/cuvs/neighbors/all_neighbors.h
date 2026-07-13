/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.h>
#include <cuvs/neighbors/ivf_pq.h>
#include <cuvs/neighbors/nn_descent.h>
#include <dlpack/dlpack.h>
#include <stdint.h>

#include <cuvs/core/export.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup all_neighbors_c_params All-neighbors C-API build parameters
 * @{
 *
 * All-neighbors constructs an approximate k-NN graph for all vectors in a dataset.
 * The All-neighbors API can either be used in single-GPU or multi-GPU mode.
 * For multi-GPU deployment, please pass a multi-GPU resources handle and
 * provide the dataset on host.
 *
 * Notes:
 * - Outputs (indices, distances) can be on host memory (numpy arrays)
 *   or device memory (CUDA arrays). core_distances can only be on device memory.
 * - Host variant accepts host-resident dataset; device variant accepts device-resident dataset.
 * - For batching, `overlap_factor < n_clusters` must hold.
 * - When `core_distances` is provided, mutual-reachability distances are produced (see alpha).
 */

/**
 * @brief Graph build algorithm selection.
 */
typedef enum {
  CUVS_ALL_NEIGHBORS_ALGO_BRUTE_FORCE = 0,  ///< Use Brute Force for local kNN subgraphs
  CUVS_ALL_NEIGHBORS_ALGO_IVF_PQ = 1,  ///< Use IVF-PQ for local kNN subgraphs (host dataset only)
  CUVS_ALL_NEIGHBORS_ALGO_NN_DESCENT = 2  ///< Use NN-Descent for local kNN subgraphs
} cuvsAllNeighborsAlgo;

/**
 * @brief Parameters controlling SNMG all-neighbors build.
 */
struct cuvsAllNeighborsIndexParams {
  cuvsAllNeighborsAlgo algo;  ///< Local kNN graph build algorithm
  size_t overlap_factor;  ///< Number of clusters each point is assigned to (must be < n_clusters)
  size_t
    n_clusters;  ///< Number of clusters/batches to partition the dataset into (> overlap_factor)
  cuvsDistanceType metric;  ///< Distance metric used for graph construction

  // Algorithm-specific parameters (only one should be set based on algo)
  cuvsIvfPqIndexParams_t ivf_pq_params;          ///< Parameters for IVF-PQ algorithm (when algo ==
                                                 ///< CUVS_ALL_NEIGHBORS_ALGO_IVF_PQ)
  cuvsNNDescentIndexParams_t nn_descent_params;  ///< Parameters for NN-Descent algorithm (when algo
                                                 ///< == CUVS_ALL_NEIGHBORS_ALGO_NN_DESCENT)
};

typedef struct cuvsAllNeighborsIndexParams* cuvsAllNeighborsIndexParams_t;

/**
 * @brief Create a default all-neighbors index parameters struct.
 *
 * @param[out] index_params  Pointer to allocated index_params struct
 *
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsAllNeighborsIndexParamsCreate(cuvsAllNeighborsIndexParams_t* index_params);

/**
 * @brief Destroy an all-neighbors index parameters struct.
 *
 * @param[in] index_params  Index parameters struct to destroy
 *
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsAllNeighborsIndexParamsDestroy(cuvsAllNeighborsIndexParams_t index_params);

/** @} */

/**
 * @defgroup all_neighbors_c_build All-neighbors C-API build
 * @{
 */

/**
 * @brief Build an all-neighbors k-NN graph automatically detecting host vs device dataset.
 *
 * @param[in] res             Can be a SNMG multi-GPU resources (`cuvsResources_t`) or single-GPU
 * resources
 * @param[in] params          Build parameters (see cuvsAllNeighborsIndexParams)
 * @param[in] dataset         2D tensor [num_rows x dim] on host or device (auto-detected)
 * @param[out] indices        2D tensor [num_rows x k] on host or device (int64)
 * @param[out] distances      Optional 2D tensor [num_rows x k] on host or device (float32); can be NULL
 * @param[out] core_distances Optional 1D tensor [num_rows] on device (float32); can be NULL
 * @param[in] alpha           Mutual-reachability scaling; used only when core_distances is provided
 *
 * The function automatically detects whether the dataset is host-resident or device-resident
 * and calls the appropriate implementation. For host datasets, it partitions data into
 * `n_clusters` clusters and assigns each row to `overlap_factor` nearest clusters. For device
 * datasets, `n_clusters` must be 1 (no batching); `overlap_factor` is ignored.
 * Outputs can be on host memory (numpy arrays) or device memory (CUDA arrays).
 */
CUVS_EXPORT cuvsError_t cuvsAllNeighborsBuild(cuvsResources_t res,
                                  cuvsAllNeighborsIndexParams_t params,
                                  DLManagedTensor* dataset,
                                  DLManagedTensor* indices,
                                  DLManagedTensor* distances,
                                  DLManagedTensor* core_distances,
                                  float alpha);

/** @} */

#ifdef __cplusplus
}
#endif
