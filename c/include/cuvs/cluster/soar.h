/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/c_api.h>
#include <dlpack/dlpack.h>
#include <stdint.h>

#include <cuvs/core/export.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup soar_c_params SOAR hyperparameters
 * @{
 */

/**
 * @brief Hyper-parameters for SOAR assignment.
 */
struct cuvsSoarParams {
  /**
   * Weight of the projection of the secondary residual onto the primary residual in the SOAR
   * loss. Larger values penalize secondary centroids whose residual is aligned with the primary
   * residual, favoring complementary assignments. `0` reduces the loss to plain squared distance,
   * which the primary centroid itself minimizes, so nothing is spilled. Default: 1.0.
   */
  float lambda;
};

typedef struct cuvsSoarParams* cuvsSoarParams_t;

/**
 * @brief Allocate SOAR params, and populate with default values
 *
 * @param[out] params cuvsSoarParams_t to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsSoarParamsCreate(cuvsSoarParams_t* params);

/**
 * @brief De-allocate SOAR params
 *
 * @param[in] params cuvsSoarParams_t to de-allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsSoarParamsDestroy(cuvsSoarParams_t params);

/**
 * @}
 */

/**
 * @defgroup soar_c SOAR assignment
 * @{
 */

/**
 * @brief Assign a secondary ("spilled") cluster to each row of the dataset.
 *
 * SOAR (Spilling with Orthogonality-Amplified Residuals) picks, for each vector, a second
 * centroid that complements the primary assignment instead of merely being the next-closest
 * one. It minimizes the loss of Theorem 3.1 of https://arxiv.org/abs/2404.00774: for a vector
 * `x` with primary residual `r = x - centroids[labels[i]]`,
 *
 *   `score(c) = ||x - c||^2 + lambda * (dot(r / ||r||, x - c))^2`
 *
 * and `soar_labels[i]` is the centroid minimizing that score. Indexing a vector under both its
 * primary and its secondary centroid improves recall for queries near a partition boundary.
 *
 * All tensors must be on device memory. `dataset` and `centroids` must be row-major float32.
 * `labels` and `soar_labels` must have the same dtype, either uint32 or int32; int32 is accepted
 * so that the output of `cuvsKMeansPredict` can be passed through without a conversion.
 *
 * The primary centroid is not excluded from the search, so `soar_labels[i] == labels[i]` is a
 * possible (and meaningful) result: it says that no other centroid is worth spilling to, which
 * is the common case for vectors in the interior of a cluster.
 *
 * Scratch memory scales as `n_rows * n_clusters * 4` bytes because scores against all centroids
 * are materialized at once and are not tiled. Process the dataset in row batches to bound the
 * peak device memory usage.
 *
 * @code{.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/cluster/soar.h>
 *
 * cuvsResources_t res;
 * cuvsResourcesCreate(&res);
 *
 * cuvsSoarParams_t params;
 * cuvsSoarParamsCreate(&params);
 *
 * // dataset, centroids and labels come from a prior k-means fit and predict
 * cuvsSoarPredict(res, params, &dataset, &centroids, &labels, &soar_labels);
 *
 * cuvsSoarParamsDestroy(params);
 * cuvsResourcesDestroy(res);
 * @endcode
 *
 * @param[in]  res          opaque C handle
 * @param[in]  params       Parameters for SOAR assignment.
 * @param[in]  dataset      The dataset. The data must be in row-major format.
 *                          [dim = n_rows x n_features]
 * @param[in]  centroids    Cluster centroids. The data must be in row-major format.
 *                          [dim = n_clusters x n_features]
 * @param[in]  labels       Index of the primary cluster each row belongs to, as produced by
 *                          k-means prediction. Every value must be in `[0, n_clusters)`.
 *                          [len = n_rows]
 * @param[out] soar_labels  Index of the secondary cluster each row is spilled to.
 *                          [len = n_rows]
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsSoarPredict(cuvsResources_t res,
                                        cuvsSoarParams_t params,
                                        DLManagedTensor* dataset,
                                        DLManagedTensor* centroids,
                                        DLManagedTensor* labels,
                                        DLManagedTensor* soar_labels);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif
