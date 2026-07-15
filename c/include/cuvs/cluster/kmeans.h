/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.h>
#include <dlpack/dlpack.h>
#include <stdint.h>

#include <cuvs/core/export.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup kmeans_c_params k-means hyperparameters
 * @{
 */

typedef enum {
  /**
   * Sample the centroids using the kmeans++ strategy
   */
  KMeansPlusPlus = 0,

  /**
   * Sample the centroids uniformly at random
   */
  Random = 1,

  /**
   * User provides the array of initial centroids
   */
  Array = 2
} cuvsKMeansInitMethod;


/**
 * @brief Hyper-parameters for the kmeans algorithm
 * NB: The inertia_check field is kept for ABI compatibility. Removed in cuvsKMeansParams_v2.
 * TODO: CalVer for the replacement: 26.08
 */
struct cuvsKMeansParams {
  cuvsDistanceType metric;

  /**
   * The number of clusters to form as well as the number of centroids to generate (default:8).
   */
  int n_clusters;

  /**
   * Method for initialization, defaults to k-means++:
   *  - cuvsKMeansInitMethod::KMeansPlusPlus (k-means++): Use scalable k-means++ algorithm
   * to select the initial cluster centers.
   *  - cuvsKMeansInitMethod::Random (random): Choose 'n_clusters' observations (rows) at
   * random from the input data for the initial centroids.
   *  - cuvsKMeansInitMethod::Array (ndarray): Use 'centroids' as initial cluster centers.
   */
  cuvsKMeansInitMethod init;

  /**
   * Maximum number of iterations of the k-means algorithm for a single run.
   */
  int max_iter;

  /**
   * Relative tolerance with regards to inertia to declare convergence.
   */
  double tol;

  /**
   * Number of instance k-means algorithm will be run with different seeds.
   */
  int n_init;

  /**
   * Oversampling factor for use in the k-means|| algorithm
   */
  double oversampling_factor;

  /**
   * batch_samples and batch_centroids are used to tile 1NN computation which is
   * useful to optimize/control the memory footprint
   * Default tile is [batch_samples x n_clusters] i.e. when batch_centroids is 0
   * then don't tile the centroids
   */
  int batch_samples;

  /**
   * if 0 then batch_centroids = n_clusters
   */
  int batch_centroids;

  /** Deprecated, ignored. Kept for ABI compatibility. */
  bool inertia_check;

  /**
   * Whether to use hierarchical (balanced) kmeans or not
   */
  bool hierarchical;

  /**
   * For hierarchical k-means , defines the number of training iterations
   */
  int hierarchical_n_iters;

  /**
   * Number of samples to process per GPU batch for the batched (host-data) API.
   * When set to 0, defaults to n_samples (process all at once).
   */
  int64_t streaming_batch_size;

  /**
   * Number of samples to draw for KMeansPlusPlus initialization.
   * When set to 0, uses heuristic min(3 * n_clusters, n_samples) for host data,
   * or n_samples for device data.
   */
  int64_t init_size;
};

/**
 * @brief Hyper-parameters for the kmeans algorithm
 * TODO: Remove this after cuvsKMeansParams is replaced in ABI 2.0
 */
 struct cuvsKMeansParams_v2 {
  cuvsDistanceType metric;

  /**
   * The number of clusters to form as well as the number of centroids to generate (default:8).
   */
  int n_clusters;

  /**
   * Method for initialization, defaults to k-means++:
   *  - cuvsKMeansInitMethod::KMeansPlusPlus (k-means++): Use scalable k-means++ algorithm
   * to select the initial cluster centers.
   *  - cuvsKMeansInitMethod::Random (random): Choose 'n_clusters' observations (rows) at
   * random from the input data for the initial centroids.
   *  - cuvsKMeansInitMethod::Array (ndarray): Use 'centroids' as initial cluster centers.
   */
  cuvsKMeansInitMethod init;

  /**
   * Maximum number of iterations of the k-means algorithm for a single run.
   */
  int max_iter;

  /**
   * Relative tolerance with regards to inertia to declare convergence.
   */
  double tol;

  /**
   * Number of instance k-means algorithm will be run with different seeds.
   */
  int n_init;

  /**
   * Oversampling factor for use in the k-means|| algorithm
   */
  double oversampling_factor;

  /**
   * batch_samples and batch_centroids are used to tile 1NN computation which is
   * useful to optimize/control the memory footprint
   * Default tile is [batch_samples x n_clusters] i.e. when batch_centroids is 0
   * then don't tile the centroids
   */
  int batch_samples;

  /**
   * if 0 then batch_centroids = n_clusters
   */
  int batch_centroids;

  /**
   * Whether to use hierarchical (balanced) kmeans or not
   */
  bool hierarchical;

  /**
   * For hierarchical k-means , defines the number of training iterations
   */
  int hierarchical_n_iters;

  /**
   * Number of samples to process per GPU batch for the batched (host-data) API.
   * When set to 0, defaults to n_samples (process all at once).
   */
  int64_t streaming_batch_size;

  /**
   * Number of samples to draw for KMeansPlusPlus initialization.
   * When set to 0, uses heuristic min(3 * n_clusters, n_samples) for host data,
   * or n_samples for device data.
   */
  int64_t init_size;
};

typedef struct cuvsKMeansParams* cuvsKMeansParams_t;
typedef struct cuvsKMeansParams_v2* cuvsKMeansParams_v2_t;

/**
 * @brief Allocate KMeans params, and populate with default values
 *
 * @note In cuVS 26.08 (next ABI major version) this signature will be
 * replaced by cuvsKMeansParamsCreate_v2.
 *
 * @param[in] params cuvsKMeansParams_t to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsKMeansParamsCreate(cuvsKMeansParams_t* params);

/**
 * @brief De-allocate KMeans params
 *
 * @note In cuVS 26.08 (next ABI major version) this signature will be
 * replaced by cuvsKMeansParamsDestroy_v2.
 *
 * @param[in] params
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsKMeansParamsDestroy(cuvsKMeansParams_t params);

/**
 * @brief Allocate KMeans params
 *
 * Mirrors cuvsKMeansParamsCreate but operates on cuvsKMeansParams_v2.
 * Will become the unsuffixed cuvsKMeansParamsCreate in cuVS 26.08.
 *
 * @param[in] params cuvsKMeansParams_v2_t to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsKMeansParamsCreate_v2(cuvsKMeansParams_v2_t* params);

/**
 * @brief De-allocate KMeans params allocated by cuvsKMeansParamsCreate_v2.
 *
 * @param[in] params
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsKMeansParamsDestroy_v2(cuvsKMeansParams_v2_t params);

/**
 * @brief Type of k-means algorithm.
 */
typedef enum { CUVS_KMEANS_TYPE_KMEANS = 0, CUVS_KMEANS_TYPE_KMEANS_BALANCED = 1 } cuvsKMeansType;

/**
 * @}
 */

/**
 * @defgroup kmeans_c k-means clustering APIs
 * @{
 */

/**
 * @brief Find clusters with k-means algorithm.
 *
 *   Initial centroids are chosen with k-means++ algorithm. Empty
 *   clusters are reinitialized by choosing new centroids with
 *   k-means++ algorithm.
 *
 *   X may reside on either host (CPU) or device (GPU) memory.
 *   When X is on the host the data is streamed to the GPU in
 *   batches controlled by params->streaming_batch_size.
 *
 * @note In cuVS 26.08 (next ABI major version) this signature will be
 * replaced by cuvsKMeansFit_v2.
 *
 * @param[in]     res           opaque C handle
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. The data must
 *                              be in row-major format. May be on host or
 *                              device memory.
 *                              [dim = n_samples x n_features]
 * @param[in]     sample_weight Optional weights for each observation in X.
 *                              Must be on the same memory space as X.
 *                              [len = n_samples]
 * @param[inout]  centroids     [in] When init is InitMethod::Array, use
 *                              centroids as the initial cluster centers.
 *                              [out] The generated centroids from the
 *                              kmeans algorithm are stored at the address
 *                              pointed by 'centroids'. Must be on device.
 *                              [dim = n_clusters x n_features]
 * @param[out]    inertia       Sum of squared distances of samples to their
 *                              closest cluster center.
 * @param[out]    n_iter        Number of iterations run.
 */
CUVS_EXPORT cuvsError_t cuvsKMeansFit(cuvsResources_t res,
                          cuvsKMeansParams_t params,
                          DLManagedTensor* X,
                          DLManagedTensor* sample_weight,
                          DLManagedTensor* centroids,
                          double* inertia,
                          int* n_iter);

/**
 * @brief Find clusters with k-means algorithm (v2 params layout).
 *
 * Mirrors cuvsKMeansFit but takes cuvsKMeansParams_v2_t. Will become the
 * unsuffixed cuvsKMeansFit in cuVS 26.08.
 *
 * @param[in]     res           opaque C handle
 * @param[in]     params        Parameters for KMeans model (v2 layout).
 * @param[in]     X             Training instances to cluster. The data must
 *                              be in row-major format. May be on host or
 *                              device memory.
 *                              [dim = n_samples x n_features]
 * @param[in]     sample_weight Optional weights for each observation in X.
 *                              Must be on the same memory space as X.
 *                              [len = n_samples]
 * @param[inout]  centroids     [in] When init is InitMethod::Array, use
 *                              centroids as the initial cluster centers.
 *                              [out] The generated centroids from the
 *                              kmeans algorithm are stored at the address
 *                              pointed by 'centroids'. Must be on device.
 *                              [dim = n_clusters x n_features]
 * @param[out]    inertia       Sum of squared distances of samples to their
 *                              closest cluster center.
 * @param[out]    n_iter        Number of iterations run.
 */
CUVS_EXPORT cuvsError_t cuvsKMeansFit_v2(cuvsResources_t res,
                             cuvsKMeansParams_v2_t params,
                             DLManagedTensor* X,
                             DLManagedTensor* sample_weight,
                             DLManagedTensor* centroids,
                             double* inertia,
                             int* n_iter);

/**
 * @brief Predict the closest cluster each sample in X belongs to.
 *
 * @note In cuVS 26.08 (next ABI major version) this signature will be
 * replaced by cuvsKMeansPredict_v2.
 *
 * @param[in]     res              opaque C handle
 * @param[in]     params           Parameters for KMeans model.
 * @param[in]     X                New data to predict.
 *                                 [dim = n_samples x n_features]
 * @param[in]     sample_weight    Optional weights for each observation in X.
 *                                 [len = n_samples]
 * @param[in]     centroids        Cluster centroids. The data must be in
 *                                 row-major format.
 *                                 [dim = n_clusters x n_features]
 * @param[in]     normalize_weight True if the weights should be normalized
 * @param[out]    labels           Index of the cluster each sample in X
 *                                 belongs to.
 *                                 [len = n_samples]
 * @param[out]    inertia          Sum of squared distances of samples to
 *                                 their closest cluster center.
 */
CUVS_EXPORT cuvsError_t cuvsKMeansPredict(cuvsResources_t res,
                              cuvsKMeansParams_t params,
                              DLManagedTensor* X,
                              DLManagedTensor* sample_weight,
                              DLManagedTensor* centroids,
                              DLManagedTensor* labels,
                              bool normalize_weight,
                              double* inertia);

/**
 * @brief Predict the closest cluster each sample in X belongs to (v2 params layout).
 *
 * Mirrors cuvsKMeansPredict but takes cuvsKMeansParams_v2_t. Will become the
 * unsuffixed cuvsKMeansPredict in cuVS 26.08.
 *
 * @param[in]     res              opaque C handle
 * @param[in]     params           Parameters for KMeans model (v2 layout).
 * @param[in]     X                New data to predict.
 *                                 [dim = n_samples x n_features]
 * @param[in]     sample_weight    Optional weights for each observation in X.
 *                                 [len = n_samples]
 * @param[in]     centroids        Cluster centroids. The data must be in
 *                                 row-major format.
 *                                 [dim = n_clusters x n_features]
 * @param[in]     normalize_weight True if the weights should be normalized
 * @param[out]    labels           Index of the cluster each sample in X
 *                                 belongs to.
 *                                 [len = n_samples]
 * @param[out]    inertia          Sum of squared distances of samples to
 *                                 their closest cluster center.
 */
CUVS_EXPORT cuvsError_t cuvsKMeansPredict_v2(cuvsResources_t res,
                                 cuvsKMeansParams_v2_t params,
                                 DLManagedTensor* X,
                                 DLManagedTensor* sample_weight,
                                 DLManagedTensor* centroids,
                                 DLManagedTensor* labels,
                                 bool normalize_weight,
                                 double* inertia);

/**
 * @brief Compute cluster cost
 *
 * @param[in]  res            opaque C handle
 * @param[in]  X              Training instances to cluster. The data must
 *                            be in row-major format.
 *                            [dim = n_samples x n_features]
 * @param[in]  centroids      Cluster centroids. The data must be in
 *                            row-major format.
 *                            [dim = n_clusters x n_features]
 * @param[out] cost           Resulting cluster cost
 *
 */
CUVS_EXPORT cuvsError_t cuvsKMeansClusterCost(cuvsResources_t res,
                                  DLManagedTensor* X,
                                  DLManagedTensor* centroids,
                                  double* cost);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif
