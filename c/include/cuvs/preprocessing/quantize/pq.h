/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/cluster/kmeans.h>
#include <cuvs/core/c_api.h>
#include <dlpack/dlpack.h>
#include <stdint.h>

#include <cuvs/core/export.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup preprocessing_c_pq C API for Product Quantizer
 * @{
 */
/**
 * @brief Product quantizer parameters.
 */
struct cuvsProductQuantizerParams {
  /**
   * The bit length of the vector element after compression by PQ.
   *
   * Possible values: within [4, 16].
   *
   * Hint: the smaller the 'pq_bits', the smaller the index size and the better the search
   * performance, but the lower the recall.
   */
  uint32_t pq_bits;
  /**
   * The dimensionality of the vector after compression by PQ.
   * When zero, an optimal value is selected using a heuristic.
   *
   * TODO: at the moment `dim` must be a multiple `pq_dim`.
   */
  uint32_t pq_dim;
  /**
   * Whether to use subspaces for product quantization (PQ).
   * When true, one PQ codebook is used for each subspace. Otherwise, a single
   * PQ codebook is used.
   */
  bool use_subspaces;
  /**
   * Whether to use Vector Quantization (KMeans) before product quantization (PQ).
   * When true, VQ is used before PQ. When false, only product quantization is used.
   */
  bool use_vq;
  /**
   * Vector Quantization (VQ) codebook size - number of "coarse cluster centers".
   * When zero, an optimal value is selected using a heuristic.
   * When one, only product quantization is used.
   */
  uint32_t vq_n_centers;
  /** The number of iterations searching for kmeans centers (both VQ & PQ phases). */
  uint32_t kmeans_n_iters;
  /**
   * The type of kmeans algorithm to use for PQ training.
   */
  cuvsKMeansType pq_kmeans_type;
  /**
   * The max number of data points to use per PQ code during PQ codebook training. Using more data
   * points per PQ code may increase the quality of PQ codebook but may also increase the build
   * time. We will use `pq_n_centers * max_train_points_per_pq_code` training
   * points to train each PQ codebook.
   */
  uint32_t max_train_points_per_pq_code;
  /**
   * The max number of data points to use per VQ cluster.
   */
  uint32_t max_train_points_per_vq_cluster;
};

typedef struct cuvsProductQuantizerParams* cuvsProductQuantizerParams_t;

/**
 * @brief Allocate Product Quantizer params, and populate with default values
 *
 * @param[in] params cuvsProductQuantizerParams_t to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsProductQuantizerParamsCreate(cuvsProductQuantizerParams_t* params);

/**
 * @brief De-allocate Product Quantizer params
 *
 * @param[in] params
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsProductQuantizerParamsDestroy(cuvsProductQuantizerParams_t params);

/**
 * @brief Defines and stores product quantizer upon training
 *
 * The quantization is performed by a linear mapping of an interval in the
 * float data type to the full range of the quantized int type.
 */
typedef struct {
  uintptr_t addr;
  DLDataType dtype;
} cuvsProductQuantizer;

typedef cuvsProductQuantizer* cuvsProductQuantizer_t;

/**
 * @brief Allocate Product Quantizer
 *
 * @param[in] quantizer cuvsProductQuantizer_t to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsProductQuantizerCreate(cuvsProductQuantizer_t* quantizer);

/**
 * @brief De-allocate Product Quantizer
 *
 * @param[in] quantizer
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsProductQuantizerDestroy(cuvsProductQuantizer_t quantizer);

/**
 * @brief Builds a product quantizer to be used later for quantizing the dataset.
 *
 * @param[in] res raft resource
 * @param[in] params Parameters for product quantizer training
 * @param[in] dataset a row-major host or device matrix
 * @param[out] quantizer trained product quantizer
 */
CUVS_EXPORT cuvsError_t cuvsProductQuantizerBuild(cuvsResources_t res,
                                      cuvsProductQuantizerParams_t params,
                                      DLManagedTensor* dataset,
                                      cuvsProductQuantizer_t quantizer);

/**
 * @brief Applies product quantization transform to the given dataset
 *
 * This applies product quantization to a dataset.
 *
 * @param[in] res raft resource
 * @param[in] quantizer product quantizer
 * @param[in] dataset a row-major host or device matrix to transform
 * @param[out] codes_out a row-major device matrix to store transformed data
 * @param[out] vq_labels a device vector to store VQ labels.
 *   Optional, can be NULL.
 */
CUVS_EXPORT cuvsError_t cuvsProductQuantizerTransform(cuvsResources_t res,
                                          cuvsProductQuantizer_t quantizer,
                                          DLManagedTensor* dataset,
                                          DLManagedTensor* codes_out,
                                          DLManagedTensor* vq_labels);

/**
 * @brief Applies product quantization inverse transform to the given quantized codes
 *
 * This applies product quantization inverse transform to the given quantized codes.
 *
 * @param[in] res raft resource
 * @param[in] quantizer product quantizer
 * @param[in] pq_codes a row-major device matrix of quantized codes
 * @param[out] out a row-major device matrix to store the original data
 * @param[out] vq_labels a device vector containing the VQ labels when VQ is used.
 *   Optional, can be NULL.
 */
CUVS_EXPORT cuvsError_t cuvsProductQuantizerInverseTransform(cuvsResources_t res,
  cuvsProductQuantizer_t quantizer,
  DLManagedTensor* pq_codes,
  DLManagedTensor* out,
  DLManagedTensor* vq_labels);

/**
 * @brief Get the bit length of the vector element after compression by PQ.
 *
 * @param[in] quantizer product quantizer
 * @param[out] pq_bits bit length of the vector element after compression by PQ
 */
CUVS_EXPORT cuvsError_t cuvsProductQuantizerGetPqBits(cuvsProductQuantizer_t quantizer, uint32_t* pq_bits);

/**
 * @brief Get the dimensionality of the vector after compression by PQ.
 *
 * @param[in] quantizer product quantizer
 * @param[out] pq_dim dimensionality of the vector after compression by PQ
 */
CUVS_EXPORT cuvsError_t cuvsProductQuantizerGetPqDim(cuvsProductQuantizer_t quantizer, uint32_t* pq_dim);

/**
 * @brief Get the PQ codebook.
 *
 * @param[in] quantizer product quantizer
 * @param[out] pq_codebook PQ codebook
 */
CUVS_EXPORT cuvsError_t cuvsProductQuantizerGetPqCodebook(cuvsProductQuantizer_t quantizer,
                                              DLManagedTensor* pq_codebook);

/**
 * @brief Get the VQ codebook.
 *
 * @param[in] quantizer product quantizer
 * @param[out] vq_codebook VQ codebook
 */
CUVS_EXPORT cuvsError_t cuvsProductQuantizerGetVqCodebook(cuvsProductQuantizer_t quantizer,
                                              DLManagedTensor* vq_codebook);
/**
 * @brief Get the encoded dimension of the quantized dataset.
 *
 * @param[in] quantizer product quantizer
 * @param[out] encoded_dim encoded dimension of the quantized dataset
 */
CUVS_EXPORT cuvsError_t cuvsProductQuantizerGetEncodedDim(cuvsProductQuantizer_t quantizer,
                                              uint32_t* encoded_dim);

/**
 * @brief Get whether VQ is used.
 *
 * @param[in] quantizer product quantizer
 * @param[out] use_vq whether VQ is used
 */
CUVS_EXPORT cuvsError_t cuvsProductQuantizerGetUseVq(cuvsProductQuantizer_t quantizer, bool* use_vq);
/**
 * @}
 */
#ifdef __cplusplus
}
#endif
