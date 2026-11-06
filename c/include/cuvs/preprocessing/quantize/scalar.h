/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @defgroup preprocessing_c_scalar C API for Scalar Quantizer
 * @{
 */
/**
 * @brief Scalar quantizer parameters.
 */
struct cuvsScalarQuantizerParams {
  /*
   * specifies how many outliers at top & bottom will be ignored
   * needs to be within range of (0, 1]
   */
  float quantile;
};

typedef struct cuvsScalarQuantizerParams* cuvsScalarQuantizerParams_t;

/**
 * @brief Allocate Scalar Quantizer params, and populate with default values
 *
 * @param[in] params cuvsScalarQuantizerParams_t to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsScalarQuantizerParamsCreate(cuvsScalarQuantizerParams_t* params);

/**
 * @brief De-allocate Scalar Quantizer params
 *
 * @param[in] params
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsScalarQuantizerParamsDestroy(cuvsScalarQuantizerParams_t params);

/**
 * @brief Defines and stores scalar for quantisation upon training
 *
 * The quantization is performed by a linear mapping of an interval in the
 * float data type to the full range of the quantized int type.
 */
typedef struct {
  double min_;
  double max_;
} cuvsScalarQuantizer;

typedef cuvsScalarQuantizer* cuvsScalarQuantizer_t;

/**
 * @brief Allocate Scalar Quantizer and populate with default values
 *
 * @param[in] quantizer cuvsScalarQuantizer_t to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsScalarQuantizerCreate(cuvsScalarQuantizer_t* quantizer);

/**
 * @brief De-allocate Scalar Quantizer
 *
 * @param[in] quantizer
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsScalarQuantizerDestroy(cuvsScalarQuantizer_t quantizer);

/**
 * @brief Trains a scalar quantizer to be used later for quantizing the dataset.
 *
 * @param[in] res raft resource
 * @param[in] params configure scalar quantizer, e.g. quantile
 * @param[in] dataset a row-major host or device matrix
 * @param[out] quantizer trained scalar quantizer
 */
CUVS_EXPORT cuvsError_t cuvsScalarQuantizerTrain(cuvsResources_t res,
                                     cuvsScalarQuantizerParams_t params,
                                     DLManagedTensor* dataset,
                                     cuvsScalarQuantizer_t quantizer);

/**
 * @brief Applies quantization transform to given dataset
 *
 * @param[in] res raft resource
 * @param[in] quantizer a scalar quantizer
 * @param[in] dataset a row-major host or device matrix to transform
 * @param[out] out a row-major host or device matrix to store transformed data
 */
CUVS_EXPORT cuvsError_t cuvsScalarQuantizerTransform(cuvsResources_t res,
                                         cuvsScalarQuantizer_t quantizer,
                                         DLManagedTensor* dataset,
                                         DLManagedTensor* out);

/**
 * @brief Perform inverse quantization step on previously quantized dataset
 *
 * Note that depending on the chosen data types train dataset the conversion is
 * not lossless.
 *
 * @param[in] res raft resource
 * @param[in] quantizer a scalar quantizer
 * @param[in] dataset a row-major host or device matrix
 * @param[out] out a row-major host or device matrix
 *
 */
CUVS_EXPORT cuvsError_t cuvsScalarQuantizerInverseTransform(cuvsResources_t res,
                                                cuvsScalarQuantizer_t quantizer,
                                                DLManagedTensor* dataset,
                                                DLManagedTensor* out);

/**
 * @brief Owns a scalar quantizer with separate min/max bounds for each dimension.
 * The fields are managed by the lifecycle and training functions.
 */
typedef struct {
  uintptr_t addr;
  DLDataType dtype;
} cuvsScalarQuantizerPerDim;

typedef cuvsScalarQuantizerPerDim* cuvsScalarQuantizerPerDim_t;

/**
 * @brief Allocate an untrained per-dimension scalar quantizer.
 * @param[out] quantizer Handle to allocate.
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsScalarQuantizerPerDimCreate(cuvsScalarQuantizerPerDim_t* quantizer);

/**
 * @brief Release the quantizer and its device bounds after outstanding stream work completes.
 * @param[in] quantizer Handle to destroy.
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsScalarQuantizerPerDimDestroy(cuvsScalarQuantizerPerDim_t quantizer);

/**
 * @brief Train per-dimension bounds from every row of a host or device matrix.
 * Supports float16, float32, and float64. The quantile is applied independently per dimension.
 * Successful retraining replaces the previous bounds; failed training preserves them.
 * Complete outstanding stream work using the quantizer before retraining.
 *
 * @param[in] res Resources used for training and allocation.
 * @param[in] params Scalar quantizer parameters.
 * @param[in] dataset Row-major host or device matrix.
 * @param[in,out] quantizer Handle created by cuvsScalarQuantizerPerDimCreate.
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsScalarQuantizerPerDimTrain(cuvsResources_t res,
                                                     cuvsScalarQuantizerParams_t params,
                                                     DLManagedTensor* dataset,
                                                     cuvsScalarQuantizerPerDim_t quantizer);

/**
 * @brief Quantize a device matrix using previously trained per-dimension bounds.
 * The input type and width must match training. Output codes are int8, in [-127, 127].
 * The quantizer must outlive stream work using its bounds.
 *
 * @param[in] res Resources used for execution.
 * @param[in] quantizer Trained per-dimension scalar quantizer.
 * @param[in] dataset Row-major device matrix to encode.
 * @param[out] out Row-major device int8 matrix with the same shape.
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsScalarQuantizerPerDimTransform(cuvsResources_t res,
                                                         cuvsScalarQuantizerPerDim_t quantizer,
                                                         DLManagedTensor* dataset,
                                                         DLManagedTensor* out);

/**
 * @brief Reconstruct a device matrix from per-dimension scalar codes.
 * The output type and width must match training.
 * The quantizer must outlive stream work using its bounds.
 *
 * @param[in] res Resources used for execution.
 * @param[in] quantizer Trained per-dimension scalar quantizer.
 * @param[in] dataset Row-major device int8 matrix to decode.
 * @param[out] out Row-major device matrix with the same shape, in the training data type.
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsScalarQuantizerPerDimInverseTransform(
  cuvsResources_t res,
  cuvsScalarQuantizerPerDim_t quantizer,
  DLManagedTensor* dataset,
  DLManagedTensor* out);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif
