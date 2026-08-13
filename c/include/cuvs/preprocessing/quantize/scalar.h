/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
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
 * @}
 */

#ifdef __cplusplus
}
#endif
