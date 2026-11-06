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
/** Select global or per-dimension scalar quantization bounds. */
enum cuvsScalarQuantizationMode {
  CUVS_SCALAR_GLOBAL = 0,
  CUVS_SCALAR_PER_DIMENSION = 1
};

/**
 * @brief Scalar quantizer parameters. Allocate with cuvsScalarQuantizerParamsCreate.
 */
struct cuvsScalarQuantizerParams {
  /**
   * specifies how many outliers at top & bottom will be ignored
   * needs to be within range of (0, 1]
   */
  float quantile;

  /** Quantization mode. Defaults to CUVS_SCALAR_GLOBAL. */
  enum cuvsScalarQuantizationMode mode;
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
 * Allocate with cuvsScalarQuantizerCreate and release with cuvsScalarQuantizerDestroy;
 * do not copy the handle's contents. The implementation also owns any per-dimension bounds.
 * In global mode, min_ and max_ are the trained bounds. In per-dimension mode, they are
 * numeric_limits<T>::min() and numeric_limits<T>::max() for the training data type T,
 * and are not used for transformation.
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
 * @brief De-allocate Scalar Quantizer after outstanding stream work completes.
 *
 * @param[in] quantizer
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsScalarQuantizerDestroy(cuvsScalarQuantizer_t quantizer);

/**
 * @brief Trains a scalar quantizer to be used later for quantizing the dataset.
 *
 * Supports float16, float32, and float64 in global or per-dimension mode.
 * Per-dimension mode applies the quantile independently to each dimension using every row.
 * Successful retraining replaces the previous bounds; failed training preserves them.
 * Complete outstanding stream work using the quantizer before retraining.
 *
 * @param[in] res raft resource
 * @param[in] params configure scalar quantizer, e.g. quantile
 * @param[in] dataset a row-major host or device matrix
 * @param[in,out] quantizer scalar quantizer allocated by cuvsScalarQuantizerCreate
 */
CUVS_EXPORT cuvsError_t cuvsScalarQuantizerTrain(cuvsResources_t res,
                                     cuvsScalarQuantizerParams_t params,
                                     DLManagedTensor* dataset,
                                     cuvsScalarQuantizer_t quantizer);

/**
 * @brief Applies quantization transform to given dataset
 *
 * Per-dimension mode requires device matrices with the same input type and width as training.
 * Its output codes are int8 in [-127, 127]. The quantizer must outlive stream work using its bounds.
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
 * not lossless. Per-dimension mode requires device matrices with the same output type and
 * width as training. The quantizer must outlive stream work using its bounds.
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
