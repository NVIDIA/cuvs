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
 * @defgroup preprocessing_c_binary C API for Binary Quantizer
 * @{
 */
/**
 * @brief In the cuvsBinaryQuantizerTransform function, a bit is set if the corresponding element in
 * the dataset vector is greater than the corresponding element in the threshold vector. The mean
 * and sampling_median thresholds are calculated separately for each dimension.
 *
 */
enum cuvsBinaryQuantizerThreshold {
  ZERO=0,
  MEAN=1,
  SAMPLING_MEDIAN=2
};

/**
 * @brief Binary quantizer parameters.
 */
struct cuvsBinaryQuantizerParams {
  /*
   * specifies the threshold to set a bit in cuvsBinaryQuantizerTransform
   */
  enum cuvsBinaryQuantizerThreshold threshold;

  /*
   * specifies the sampling ratio
   */
  float sampling_ratio;
};

typedef struct cuvsBinaryQuantizerParams* cuvsBinaryQuantizerParams_t;

/**
 * @brief Allocate Binary Quantizer params, and populate with default values
 *
 * @param[in] params cuvsBinaryQuantizerParams_t to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsBinaryQuantizerParamsCreate(cuvsBinaryQuantizerParams_t* params);

/**
 * @brief De-allocate Binary Quantizer params
 *
 * @param[in] params
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsBinaryQuantizerParamsDestroy(cuvsBinaryQuantizerParams_t params);

/**
 * @brief Defines and stores threshold for quantization upon training
 *
 * The quantization is performed by a linear mapping of an interval in the
 * float data type to the full range of the quantized int type.
 */
typedef struct {
  uintptr_t addr;
  DLDataType dtype;
} cuvsBinaryQuantizer;

typedef cuvsBinaryQuantizer* cuvsBinaryQuantizer_t;

/**
 * @brief Allocate Binary Quantizer and populate with default values
 *
 * @param[in] quantizer cuvsBinaryQuantizer_t to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsBinaryQuantizerCreate(cuvsBinaryQuantizer_t* quantizer);

/**
 * @brief De-allocate Binary Quantizer
 *
 * @param[in] quantizer
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsBinaryQuantizerDestroy(cuvsBinaryQuantizer_t quantizer);

/**
 * @brief Trains a binary quantizer to be used later for quantizing the dataset.
 *
 * @param[in] res raft resource
 * @param[in] params configure binary quantizer, e.g. threshold
 * @param[in] dataset a row-major host or device matrix
 * @param[out] quantizer trained binary quantizer
 */
CUVS_EXPORT cuvsError_t cuvsBinaryQuantizerTrain(cuvsResources_t res,
                                     cuvsBinaryQuantizerParams_t params,
                                     DLManagedTensor* dataset,
                                     cuvsBinaryQuantizer_t quantizer);

/**
 * @brief Applies binary quantization transform to the given dataset
 *
 * This applies binary quantization to a dataset, changing any positive
 * values to a bitwise 1. This is useful for searching with the
 * BitwiseHamming distance type.
 *
 * @param[in] res raft resource
 * @param[in] dataset a row-major host or device matrix to transform
 * @param[out] out a row-major host or device matrix to store transformed data
 */
CUVS_EXPORT cuvsError_t cuvsBinaryQuantizerTransform(cuvsResources_t res,
                                         DLManagedTensor* dataset,
                                         DLManagedTensor* out);

/**
 * @brief Applies binary quantization transform to the given dataset
 *
 * This applies binary quantization to a dataset, changing any values that are larger than the
 * threshold specified in the param to a bitwise 1. This is useful for searching with the
 * BitwiseHamming distance type.
 *
 * @param[in] res raft resource
 * @param[in] quantizer binary quantizer
 * @param[in] dataset a row-major host or device matrix to transform
 * @param[out] out a row-major host or device matrix to store transformed data
 */
CUVS_EXPORT cuvsError_t cuvsBinaryQuantizerTransformWithParams(cuvsResources_t res,
                                                   cuvsBinaryQuantizer_t quantizer,
                                                   DLManagedTensor* dataset,
                                                   DLManagedTensor* out);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif
