/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/c_api.h>
#include <dlpack/dlpack.h>
#include <stdbool.h>
#include <stdint.h>

#include <cuvs/core/export.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup preprocessing_c_pca C API for PCA (Principal Component Analysis)
 * @{
 */

/**
 * @brief Solver algorithm for PCA eigen decomposition.
 */
enum cuvsPcaSolver {
  /** Covariance + divide-and-conquer eigen decomposition */
  CUVS_PCA_COV_EIG_DQ = 0,
  /** Covariance + Jacobi eigen decomposition */
  CUVS_PCA_COV_EIG_JACOBI = 1
};

/**
 * @brief Parameters for PCA decomposition.
 */
struct cuvsPcaParams {
  /** Number of principal components to keep. */
  int n_components;

  /**
   * If false, data passed to fit are overwritten and running fit(X).transform(X) will
   * not yield the expected results; use fit_transform(X) instead.
   */
  bool copy;

  /**
   * When true the component vectors are multiplied by the square root of n_samples and then
   * divided by the singular values to ensure uncorrelated outputs with unit component-wise
   * variances.
   */
  bool whiten;

  /** Solver algorithm to use. */
  enum cuvsPcaSolver algorithm;

  /** Tolerance for singular values (used by Jacobi solver). */
  float tol;

  /** Number of iterations for the power method (Jacobi solver). */
  int n_iterations;
};

typedef struct cuvsPcaParams* cuvsPcaParams_t;

/**
 * @brief Allocate PCA params and populate with default values.
 *
 * @param[out] params cuvsPcaParams_t to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsPcaParamsCreate(cuvsPcaParams_t* params);

/**
 * @brief De-allocate PCA params.
 *
 * @param[in] params cuvsPcaParams_t to de-allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsPcaParamsDestroy(cuvsPcaParams_t params);

/**
 * @brief Perform PCA fit operation.
 *
 * Computes the principal components, explained variances, singular values, and column means
 * from the input data.
 *
 * The layout of `input` (C-contiguous / row-major or F-contiguous / col-major) is detected
 * from its DLPack strides; `components` must use the same layout as `input`.
 *
 * @code {.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/preprocessing/pca.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsResourcesCreate(&res);
 *
 * // Create PCA params
 * cuvsPcaParams_t params;
 * cuvsPcaParamsCreate(&params);
 * params->n_components = 2;
 *
 * // Assume populated DLManagedTensor objects (float32, device memory)
 * DLManagedTensor input;          // [n_rows x n_cols] (C- or F-contiguous)
 * DLManagedTensor components;     // [n_components x n_cols] (same layout as input)
 * DLManagedTensor explained_var;  // [n_components]
 * DLManagedTensor explained_var_ratio; // [n_components]
 * DLManagedTensor singular_vals;  // [n_components]
 * DLManagedTensor mu;             // [n_cols]
 * DLManagedTensor noise_vars;     // [1] (scalar)
 *
 * cuvsPcaFit(res, params, &input, &components, &explained_var,
 *            &explained_var_ratio, &singular_vals, &mu, &noise_vars, false);
 *
 * // Cleanup
 * cuvsPcaParamsDestroy(params);
 * cuvsResourcesDestroy(res);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] params PCA parameters
 * @param[inout] input input data [n_rows x n_cols] (C- or F-contiguous, float32, device)
 * @param[out] components principal components [n_components x n_cols] (same layout as input)
 * @param[out] explained_var explained variances [n_components] (float32, device)
 * @param[out] explained_var_ratio explained variance ratios [n_components] (float32, device)
 * @param[out] singular_vals singular values [n_components] (float32, device)
 * @param[out] mu column means [n_cols] (float32, device)
 * @param[out] noise_vars noise variance [1] (float32, device)
 * @param[in] flip_signs_based_on_U whether to determine signs by U (true) or V.T (false)
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsPcaFit(cuvsResources_t res,
                       cuvsPcaParams_t params,
                       DLManagedTensor* input,
                       DLManagedTensor* components,
                       DLManagedTensor* explained_var,
                       DLManagedTensor* explained_var_ratio,
                       DLManagedTensor* singular_vals,
                       DLManagedTensor* mu,
                       DLManagedTensor* noise_vars,
                       bool flip_signs_based_on_U);

/**
 * @brief Perform PCA fit and transform in a single operation.
 *
 * Computes the principal components and transforms the input data into the eigenspace.
 * The layout of `input` (C- or F-contiguous) is detected from its DLPack strides; all
 * other matrix tensors must use the same layout.
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] params PCA parameters
 * @param[inout] input input data [n_rows x n_cols] (C- or F-contiguous, float32, device)
 * @param[out] trans_input transformed data [n_rows x n_components] (same layout as input)
 * @param[out] components principal components [n_components x n_cols] (same layout as input)
 * @param[out] explained_var explained variances [n_components] (float32, device)
 * @param[out] explained_var_ratio explained variance ratios [n_components] (float32, device)
 * @param[out] singular_vals singular values [n_components] (float32, device)
 * @param[out] mu column means [n_cols] (float32, device)
 * @param[out] noise_vars noise variance [1] (float32, device)
 * @param[in] flip_signs_based_on_U whether to determine signs by U (true) or V.T (false)
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsPcaFitTransform(cuvsResources_t res,
                                cuvsPcaParams_t params,
                                DLManagedTensor* input,
                                DLManagedTensor* trans_input,
                                DLManagedTensor* components,
                                DLManagedTensor* explained_var,
                                DLManagedTensor* explained_var_ratio,
                                DLManagedTensor* singular_vals,
                                DLManagedTensor* mu,
                                DLManagedTensor* noise_vars,
                                bool flip_signs_based_on_U);

/**
 * @brief Perform PCA transform operation.
 *
 * Transforms the input data into the eigenspace using previously computed principal components.
 * The layout of `input` (C- or F-contiguous) is detected from its DLPack strides; all other
 * matrix tensors must use the same layout.
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] params PCA parameters
 * @param[inout] input data to transform [n_rows x n_cols] (C- or F-contiguous, float32, device)
 * @param[in] components principal components [n_components x n_cols] (same layout as input)
 * @param[in] singular_vals singular values [n_components] (float32, device)
 * @param[in] mu column means [n_cols] (float32, device)
 * @param[out] trans_input transformed data [n_rows x n_components] (same layout as input)
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsPcaTransform(cuvsResources_t res,
                             cuvsPcaParams_t params,
                             DLManagedTensor* input,
                             DLManagedTensor* components,
                             DLManagedTensor* singular_vals,
                             DLManagedTensor* mu,
                             DLManagedTensor* trans_input);

/**
 * @brief Perform PCA inverse transform operation.
 *
 * Transforms data from the eigenspace back to the original space.
 * The layout of `trans_input` (C- or F-contiguous) is detected from its DLPack strides;
 * all other matrix tensors must use the same layout.
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] params PCA parameters
 * @param[in] trans_input transformed data [n_rows x n_components] (C- or F-contiguous,
 * float32, device)
 * @param[in] components principal components [n_components x n_cols] (same layout as trans_input)
 * @param[in] singular_vals singular values [n_components] (float32, device)
 * @param[in] mu column means [n_cols] (float32, device)
 * @param[out] output reconstructed data [n_rows x n_cols] (same layout as trans_input)
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsPcaInverseTransform(cuvsResources_t res,
                                    cuvsPcaParams_t params,
                                    DLManagedTensor* trans_input,
                                    DLManagedTensor* components,
                                    DLManagedTensor* singular_vals,
                                    DLManagedTensor* mu,
                                    DLManagedTensor* output);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif
