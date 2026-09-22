/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/export.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/pca_types.hpp>

namespace CUVS_EXPORT cuvs {
namespace preprocessing {
namespace pca {

using solver = raft::linalg::solver;

/**
 * @brief Parameters for PCA decomposition. Ref:
 * http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
 */
struct params {
  /** @brief Number of components to keep. */
  int n_components = 1;

  /**
   * @brief If false, data passed to fit are overwritten and running fit(X).transform(X) will
   * not yield the expected results, use fit_transform(X) instead.
   */
  bool copy = true;

  /**
   * @brief When true (false by default) the components vectors are multiplied by the square
   * root of n_samples and then divided by the singular values to ensure uncorrelated outputs with
   * unit component-wise variances.
   */
  bool whiten = false;

  /** @brief The solver algorithm to use. */
  solver algorithm = solver::COV_EIG_DQ;

  /**
   * @brief Tolerance for singular values computed by svd_solver == 'arpack' or
   * the Jacobi solver.
   */
  float tol = 0.0f;

  /**
   * @brief Number of iterations for the power method computed by the Jacobi solver.
   */
  int n_iterations = 15;
};

/**
 * @defgroup pca PCA (Principal Component Analysis)
 * @{
 */

/**
 * @brief Perform PCA fit operation (col-major input).
 *
 * Computes the principal components, explained variances, singular values, and column means
 * from the input data.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/preprocessing/pca.hpp>
 *
 * raft::resources handle;
 *
 * cuvs::preprocessing::pca::params params;
 * params.n_components = 2;
 *
 * auto input = raft::make_device_matrix<float, int>(handle, n_rows, n_cols);
 * // ... fill input ...
 *
 * auto components       = raft::make_device_matrix<float, int, raft::col_major>(
 *     handle, params.n_components, n_cols);
 * auto explained_var    = raft::make_device_vector<float, int>(handle, params.n_components);
 * auto explained_var_ratio = raft::make_device_vector<float, int>(handle, params.n_components);
 * auto singular_vals    = raft::make_device_vector<float, int>(handle, params.n_components);
 * auto mu               = raft::make_device_vector<float, int>(handle, n_cols);
 * auto noise_vars       = raft::make_device_scalar<float>(handle);
 *
 * cuvs::preprocessing::pca::fit(handle, params,
 *     input.view(), components.view(), explained_var.view(),
 *     explained_var_ratio.view(), singular_vals.view(), mu.view(), noise_vars.view());
 * @endcode
 *
 * @param[in] handle raft resource handle
 * @param[in] config PCA parameters
 * @param[inout] input input data [n_rows x n_cols] (col-major). Modified temporarily.
 * @param[out] components principal components [n_components x n_cols] (col-major)
 * @param[out] explained_var explained variances [n_components]
 * @param[out] explained_var_ratio explained variance ratios [n_components]
 * @param[out] singular_vals singular values [n_components]
 * @param[out] mu column means [n_cols]
 * @param[out] noise_vars noise variance (scalar)
 * @param[in] flip_signs_based_on_U whether to determine signs by U (true) or V.T (false)
 */
void fit(raft::resources const& handle,
         const params& config,
         raft::device_matrix_view<float, int64_t, raft::col_major> input,
         raft::device_matrix_view<float, int64_t, raft::col_major> components,
         raft::device_vector_view<float, int64_t> explained_var,
         raft::device_vector_view<float, int64_t> explained_var_ratio,
         raft::device_vector_view<float, int64_t> singular_vals,
         raft::device_vector_view<float, int64_t> mu,
         raft::device_scalar_view<float, int64_t> noise_vars,
         bool flip_signs_based_on_U = false);

/**
 * @brief Perform PCA fit operation (row-major input).
 *
 * Same as the col-major overload, but operates natively on row-major (C-contiguous) data
 * with no internal copy/transpose of the input. The output `components` matrix is also
 * row-major.
 *
 * @param[in] handle raft resource handle
 * @param[in] config PCA parameters
 * @param[inout] input input data [n_rows x n_cols] (row-major). Modified temporarily.
 * @param[out] components principal components [n_components x n_cols] (row-major)
 * @param[out] explained_var explained variances [n_components]
 * @param[out] explained_var_ratio explained variance ratios [n_components]
 * @param[out] singular_vals singular values [n_components]
 * @param[out] mu column means [n_cols]
 * @param[out] noise_vars noise variance (scalar)
 * @param[in] flip_signs_based_on_U whether to determine signs by U (true) or V.T (false)
 */
void fit(raft::resources const& handle,
         const params& config,
         raft::device_matrix_view<float, int64_t, raft::row_major> input,
         raft::device_matrix_view<float, int64_t, raft::row_major> components,
         raft::device_vector_view<float, int64_t> explained_var,
         raft::device_vector_view<float, int64_t> explained_var_ratio,
         raft::device_vector_view<float, int64_t> singular_vals,
         raft::device_vector_view<float, int64_t> mu,
         raft::device_scalar_view<float, int64_t> noise_vars,
         bool flip_signs_based_on_U = false);

/**
 * @brief Perform PCA fit and transform operations (col-major).
 *
 * Computes the principal components and transforms the input data into the eigenspace
 * in a single operation.
 *
 * @param[in] handle raft resource handle
 * @param[in] config PCA parameters
 * @param[inout] input input data [n_rows x n_cols] (col-major). Modified temporarily.
 * @param[out] trans_input transformed data [n_rows x n_components] (col-major)
 * @param[out] components principal components [n_components x n_cols] (col-major)
 * @param[out] explained_var explained variances [n_components]
 * @param[out] explained_var_ratio explained variance ratios [n_components]
 * @param[out] singular_vals singular values [n_components]
 * @param[out] mu column means [n_cols]
 * @param[out] noise_vars noise variance (scalar)
 * @param[in] flip_signs_based_on_U whether to determine signs by U (true) or V.T (false)
 */
void fit_transform(raft::resources const& handle,
                   const params& config,
                   raft::device_matrix_view<float, int64_t, raft::col_major> input,
                   raft::device_matrix_view<float, int64_t, raft::col_major> trans_input,
                   raft::device_matrix_view<float, int64_t, raft::col_major> components,
                   raft::device_vector_view<float, int64_t> explained_var,
                   raft::device_vector_view<float, int64_t> explained_var_ratio,
                   raft::device_vector_view<float, int64_t> singular_vals,
                   raft::device_vector_view<float, int64_t> mu,
                   raft::device_scalar_view<float, int64_t> noise_vars,
                   bool flip_signs_based_on_U = false);

/**
 * @brief Perform PCA fit and transform operations (row-major).
 *
 * Same as the col-major overload but operates natively on row-major data.
 *
 * @param[in] handle raft resource handle
 * @param[in] config PCA parameters
 * @param[inout] input input data [n_rows x n_cols] (row-major). Modified temporarily.
 * @param[out] trans_input transformed data [n_rows x n_components] (row-major)
 * @param[out] components principal components [n_components x n_cols] (row-major)
 * @param[out] explained_var explained variances [n_components]
 * @param[out] explained_var_ratio explained variance ratios [n_components]
 * @param[out] singular_vals singular values [n_components]
 * @param[out] mu column means [n_cols]
 * @param[out] noise_vars noise variance (scalar)
 * @param[in] flip_signs_based_on_U whether to determine signs by U (true) or V.T (false)
 */
void fit_transform(raft::resources const& handle,
                   const params& config,
                   raft::device_matrix_view<float, int64_t, raft::row_major> input,
                   raft::device_matrix_view<float, int64_t, raft::row_major> trans_input,
                   raft::device_matrix_view<float, int64_t, raft::row_major> components,
                   raft::device_vector_view<float, int64_t> explained_var,
                   raft::device_vector_view<float, int64_t> explained_var_ratio,
                   raft::device_vector_view<float, int64_t> singular_vals,
                   raft::device_vector_view<float, int64_t> mu,
                   raft::device_scalar_view<float, int64_t> noise_vars,
                   bool flip_signs_based_on_U = false);

/**
 * @brief Perform PCA transform operation (col-major).
 *
 * Transforms the input data into the eigenspace using previously computed principal components.
 *
 * @param[in] handle raft resource handle
 * @param[in] config PCA parameters
 * @param[inout] input data to transform [n_rows x n_cols] (col-major). Modified temporarily
 * (mean-centered then restored).
 * @param[in] components principal components [n_components x n_cols] (col-major)
 * @param[in] singular_vals singular values [n_components]
 * @param[in] mu column means [n_cols]
 * @param[out] trans_input transformed data [n_rows x n_components] (col-major)
 */
void transform(raft::resources const& handle,
               const params& config,
               raft::device_matrix_view<float, int64_t, raft::col_major> input,
               raft::device_matrix_view<float, int64_t, raft::col_major> components,
               raft::device_vector_view<float, int64_t> singular_vals,
               raft::device_vector_view<float, int64_t> mu,
               raft::device_matrix_view<float, int64_t, raft::col_major> trans_input);

/**
 * @brief Perform PCA transform operation (row-major).
 *
 * @param[in] handle raft resource handle
 * @param[in] config PCA parameters
 * @param[inout] input data to transform [n_rows x n_cols] (row-major). Modified temporarily
 * (mean-centered then restored).
 * @param[in] components principal components [n_components x n_cols] (row-major)
 * @param[in] singular_vals singular values [n_components]
 * @param[in] mu column means [n_cols]
 * @param[out] trans_input transformed data [n_rows x n_components] (row-major)
 */
void transform(raft::resources const& handle,
               const params& config,
               raft::device_matrix_view<float, int64_t, raft::row_major> input,
               raft::device_matrix_view<float, int64_t, raft::row_major> components,
               raft::device_vector_view<float, int64_t> singular_vals,
               raft::device_vector_view<float, int64_t> mu,
               raft::device_matrix_view<float, int64_t, raft::row_major> trans_input);

/**
 * @brief Perform PCA inverse transform operation (col-major).
 *
 * Transforms data from the eigenspace back to the original space.
 *
 * @param[in] handle raft resource handle
 * @param[in] config PCA parameters
 * @param[in] trans_input transformed data [n_rows x n_components] (col-major)
 * @param[in] components principal components [n_components x n_cols] (col-major)
 * @param[in] singular_vals singular values [n_components]
 * @param[in] mu column means [n_cols]
 * @param[out] output reconstructed data [n_rows x n_cols] (col-major)
 */
void inverse_transform(raft::resources const& handle,
                       const params& config,
                       raft::device_matrix_view<float, int64_t, raft::col_major> trans_input,
                       raft::device_matrix_view<float, int64_t, raft::col_major> components,
                       raft::device_vector_view<float, int64_t> singular_vals,
                       raft::device_vector_view<float, int64_t> mu,
                       raft::device_matrix_view<float, int64_t, raft::col_major> output);

/**
 * @brief Perform PCA inverse transform operation (row-major).
 *
 * @param[in] handle raft resource handle
 * @param[in] config PCA parameters
 * @param[in] trans_input transformed data [n_rows x n_components] (row-major)
 * @param[in] components principal components [n_components x n_cols] (row-major)
 * @param[in] singular_vals singular values [n_components]
 * @param[in] mu column means [n_cols]
 * @param[out] output reconstructed data [n_rows x n_cols] (row-major)
 */
void inverse_transform(raft::resources const& handle,
                       const params& config,
                       raft::device_matrix_view<float, int64_t, raft::row_major> trans_input,
                       raft::device_matrix_view<float, int64_t, raft::row_major> components,
                       raft::device_vector_view<float, int64_t> singular_vals,
                       raft::device_vector_view<float, int64_t> mu,
                       raft::device_matrix_view<float, int64_t, raft::row_major> output);

/** @} */  // end group pca

}  // namespace pca
}  // namespace preprocessing
}  // namespace CUVS_EXPORT cuvs
