/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <cuvs/core/cuda_fp16.hpp>
#include <cuvs/core/export.hpp>
#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

namespace CUVS_EXPORT cuvs {
namespace distance {

/** enum to tell how to compute distance */
enum class DistanceType : int {

  /** evaluate as dist_ij = sum(x_ik^2) + sum(y_ij)^2 - 2*sum(x_ik * y_jk) */
  L2Expanded = 0,
  /** same as above, but inside the epilogue, perform square root operation */
  L2SqrtExpanded = 1,
  /** cosine distance */
  CosineExpanded = 2,
  /** L1 distance */
  L1 = 3,
  /** evaluate as dist_ij += (x_ik - y-jk)^2 */
  L2Unexpanded = 4,
  /** same as above, but inside the epilogue, perform square root operation */
  L2SqrtUnexpanded = 5,
  /** basic inner product **/
  InnerProduct = 6,
  /** Chebyshev (Linf) distance **/
  Linf = 7,
  /** Canberra distance **/
  Canberra = 8,
  /** Generalized Minkowski distance **/
  LpUnexpanded = 9,
  /** Correlation distance **/
  CorrelationExpanded = 10,
  /** Jaccard distance **/
  JaccardExpanded = 11,
  /** Hellinger distance **/
  HellingerExpanded = 12,
  /** Haversine distance **/
  Haversine = 13,
  /** Bray-Curtis distance **/
  BrayCurtis = 14,
  /** Jensen-Shannon distance**/
  JensenShannon = 15,
  /** Hamming distance **/
  HammingUnexpanded = 16,
  /** KLDivergence **/
  KLDivergence = 17,
  /** RusselRao **/
  RusselRaoExpanded = 18,
  /** Dice-Sorensen distance **/
  DiceExpanded = 19,
  /** Bitstring Hamming distance **/
  BitwiseHamming = 20,
  /** Precomputed (special value) **/
  Precomputed = 100,
  /** Custom metric UDF **/
  CustomUDF = 101
};

/**
 * Whether minimal distance corresponds to similar elements (using the given metric).
 */
inline bool is_min_close(DistanceType metric)
{
  bool select_min;
  switch (metric) {
    case DistanceType::InnerProduct:
      // Similarity metrics have the opposite meaning, i.e. nearest neighbors are those with larger
      // similarity (See the same logic at cpp/include/raft/sparse/spatial/detail/knn.cuh:362
      // {perform_k_selection})
      select_min = false;
      break;
    default: select_min = true;
  }
  return select_min;
}

/**
 * @brief Density kernel type for Kernel Density Estimation.
 *
 * These are the smoothing kernels used in KDE — distinct from the dot-product
 * kernels (RBF, Polynomial, etc.) in cuvs::distance::kernels used by SVMs.
 */
enum class DensityKernelType : int {
  Gaussian     = 0,
  Tophat       = 1,
  Epanechnikov = 2,
  Exponential  = 3,
  Linear       = 4,
  Cosine       = 5
};

namespace kernels {
enum KernelType { LINEAR, POLYNOMIAL, RBF, TANH };

/**
 * Parameters for kernel matrices.
 * The following kernels are implemented:
 * - LINEAR \f[ K(x_1,x_2) = <x_1,x_2>, \f] where \f$< , >\f$ is the dot product
 * - POLYNOMIAL \f[ K(x_1, x_2) = (\gamma <x_1,x_2> + \mathrm{coef0})^\mathrm{degree} \f]
 * - RBF \f[ K(x_1, x_2) = \exp(- \gamma |x_1-x_2|^2) \f]
 * - TANH \f[ K(x_1, x_2) = \tanh(\gamma <x_1,x_2> + \mathrm{coef0}) \f]
 */
struct KernelParams {
  // Kernel function parameters
  KernelType kernel;  //!< Type of the kernel function
  int degree;         //!< Degree of polynomial kernel (ignored by others)
  double gamma;       //!< multiplier in the
  double coef0;       //!< additive constant in poly and tanh kernels
};
}  // end namespace kernels

/**
 * @defgroup pairwise_distance Pairwise Distances API
 * @{
 */

/**
 * @brief Compute pairwise distances for two matrices
 *
 * Note: Only contiguous row- or column-major layouts supported currently.
 *
 * Usage example:
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <raft/core/device_mdarray.hpp>
 * #include <cuvs/distance/distance.hpp>
 *
 * raft::resources handle;
 * int n_samples = 5000;
 * int n_features = 50;
 *
 * auto input = raft::make_device_matrix<float>(handle, n_samples, n_features);
 *
 * // ... fill input with data ...
 *
 * auto output = raft::make_device_matrix<float>(handle, n_samples, n_samples);
 *
 * auto metric = cuvs::distance::DistanceType::L2SqrtExpanded;
 * cuvs::distance::pairwise_distance(handle,
 *                                   raft::make_const(input.view()),
 *                                   raft::make_const(input.view()),
 *                                   output.view(),
 *                                   metric);
 * @endcode
 *
 * @param[in] handle raft handle for managing expensive resources
 * @param[in] x first set of points (size n*k)
 * @param[in] y second set of points (size m*k)
 * @param[out] dist output distance matrix (size n*m)
 * @param[in] metric distance to evaluate
 * @param[in] metric_arg metric argument (used for Minkowski distance)
 */
void pairwise_distance(
  raft::resources const& handle,
  raft::device_matrix_view<const float, std::int64_t, raft::layout_c_contiguous> const x,
  raft::device_matrix_view<const float, std::int64_t, raft::layout_c_contiguous> const y,
  raft::device_matrix_view<float, std::int64_t, raft::layout_c_contiguous> dist,
  cuvs::distance::DistanceType metric,
  float metric_arg = 2.0f);

/**
 * @brief Compute pairwise distances for two matrices
 *
 * Note: Only contiguous row- or column-major layouts supported currently.
 *
 * Usage example:
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <raft/core/device_mdarray.hpp>
 * #include <cuvs/distance/distance.hpp>
 *
 * raft::resources handle;
 * int n_samples = 5000;
 * int n_features = 50;
 *
 * auto input = raft::make_device_matrix<double>(handle, n_samples, n_features);
 *
 * // ... fill input with data ...
 *
 * auto output = raft::make_device_matrix<double>(handle, n_samples, n_samples);
 *
 * auto metric = cuvs::distance::DistanceType::L2SqrtExpanded;
 * cuvs::distance::pairwise_distance(handle,
 *                                   raft::make_const(input.view()),
 *                                   raft::make_const(input.view()),
 *                                   output.view(),
 *                                   metric);
 * @endcode
 *
 * @param[in] handle raft handle for managing expensive resources
 * @param[in] x first set of points (size n*k)
 * @param[in] y second set of points (size m*k)
 * @param[out] dist output distance matrix (size n*m)
 * @param[in] metric distance to evaluate
 * @param[in] metric_arg metric argument (used for Minkowski distance)
 */
void pairwise_distance(
  raft::resources const& handle,
  raft::device_matrix_view<const double, std::int64_t, raft::layout_c_contiguous> const x,
  raft::device_matrix_view<const double, std::int64_t, raft::layout_c_contiguous> const y,
  raft::device_matrix_view<double, std::int64_t, raft::layout_c_contiguous> dist,
  cuvs::distance::DistanceType metric,
  double metric_arg = 2.0f);
/**
 * @brief Compute pairwise distances for two matrices
 *
 * Note: Only contiguous row- or column-major layouts supported currently.
 *
 * Usage example:
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <raft/core/device_mdarray.hpp>
 * #include <cuvs/distance/distance.hpp>
 *
 * raft::resources handle;
 * int n_samples = 5000;
 * int n_features = 50;
 *
 * auto input = raft::make_device_matrix<float>(handle, n_samples, n_features);
 *
 * // ... fill input with data ...
 *
 * auto output = raft::make_device_matrix<float>(handle, n_samples, n_samples);
 *
 * auto metric = cuvs::distance::DistanceType::L2SqrtExpanded;
 * cuvs::distance::pairwise_distance(handle,
 *                                   raft::make_const(input.view()),
 *                                   raft::make_const(input.view()),
 *                                   output.view(),
 *                                   metric);
 * @endcode
 *
 * @param[in] handle raft handle for managing expensive resources
 * @param[in] x first set of points (size n*k)
 * @param[in] y second set of points (size m*k)
 * @param[out] dist output distance matrix (size n*m)
 * @param[in] metric distance to evaluate
 * @param[in] metric_arg metric argument (used for Minkowski distance)
 */
void pairwise_distance(
  raft::resources const& handle,
  raft::device_matrix_view<const half, std::int64_t, raft::layout_c_contiguous> const x,
  raft::device_matrix_view<const half, std::int64_t, raft::layout_c_contiguous> const y,
  raft::device_matrix_view<float, std::int64_t, raft::layout_c_contiguous> dist,
  cuvs::distance::DistanceType metric,
  float metric_arg = 2.0f);

/**
 * @brief Compute pairwise distances for two matrices
 *
 * Note: Only contiguous row- or column-major layouts supported currently.
 *
 * Usage example:
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <raft/core/device_mdarray.hpp>
 * #include <cuvs/distance/distance.hpp>
 *
 * raft::resources handle;
 * int n_samples = 5000;
 * int n_features = 50;
 *
 * auto input = raft::make_device_matrix<float>(handle, n_samples, n_features);
 *
 * // ... fill input with data ...
 *
 * auto output = raft::make_device_matrix<float>(handle, n_samples, n_samples);
 *
 * auto metric = cuvs::distance::DistanceType::L2SqrtExpanded;
 * cuvs::distance::pairwise_distance(handle,
 *                                   raft::make_const(input.view()),
 *                                   raft::make_const(input.view()),
 *                                   output.view(),
 *                                   metric);
 * @endcode
 *
 * @param[in] handle raft handle for managing expensive resources
 * @param[in] x first set of points (size n*k)
 * @param[in] y second set of points (size m*k)
 * @param[out] dist output distance matrix (size n*m)
 * @param[in] metric distance to evaluate
 * @param[in] metric_arg metric argument (used for Minkowski distance)
 */
void pairwise_distance(
  raft::resources const& handle,
  raft::device_matrix_view<const float, std::int64_t, raft::layout_f_contiguous> const x,
  raft::device_matrix_view<const float, std::int64_t, raft::layout_f_contiguous> const y,
  raft::device_matrix_view<float, std::int64_t, raft::layout_f_contiguous> dist,
  cuvs::distance::DistanceType metric,
  float metric_arg = 2.0f);
/**
 * @brief Compute pairwise distances for two matrices
 *
 * Note: Only contiguous row- or column-major layouts supported currently.
 *
 * Usage example:
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <raft/core/device_mdarray.hpp>
 * #include <cuvs/distance/distance.hpp>
 *
 * raft::resources handle;
 * int n_samples = 5000;
 * int n_features = 50;
 *
 * auto input = raft::make_device_matrix<double>(handle, n_samples, n_features);
 *
 * // ... fill input with data ...
 *
 * auto output = raft::make_device_matrix<double>(handle, n_samples, n_samples);
 *
 * auto metric = cuvs::distance::DistanceType::L2SqrtExpanded;
 * cuvs::distance::pairwise_distance(handle,
 *                                   raft::make_const(input.view()),
 *                                   raft::make_const(input.view()),
 *                                   output.view(),
 *                                   metric);
 * @endcode
 *
 * @param[in] handle raft handle for managing expensive resources
 * @param[in] x first set of points (size n*k)
 * @param[in] y second set of points (size m*k)
 * @param[out] dist output distance matrix (size n*m)
 * @param[in] metric distance to evaluate
 * @param[in] metric_arg metric argument (used for Minkowski distance)
 */
void pairwise_distance(
  raft::resources const& handle,
  raft::device_matrix_view<const double, std::int64_t, raft::layout_f_contiguous> const x,
  raft::device_matrix_view<const double, std::int64_t, raft::layout_f_contiguous> const y,
  raft::device_matrix_view<double, std::int64_t, raft::layout_f_contiguous> dist,
  cuvs::distance::DistanceType metric,
  double metric_arg = 2.0f);
/**
 * @brief Compute pairwise distances for two matrices
 *
 * Note: Only contiguous row- or column-major layouts supported currently.
 *
 * Usage example:
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <raft/core/device_mdarray.hpp>
 * #include <cuvs/distance/distance.hpp>
 *
 * raft::resources handle;
 * int n_samples = 5000;
 * int n_features = 50;
 *
 * auto input = raft::make_device_matrix<float>(handle, n_samples, n_features);
 *
 * // ... fill input with data ...
 *
 * auto output = raft::make_device_matrix<float>(handle, n_samples, n_samples);
 *
 * auto metric = cuvs::distance::DistanceType::L2SqrtExpanded;
 * cuvs::distance::pairwise_distance(handle,
 *                                   raft::make_const(input.view()),
 *                                   raft::make_const(input.view()),
 *                                   output.view(),
 *                                   metric);
 * @endcode
 *
 * @param[in] handle raft handle for managing expensive resources
 * @param[in] x first set of points (size n*k)
 * @param[in] y second set of points (size m*k)
 * @param[out] dist output distance matrix (size n*m)
 * @param[in] metric distance to evaluate
 * @param[in] metric_arg metric argument (used for Minkowski distance)
 */
void pairwise_distance(
  raft::resources const& handle,
  raft::device_matrix_view<const half, std::int64_t, raft::layout_f_contiguous> const x,
  raft::device_matrix_view<const half, std::int64_t, raft::layout_f_contiguous> const y,
  raft::device_matrix_view<float, std::int64_t, raft::layout_f_contiguous> dist,
  cuvs::distance::DistanceType metric,
  float metric_arg = 2.0f);

/**
 * @brief Compute sparse pairwise distances between x and y, using the provided
 * input configuration and distance function.
 *
 * @code{.cpp}
 * #include <raft/core/device_resources.hpp>
 * #include <raft/core/device_csr_matrix.hpp>
 * #include <raft/core/device_mdspan.hpp>
 *
 * int x_n_rows = 100000;
 * int y_n_rows = 50000;
 * int n_cols = 10000;
 *
 * raft::device_resources handle;
 * auto x = raft::make_device_csr_matrix<float>(handle, x_n_rows, n_cols);
 * auto y = raft::make_device_csr_matrix<float>(handle, y_n_rows, n_cols);
 *
 * ...
 * // populate data
 * ...
 *
 * auto out = raft::make_device_matrix<float>(handle, x_nrows, y_nrows);
 * auto metric = cuvs::distance::DistanceType::L2Expanded;
 * cuvs::pairwise_distance(handle, x.view(), y.view(), out, metric);
 * @endcode
 *
 * @param[in] handle raft::resources
 * @param[in] x raft::device_csr_matrix_view
 * @param[in] y raft::device_csr_matrix_view
 * @param[out] dist raft::device_matrix_view dense matrix
 * @param[in] metric distance metric to use
 * @param[in] metric_arg metric argument (used for Minkowski distance)
 */
void pairwise_distance(raft::resources const& handle,
                       raft::device_csr_matrix_view<const float, int, int, int> x,
                       raft::device_csr_matrix_view<const float, int, int, int> y,
                       raft::device_matrix_view<float, int, raft::row_major> dist,
                       cuvs::distance::DistanceType metric,
                       float metric_arg = 2.0f);

/**
 * @brief Compute sparse pairwise distances between x and y, using the provided
 * input configuration and distance function.
 *
 * @code{.cpp}
 * #include <raft/core/device_resources.hpp>
 * #include <raft/core/device_csr_matrix.hpp>
 * #include <raft/core/device_mdspan.hpp>
 *
 * int x_n_rows = 100000;
 * int y_n_rows = 50000;
 * int n_cols = 10000;
 *
 * raft::device_resources handle;
 * auto x = raft::make_device_csr_matrix<double>(handle, x_n_rows, n_cols);
 * auto y = raft::make_device_csr_matrix<double>(handle, y_n_rows, n_cols);
 *
 * ...
 * // populate data
 * ...
 *
 * auto out = raft::make_device_matrix<double>(handle, x_nrows, y_nrows);
 * auto metric = cuvs::distance::DistanceType::L2Expanded;
 * cuvs::distance::pairwise_distance(handle, x.view(), y.view(), out, metric);
 * @endcode
 *
 * @param[in] handle raft::resources
 * @param[in] x raft::device_csr_matrix_view
 * @param[in] y raft::device_csr_matrix_view
 * @param[out] dist raft::device_matrix_view dense matrix
 * @param[in] metric distance metric to use
 * @param[in] metric_arg metric argument (used for Minkowski distance)
 */
void pairwise_distance(raft::resources const& handle,
                       raft::device_csr_matrix_view<const double, int, int, int> x,
                       raft::device_csr_matrix_view<const double, int, int, int> y,
                       raft::device_matrix_view<double, int, raft::row_major> dist,
                       cuvs::distance::DistanceType metric,
                       float metric_arg = 2.0f);

/** @} */  // end group pairwise_distance_runtime

}  // namespace distance
}  // namespace CUVS_EXPORT cuvs
