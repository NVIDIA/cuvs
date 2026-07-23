/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <cuvs/core/export.hpp>

#include <cstdint>

namespace CUVS_EXPORT cuvs {
namespace cluster {
namespace gmm {

/**
 * @defgroup gmm_params Gaussian mixture hyperparameters
 * @{
 */

/** Covariance parameterization of the mixture components. */
enum class covariance_type {
  /** Each component has its own general covariance matrix; shape (K, d, d). */
  FULL = 0,
  /** All components share one general covariance matrix; shape (d, d). */
  TIED = 1,
  /** Each component has its own diagonal covariance matrix; shape (K, d). */
  DIAG = 2,
  /** Each component has its own single variance (isotropic); shape (K,). */
  SPHERICAL = 3,
};

/** Strategy used to initialize the responsibilities before EM. */
enum class init_method {
  /** Run k-means (itself seeded with k-means++) and use the hard labels. */
  KMeans = 0,
  /** Use the k-means++ seeding labels directly: every sample is assigned to
   *  its nearest seed. (scikit-learn instead one-hots only the seed rows.) */
  KMeansPlusPlus = 1,
  /** Random per-sample-normalized responsibilities. */
  Random = 2,
  /** Pick n_components samples at random as one-hot responsibilities. */
  RandomFromData = 3,
};

/** Hyper-parameters for the Gaussian mixture EM solver.
 *
 *  All defaults match scikit-learn's ``GaussianMixture`` so that results are
 *  directly comparable out of the box.
 */
struct params {
  /** The number of mixture components (at most 65535). Default: 1. */
  int n_components = 1;
  /** Covariance parameterization of the mixture components. Default: FULL. */
  covariance_type cov_type = covariance_type::FULL;
  /** Convergence threshold on the change of the per-sample average
   *  log-likelihood (lower bound). Default: 1e-3. */
  double tol = 1e-3;
  /** Non-negative regularization added to the diagonal of covariance.
   *  Default: 1e-6. */
  double reg_covar = 1e-6;
  /** Maximum number of EM iterations for a single run. Default: 100. */
  int max_iter = 100;
  /** Number of initializations to perform; the best result is kept.
   *  Default: 1. */
  int n_init = 1;
  /** Strategy used to initialize the responsibilities before EM.
   *  Default: KMeans. */
  init_method init = init_method::KMeans;
  /** Seed to the random number generator. Default: 0. */
  uint64_t seed = 0;
};

/**
 * @}
 */

/**
 * @defgroup gmm Gaussian mixture model APIs
 * @{
 *
 * The covariance-shaped buffers (``covariances``, ``precisions_chol``,
 * ``precisions``) are passed as flat device vectors because their logical
 * shape depends on ``params::cov_type``. With ``K = n_components`` and
 * ``d = n_features`` the expected lengths are (row-major):
 *
 *   - ``FULL``:      K * d * d   (logically (K, d, d))
 *   - ``TIED``:      d * d       (logically (d, d))
 *   - ``DIAG``:      K * d       (logically (K, d))
 *   - ``SPHERICAL``: K           (logically (K,))
 *
 * For ``FULL``/``TIED``, ``precisions_chol`` holds the upper-triangular
 * factor ``U`` of each precision matrix (precision ``= U @ Uᵀ``); for
 * ``DIAG``/``SPHERICAL`` it holds reciprocal standard deviations. These
 * conventions match scikit-learn's ``GaussianMixture``.
 */

/**
 * @brief Fit a Gaussian mixture with the EM algorithm.
 *
 * Runs ``params.n_init`` random restarts (unless @p warm_start is true) and
 * keeps the parameters with the largest lower bound. Writes the fitted
 * ``weights``, ``means``, ``covariances``, ``precisions_chol`` and
 * ``precisions``, the per-sample hard ``labels`` (argmax of the final
 * responsibilities), and the scalar ``lower_bound`` / ``n_iter`` /
 * ``converged`` diagnostics.
 *
 * When @p warm_start is true the incoming ``weights`` / ``means`` /
 * ``covariances`` are used as the single initialization and ``params.n_init``
 * is ignored.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/gmm.hpp>
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::gmm::params params;
 *   params.n_components = 3;
 *
 *   int64_t K = params.n_components, d = X.extent(1);
 *   auto weights = raft::make_device_vector<float, int64_t>(handle, K);
 *   auto means   = raft::make_device_matrix<float, int64_t>(handle, K, d);
 *   auto covs    = raft::make_device_vector<float, int64_t>(handle, K * d * d);
 *   auto pchol   = raft::make_device_vector<float, int64_t>(handle, K * d * d);
 *   auto precs   = raft::make_device_vector<float, int64_t>(handle, K * d * d);
 *   auto labels  = raft::make_device_vector<int, int64_t>(handle, X.extent(0));
 *   float lower_bound;
 *   int n_iter;
 *   bool converged;
 *
 *   cuvs::cluster::gmm::fit(handle, params, X, weights.view(), means.view(),
 *                           covs.view(), pchol.view(), precs.view(), labels.view(),
 *                           raft::make_host_scalar_view(&lower_bound),
 *                           raft::make_host_scalar_view(&n_iter),
 *                           raft::make_host_scalar_view(&converged));
 * @endcode
 *
 * @param[in]    handle          The raft resources handle.
 * @param[in]    params          Hyper-parameters of the EM solver.
 * @param[in]    X               Training data, row-major.
 *                               [dim = n_samples x n_features]
 * @param[inout] weights         Mixture weights. [len = n_components]
 * @param[inout] means           Component means, row-major.
 *                               [dim = n_components x n_features]
 * @param[inout] covariances     Component covariances, flat. Length depends on
 *                               cov_type (K=n_components, d=n_features): FULL
 *                               K*d*d, TIED d*d, DIAG K*d, SPHERICAL K.
 * @param[out]   precisions_chol Precision Cholesky factors, same flat layout as
 *                               covariances. FULL/TIED hold the upper-triangular
 *                               factor U (precision = U @ Uᵀ); DIAG/SPHERICAL
 *                               hold reciprocal standard deviations.
 * @param[out]   precisions      Precision matrices, same flat layout as
 *                               covariances.
 * @param[out]   labels          Hard component assignment per sample.
 *                               [len = n_samples]
 * @param[out]   lower_bound     Per-sample average log-likelihood of the best
 *                               fit.
 * @param[out]   n_iter          Number of EM iterations of the best fit.
 * @param[out]   converged       Whether the best fit converged within
 *                               ``params.tol``.
 * @param[in]    warm_start      Use the incoming weights/means/covariances as
 *                               the single initialization.
 */
template <typename T>
void fit(raft::resources const& handle,
         const params& params,
         raft::device_matrix_view<const T, int64_t> X,
         raft::device_vector_view<T, int64_t> weights,
         raft::device_matrix_view<T, int64_t> means,
         raft::device_vector_view<T, int64_t> covariances,
         raft::device_vector_view<T, int64_t> precisions_chol,
         raft::device_vector_view<T, int64_t> precisions,
         raft::device_vector_view<int, int64_t> labels,
         raft::host_scalar_view<T> lower_bound,
         raft::host_scalar_view<int> n_iter,
         raft::host_scalar_view<bool> converged,
         bool warm_start = false);

extern template void fit<float>(raft::resources const& handle,
                                const params& params,
                                raft::device_matrix_view<const float, int64_t> X,
                                raft::device_vector_view<float, int64_t> weights,
                                raft::device_matrix_view<float, int64_t> means,
                                raft::device_vector_view<float, int64_t> covariances,
                                raft::device_vector_view<float, int64_t> precisions_chol,
                                raft::device_vector_view<float, int64_t> precisions,
                                raft::device_vector_view<int, int64_t> labels,
                                raft::host_scalar_view<float> lower_bound,
                                raft::host_scalar_view<int> n_iter,
                                raft::host_scalar_view<bool> converged,
                                bool warm_start);

extern template void fit<double>(raft::resources const& handle,
                                 const params& params,
                                 raft::device_matrix_view<const double, int64_t> X,
                                 raft::device_vector_view<double, int64_t> weights,
                                 raft::device_matrix_view<double, int64_t> means,
                                 raft::device_vector_view<double, int64_t> covariances,
                                 raft::device_vector_view<double, int64_t> precisions_chol,
                                 raft::device_vector_view<double, int64_t> precisions,
                                 raft::device_vector_view<int, int64_t> labels,
                                 raft::host_scalar_view<double> lower_bound,
                                 raft::host_scalar_view<int> n_iter,
                                 raft::host_scalar_view<bool> converged,
                                 bool warm_start);

/**
 * @brief Hard component labels (argmax responsibility) for new data.
 *
 * @param[in]  handle          The raft resources handle.
 * @param[in]  params          Fit hyper-parameters; only n_components and
 *                             cov_type are consulted at inference time.
 * @param[in]  X               Data to assign, row-major.
 *                             [dim = n_samples x n_features]
 * @param[in]  weights         Fitted mixture weights. [len = n_components]
 * @param[in]  means           Fitted component means.
 *                             [dim = n_components x n_features]
 * @param[in]  precisions_chol Fitted precision Cholesky factors, flat. Length
 *                             by cov_type (K=n_components, d=n_features): FULL
 *                             K*d*d, TIED d*d, DIAG K*d, SPHERICAL K.
 * @param[out] labels          Hard component assignment per sample.
 *                             [len = n_samples]
 */
template <typename T>
void predict(raft::resources const& handle,
             const params& params,
             raft::device_matrix_view<const T, int64_t> X,
             raft::device_vector_view<const T, int64_t> weights,
             raft::device_matrix_view<const T, int64_t> means,
             raft::device_vector_view<const T, int64_t> precisions_chol,
             raft::device_vector_view<int, int64_t> labels);

extern template void predict<float>(raft::resources const& handle,
                                    const params& params,
                                    raft::device_matrix_view<const float, int64_t> X,
                                    raft::device_vector_view<const float, int64_t> weights,
                                    raft::device_matrix_view<const float, int64_t> means,
                                    raft::device_vector_view<const float, int64_t> precisions_chol,
                                    raft::device_vector_view<int, int64_t> labels);

extern template void predict<double>(
  raft::resources const& handle,
  const params& params,
  raft::device_matrix_view<const double, int64_t> X,
  raft::device_vector_view<const double, int64_t> weights,
  raft::device_matrix_view<const double, int64_t> means,
  raft::device_vector_view<const double, int64_t> precisions_chol,
  raft::device_vector_view<int, int64_t> labels);

/**
 * @brief Posterior responsibilities for new data.
 *
 * @param[in]  handle          The raft resources handle.
 * @param[in]  params          Fit hyper-parameters; only n_components and
 *                             cov_type are consulted at inference time.
 * @param[in]  X               Data to evaluate, row-major.
 *                             [dim = n_samples x n_features]
 * @param[in]  weights         Fitted mixture weights. [len = n_components]
 * @param[in]  means           Fitted component means.
 *                             [dim = n_components x n_features]
 * @param[in]  precisions_chol Fitted precision Cholesky factors, flat. Length
 *                             by cov_type (K=n_components, d=n_features): FULL
 *                             K*d*d, TIED d*d, DIAG K*d, SPHERICAL K.
 * @param[out] resp            Posterior probability of each component for
 *                             each sample, row-major.
 *                             [dim = n_samples x n_components]
 */
template <typename T>
void predict_proba(raft::resources const& handle,
                   const params& params,
                   raft::device_matrix_view<const T, int64_t> X,
                   raft::device_vector_view<const T, int64_t> weights,
                   raft::device_matrix_view<const T, int64_t> means,
                   raft::device_vector_view<const T, int64_t> precisions_chol,
                   raft::device_matrix_view<T, int64_t> resp);

extern template void predict_proba<float>(
  raft::resources const& handle,
  const params& params,
  raft::device_matrix_view<const float, int64_t> X,
  raft::device_vector_view<const float, int64_t> weights,
  raft::device_matrix_view<const float, int64_t> means,
  raft::device_vector_view<const float, int64_t> precisions_chol,
  raft::device_matrix_view<float, int64_t> resp);

extern template void predict_proba<double>(
  raft::resources const& handle,
  const params& params,
  raft::device_matrix_view<const double, int64_t> X,
  raft::device_vector_view<const double, int64_t> weights,
  raft::device_matrix_view<const double, int64_t> means,
  raft::device_vector_view<const double, int64_t> precisions_chol,
  raft::device_matrix_view<double, int64_t> resp);

/**
 * @brief Per-sample log-likelihood log p(x_i) for new data.
 *
 * @param[in]  handle          The raft resources handle.
 * @param[in]  params          Fit hyper-parameters; only n_components and
 *                             cov_type are consulted at inference time.
 * @param[in]  X               Data to evaluate, row-major.
 *                             [dim = n_samples x n_features]
 * @param[in]  weights         Fitted mixture weights. [len = n_components]
 * @param[in]  means           Fitted component means.
 *                             [dim = n_components x n_features]
 * @param[in]  precisions_chol Fitted precision Cholesky factors, flat. Length
 *                             by cov_type (K=n_components, d=n_features): FULL
 *                             K*d*d, TIED d*d, DIAG K*d, SPHERICAL K.
 * @param[out] log_prob_norm   Log-likelihood of each sample under the model.
 *                             [len = n_samples]
 */
template <typename T>
void score_samples(raft::resources const& handle,
                   const params& params,
                   raft::device_matrix_view<const T, int64_t> X,
                   raft::device_vector_view<const T, int64_t> weights,
                   raft::device_matrix_view<const T, int64_t> means,
                   raft::device_vector_view<const T, int64_t> precisions_chol,
                   raft::device_vector_view<T, int64_t> log_prob_norm);

extern template void score_samples<float>(
  raft::resources const& handle,
  const params& params,
  raft::device_matrix_view<const float, int64_t> X,
  raft::device_vector_view<const float, int64_t> weights,
  raft::device_matrix_view<const float, int64_t> means,
  raft::device_vector_view<const float, int64_t> precisions_chol,
  raft::device_vector_view<float, int64_t> log_prob_norm);

extern template void score_samples<double>(
  raft::resources const& handle,
  const params& params,
  raft::device_matrix_view<const double, int64_t> X,
  raft::device_vector_view<const double, int64_t> weights,
  raft::device_matrix_view<const double, int64_t> means,
  raft::device_vector_view<const double, int64_t> precisions_chol,
  raft::device_vector_view<double, int64_t> log_prob_norm);

/**
 * @}
 */

}  // namespace gmm
}  // namespace cluster
}  // namespace CUVS_EXPORT cuvs
