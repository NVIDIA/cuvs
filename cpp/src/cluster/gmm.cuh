/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gmm_impl.cuh"

#include <cuvs/cluster/gmm.hpp>

#include <raft/core/error.hpp>

#include <limits>

namespace cuvs::cluster::gmm {

namespace {

// Validate the shared (X, weights, means, precisions_chol) arguments and
// write (n, d, K) as ints (the internal kernels index with 32-bit math).
template <typename T>
void check_common_args(const params& params,
                       raft::device_matrix_view<const T, int64_t> X,
                       raft::device_vector_view<const T, int64_t> weights,
                       raft::device_matrix_view<const T, int64_t> means,
                       raft::device_vector_view<const T, int64_t> precisions_chol,
                       int& n,
                       int& d,
                       int& K)
{
  int64_t n64 = X.extent(0);
  int64_t d64 = X.extent(1);
  int64_t K64 = params.n_components;
  RAFT_EXPECTS(static_cast<int>(params.cov_type) >= 0 && static_cast<int>(params.cov_type) <= 3,
               "cov_type must be one of FULL, TIED, DIAG, SPHERICAL");
  RAFT_EXPECTS(n64 > 0 && d64 > 0, "X must be non-empty");
  RAFT_EXPECTS(K64 > 0, "n_components must be positive");
  RAFT_EXPECTS(K64 <= 65535, "gmm currently supports up to 65535 components");
  RAFT_EXPECTS(n64 <= std::numeric_limits<int>::max() && d64 <= std::numeric_limits<int>::max(),
               "gmm currently supports up to 2^31-1 samples / features");
  RAFT_EXPECTS(weights.extent(0) == K64, "weights must have n_components elements");
  RAFT_EXPECTS(means.extent(0) == K64 && means.extent(1) == d64,
               "means must be of shape (n_components, n_features)");
  auto expected = detail::cov_elems(params.cov_type, (int)d64, (int)K64);
  RAFT_EXPECTS((size_t)precisions_chol.extent(0) == expected,
               "precisions_chol has the wrong number of elements for the covariance type");
  n = (int)n64;
  d = (int)d64;
  K = (int)K64;
}

}  // namespace

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
         bool warm_start)
{
  int n, d, K;
  check_common_args<T>(
    params,
    X,
    raft::make_device_vector_view<const T, int64_t>(weights.data_handle(), weights.extent(0)),
    raft::make_device_matrix_view<const T, int64_t>(
      means.data_handle(), means.extent(0), means.extent(1)),
    raft::make_device_vector_view<const T, int64_t>(precisions_chol.data_handle(),
                                                    precisions_chol.extent(0)),
    n,
    d,
    K);
  auto expected = detail::cov_elems(params.cov_type, d, K);
  RAFT_EXPECTS((size_t)covariances.extent(0) == expected,
               "covariances has the wrong number of elements for the covariance type");
  RAFT_EXPECTS((size_t)precisions.extent(0) == expected,
               "precisions has the wrong number of elements for the covariance type");
  RAFT_EXPECTS(labels.extent(0) == X.extent(0), "labels must have n_samples elements");
  RAFT_EXPECTS(static_cast<int>(params.init) >= 0 && static_cast<int>(params.init) <= 3,
               "init must be one of KMeans, KMeansPlusPlus, Random, RandomFromData");
  RAFT_EXPECTS(params.n_init > 0, "n_init must be positive");
  RAFT_EXPECTS(params.max_iter >= 0, "max_iter must be non-negative");
  RAFT_EXPECTS(params.tol >= 0.0, "tol must be non-negative");
  RAFT_EXPECTS(params.reg_covar >= 0.0, "reg_covar must be non-negative");
  RAFT_EXPECTS(K <= n, "n_components must be <= n_samples");

  detail::fit_impl<T>(handle,
                      params,
                      X.data_handle(),
                      n,
                      d,
                      weights.data_handle(),
                      means.data_handle(),
                      covariances.data_handle(),
                      precisions_chol.data_handle(),
                      precisions.data_handle(),
                      labels.data_handle(),
                      *lower_bound.data_handle(),
                      *n_iter.data_handle(),
                      *converged.data_handle(),
                      warm_start);
}

template <typename T>
void predict(raft::resources const& handle,
             const params& params,
             raft::device_matrix_view<const T, int64_t> X,
             raft::device_vector_view<const T, int64_t> weights,
             raft::device_matrix_view<const T, int64_t> means,
             raft::device_vector_view<const T, int64_t> precisions_chol,
             raft::device_vector_view<int, int64_t> labels)
{
  int n, d, K;
  check_common_args<T>(params, X, weights, means, precisions_chol, n, d, K);
  RAFT_EXPECTS(labels.extent(0) == X.extent(0), "labels must have n_samples elements");
  detail::predict_impl<T>(handle,
                          params,
                          X.data_handle(),
                          n,
                          d,
                          weights.data_handle(),
                          means.data_handle(),
                          precisions_chol.data_handle(),
                          labels.data_handle());
}

template <typename T>
void predict_proba(raft::resources const& handle,
                   const params& params,
                   raft::device_matrix_view<const T, int64_t> X,
                   raft::device_vector_view<const T, int64_t> weights,
                   raft::device_matrix_view<const T, int64_t> means,
                   raft::device_vector_view<const T, int64_t> precisions_chol,
                   raft::device_matrix_view<T, int64_t> resp)
{
  int n, d, K;
  check_common_args<T>(params, X, weights, means, precisions_chol, n, d, K);
  RAFT_EXPECTS(resp.extent(0) == X.extent(0) && resp.extent(1) == (int64_t)K,
               "resp must be of shape (n_samples, n_components)");
  detail::predict_proba_impl<T>(handle,
                                params,
                                X.data_handle(),
                                n,
                                d,
                                weights.data_handle(),
                                means.data_handle(),
                                precisions_chol.data_handle(),
                                resp.data_handle());
}

template <typename T>
void score_samples(raft::resources const& handle,
                   const params& params,
                   raft::device_matrix_view<const T, int64_t> X,
                   raft::device_vector_view<const T, int64_t> weights,
                   raft::device_matrix_view<const T, int64_t> means,
                   raft::device_vector_view<const T, int64_t> precisions_chol,
                   raft::device_vector_view<T, int64_t> log_prob_norm)
{
  int n, d, K;
  check_common_args<T>(params, X, weights, means, precisions_chol, n, d, K);
  RAFT_EXPECTS(log_prob_norm.extent(0) == X.extent(0),
               "log_prob_norm must have n_samples elements");
  detail::score_samples_impl<T>(handle,
                                params,
                                X.data_handle(),
                                n,
                                d,
                                weights.data_handle(),
                                means.data_handle(),
                                precisions_chol.data_handle(),
                                log_prob_norm.data_handle());
}

}  // namespace cuvs::cluster::gmm
