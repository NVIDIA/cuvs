/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gmm.cuh"

#include <cuvs/cluster/gmm.hpp>

namespace cuvs::cluster::gmm {

void fit(raft::resources const& handle,
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
         bool warm_start)
{
  fit<double>(handle,
              params,
              X,
              weights,
              means,
              covariances,
              precisions_chol,
              precisions,
              labels,
              lower_bound,
              n_iter,
              converged,
              warm_start);
}

void predict(raft::resources const& handle,
             const params& params,
             raft::device_matrix_view<const double, int64_t> X,
             raft::device_vector_view<const double, int64_t> weights,
             raft::device_matrix_view<const double, int64_t> means,
             raft::device_vector_view<const double, int64_t> precisions_chol,
             raft::device_vector_view<int, int64_t> labels)
{
  predict<double>(handle, params, X, weights, means, precisions_chol, labels);
}

void predict_proba(raft::resources const& handle,
                   const params& params,
                   raft::device_matrix_view<const double, int64_t> X,
                   raft::device_vector_view<const double, int64_t> weights,
                   raft::device_matrix_view<const double, int64_t> means,
                   raft::device_vector_view<const double, int64_t> precisions_chol,
                   raft::device_matrix_view<double, int64_t> resp)
{
  predict_proba<double>(handle, params, X, weights, means, precisions_chol, resp);
}

void score_samples(raft::resources const& handle,
                   const params& params,
                   raft::device_matrix_view<const double, int64_t> X,
                   raft::device_vector_view<const double, int64_t> weights,
                   raft::device_matrix_view<const double, int64_t> means,
                   raft::device_vector_view<const double, int64_t> precisions_chol,
                   raft::device_vector_view<double, int64_t> log_prob_norm)
{
  score_samples<double>(handle, params, X, weights, means, precisions_chol, log_prob_norm);
}

}  // namespace cuvs::cluster::gmm
