/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/preprocessing/pca.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/pca.cuh>

namespace cuvs::preprocessing::pca::detail {

/**
 * @brief Convert cuvs::preprocessing::pca::params to raft::linalg::paramsPCA.
 */
inline auto to_raft_params(const params& config) -> raft::linalg::paramsPCA
{
  raft::linalg::paramsPCA prms;
  prms.algorithm    = config.algorithm;
  prms.tol          = config.tol;
  prms.n_iterations = config.n_iterations;
  prms.copy         = config.copy;
  prms.whiten       = config.whiten;
  return prms;
}

template <typename DataT, typename IndexT, typename LayoutPolicy>
void fit(raft::resources const& handle,
         const params& config,
         raft::device_matrix_view<DataT, IndexT, LayoutPolicy> input,
         raft::device_matrix_view<DataT, IndexT, LayoutPolicy> components,
         raft::device_vector_view<DataT, IndexT> explained_var,
         raft::device_vector_view<DataT, IndexT> explained_var_ratio,
         raft::device_vector_view<DataT, IndexT> singular_vals,
         raft::device_vector_view<DataT, IndexT> mu,
         raft::device_scalar_view<DataT, IndexT> noise_vars,
         bool flip_signs_based_on_U)
{
  auto raft_prms = to_raft_params(config);
  raft::linalg::pca_fit(handle,
                        raft_prms,
                        input,
                        components,
                        explained_var,
                        explained_var_ratio,
                        singular_vals,
                        mu,
                        noise_vars,
                        flip_signs_based_on_U);
}

template <typename DataT, typename IndexT, typename LayoutPolicy>
void fit_transform(raft::resources const& handle,
                   const params& config,
                   raft::device_matrix_view<DataT, IndexT, LayoutPolicy> input,
                   raft::device_matrix_view<DataT, IndexT, LayoutPolicy> trans_input,
                   raft::device_matrix_view<DataT, IndexT, LayoutPolicy> components,
                   raft::device_vector_view<DataT, IndexT> explained_var,
                   raft::device_vector_view<DataT, IndexT> explained_var_ratio,
                   raft::device_vector_view<DataT, IndexT> singular_vals,
                   raft::device_vector_view<DataT, IndexT> mu,
                   raft::device_scalar_view<DataT, IndexT> noise_vars,
                   bool flip_signs_based_on_U)
{
  auto raft_prms = to_raft_params(config);
  raft::linalg::pca_fit_transform(handle,
                                  raft_prms,
                                  input,
                                  trans_input,
                                  components,
                                  explained_var,
                                  explained_var_ratio,
                                  singular_vals,
                                  mu,
                                  noise_vars,
                                  flip_signs_based_on_U);
}

template <typename DataT, typename IndexT, typename LayoutPolicy>
void transform(raft::resources const& handle,
               const params& config,
               raft::device_matrix_view<DataT, IndexT, LayoutPolicy> input,
               raft::device_matrix_view<DataT, IndexT, LayoutPolicy> components,
               raft::device_vector_view<DataT, IndexT> singular_vals,
               raft::device_vector_view<DataT, IndexT> mu,
               raft::device_matrix_view<DataT, IndexT, LayoutPolicy> trans_input)
{
  auto raft_prms = to_raft_params(config);
  raft::linalg::pca_transform(handle, raft_prms, input, components, singular_vals, mu, trans_input);
}

template <typename DataT, typename IndexT, typename LayoutPolicy>
void inverse_transform(raft::resources const& handle,
                       const params& config,
                       raft::device_matrix_view<DataT, IndexT, LayoutPolicy> trans_input,
                       raft::device_matrix_view<DataT, IndexT, LayoutPolicy> components,
                       raft::device_vector_view<DataT, IndexT> singular_vals,
                       raft::device_vector_view<DataT, IndexT> mu,
                       raft::device_matrix_view<DataT, IndexT, LayoutPolicy> output)
{
  auto raft_prms = to_raft_params(config);
  raft::linalg::pca_inverse_transform(
    handle, raft_prms, trans_input, components, singular_vals, mu, output);
}

}  // namespace cuvs::preprocessing::pca::detail
