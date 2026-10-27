/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/pca_types.hpp>

namespace cuvs::detail {
template <typename math_t, typename idx_t, typename LayoutPolicy>
void pca_fit(raft::resources const& handle,
             const raft::linalg::paramsPCA& prms,
             raft::device_matrix_view<math_t, idx_t, LayoutPolicy> input,
             raft::device_matrix_view<math_t, idx_t, LayoutPolicy> components,
             raft::device_vector_view<math_t, idx_t> explained_var,
             raft::device_vector_view<math_t, idx_t> explained_var_ratio,
             raft::device_vector_view<math_t, idx_t> singular_vals,
             raft::device_vector_view<math_t, idx_t> mu,
             raft::device_scalar_view<math_t, idx_t> noise_vars,
             bool flip_signs_based_on_U = false);

template <typename math_t, typename idx_t, typename LayoutPolicy>
void pca_fit_transform(raft::resources const& handle,
                       const raft::linalg::paramsPCA& prms,
                       raft::device_matrix_view<math_t, idx_t, LayoutPolicy> input,
                       raft::device_matrix_view<math_t, idx_t, LayoutPolicy> trans_input,
                       raft::device_matrix_view<math_t, idx_t, LayoutPolicy> components,
                       raft::device_vector_view<math_t, idx_t> explained_var,
                       raft::device_vector_view<math_t, idx_t> explained_var_ratio,
                       raft::device_vector_view<math_t, idx_t> singular_vals,
                       raft::device_vector_view<math_t, idx_t> mu,
                       raft::device_scalar_view<math_t, idx_t> noise_vars,
                       bool flip_signs_based_on_U = false);

template <typename math_t, typename idx_t, typename LayoutPolicy>
void pca_inverse_transform(raft::resources const& handle,
                           const raft::linalg::paramsPCA& prms,
                           raft::device_matrix_view<math_t, idx_t, LayoutPolicy> trans_input,
                           raft::device_matrix_view<math_t, idx_t, LayoutPolicy> components,
                           raft::device_vector_view<math_t, idx_t> singular_vals,
                           raft::device_vector_view<math_t, idx_t> mu,
                           raft::device_matrix_view<math_t, idx_t, LayoutPolicy> output);

template <typename math_t, typename idx_t, typename LayoutPolicy>
void pca_transform(raft::resources const& handle,
                   const raft::linalg::paramsPCA& prms,
                   raft::device_matrix_view<math_t, idx_t, LayoutPolicy> input,
                   raft::device_matrix_view<math_t, idx_t, LayoutPolicy> components,
                   raft::device_vector_view<math_t, idx_t> singular_vals,
                   raft::device_vector_view<math_t, idx_t> mu,
                   raft::device_matrix_view<math_t, idx_t, LayoutPolicy> trans_input);
}  // namespace cuvs::detail
