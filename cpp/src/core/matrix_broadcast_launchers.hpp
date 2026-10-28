/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/types.hpp>

namespace cuvs::detail {
using raft::Apply;
template <Apply apply,
          typename MatValueType,
          typename VecValueType,
          typename LayoutPolicy,
          typename Lambda,
          typename IndexType>
void matrix_vector_op(raft::resources const& handle,
                      raft::device_matrix_view<const MatValueType, IndexType, LayoutPolicy> matrix,
                      raft::device_vector_view<const VecValueType, IndexType> vec,
                      raft::device_matrix_view<MatValueType, IndexType, LayoutPolicy> out,
                      Lambda op);

template <Apply apply,
          typename MatValueType,
          typename Vec1ValueType,
          typename Vec2ValueType,
          typename LayoutPolicy,
          typename Lambda,
          typename IndexType>
void matrix_vector_op(raft::resources const& handle,
                      raft::device_matrix_view<const MatValueType, IndexType, LayoutPolicy> matrix,
                      raft::device_vector_view<const Vec1ValueType, IndexType> vec1,
                      raft::device_vector_view<const Vec2ValueType, IndexType> vec2,
                      raft::device_matrix_view<MatValueType, IndexType, LayoutPolicy> out,
                      Lambda op);

template <Apply apply, typename math_t, typename idx_t, typename layout_t>
void binary_mult_skip_zero(raft::resources const& handle,
                           raft::device_matrix_view<math_t, idx_t, layout_t> data,
                           raft::device_vector_view<const math_t, idx_t> vec);
}  // namespace cuvs::detail

namespace cuvs::detail {
template <raft::Apply apply,
          typename m_t,
          typename idx_t,
          typename layout,
          typename Lambda,
          typename... vec_t,
          typename = raft::enable_if_device_mdspan<vec_t...>>
void linewise_op(raft::resources const& handle,
                 raft::device_matrix_view<const m_t, idx_t, layout> in,
                 raft::device_matrix_view<m_t, idx_t, layout> out,
                 Lambda op,
                 vec_t... vecs);

template <raft::Apply apply,
          typename m_t,
          typename idx_t,
          typename layout,
          typename Lambda,
          typename... vec_t,
          typename = raft::enable_if_device_mdspan<vec_t...>>
void linewise_op(raft::resources const& handle,
                 raft::device_aligned_matrix_view<const m_t, idx_t, layout> in,
                 raft::device_aligned_matrix_view<m_t, idx_t, layout> out,
                 Lambda op,
                 vec_t... vecs);

template <typename math_t, typename idx_t, typename layout_t>
void eye(const raft::resources& handle, raft::device_matrix_view<math_t, idx_t, layout_t> out);
}  // namespace cuvs::detail
