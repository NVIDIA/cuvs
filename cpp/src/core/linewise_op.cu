/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "matrix_broadcast_launchers.hpp"
#include <cuvs/core/cuda_fp16.hpp>
#include <raft/matrix/linewise_op.cuh>

extern template void
raft::matrix::detail::matrixLinewiseVecRows<float, long, 16ul, 256, raft::sub_op, float>(
  float*, float const*, long, long, raft::sub_op, CUstream_st*, float const*);

extern template void
raft::matrix::detail::matrixLinewiseVecRows<float, long, 4ul, 256, raft::sub_op, float>(
  float*, float const*, long, long, raft::sub_op, CUstream_st*, float const*);

extern template void
raft::matrix::detail::matrixLinewiseVecRows<float, long, 8ul, 256, raft::sub_op, float>(
  float*, float const*, long, long, raft::sub_op, CUstream_st*, float const*);

namespace cuvs::detail {

template <raft::Apply apply,
          typename m_t,
          typename idx_t,
          typename layout,
          typename Lambda,
          typename... vec_t,
          typename>
void linewise_op(raft::resources const& handle,
                 raft::device_matrix_view<const m_t, idx_t, layout> in,
                 raft::device_matrix_view<m_t, idx_t, layout> out,
                 Lambda op,
                 vec_t... vecs)

{
  return raft::matrix::linewise_op<apply, m_t, idx_t, layout, Lambda, vec_t...>(
    handle, in, out, op, vecs...);
}

template <raft::Apply apply,
          typename m_t,
          typename idx_t,
          typename layout,
          typename Lambda,
          typename... vec_t,
          typename>
void linewise_op(raft::resources const& handle,
                 raft::device_aligned_matrix_view<const m_t, idx_t, layout> in,
                 raft::device_aligned_matrix_view<m_t, idx_t, layout> out,
                 Lambda op,
                 vec_t... vecs)

{
  return raft::matrix::linewise_op<apply, m_t, idx_t, layout, Lambda, vec_t...>(
    handle, in, out, op, vecs...);
}

}  // namespace cuvs::detail

template void cuvs::detail::linewise_op<(raft::Apply)0>(
  raft::resources const&,
  raft::device_matrix_view<float const, long, raft::row_major>,
  raft::device_matrix_view<float, long, raft::row_major>,
  raft::sub_op,
  raft::device_vector_view<float const, long, raft::row_major>);
