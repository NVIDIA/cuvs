/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "matrix_broadcast_launchers.hpp"
#include <cuvs/core/cuda_fp16.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/linalg/map.cuh>
#include <raft/matrix/diagonal.cuh>

extern template void raft::linalg::detail::map<false, float, long, raft::const_op<float>>(
  cuda::stream_ref, float*, long, raft::const_op<float>);

namespace cuvs::detail {

template <typename math_t, typename idx_t, typename layout_t>
void eye(const raft::resources& handle, raft::device_matrix_view<math_t, idx_t, layout_t> out)

{
  return raft::matrix::eye(handle, out);
}

}  // namespace cuvs::detail

template void cuvs::detail::eye(raft::resources const&,
                                raft::device_matrix_view<float, long, raft::col_major>);
