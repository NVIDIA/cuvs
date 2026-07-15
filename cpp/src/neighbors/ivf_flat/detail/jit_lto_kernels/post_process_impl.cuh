/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/operators.hpp>

namespace cuvs::neighbors::ivf_flat::detail {

template <typename T>
__device__ T post_process_identity_impl(T val)
{
  return raft::identity_op{}(val);
}

template <typename T>
__device__ T post_process_sqrt_impl(T val)
{
  return raft::sqrt_op{}(val);
}

template <typename T>
__device__ T post_process_compose_impl(T val)
{
  // This is for cosine distance: compose(add_const(1.0), mul_const(-1.0))
  // which computes: 1.0 + (-1.0 * val) = 1.0 - val
  return raft::compose_op(raft::add_const_op<float>{1.0f}, raft::mul_const_op<float>{-1.0f})(val);
}

}  // namespace cuvs::neighbors::ivf_flat::detail
