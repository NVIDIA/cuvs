/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cub_ptx_query.hpp"

#include "cub_runtime_launchers.hpp"
#include <cuda_runtime.h>
#include <raft/core/resource/thrust_policy.hpp>
#include <thrust/logical.h>

namespace cuvs::detail {

bool has_non_finite(raft::resources const& handle, const float* values, int64_t count)
{
  auto is_non_finite_op = [] __device__(float v) { return isnan(v) || isinf(v); };
  return thrust::any_of(
    raft::resource::get_thrust_policy(handle), values, values + count, is_non_finite_op);
}

}  // namespace cuvs::detail
