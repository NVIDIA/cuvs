/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cub_ptx_query.hpp"

#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/exec_policy.hpp>

#include "../distance/detail/sparse/coo_spmv_strategies/coo_mask_row_iterators.cuh"
#include "cub_runtime_launchers.hpp"
#include <thrust/transform.h>

namespace cuvs::detail {

template <typename Policy, typename Input, typename Output, typename Op>
Output thrust_transform(Policy const& policy, Input first, Input last, Output output, Op op)
{
  return thrust::transform(policy, first, last, output, op);
}

}  // namespace cuvs::detail

template int* cuvs::detail::thrust_transform(
  rmm::exec_policy const&,
  int const*,
  int const*,
  int*,
  cuvs::distance::detail::sparse::chunked_mask_row_it<int>::n_chunks_per_row_functor);
