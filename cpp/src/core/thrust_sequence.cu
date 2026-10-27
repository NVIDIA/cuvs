/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cub_ptx_query.hpp"
#include "thrust_fill_launchers.hpp"
#include <cuvs/core/cuda_fp16.hpp>
#include <raft/matrix/init.cuh>
#include <rmm/exec_policy.hpp>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/system/cuda/detail/tabulate.h>

namespace cuvs::detail {

template <typename Policy, typename Iterator>
void thrust_sequence(Policy const& policy, Iterator first, Iterator last)
{
  return thrust::sequence(policy, first, last);
}

template <typename Policy, typename Iterator, typename Init>
void thrust_sequence(Policy const& policy, Iterator first, Iterator last, Init init)
{
  return thrust::sequence(policy, first, last, init);
}

template <typename Policy, typename Iterator, typename Init, typename Step>
void thrust_sequence(Policy const& policy, Iterator first, Iterator last, Init init, Step step)
{
  return thrust::sequence(policy, first, last, init, step);
}

}  // namespace cuvs::detail
