/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cub_ptx_query.hpp"

#include "../neighbors/detail/cagra/cagra_merge_scaffold.cuh"
#include "cub_runtime_launchers.hpp"
#include <rmm/exec_policy.hpp>
#include <thrust/sort.h>

namespace cuvs::detail {

template <typename Policy, typename Key, typename Value>
void thrust_stable_sort_by_key(Policy const& policy, Key first, Key last, Value values)
{
  thrust::stable_sort_by_key(policy, first, last, values);
}

}  // namespace cuvs::detail

template void cuvs::detail::thrust_stable_sort_by_key(
  rmm::exec_policy_nosync const&,
  thrust::device_ptr<unsigned int>,
  thrust::device_ptr<unsigned int>,
  thrust::device_ptr<cuvs::neighbors::cagra::detail::merge_scaffold::partition_membership>);
