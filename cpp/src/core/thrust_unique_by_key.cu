/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cub_unique_launchers.hpp"

#include "cub_compact_launchers.hpp"

#include "cub_ptx_query.hpp"
#include "unique_by_key_launchers.hpp"
#include <cuvs/core/cuda_fp16.hpp>
#include <rmm/exec_policy.hpp>
#include <thrust/unique.h>

namespace cuvs::detail {

template <typename Policy, typename Key, typename Value>
cuda::std::pair<Key, Value> thrust_unique_by_key(Policy const& policy,
                                                 Key first,
                                                 Key last,
                                                 Value values)
{
  return thrust::unique_by_key(policy, first, last, values);
}

template <typename Policy, typename Key, typename Value, typename Predicate>
cuda::std::pair<Key, Value> thrust_unique_by_key(
  Policy const& policy, Key first, Key last, Value values, Predicate predicate)
{
  return thrust::unique_by_key(policy, first, last, values, predicate);
}

}  // namespace cuvs::detail

template cuda::std::pair<thrust::device_ptr<unsigned int>, thrust::device_ptr<int>>
cuvs::detail::thrust_unique_by_key<rmm::exec_policy_nosync,
                                   thrust::device_ptr<unsigned int>,
                                   thrust::device_ptr<int>>(rmm::exec_policy_nosync const&,
                                                            thrust::device_ptr<unsigned int>,
                                                            thrust::device_ptr<unsigned int>,
                                                            thrust::device_ptr<int>);
