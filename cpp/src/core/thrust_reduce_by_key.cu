/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cub_ptx_query.hpp"

#include "cub_runtime_launchers.hpp"
#include <rmm/exec_policy.hpp>
#include <thrust/device_ptr.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>

namespace cuvs::detail {

template <typename Policy, typename Key, typename Value, typename OutputKey, typename OutputValue>
cuda::std::pair<OutputKey, OutputValue> thrust_reduce_by_key(Policy const& policy,
                                                             Key first,
                                                             Key last,
                                                             Value values,
                                                             OutputKey keys_out,
                                                             OutputValue values_out)
{
  return thrust::reduce_by_key(policy, first, last, values, keys_out, values_out);
}

}  // namespace cuvs::detail

template cuda::std::pair<thrust::device_ptr<unsigned int>, thrust::device_ptr<unsigned int>>
cuvs::detail::thrust_reduce_by_key(
  rmm::exec_policy_nosync const&,
  thrust::device_ptr<unsigned int>,
  thrust::device_ptr<unsigned int>,
  thrust::constant_iterator<unsigned int, thrust::use_default, thrust::use_default>,
  thrust::device_ptr<unsigned int>,
  thrust::device_ptr<unsigned int>);
