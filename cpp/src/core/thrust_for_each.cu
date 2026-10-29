/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cub_ptx_query.hpp"

#include "../cluster/detail/agglomerative.cuh"
#include "cub_runtime_launchers.hpp"
#include <cuda/iterator>
#include <rmm/exec_policy.hpp>
#include <thrust/for_each.h>

namespace cuvs::detail {

template <typename Policy, typename Iterator, typename Op>
void thrust_for_each(Policy const& policy, Iterator first, Iterator last, Op op)
{
  thrust::for_each(policy, first, last, op);
}

}  // namespace cuvs::detail

template void cuvs::detail::thrust_for_each(
  rmm::exec_policy_nosync const&,
  thrust::zip_iterator<cuda::std::tuple<cuda::counting_iterator<unsigned int, long>, int*>>,
  thrust::zip_iterator<cuda::std::tuple<cuda::counting_iterator<unsigned int, long>, int*>>,
  cuvs::cluster::agglomerative::detail::init_label_roots<int>);
