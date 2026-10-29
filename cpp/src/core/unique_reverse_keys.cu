/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cub_unique_launchers.hpp"

#include "cub_compact_launchers.hpp"

#include "cub_ptx_query.hpp"

#include "../neighbors/detail/vamana/build_helpers.hpp"
#include "../neighbors/detail/vamana/vamana_structs.cuh"
#include <cub/device/device_merge_sort.cuh>
#include <rmm/exec_policy.hpp>
#include <thrust/device_ptr.h>
#include <thrust/unique.h>

namespace cuvs::neighbors::vamana::detail {

template <typename Policy, typename Key, typename Value>
void unique_reverse_keys(Policy&& policy, Key begin, Key end, Value values)
{
  thrust::unique_by_key(policy, begin, end, values);
}

template void cuvs::neighbors::vamana::detail::unique_reverse_keys<rmm::exec_policy_nosync,
                                                                   thrust::device_ptr<unsigned int>,
                                                                   thrust::device_ptr<int>>(
  rmm::exec_policy_nosync&&,
  thrust::device_ptr<unsigned int>,
  thrust::device_ptr<unsigned int>,
  thrust::device_ptr<int>);

}  // namespace cuvs::neighbors::vamana::detail
