/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/cluster/assignment.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>

#include "ivf_pq_build.cuh"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/device_memory_resource.hpp>

namespace cuvs::neighbors::ivf_pq {

auto build_ivf_assignment_index(raft::resources const& handle,
                                const cuvs::neighbors::ivf_pq::index<int64_t>& index,
                                const cuvs::cluster::ivf_assignment_params& params)
  -> cuvs::cluster::ivf_assignment_index
{
  auto centers = raft::make_device_matrix<float, int64_t>(
    handle, static_cast<int64_t>(index.n_lists()), static_cast<int64_t>(index.dim()));
  detail::extract_centers(handle, index, centers.view());
  return cuvs::cluster::build_ivf_assignment_index(
    handle,
    params,
    raft::make_const_mdspan(centers.view()),
    detail::coarse_clustering_metric(index.metric()));
}

}  // namespace cuvs::neighbors::ivf_pq
