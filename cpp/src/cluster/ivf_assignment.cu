/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/ivf_assignment_impl.hpp"

#include <cuvs/cluster/assignment.hpp>

namespace cuvs::cluster {

ivf_assignment_index::ivf_assignment_index(std::unique_ptr<impl> i) : impl_(std::move(i)) {}
ivf_assignment_index::ivf_assignment_index(ivf_assignment_index&&) noexcept            = default;
ivf_assignment_index& ivf_assignment_index::operator=(ivf_assignment_index&&) noexcept = default;
ivf_assignment_index::~ivf_assignment_index()                                          = default;

uint32_t ivf_assignment_index::n_centroids() const noexcept { return impl_->idx.n_centroids; }
uint32_t ivf_assignment_index::dim() const noexcept { return impl_->idx.dim; }
uint32_t ivf_assignment_index::n_lists() const noexcept { return impl_->idx.n_lists; }
cuvs::distance::DistanceType ivf_assignment_index::metric() const noexcept { return impl_->metric; }

auto build_ivf_assignment_index(
  raft::resources const& res,
  const ivf_assignment_params& params,
  raft::device_matrix_view<const float, int64_t, raft::row_major> centroids,
  cuvs::distance::DistanceType metric) -> ivf_assignment_index
{
  kmeans::detail::ivf_assign::index_params ip;
  ip.n_lists        = params.n_lists;
  ip.kmeans_n_iters = params.kmeans_n_iters;
  ip.metric         = metric;
  return ivf_assignment_index(std::make_unique<ivf_assignment_index::impl>(
    kmeans::detail::ivf_assign::build(res, ip, centroids), metric));
}

}  // namespace cuvs::cluster
