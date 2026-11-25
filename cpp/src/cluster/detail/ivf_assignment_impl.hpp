/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ivf_assign.hpp"

#include <cuvs/cluster/assignment.hpp>

#include <raft/core/error.hpp>

#include <utility>

namespace cuvs::cluster {

struct ivf_assignment_index::impl {
  impl(kmeans::detail::ivf_assign::index&& i, cuvs::distance::DistanceType m)
    : idx(std::move(i)), metric(m)
  {
  }
  kmeans::detail::ivf_assign::index idx;
  cuvs::distance::DistanceType metric;
};

namespace detail {

/** The index a call should use: the caller's prebuilt one, validated, or null if there is none. */
inline auto prebuilt_ivf_index(const ivf_assignment_params& params,
                               uint32_t n_centroids,
                               uint32_t dim,
                               cuvs::distance::DistanceType metric)
  -> const kmeans::detail::ivf_assign::index*
{
  if (params.index == nullptr) { return nullptr; }
  const auto& impl = params.index->get();
  RAFT_EXPECTS(impl.idx.n_centroids == n_centroids,
               "The IVF assignment index was built from %u centroids, expected %u",
               impl.idx.n_centroids,
               n_centroids);
  RAFT_EXPECTS(impl.idx.dim == dim,
               "The IVF assignment index has dimension %u, expected %u",
               impl.idx.dim,
               dim);
  RAFT_EXPECTS(impl.metric == metric, "The IVF assignment index was built for another metric");
  return &impl.idx;
}

}  // namespace detail
}  // namespace cuvs::cluster
