/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/soar.cuh"

#include <cuvs/cluster/soar.hpp>

#include <raft/core/error.hpp>

namespace cuvs::cluster::soar {

void predict(raft::resources const& handle,
             const soar::params& params,
             raft::device_matrix_view<const float, int64_t> dataset,
             raft::device_matrix_view<const float, int64_t> centroids,
             raft::device_vector_view<const uint32_t, int64_t> labels,
             raft::device_vector_view<uint32_t, int64_t> soar_labels)
{
  int64_t n_rows     = dataset.extent(0);
  int64_t dim        = dataset.extent(1);
  int64_t n_clusters = centroids.extent(0);

  RAFT_EXPECTS(centroids.extent(1) == dim,
               "Number of features in the dataset (%zd) and in the centroids (%zd) must match.",
               dim,
               centroids.extent(1));
  RAFT_EXPECTS(n_clusters > 0, "The number of centroids must be positive.");
  RAFT_EXPECTS(dim > 0, "The number of features must be positive.");
  RAFT_EXPECTS(labels.extent(0) == n_rows,
               "The number of labels (%zd) must match the number of rows in the dataset (%zd).",
               labels.extent(0),
               n_rows);
  RAFT_EXPECTS(
    soar_labels.extent(0) == n_rows,
    "The number of soar labels (%zd) must match the number of rows in the dataset (%zd).",
    soar_labels.extent(0),
    n_rows);

  if (n_rows == 0) { return; }

  auto residuals = detail::compute_residuals<float, uint32_t>(handle, dataset, centroids, labels);

  detail::compute_soar_labels<float, uint32_t>(handle,
                                               dataset,
                                               raft::make_const_mdspan(residuals.view()),
                                               centroids,
                                               labels,
                                               soar_labels,
                                               params.lambda);
}

}  // namespace cuvs::cluster::soar
