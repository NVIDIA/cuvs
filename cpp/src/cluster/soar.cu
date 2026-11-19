/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/soar.cuh"

#include <cuvs/cluster/soar.hpp>

namespace cuvs::cluster::soar {

void predict(raft::resources const& handle,
             const soar::params& params,
             raft::device_matrix_view<const float, int64_t> dataset,
             raft::device_matrix_view<const float, int64_t> centroids,
             raft::device_vector_view<const uint32_t, int64_t> labels,
             raft::device_vector_view<uint32_t, int64_t> soar_labels)
{
  detail::predict(handle, params, dataset, centroids, labels, soar_labels);
}

}  // namespace cuvs::cluster::soar
