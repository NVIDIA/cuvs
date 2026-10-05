/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cagra.cuh"
#include <cuvs/neighbors/cagra.hpp>

namespace cuvs::neighbors::cagra::helpers {

void optimize(raft::resources const& handle,
              raft::host_matrix_view<uint32_t, int64_t, raft::row_major> knn_graph,
              raft::host_matrix_view<uint32_t, int64_t, raft::row_major> new_graph,
              bool guarantee_connectivity)
{
  cuvs::neighbors::cagra::optimize(handle, knn_graph, new_graph, guarantee_connectivity);
}

void optimize(raft::resources const& handle,
              raft::device_matrix_view<uint32_t, int64_t, raft::row_major> knn_graph,
              raft::device_matrix_view<uint32_t, int64_t, raft::row_major> new_graph,
              bool guarantee_connectivity)
{
  cuvs::neighbors::cagra::optimize(handle, knn_graph, new_graph, guarantee_connectivity);
}

}  // namespace cuvs::neighbors::cagra::helpers
