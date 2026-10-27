/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <neighbors/detail/knn_brute_force.cuh>

namespace cuvs::neighbors::detail {

void apply_tiled_cosine_distance(raft::resources const& handle,
                                 float* dist,
                                 size_t current_query_size,
                                 size_t current_centroid_size,
                                 size_t i,
                                 size_t j,
                                 float const* row_norms,
                                 float const* col_norms)
{
  using IndexType = int64_t;
  using DistanceT = float;
  raft::identity_op distance_epilogue{};
  raft::linalg::map_offset(
    handle,
    raft::make_device_vector_view(dist, current_query_size * current_centroid_size),
    [=] __device__(IndexType idx) {
      IndexType row = i + (idx / current_centroid_size);
      IndexType col = j + (idx % current_centroid_size);
      auto val      = DistanceT(1.0) - dist[idx] / DistanceT(row_norms[row] * col_norms[col]);
      return distance_epilogue(val, row, col);
    });
}

}  // namespace cuvs::neighbors::detail
