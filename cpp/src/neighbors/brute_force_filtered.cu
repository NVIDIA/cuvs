/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Filtered search uses cuSPARSE (masked matmul); keeping it out of brute_force.cu lets
// unfiltered static-library users avoid linking cuSPARSE.

#include <cuvs/core/cuda_fp16.hpp>

#include "./detail/knn_brute_force.cuh"

#include <cuvs/neighbors/brute_force.hpp>

namespace CUVS_EXPORT cuvs {
namespace neighbors {
namespace brute_force {

#define CUVS_INST_BFKNN_FILTERED(T, DistT)                                           \
  void search(raft::resources const& res,                                            \
              const cuvs::neighbors::brute_force::search_params& params,             \
              const cuvs::neighbors::brute_force::index<T, DistT>& idx,              \
              raft::device_matrix_view<const T, int64_t, raft::row_major> queries,   \
              raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors, \
              raft::device_matrix_view<DistT, int64_t, raft::row_major> distances,   \
              const cuvs::neighbors::filtering::base_filter& sample_filter)          \
  {                                                                                  \
    detail::search<T, int64_t, DistT, raft::row_major>(                              \
      res, idx, queries, neighbors, distances, sample_filter);                       \
  }                                                                                  \
  void search(raft::resources const& res,                                            \
              const cuvs::neighbors::brute_force::index<T, DistT>& idx,              \
              raft::device_matrix_view<const T, int64_t, raft::row_major> queries,   \
              raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors, \
              raft::device_matrix_view<DistT, int64_t, raft::row_major> distances,   \
              const cuvs::neighbors::filtering::base_filter& sample_filter)          \
  {                                                                                  \
    detail::search<T, int64_t, DistT, raft::row_major>(                              \
      res, idx, queries, neighbors, distances, sample_filter);                       \
  }                                                                                  \
  void search(raft::resources const& res,                                            \
              const cuvs::neighbors::brute_force::search_params& params,             \
              const cuvs::neighbors::brute_force::index<T, DistT>& idx,              \
              raft::device_matrix_view<const T, int64_t, raft::col_major> queries,   \
              raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors, \
              raft::device_matrix_view<DistT, int64_t, raft::row_major> distances,   \
              const cuvs::neighbors::filtering::base_filter& sample_filter)          \
  {                                                                                  \
    detail::search<T, int64_t, DistT, raft::col_major>(                              \
      res, idx, queries, neighbors, distances, sample_filter);                       \
  }                                                                                  \
  void search(raft::resources const& res,                                            \
              const cuvs::neighbors::brute_force::index<T, DistT>& idx,              \
              raft::device_matrix_view<const T, int64_t, raft::col_major> queries,   \
              raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors, \
              raft::device_matrix_view<DistT, int64_t, raft::row_major> distances,   \
              const cuvs::neighbors::filtering::base_filter& sample_filter)          \
  {                                                                                  \
    detail::search<T, int64_t, DistT, raft::col_major>(                              \
      res, idx, queries, neighbors, distances, sample_filter);                       \
  }

CUVS_INST_BFKNN_FILTERED(float, float);
CUVS_INST_BFKNN_FILTERED(half, float);

#undef CUVS_INST_BFKNN_FILTERED

}  // namespace brute_force
}  // namespace neighbors
}  // namespace CUVS_EXPORT cuvs
