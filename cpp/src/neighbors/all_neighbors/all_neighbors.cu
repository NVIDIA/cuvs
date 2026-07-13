/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "all_neighbors.cuh"

namespace cuvs::neighbors::all_neighbors {

#define CUVS_INST_ALL_NEIGHBORS(T, IdxT)                                                 \
  void build(const raft::resources& handle,                                              \
             const all_neighbors_params& params,                                         \
             raft::host_matrix_view<const T, IdxT, row_major> dataset,                   \
             raft::device_matrix_view<IdxT, IdxT, row_major> indices,                    \
             std::optional<raft::device_matrix_view<T, IdxT, row_major>> distances,      \
             std::optional<raft::device_vector_view<T, IdxT, row_major>> core_distances, \
             T alpha)                                                                    \
  {                                                                                      \
    return all_neighbors::detail::build<T, IdxT>(                                        \
      handle, params, dataset, indices, distances, core_distances, alpha);               \
  }                                                                                      \
                                                                                         \
  void build(const raft::resources& handle,                                              \
             const all_neighbors_params& params,                                         \
             raft::device_matrix_view<const T, IdxT, row_major> dataset,                 \
             raft::device_matrix_view<IdxT, IdxT, row_major> indices,                    \
             std::optional<raft::device_matrix_view<T, IdxT, row_major>> distances,      \
             std::optional<raft::device_vector_view<T, IdxT, row_major>> core_distances, \
             T alpha)                                                                    \
  {                                                                                      \
    return all_neighbors::detail::build<T, IdxT>(                                        \
      handle, params, dataset, indices, distances, core_distances, alpha);               \
  }                                                                                      \
                                                                                         \
  void build(const raft::resources& handle,                                              \
             const all_neighbors_params& params,                                         \
             raft::host_matrix_view<const T, IdxT, row_major> dataset,                   \
             raft::host_matrix_view<IdxT, IdxT, row_major> indices,                      \
             std::optional<raft::host_matrix_view<T, IdxT, row_major>> distances,        \
             std::optional<raft::host_vector_view<T, IdxT, row_major>> core_distances,   \
             T alpha)                                                                    \
  {                                                                                      \
    return all_neighbors::detail::build(                                                 \
      handle, params, dataset, indices, distances, core_distances, alpha);               \
  }                                                                                      \
                                                                                         \
  void build(const raft::resources& handle,                                              \
             const all_neighbors_params& params,                                         \
             raft::device_matrix_view<const T, IdxT, row_major> dataset,                 \
             raft::host_matrix_view<IdxT, IdxT, row_major> indices,                      \
             std::optional<raft::host_matrix_view<T, IdxT, row_major>> distances,        \
             std::optional<raft::host_vector_view<T, IdxT, row_major>> core_distances,   \
             T alpha)                                                                    \
  {                                                                                      \
    return all_neighbors::detail::build(                                                 \
      handle, params, dataset, indices, distances, core_distances, alpha);               \
  }

CUVS_INST_ALL_NEIGHBORS(float, int64_t);

#undef CUVS_INST_ALL_NEIGHBORS

}  // namespace cuvs::neighbors::all_neighbors
