/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/ivf_sq.hpp>

#include "ivf_sq_build.cuh"

namespace cuvs::neighbors::ivf_sq {

#define CUVS_INST_IVF_SQ_BUILD_EXTEND(T, CodeT)                                               \
  auto build(raft::resources const& handle,                                                   \
             const cuvs::neighbors::ivf_sq::index_params& params,                             \
             raft::device_matrix_view<const T, int64_t, raft::row_major> dataset)             \
    -> cuvs::neighbors::ivf_sq::index<CodeT>                                                  \
  {                                                                                           \
    return cuvs::neighbors::ivf_sq::index<CodeT>(                                             \
      std::move(cuvs::neighbors::ivf_sq::detail::build<T, CodeT>(handle, params, dataset)));  \
  }                                                                                           \
                                                                                              \
  auto build(raft::resources const& handle,                                                   \
             const cuvs::neighbors::ivf_sq::index_params& params,                             \
             raft::host_matrix_view<const T, int64_t, raft::row_major> dataset)               \
    -> cuvs::neighbors::ivf_sq::index<CodeT>                                                  \
  {                                                                                           \
    return cuvs::neighbors::ivf_sq::index<CodeT>(                                             \
      std::move(cuvs::neighbors::ivf_sq::detail::build<T, CodeT>(handle, params, dataset)));  \
  }                                                                                           \
                                                                                              \
  void extend(raft::resources const& handle,                                                  \
              raft::device_matrix_view<const T, int64_t, raft::row_major> new_vectors,        \
              std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,    \
              cuvs::neighbors::ivf_sq::index<CodeT>* idx)                                     \
  {                                                                                           \
    cuvs::neighbors::ivf_sq::detail::extend<T, CodeT>(handle, new_vectors, new_indices, idx); \
  }                                                                                           \
                                                                                              \
  void extend(raft::resources const& handle,                                                  \
              raft::host_matrix_view<const T, int64_t, raft::row_major> new_vectors,          \
              std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,      \
              cuvs::neighbors::ivf_sq::index<CodeT>* idx)                                     \
  {                                                                                           \
    cuvs::neighbors::ivf_sq::detail::extend<T, CodeT>(handle, new_vectors, new_indices, idx); \
  }

CUVS_INST_IVF_SQ_BUILD_EXTEND(float, uint8_t);

#undef CUVS_INST_IVF_SQ_BUILD_EXTEND

}  // namespace cuvs::neighbors::ivf_sq
