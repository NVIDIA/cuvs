/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "scann.cuh"
#include <cuvs/neighbors/scann.hpp>

namespace cuvs::neighbors::scann {

#define CUVS_INST_SCANN_BUILD(T, IdxT)                                         \
  auto build(raft::resources const& handle,                                    \
             const cuvs::neighbors::scann::index_params& params,               \
             raft::device_matrix_view<const T, IdxT, raft::row_major> dataset) \
    -> cuvs::neighbors::scann::index<T, IdxT>                                  \
  {                                                                            \
    return cuvs::neighbors::scann::build<T, IdxT>(handle, params, dataset);    \
  }                                                                            \
                                                                               \
  auto build(raft::resources const& handle,                                    \
             const cuvs::neighbors::scann::index_params& params,               \
             raft::host_matrix_view<const T, IdxT, raft::row_major> dataset)   \
    -> cuvs::neighbors::scann::index<T, IdxT>                                  \
  {                                                                            \
    return cuvs::neighbors::scann::build<T, IdxT>(handle, params, dataset);    \
  }

CUVS_INST_SCANN_BUILD(float, int64_t);

#undef CUVS_INST_SCANN_BUILD

}  // namespace cuvs::neighbors::scann
