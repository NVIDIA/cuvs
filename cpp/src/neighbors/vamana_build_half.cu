/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vamana.cuh"
#include <cuda_fp16.h>
#include <cuvs/neighbors/vamana.hpp>

namespace cuvs::neighbors::vamana {

#define RAFT_INST_VAMANA_BUILD(T, IdxT)                                           \
  auto build(raft::resources const& handle,                                       \
             const cuvs::neighbors::vamana::index_params& params,                 \
             raft::device_matrix_view<const T, int64_t, raft::row_major> dataset) \
    -> cuvs::neighbors::vamana::index<T, IdxT>                                    \
  {                                                                               \
    return cuvs::neighbors::vamana::build<T, IdxT>(handle, params, dataset);      \
  }                                                                               \
                                                                                  \
  auto build(raft::resources const& handle,                                       \
             const cuvs::neighbors::vamana::index_params& params,                 \
             raft::host_matrix_view<const T, int64_t, raft::row_major> dataset)   \
    -> cuvs::neighbors::vamana::index<T, IdxT>                                    \
  {                                                                               \
    return cuvs::neighbors::vamana::build<T, IdxT>(handle, params, dataset);      \
  }

RAFT_INST_VAMANA_BUILD(half, uint32_t);

#undef RAFT_INST_VAMANA_BUILD

}  // namespace cuvs::neighbors::vamana
