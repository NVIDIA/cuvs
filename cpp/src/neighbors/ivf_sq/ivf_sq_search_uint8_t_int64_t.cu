/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/ivf_sq.hpp>

#include "ivf_sq_search.cuh"

namespace cuvs::neighbors::ivf_sq {

#define CUVS_INST_IVF_SQ_SEARCH(T, CodeT)                                            \
  void search(raft::resources const& handle,                                         \
              const cuvs::neighbors::ivf_sq::search_params& params,                  \
              const cuvs::neighbors::ivf_sq::index<CodeT>& index,                    \
              raft::device_matrix_view<const T, int64_t, raft::row_major> queries,   \
              raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors, \
              raft::device_matrix_view<float, int64_t, raft::row_major> distances,   \
              const cuvs::neighbors::filtering::base_filter& sample_filter)          \
  {                                                                                  \
    cuvs::neighbors::ivf_sq::detail::search(                                         \
      handle, params, index, queries, neighbors, distances, sample_filter);          \
  }

CUVS_INST_IVF_SQ_SEARCH(float, uint8_t);
CUVS_INST_IVF_SQ_SEARCH(half, uint8_t);

#undef CUVS_INST_IVF_SQ_SEARCH

}  // namespace cuvs::neighbors::ivf_sq
