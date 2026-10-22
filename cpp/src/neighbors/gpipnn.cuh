/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "detail/gpipnn/gpipnn_build.cuh"

#include <cuvs/neighbors/gpipnn.hpp>

namespace cuvs::neighbors::experimental::gpipnn {

template <typename DataT>
void build_graph(raft::resources const& res,
                 const index_params& params,
                 raft::device_matrix_view<const DataT, int64_t, raft::row_major> dataset,
                 raft::device_matrix_view<uint32_t, int64_t, raft::row_major> graph)
{
  detail::build_graph<DataT>(res, params, dataset, graph);
}

}  // namespace cuvs::neighbors::experimental::gpipnn
