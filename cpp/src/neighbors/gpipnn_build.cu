/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gpipnn.cuh"
#include <cuvs/neighbors/gpipnn.hpp>

#include <cstdint>

/*
 * Every supported input type is instantiated in this one translation unit, rather than one file
 * each as most of cuVS does.
 *
 * CAGRA splits per type because its distance operators are templated on the data type, so each one
 * really does produce different kernels. None of GPiPNN's do: `detail::build_graph` converts the
 * caller's dataset to `half` before anything is launched, so the binary size and build time do not
 * increase with each additional type.
 */

namespace cuvs::neighbors::experimental::gpipnn {

auto is_supported() -> bool { return detail::gpipnn_arch_supported(); }

void build_graph(raft::resources const& res,
                 const index_params& params,
                 raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
                 raft::device_matrix_view<uint32_t, int64_t, raft::row_major> graph)
{
  cuvs::neighbors::experimental::gpipnn::build_graph<float>(res, params, dataset, graph);
}

void build_graph(raft::resources const& res,
                 const index_params& params,
                 raft::device_matrix_view<const half, int64_t, raft::row_major> dataset,
                 raft::device_matrix_view<uint32_t, int64_t, raft::row_major> graph)
{
  cuvs::neighbors::experimental::gpipnn::build_graph<half>(res, params, dataset, graph);
}

void build_graph(raft::resources const& res,
                 const index_params& params,
                 raft::device_matrix_view<const int8_t, int64_t, raft::row_major> dataset,
                 raft::device_matrix_view<uint32_t, int64_t, raft::row_major> graph)
{
  cuvs::neighbors::experimental::gpipnn::build_graph<int8_t>(res, params, dataset, graph);
}

void build_graph(raft::resources const& res,
                 const index_params& params,
                 raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> dataset,
                 raft::device_matrix_view<uint32_t, int64_t, raft::row_major> graph)
{
  cuvs::neighbors::experimental::gpipnn::build_graph<uint8_t>(res, params, dataset, graph);
}

}  // namespace cuvs::neighbors::experimental::gpipnn
