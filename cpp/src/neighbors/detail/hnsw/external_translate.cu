/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "external_translate.hpp"

#include <raft/core/device_mdspan.hpp>
#include <raft/linalg/map.cuh>

namespace cuvs::neighbors::hnsw::detail::external {
namespace {

struct translate_graph_id {
  const uint32_t* local_to_global;
  size_t mapping_size;

  __device__ uint32_t operator()(uint32_t local) const
  {
    return local < mapping_size ? local_to_global[local] : UINT32_MAX;
  }
};

}  // namespace

void translate_graph_ids(raft::resources const& res,
                         uint32_t* graph,
                         size_t graph_size,
                         const uint32_t* local_to_global,
                         size_t mapping_size)
{
  auto graph_view = raft::make_device_vector_view<uint32_t, size_t>(graph, graph_size);
  raft::linalg::map(res,
                    graph_view,
                    translate_graph_id{local_to_global, mapping_size},
                    raft::make_const_mdspan(graph_view));
}

}  // namespace cuvs::neighbors::hnsw::detail::external
