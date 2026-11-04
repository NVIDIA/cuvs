/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuvs/core/export.hpp>

#include <raft/core/resources.hpp>

#include <cstddef>
#include <cstdint>

namespace cuvs::neighbors::hnsw::detail::external {

CUVS_EXPORT void translate_graph_ids(raft::resources const& res,
                                     uint32_t* graph,
                                     size_t graph_size,
                                     const uint32_t* local_to_global,
                                     size_t mapping_size);

}  // namespace cuvs::neighbors::hnsw::detail::external
