/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <raft/core/resources.hpp>

namespace cuvs::neighbors::hnsw::detail {
// Gather only selected device rows to a compact host matrix using bounded device staging.
// Row IDs must be valid source rows; host_output must hold rows * row_bytes bytes.
void gather_layered_hnsw_vectors(raft::resources const& res,
                                 const void* source,
                                 size_t source_stride_bytes,
                                 size_t row_bytes,
                                 const size_t* host_row_ids,
                                 size_t rows,
                                 void* host_output);
}  // namespace cuvs::neighbors::hnsw::detail
