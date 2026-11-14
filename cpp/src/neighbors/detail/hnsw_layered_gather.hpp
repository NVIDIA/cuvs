/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <raft/core/resources.hpp>

namespace cuvs::neighbors::hnsw::detail {
/**
 * @brief Copy selected device vectors to host memory for HNSW upper-layer construction.
 *
 * This wrapper manages host/device transfers and bounded device staging. Row selection is a
 * generic gather operation; no HNSW topology is interpreted here. Only the logical row bytes
 * are copied, excluding any padding in the source. Copies use the resource stream and complete
 * before this function returns. A request with zero rows performs no work.
 *
 * @param[in] res Resources supplying the CUDA stream and current device.
 * @param[in] source Device-accessible source storage on the current device. Each selected row
 * must have at least `row_bytes` readable bytes at `source + row_id * source_stride_bytes`.
 * @param[in] source_stride_bytes Physical distance between source rows in bytes; must be at
 * least `row_bytes` for a nonempty request.
 * @param[in] row_bytes Logical bytes per vector; must be positive for a nonempty request.
 * @param[in] host_row_ids Host-resident array of `rows` source-row indices. Every index must
 * identify a valid source row; bounds are the caller's responsibility. Order and duplicates
 * are preserved.
 * @param[in] rows Number of vectors to gather, not the source row count.
 * @param[out] host_output Host-resident writable storage of at least `rows * row_bytes` bytes.
 * Output rows are contiguous, with no padding. Input and output storage must remain valid
 * until the call returns.
 */
void gather_layered_hnsw_vectors(raft::resources const& res,
                                 const void* source,
                                 size_t source_stride_bytes,
                                 size_t row_bytes,
                                 const size_t* host_row_ids,
                                 size_t rows,
                                 void* host_output);
}  // namespace cuvs::neighbors::hnsw::detail
