/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>

namespace cuvs::neighbors::cagra::helpers {

/** The batch size for the CAGRA optimize stage. */
constexpr static size_t kOptimizeBatchSize = 256 * 1024;

/** The query chunk size (search batch size) of the iterative CAGRA graph build. */
constexpr static size_t kIterativeBuildChunkSize = 8192;

/** The largest node sample of a graph bridging pass; its candidate pools are brute-force results.
 */
constexpr static size_t kBridgeMaxSample = size_t{1} << 17;

/** The device memory of one batch of candidate pools in a graph bridging pass. */
constexpr static size_t kBridgePoolBatchBytes = size_t{256} << 20;

/** Whether builds bridge disconnected graphs, which replaces the MST of guarantee_connectivity. */
bool bridge_enabled();

/** Peak device memory of bridging a disconnected graph after optimize. */
size_t bridge_workspace_size(size_t n_rows, size_t dim, size_t dtype_size, size_t graph_degree);

}  // namespace cuvs::neighbors::cagra::helpers
