/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuvs/core/export.hpp>
#include <cuvs/distance/distance.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <cuda_fp16.h>

#include <cstdint>

namespace cuvs::neighbors::cagra::detail::graph {

// kern_sort capacity is WarpSize * numElementsPerThread; largest specialization uses 32.
inline constexpr uint64_t kMaxSortDegree = 32 * 32;

#define CUVS_DECL_CAGRA_GRAPH_SORT(DataT)                              \
  CUVS_EXPORT void launch_sort_knn_graph(raft::resources const& res,   \
                                         cuvs::distance::DistanceType, \
                                         DataT const* dataset,         \
                                         uint32_t dataset_size,        \
                                         uint32_t dataset_dim,         \
                                         uint32_t* knn_graph,          \
                                         uint32_t graph_degree)

CUVS_DECL_CAGRA_GRAPH_SORT(float);
CUVS_DECL_CAGRA_GRAPH_SORT(half);
CUVS_DECL_CAGRA_GRAPH_SORT(int8_t);
CUVS_DECL_CAGRA_GRAPH_SORT(uint8_t);

#undef CUVS_DECL_CAGRA_GRAPH_SORT

/** Run the existing CAGRA optimizer through one compiled instantiation instead of rematerializing
 *  its reverse-graph, prune, merge, and MST kernels in every Fastener dtype TU. */
CUVS_EXPORT void optimize_device_graph(
  raft::resources const& res,
  raft::device_matrix_view<uint32_t, int64_t, raft::row_major> knn_graph,
  raft::device_matrix_view<uint32_t, int64_t, raft::row_major> output_graph,
  bool guarantee_connectivity);

}  // namespace cuvs::neighbors::cagra::detail::graph
