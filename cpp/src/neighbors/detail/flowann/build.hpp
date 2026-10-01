/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/error.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace cuvs::neighbors::cagra::experimental::flowann::detail {

struct uniform_layout_plan {
  std::uint16_t n_bits;
  std::uint32_t resident_degree;
  std::uint32_t cross_degree;
  std::uint32_t inner_row_bytes;
  std::uint64_t inner_graph_bytes;
};

inline auto required_id_bits(std::uint64_t n_rows) -> std::uint16_t
{
  RAFT_EXPECTS(n_rows > 0, "FlowANN cannot plan a graph layout for zero rows");
  auto value = n_rows - 1;
  auto bits  = std::uint16_t{0};
  while (value != 0) {
    ++bits;
    value >>= 1;
  }
  return std::max<std::uint16_t>(1, bits);
}

inline auto packed_row_bytes(std::uint16_t node_per_cacheline,
                             std::uint16_t n_bits,
                             std::uint32_t resident_degree) -> std::uint32_t
{
  RAFT_EXPECTS(node_per_cacheline > 0, "FlowANN node_per_cacheline must be positive");
  RAFT_EXPECTS(n_bits > 0 && n_bits <= 32, "FlowANN packed ID width must be in [1, 32]");
  if (resident_degree == 0) { return 0; }
  auto const header_bits = static_cast<std::uint64_t>(node_per_cacheline) * 8;
  auto const edge_bits   = static_cast<std::uint64_t>(node_per_cacheline) * resident_degree *
                         static_cast<std::uint64_t>(n_bits);
  auto const bytes = (header_bits + edge_bits + 7) / 8;
  RAFT_EXPECTS(bytes <= std::numeric_limits<std::uint32_t>::max(),
               "FlowANN packed row length exceeds uint32");
  return static_cast<std::uint32_t>(bytes);
}

inline auto plan_uniform_layout(std::uint64_t n_rows,
                                std::uint32_t graph_degree,
                                std::uint16_t node_per_cacheline,
                                std::uint64_t device_graph_budget_bytes) -> uniform_layout_plan
{
  RAFT_EXPECTS(graph_degree > 0 && graph_degree <= std::numeric_limits<std::uint8_t>::max(),
               "FlowANN complete graph degree must be in [1, 255]");
  RAFT_EXPECTS(node_per_cacheline > 0, "FlowANN node_per_cacheline must be positive");
  RAFT_EXPECTS(n_rows <= std::numeric_limits<std::uint32_t>::max(),
               "FlowANN graph row count must fit uint32 IDs");

  auto const n_bits      = required_id_bits(n_rows);
  auto const packed_rows = n_rows / node_per_cacheline + (n_rows % node_per_cacheline != 0);

  for (auto resident = graph_degree; resident > 0; --resident) {
    auto const row_bytes = packed_row_bytes(node_per_cacheline, n_bits, resident);
    RAFT_EXPECTS(packed_rows <= std::numeric_limits<std::uint64_t>::max() / row_bytes,
                 "FlowANN packed graph size overflows uint64");
    auto const total_bytes = packed_rows * row_bytes;
    if (total_bytes <= device_graph_budget_bytes) {
      return {n_bits, resident, graph_degree - resident, row_bytes, total_bytes};
    }
  }

  return {n_bits, 0, graph_degree, 0, 0};
}

/** Budgeted grouped rows share capacity between nodes; do not cap them at average local degree. */
inline auto budgeted_grouped_row_bytes(std::uint64_t n_rows,
                                       std::uint32_t graph_degree,
                                       std::uint16_t nodes_per_row,
                                       std::uint16_t n_bits,
                                       std::uint64_t budget_bytes) -> std::uint32_t
{
  RAFT_EXPECTS(n_rows > 0 && nodes_per_row > 0, "FlowANN grouped layout must have rows and nodes");
  RAFT_EXPECTS(graph_degree > 0 && graph_degree <= std::numeric_limits<std::uint8_t>::max(),
               "FlowANN grouped degree must be in [1, 255]");
  auto const rows  = n_rows / nodes_per_row + (n_rows % nodes_per_row != 0);
  auto const bytes = std::min<std::uint64_t>(budget_bytes / rows,
                                             packed_row_bytes(nodes_per_row, n_bits, graph_degree));
  // A header without space for even one edge is equivalent to an empty resident graph.
  if (bytes < nodes_per_row + (n_bits + 7u) / 8u) { return 0; }
  return static_cast<std::uint32_t>(bytes);
}

}  // namespace cuvs::neighbors::cagra::experimental::flowann::detail
