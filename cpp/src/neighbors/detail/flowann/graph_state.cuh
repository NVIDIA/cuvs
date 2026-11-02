/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * FlowANN is described in "Disentangling Graph Dependencies for Efficient
 * Billion-Scale GPU Vector Search" (OSDI 2026).
 */

#pragma once

#include "kernel.cuh"

namespace cuvs::neighbors::cagra::experimental::flowann::detail {

inline constexpr std::uint32_t pending_capacity = 512;
inline constexpr std::uint32_t pending_mask     = pending_capacity - 1;

static_assert((pending_capacity & pending_mask) == 0,
              "FlowANN pending capacity must be a power of two");

struct pending_entry {
  std::uint32_t parent;
  queue_reservation reservation;
  std::uint16_t sync_window;
};

struct pending_queue {
  std::uint16_t head;
  std::uint16_t tail;
  std::uint32_t consumed;
  pending_entry entries[pending_capacity];

  RAFT_DEVICE_INLINE_FUNCTION void initialize()
  {
    head     = 0;
    tail     = 0;
    consumed = 0;
  }

  [[nodiscard]] RAFT_DEVICE_INLINE_FUNCTION auto size() const -> std::uint16_t
  {
    return static_cast<std::uint16_t>((head - tail) & pending_mask);
  }

  [[nodiscard]] RAFT_DEVICE_INLINE_FUNCTION auto available() const -> std::uint16_t
  {
    return static_cast<std::uint16_t>(pending_mask - size());
  }

  [[nodiscard]] RAFT_DEVICE_INLINE_FUNCTION auto empty() const -> bool { return head == tail; }

  RAFT_DEVICE_INLINE_FUNCTION void push(pending_entry entry)
  {
    entries[head] = entry;
    head          = static_cast<std::uint16_t>((head + 1) & pending_mask);
  }

  [[nodiscard]] RAFT_DEVICE_INLINE_FUNCTION auto front() -> pending_entry& { return entries[tail]; }

  RAFT_DEVICE_INLINE_FUNCTION void pop()
  {
    tail = static_cast<std::uint16_t>((tail + 1) & pending_mask);
  }
};

template <typename IndexT>
struct graph_state {
  pending_queue pending;
  IndexT parent_ids[max_search_width];
  bool parent_valid[max_search_width];
  std::uint32_t parent_seed_positions[max_search_width];
  std::uint32_t parent_inner_counts[max_search_width];
  std::uint64_t parent_bit_offsets[max_search_width];
  std::uint32_t parent_subgraph_offsets[max_search_width];
  const std::uint32_t* ready_cross_data[max_search_width];
  std::uint32_t ready_cross_edge_counts[max_search_width];
  queue_reservation ready_reservations[max_search_width];
  std::uint32_t ready_cross_count;
  std::uint32_t total_inner_children;
  std::uint32_t total_children;
  std::uint32_t previous_parent_position;
  std::uint32_t iteration;
  bool use_seed_cache;
};

}  // namespace cuvs::neighbors::cagra::experimental::flowann::detail
