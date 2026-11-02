/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * FlowANN is described in "Disentangling Graph Dependencies for Efficient
 * Billion-Scale GPU Vector Search" (OSDI 2026).
 */

#pragma once

#include "kernel.cuh"

#include <neighbors/detail/cagra/hashmap.hpp>

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

#ifdef __CUDACC__
/** Submit a complete parent batch or restore its traversal state for retry. */
template <typename IndexT>
RAFT_DEVICE_INLINE_FUNCTION void submit_parent_requests(queue_view const& queue,
                                                        graph_state<IndexT>* state,
                                                        const IndexT* parent_indices,
                                                        const IndexT* internal_topk,
                                                        std::uint32_t search_width,
                                                        std::uint16_t sync_window,
                                                        IndexT* traversed_hashmap,
                                                        std::uint32_t traversed_hash_bitlen)
{
  constexpr IndexT index_msb =
    cuvs::neighbors::cagra::detail::utils::gen_index_msb_1_mask<IndexT>::value;
  std::uint32_t commands[max_search_width];
  std::uint32_t command_count = 0;
  for (std::uint32_t i = 0; i < search_width; ++i) {
    if (!state->parent_valid[i]) { continue; }
    commands[command_count++] = static_cast<std::uint32_t>(state->parent_ids[i]);
  }

  queue_reservation first{};
  bool reserved = command_count == 0;
  // One slot is reserved to distinguish a full pending ring from an empty one.
  // Do not publish commands unless every reservation can be tracked until release.
  if (command_count <= state->pending.available()) {
    for (std::uint32_t attempt = 0; attempt < 200 && !reserved; ++attempt) {
      reserved = try_reserve_deferred(queue, command_count, &first);
    }
  }
  if (command_count > 0 && reserved) {
    submit_batch(queue, first, commands, command_count);
    for (std::uint32_t i = 0; i < command_count; ++i) {
      state->pending.push({commands[i], reservation_at(first, i), sync_window});
    }
  } else if (!reserved) {
    // Parent selection marked these nodes as traversed. Retry their cross rows
    // after making progress on older requests rather than silently losing edges.
    for (std::uint32_t i = 0; i < search_width; ++i) {
      if (!state->parent_valid[i]) { continue; }
      const_cast<IndexT*>(internal_topk)[parent_indices[i]] &= ~index_msb;
      if (traversed_hashmap != nullptr) {
        cuvs::neighbors::cagra::detail::hashmap::remove(
          traversed_hashmap, traversed_hash_bitlen, state->parent_ids[i]);
      }
    }
  }
}
#endif

}  // namespace cuvs::neighbors::cagra::experimental::flowann::detail
