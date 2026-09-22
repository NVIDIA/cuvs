/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "queue.cuh"

#include <neighbors/detail/cagra/jit_lto_kernels/kernel_def.hpp>
#include <neighbors/detail/cagra/jit_lto_kernels/search_single_cta_jit.cuh>

#include <raft/core/detail/macros.hpp>
#include <raft/core/operators.hpp>
#include <raft/util/integer_utils.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>

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

RAFT_DEVICE_INLINE_FUNCTION auto load_packed_neighbor(const std::uint8_t* inner_graph,
                                                      std::uint64_t bit_position,
                                                      std::uint32_t neighbor,
                                                      std::uint32_t n_bits) -> std::uint32_t
{
  bit_position += static_cast<std::uint64_t>(neighbor) * n_bits;
  auto const byte_position = bit_position / 8;
  auto const bit_offset    = static_cast<std::uint32_t>(bit_position % 8);
  auto const byte_count    = (bit_offset + n_bits + 7) / 8;

  // The legacy 24-bit layout is byte aligned. Issue the three independent byte
  // reads without a runtime loop; never read beyond the encoded neighbor.
  if (n_bits == 24 && bit_offset == 0) {
    return static_cast<std::uint32_t>(inner_graph[byte_position]) |
           (static_cast<std::uint32_t>(inner_graph[byte_position + 1]) << 8) |
           (static_cast<std::uint32_t>(inner_graph[byte_position + 2]) << 16);
  }

  std::uint64_t packed = 0;
  for (std::uint32_t i = 0; i < byte_count; ++i) {
    packed |= static_cast<std::uint64_t>(inner_graph[byte_position + i]) << (8 * i);
  }
  auto const mask = n_bits == 32 ? std::uint64_t{0xffffffffu} : ((std::uint64_t{1} << n_bits) - 1);
  return static_cast<std::uint32_t>((packed >> bit_offset) & mask);
}

template <typename DataT, typename IndexT, typename DistanceT>
struct tiered_graph_provider {
  const std::uint8_t* inner_graph;
  std::uint32_t inner_row_length;
  std::uint32_t node_per_cacheline;
  std::uint32_t n_bits;
  const std::uint32_t* subgraph_offsets;
  const std::uint32_t* subgraph_ids;
  const IndexT* seeds;
  const std::uint32_t* seed_cross_graph;
  const IndexT* seed_lookup_keys;
  const std::uint32_t* seed_lookup_values;
  std::uint32_t seed_lookup_capacity;
  std::uint32_t num_seeds;
  queue_view queue;
  std::uint32_t graph_degree;
  std::uint32_t cross_graph_degree;
  IndexT graph_size;
  float sync_window_scale;
  std::uint32_t sync_drop_threshold;
  graph_state<IndexT>* state;

  RAFT_DEVICE_INLINE_FUNCTION void initialize() const
  {
    if (threadIdx.x == 0) {
      state->pending.initialize();
      state->previous_parent_position = 0;
      state->iteration                = 0;
      state->use_seed_cache           = false;
    }
  }

  [[nodiscard]] RAFT_DEVICE_INLINE_FUNCTION auto candidate_capacity(
    std::uint32_t search_width) const -> std::uint32_t
  {
    return (search_width + additional_buffer_width) * graph_degree;
  }

  [[nodiscard]] RAFT_DEVICE_INLINE_FUNCTION auto result_buffer_size(
    std::uint32_t base_size, std::uint32_t requested_seeds) const -> std::uint32_t
  {
    return seeds == nullptr ? base_size : std::max(base_size, requested_seeds);
  }

  [[nodiscard]] RAFT_DEVICE_INLINE_FUNCTION auto seed_pointer(const IndexT* seeds,
                                                              std::uint32_t,
                                                              std::uint32_t) const -> const IndexT*
  {
    return seeds;
  }

  RAFT_DEVICE_INLINE_FUNCTION auto seed_block_id(std::uint32_t cta_id,
                                                 std::uint32_t num_cta_per_query,
                                                 std::uint32_t row) const -> std::uint32_t
  {
    return seeds != nullptr && num_seeds > 0 ? 0 : cta_id + num_cta_per_query * row;
  }

  RAFT_DEVICE_INLINE_FUNCTION auto seed_num_blocks(std::uint32_t num_cta_per_query,
                                                   std::uint32_t num_queries,
                                                   std::uint32_t num_partitions) const
    -> std::uint32_t
  {
    return seeds != nullptr && num_seeds > 0 ? 1 : num_cta_per_query * num_queries * num_partitions;
  }

  [[nodiscard]] RAFT_DEVICE_INLINE_FUNCTION auto seed_position(IndexT key) const -> std::uint32_t
  {
    if (seed_lookup_keys == nullptr || seed_lookup_values == nullptr || seed_lookup_capacity == 0) {
      return num_seeds;
    }

    std::uint64_t hash = static_cast<std::uint64_t>(key);
    hash ^= hash >> 16;
    hash *= 0x85ebca77u;
    hash ^= hash >> 13;
    hash *= 0xc2b2ae3du;
    hash ^= hash >> 16;
    auto slot = static_cast<std::uint32_t>(hash % seed_lookup_capacity);

    constexpr auto empty_key = ~IndexT{0};
    for (std::uint32_t probe = 0; probe < seed_lookup_capacity; ++probe) {
      auto const stored = seed_lookup_keys[slot];
      if (stored == empty_key) { return num_seeds; }
      if (stored == key) { return seed_lookup_values[slot]; }
      slot = (slot + 1) % seed_lookup_capacity;
    }
    return num_seeds;
  }

  RAFT_DEVICE_INLINE_FUNCTION auto initial_candidate_count(std::uint32_t candidate_count,
                                                           std::uint32_t requested_seeds,
                                                           std::uint32_t block_id,
                                                           std::uint32_t num_blocks) const
    -> std::uint32_t
  {
    if (seeds == nullptr || requested_seeds == 0) { return candidate_count; }
    if (block_id >= requested_seeds) { return 0; }
    auto const seeds_for_block = (requested_seeds - block_id + num_blocks - 1) / num_blocks;
    // The result buffer includes space for the explicit seeds. The graph degree only limits
    // random initialization; applying it here silently discards later medoids.
    return seeds_for_block;
  }

  RAFT_DEVICE_INLINE_FUNCTION auto initial_distillation_count(std::uint32_t count,
                                                              std::uint32_t requested_seeds) const
    -> std::uint32_t
  {
    return seeds != nullptr && requested_seeds > 0 ? 1u : count;
  }

  RAFT_DEVICE_INLINE_FUNCTION auto expand(
    IndexT* result_indices,
    DistanceT* result_distances,
    const cuvs::neighbors::cagra::detail::dataset_descriptor_base_t<DataT, IndexT, DistanceT>*
      dataset_desc,
    IndexT* visited_hashmap,
    std::uint32_t visited_hash_bitlen,
    const IndexT* parent_indices,
    const IndexT* internal_topk,
    std::uint32_t search_width,
    IndexT* traversed_hashmap           = nullptr,
    std::uint32_t traversed_hash_bitlen = 0,
    int* result_position                = nullptr,
    int max_result_position             = 0) const -> std::uint32_t
  {
    using cuvs::neighbors::cagra::detail::device::team_sum;
    using cuvs::neighbors::cagra::detail::utils::gen_index_msb_1_mask;
    constexpr IndexT invalid_index = ~static_cast<IndexT>(0);
    constexpr IndexT index_msb     = gen_index_msb_1_mask<IndexT>::value;

    auto store_child = [&](IndexT child, std::uint32_t i) {
      if (child != invalid_index && cuvs::neighbors::cagra::detail::hashmap::insert(
                                      visited_hashmap, visited_hash_bitlen, child) == 0) {
        child = invalid_index;
      } else if (child != invalid_index && traversed_hashmap != nullptr &&
                 cuvs::neighbors::cagra::detail::hashmap::search<IndexT, 1>(
                   traversed_hashmap, traversed_hash_bitlen, child)) {
        child = invalid_index;
      }
      if (result_position == nullptr) {
        result_indices[i] = child;
      } else if (child != invalid_index) {
        auto const position      = atomicSub(result_position, 1) - 1;
        result_indices[position] = child;
      }
    };

    // A one-parent compacted expansion fits in one warp. Decode and deduplicate
    // its inner neighbors while warp 0 publishes requests. Compaction writes
    // only the candidate tail, beyond the retained parent pool.
    bool const overlap_inner =
      result_position != nullptr && search_width == 1 && graph_degree <= 32;

    // Read packed-row metadata independently while warp 0 publishes cross-graph requests.
    // These lanes derive parents directly; they do not depend on warp 0's shared writes.
    if (threadIdx.x >= 32 && threadIdx.x - 32 < search_width) {
      auto const p        = threadIdx.x - 32;
      auto const position = parent_indices[p];
      auto const valid =
        position != invalid_index && inner_graph != nullptr && inner_row_length != 0;
      std::uint32_t count = 0;
      std::uint64_t bits  = 0;
      std::uint32_t base  = 0;
      if (valid) {
        auto const parent = static_cast<IndexT>(internal_topk[position] & ~index_msb);
        auto const row    = static_cast<std::uint64_t>(parent) / node_per_cacheline;
        auto const col    = static_cast<std::uint32_t>(parent % node_per_cacheline);
        auto const start  = row * inner_row_length;
        count             = inner_graph[start + col];
        bits              = (start + node_per_cacheline) * 8;
        for (std::uint32_t j = 0; j < col; ++j) {
          bits += static_cast<std::uint64_t>(inner_graph[start + j]) * n_bits;
        }
        if (count > 0) { base = subgraph_offsets[subgraph_ids[parent]]; }
      }
      state->parent_inner_counts[p]     = count;
      state->parent_bit_offsets[p]      = bits;
      state->parent_subgraph_offsets[p] = base;
    }

    if (overlap_inner && threadIdx.x >= 32 && threadIdx.x < 64) {
      __syncwarp();
      auto const lane = threadIdx.x - 32;
      if (lane < state->parent_inner_counts[0]) {
        auto const local =
          load_packed_neighbor(inner_graph, state->parent_bit_offsets[0], lane, n_bits);
        auto const decoded = static_cast<std::uint64_t>(state->parent_subgraph_offsets[0]) + local;
        if (decoded < graph_size) { store_child(static_cast<IndexT>(decoded), lane); }
      }
    }

    if (threadIdx.x == 0) {
      auto const current_parent_position =
        parent_indices[0] == invalid_index ? 0u : static_cast<std::uint32_t>(parent_indices[0]);
      auto const sharp_drop =
        state->previous_parent_position > current_parent_position &&
        state->previous_parent_position - current_parent_position >= sync_drop_threshold;
      auto const window_value =
        sharp_drop
          ? 0u
          : std::max(search_width,
                     static_cast<std::uint32_t>(sync_window_scale * current_parent_position));
      auto const sync_window = static_cast<std::uint16_t>(std::min(window_value, 0xffffu));

      for (std::uint32_t i = 0; i < search_width; ++i) {
        auto const parent_position = parent_indices[i];
        auto const valid           = parent_position != invalid_index;
        auto const parent =
          valid ? static_cast<IndexT>(internal_topk[parent_position] & ~index_msb) : IndexT{};
        state->parent_ids[i]   = parent;
        state->parent_valid[i] = valid;
      }

      state->use_seed_cache = state->iteration == 0 && seeds != nullptr &&
                              seed_cross_graph != nullptr && seed_lookup_keys != nullptr &&
                              seed_lookup_values != nullptr && seed_lookup_capacity > 0 &&
                              num_seeds > 0;
      if (state->use_seed_cache) {
        for (std::uint32_t i = 0; i < search_width; ++i) {
          if (!state->parent_valid[i]) { continue; }
          auto const seed_position        = this->seed_position(state->parent_ids[i]);
          state->parent_seed_positions[i] = seed_position;
          if (seed_position == num_seeds) {
            state->use_seed_cache = false;
            break;
          }
        }
      }

      if (!state->use_seed_cache) {
        std::uint32_t commands[max_search_width];
        std::uint32_t command_count = 0;
        for (std::uint32_t i = 0; i < search_width; ++i) {
          if (!state->parent_valid[i]) { continue; }
          commands[command_count++] = static_cast<std::uint32_t>(state->parent_ids[i]);
        }

        queue_reservation first{};
        bool reserved = command_count == 0;
        for (std::uint32_t attempt = 0; attempt < 200 && !reserved; ++attempt) {
          reserved = try_reserve_deferred(queue, command_count, &first);
        }
        if (command_count > 0 && reserved) {
          submit_batch(queue, first, commands, command_count);
          for (std::uint32_t i = 0; i < command_count; ++i) {
            state->pending.push({commands[i], reservation_at(first, i), sync_window});
          }
        }
      }
    }
    // The one-parent path can consume ready rows while warp 1 prepares inner
    // neighbors. Warp 0 reads the raw inner count independently below; the
    // following join still protects all candidate writes and row descriptors.
    if (!overlap_inner) { __syncthreads(); }

    auto decode_inner = [&](std::uint32_t i) {
      IndexT child               = invalid_index;
      auto local                 = i;
      std::uint32_t parent_index = 0;
      for (; parent_index < search_width; ++parent_index) {
        auto const count = state->parent_inner_counts[parent_index];
        if (local < count) { break; }
        local -= count;
      }
      if (parent_index < search_width && state->parent_valid[parent_index]) {
        auto const local_neighbor =
          load_packed_neighbor(inner_graph, state->parent_bit_offsets[parent_index], local, n_bits);
        auto const decoded =
          static_cast<std::uint64_t>(state->parent_subgraph_offsets[parent_index]) + local_neighbor;
        if (decoded < graph_size) { child = static_cast<IndexT>(decoded); }
      }
      return child;
    };

    // The caller has acquired this row's ready sequence. A partial row remains
    // pending until the rest of its neighbors fit in a later expansion.
    auto consume_cross_row = [&](std::uint32_t& remaining) {
      auto const& entry                     = state->pending.front();
      auto const* row                       = result_row(queue, entry.reservation);
      auto edge_count                       = std::min(row[0], cross_graph_degree);
      auto const consumed                   = state->pending.consumed;
      edge_count                            = edge_count > consumed ? edge_count - consumed : 0;
      auto const take                       = std::min(edge_count, remaining);
      auto const ready                      = state->ready_cross_count;
      state->ready_cross_data[ready]        = row + 1 + consumed;
      state->ready_cross_edge_counts[ready] = take;
      ++state->ready_cross_count;
      remaining -= take;

      if (take < edge_count) {
        state->pending.consumed += take;
        return false;
      }
      state->pending.consumed          = 0;
      state->ready_reservations[ready] = entry.reservation;
      state->pending.pop();
      return true;
    };

    if (threadIdx.x == 0) {
      std::uint32_t inner_count = 0;
      if (overlap_inner) {
        if (state->parent_valid[0] && inner_graph != nullptr && inner_row_length != 0) {
          auto const parent = state->parent_ids[0];
          auto const row    = static_cast<std::uint64_t>(parent) / node_per_cacheline;
          auto const col    = static_cast<std::uint32_t>(parent % node_per_cacheline);
          inner_count       = inner_graph[row * inner_row_length + col];
        }
      } else {
        for (std::uint32_t i = 0; i < search_width; ++i) {
          inner_count += state->parent_inner_counts[i];
        }
      }
      state->total_inner_children = inner_count;

      auto remaining           = candidate_capacity(search_width) - inner_count;
      state->ready_cross_count = 0;

      if (state->use_seed_cache) {
        auto const row_length = cross_graph_degree + 1;
        for (std::uint32_t i = 0;
             i < search_width && state->ready_cross_count < max_search_width && remaining > 0;
             ++i) {
          if (!state->parent_valid[i]) { continue; }
          auto const* row = seed_cross_graph +
                            static_cast<std::size_t>(state->parent_seed_positions[i]) * row_length;
          auto const take = std::min(std::min(row[0], cross_graph_degree), remaining);
          if (take == 0) { continue; }
          auto const ready                      = state->ready_cross_count;
          state->ready_cross_data[ready]        = row + 1;
          state->ready_cross_edge_counts[ready] = take;
          ++state->ready_cross_count;
          remaining -= take;
        }
      } else {
        auto available = state->pending.available();
        auto deferred  = state->pending.size();
        while (!state->pending.empty() && state->ready_cross_count < max_search_width &&
               remaining > 0) {
          auto& entry    = state->pending.front();
          auto need_wait = available < search_width || deferred > entry.sync_window;
          if (need_wait) {
            while (!result_ready(queue, entry.reservation)) {}
            ++available;
            --deferred;
          } else {
            if (!result_ready(queue, entry.reservation)) { break; }
          }

          if (!consume_cross_row(remaining)) { break; }
        }
      }
      state->total_children = candidate_capacity(search_width) - remaining;
      state->previous_parent_position =
        parent_indices[0] == invalid_index ? 0u : static_cast<std::uint32_t>(parent_indices[0]);
    } else if (!overlap_inner && threadIdx.x >= 32) {
      std::uint32_t inner_count = 0;
      for (std::uint32_t p = 0; p < search_width; ++p) {
        inner_count += state->parent_inner_counts[p];
      }
      for (std::uint32_t i = threadIdx.x - 32; i < inner_count; i += blockDim.x - 32) {
        store_child(decode_inner(i), i);
      }
    }
    __syncthreads();

    // In the compacted multi-CTA path, include rows that became ready during
    // inner-neighbor work. This pass never waits or expands the candidate budget.
    // The single-CTA path retains its single readiness pass.
    if (result_position != nullptr) {
      if (threadIdx.x == 0 && !state->use_seed_cache) {
        auto remaining = candidate_capacity(search_width) - state->total_children;
        while (!state->pending.empty() && state->ready_cross_count < max_search_width &&
               remaining > 0) {
          if (!result_ready(queue, state->pending.front().reservation)) { break; }
          if (!consume_cross_row(remaining)) { break; }
        }
        state->total_children = candidate_capacity(search_width) - remaining;
      }
      __syncthreads();
    }

    for (std::uint32_t i = threadIdx.x + state->total_inner_children; i < state->total_children;
         i += blockDim.x) {
      IndexT child = invalid_index;
      auto local   = i - state->total_inner_children;
      for (std::uint32_t ready = 0; ready < state->ready_cross_count; ++ready) {
        auto const count = state->ready_cross_edge_counts[ready];
        if (local < count) {
          auto const decoded = state->ready_cross_data[ready][local];
          if (decoded < graph_size) { child = static_cast<IndexT>(decoded); }
          break;
        }
        local -= count;
      }
      store_child(child, i);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      if (!state->use_seed_cache) {
        auto release_count = state->ready_cross_count;
        if (state->pending.consumed != 0 && release_count > 0) { --release_count; }
        for (std::uint32_t i = 0; i < release_count; ++i) {
          release_deferred(queue, state->ready_reservations[i]);
        }
      }
      ++state->iteration;
    }
    // PQ uses only the copied child IDs. The caller joins the CTA after expansion,
    // before any thread can reuse iteration state or consume another queue row.

    auto const team_size_bits = dataset_desc->team_size_bitshift_from_smem();
    auto const offset =
      result_position == nullptr ? 0u : static_cast<std::uint32_t>(result_position[0]);
    auto const max_i = raft::round_up_safe(
      state->total_children, cuvs::neighbors::cagra::detail::device::warp_size >> team_size_bits);
    auto const args      = dataset_desc->args.load();
    auto const lead_lane = (threadIdx.x & ((1u << team_size_bits) - 1u)) == 0;
    for (std::uint32_t i = threadIdx.x >> team_size_bits; i < max_i;
         i += blockDim.x >> team_size_bits) {
      auto const position = i + offset;
      auto const valid    = i < state->total_children &&
                         (result_position == nullptr || position < max_result_position) &&
                         result_indices[position] != invalid_index;
      auto const distance = team_sum(
        valid
          ? cuvs::neighbors::cagra::detail::compute_distance_per_thread<DataT, IndexT, DistanceT>(
              args, result_indices[position])
          : (lead_lane ? raft::upper_bound<DistanceT>() : DistanceT{}),
        team_size_bits);
      __syncwarp();
      if (i < state->total_children &&
          (result_position == nullptr || position < max_result_position) && lead_lane) {
        result_distances[position] = distance;
      }
    }
    return state->total_children;
  }

  RAFT_DEVICE_INLINE_FUNCTION auto expand(
    IndexT* result_indices,
    DistanceT* result_distances,
    const cuvs::neighbors::cagra::detail::dataset_descriptor_base_t<DataT, IndexT, DistanceT>*
      dataset_desc,
    IndexT* visited_hashmap,
    std::uint32_t visited_hash_bitlen,
    IndexT* traversed_hashmap,
    std::uint32_t traversed_hash_bitlen,
    const IndexT* parent_indices,
    const IndexT* internal_topk,
    int* result_position,
    int max_result_position) const -> std::uint32_t
  {
    return expand(result_indices,
                  result_distances,
                  dataset_desc,
                  visited_hashmap,
                  visited_hash_bitlen,
                  parent_indices,
                  internal_topk,
                  1,
                  traversed_hashmap,
                  traversed_hash_bitlen,
                  result_position,
                  max_result_position);
  }

  RAFT_DEVICE_INLINE_FUNCTION void finalize() const
  {
    if (threadIdx.x == 0) {
      while (!state->pending.empty()) {
        auto const reservation = state->pending.front().reservation;
        while (!result_ready(queue, reservation)) {}
        release_deferred(queue, reservation);
        state->pending.pop();
      }
      state->pending.consumed = 0;
    }
    __syncthreads();
    if (threadIdx.x < 32) { reclaim_released_warp(queue); }
    __syncthreads();
  }
};

template <bool TopkByBitonicSort,
          bool BitonicSortAndMergeMultiWarps,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT>
__device__ void search_kernel_jit(
  uintptr_t result_indices_ptr,
  DistanceT* result_distances_ptr,
  std::uint32_t top_k,
  const DataT* queries_ptr,
  const std::uint8_t* inner_graph,
  std::uint32_t inner_row_length,
  std::uint32_t node_per_cacheline,
  std::uint32_t n_bits,
  const std::uint32_t* subgraph_offsets,
  const std::uint32_t* subgraph_ids,
  queue_view* queues,
  std::uint32_t num_queues,
  std::uint32_t graph_degree,
  std::uint32_t cross_graph_degree,
  const SourceIndexT* source_indices,
  unsigned num_distillation,
  std::uint64_t rand_xor_mask,
  const IndexT* seeds,
  std::uint32_t num_seeds,
  const std::uint32_t* seed_cross_graph,
  const IndexT* seed_lookup_keys,
  const std::uint32_t* seed_lookup_values,
  std::uint32_t seed_lookup_capacity,
  IndexT* visited_hashmap,
  std::uint32_t max_candidates,
  std::uint32_t max_itopk,
  std::uint32_t internal_topk,
  std::uint32_t search_width,
  std::uint32_t min_iterations,
  std::uint32_t max_iterations,
  std::uint32_t* num_executed_iterations,
  std::uint32_t hash_bitlen,
  std::uint32_t small_hash_bitlen,
  std::uint32_t small_hash_reset_interval,
  std::uint32_t query_id_offset,
  const cuvs::neighbors::cagra::detail::dataset_descriptor_base_t<DataT, IndexT, DistanceT>*
    dataset_desc,
  IndexT graph_size,
  float sync_window_scale,
  std::uint32_t sync_drop_threshold,
  cuvs::neighbors::cagra::detail::cagra_sample_filter<SourceIndexT> filter_payload)
{
  __shared__ graph_state<IndexT> state;
  auto const query_id = blockIdx.y;
  auto const queue_id = query_id % num_queues;
  tiered_graph_provider<DataT, IndexT, DistanceT> provider{inner_graph,
                                                           inner_row_length,
                                                           node_per_cacheline,
                                                           n_bits,
                                                           subgraph_offsets,
                                                           subgraph_ids,
                                                           seeds,
                                                           seed_cross_graph,
                                                           seed_lookup_keys,
                                                           seed_lookup_values,
                                                           seed_lookup_capacity,
                                                           num_seeds,
                                                           queues[queue_id],
                                                           graph_degree,
                                                           cross_graph_degree,
                                                           graph_size,
                                                           sync_window_scale,
                                                           sync_drop_threshold,
                                                           &state};
  cuvs::neighbors::cagra::detail::single_cta_search::search_core<TopkByBitonicSort,
                                                                 BitonicSortAndMergeMultiWarps,
                                                                 DataT,
                                                                 IndexT,
                                                                 DistanceT,
                                                                 SourceIndexT>(
    result_indices_ptr,
    result_distances_ptr,
    top_k,
    queries_ptr,
    provider,
    graph_degree,
    source_indices,
    num_distillation,
    rand_xor_mask,
    seeds,
    num_seeds,
    visited_hashmap,
    max_candidates,
    max_itopk,
    internal_topk,
    search_width,
    min_iterations,
    max_iterations,
    num_executed_iterations,
    hash_bitlen,
    small_hash_bitlen,
    small_hash_reset_interval,
    query_id,
    query_id_offset,
    dataset_desc,
    filter_payload,
    graph_size);
}

}  // namespace cuvs::neighbors::cagra::experimental::flowann::detail
