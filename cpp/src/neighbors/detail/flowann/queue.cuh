/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/neighbors/flowann.hpp>

#include <raft/core/detail/macros.hpp>

#include <cuda/atomic>

#include <cstddef>
#include <cstdint>

namespace cuvs::neighbors::cagra::experimental::flowann::detail {

inline constexpr std::uint32_t queue_capacity     = 1u << 16;
inline constexpr std::uint32_t queue_mask         = queue_capacity - 1;
inline constexpr unsigned long long empty_command = 0ull;

static_assert((queue_capacity & queue_mask) == 0, "FlowANN queue capacity must be a power of two");

// Each word carries the row ID and its complement. Empty or inconsistent
// words fail validation; every uint32 row ID remains representable.
RAFT_INLINE_FUNCTION constexpr auto make_command(std::uint32_t row_id) -> unsigned long long
{
  return (static_cast<unsigned long long>(~row_id) << 32) | row_id;
}

RAFT_INLINE_FUNCTION constexpr auto command_ready(unsigned long long word) -> bool
{
  return static_cast<std::uint32_t>(word >> 32) == ~static_cast<std::uint32_t>(word);
}

struct alignas(128) queue_state {
  alignas(128) unsigned long long reservation_head;
  alignas(128) unsigned long long cpu_tail;
  alignas(128) unsigned long long gpu_tail;
  std::uint32_t capacity;
  std::uint32_t row_length;
};

/** Device pointers for one host-serviced cross-graph queue. */
struct queue_view {
  queue_state* state;
  std::uint32_t* rows;
  unsigned long long* command_words;
  unsigned long long* result_sequence;
  unsigned long long* released_sequence;
};

struct queue_reservation {
  unsigned long long sequence;
  std::uint32_t slot;
};

#ifdef __CUDACC__

RAFT_DEVICE_INLINE_FUNCTION auto load_counter(unsigned long long const* value) -> unsigned long long
{
  cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> counter(
    *const_cast<unsigned long long*>(value));
  return counter.load(cuda::memory_order_relaxed);
}

[[nodiscard]] RAFT_DEVICE_INLINE_FUNCTION auto reservation_at(queue_reservation first,
                                                              std::uint32_t offset)
  -> queue_reservation
{
  auto const sequence = first.sequence + offset;
  return {sequence, static_cast<std::uint32_t>(sequence) & queue_mask};
}

RAFT_DEVICE_INLINE_FUNCTION auto load_published_counter(unsigned long long const* value)
  -> unsigned long long
{
  cuda::atomic_ref<unsigned long long, cuda::thread_scope_system> counter(
    *const_cast<unsigned long long*>(value));
  return counter.load(cuda::memory_order_acquire);
}

/**
 * Reclaim only the contiguous released prefix. A failed CAS means another reclaimer advanced
 * the tail; restart from that tail so no slot is returned to producers twice.
 * This scalar fallback is used when a deferred producer runs out of queue capacity.
 */
RAFT_DEVICE_INLINE_FUNCTION void reclaim_released(queue_view queue)
{
  while (true) {
    auto const tail = load_counter(&queue.state->gpu_tail);
    auto const head = load_counter(&queue.state->reservation_head);
    auto end        = tail;
    while (end < head) {
      auto const slot = static_cast<std::uint32_t>(end) & queue_mask;
      cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> sequence(
        queue.released_sequence[slot]);
      if (sequence.load(cuda::memory_order_relaxed) != end + 1) { break; }
      ++end;
    }
    if (end == tail) { return; }
    if (atomicCAS(&queue.state->gpu_tail, tail, end) == tail) { return; }
  }
}

/**
 * Called by all 32 lanes of a warp after the CTA has consumed its pending rows. The first
 * unreleased sequence stops reclamation, including when the scan crosses the ring boundary.
 * The tail CAS also permits concurrent reclaimers and users of the ordinary release path.
 */
RAFT_DEVICE_INLINE_FUNCTION void reclaim_released_warp(queue_view queue)
{
  auto const lane = threadIdx.x & 31u;
  while (true) {
    auto const tail =
      __shfl_sync(0xffffffffu, lane == 0 ? load_counter(&queue.state->gpu_tail) : 0ull, 0);
    auto const head =
      __shfl_sync(0xffffffffu, lane == 0 ? load_counter(&queue.state->reservation_head) : 0ull, 0);
    auto end = tail;
    while (end < head) {
      auto const sequence = end + lane;
      auto const slot     = static_cast<std::uint32_t>(sequence) & queue_mask;
      cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> released(
        queue.released_sequence[slot]);
      auto const ready =
        sequence < head && released.load(cuda::memory_order_relaxed) == sequence + 1;
      auto const mask  = __ballot_sync(0xffffffffu, ready);
      auto const count = mask == 0xffffffffu ? 32u : static_cast<unsigned>(__ffs(~mask) - 1);
      end += count;
      if (count < 32) { break; }
    }
    if (end == tail) { return; }
    int success = lane == 0 && atomicCAS(&queue.state->gpu_tail, tail, end) == tail;
    if (__shfl_sync(0xffffffffu, success, 0)) { return; }
  }
}

/** Mark a consumed row; the caller must reclaim at capacity pressure and on completion. */
RAFT_DEVICE_INLINE_FUNCTION void release_deferred(queue_view queue, queue_reservation reservation)
{
  atomicExch(&queue.released_sequence[reservation.slot], reservation.sequence + 1);
}

/** Try to reserve a contiguous batch of queue slots atomically. */
RAFT_DEVICE_INLINE_FUNCTION auto try_reserve_batch(queue_view queue,
                                                   std::uint32_t count,
                                                   queue_reservation* first) -> bool
{
  auto* state = queue.state;
  if (count == 0 || count > state->capacity) { return false; }

  auto const head = load_counter(&state->reservation_head);
  auto const tail = load_counter(&state->gpu_tail);
  if (head - tail > state->capacity - count) { return false; }
  if (atomicCAS(&state->reservation_head, head, head + count) != head) { return false; }

  first->sequence = head;
  first->slot     = static_cast<std::uint32_t>(head) & queue_mask;
  return true;
}

RAFT_DEVICE_INLINE_FUNCTION auto try_reserve_deferred(queue_view queue,
                                                      std::uint32_t count,
                                                      queue_reservation* first) -> bool
{
  auto* state = queue.state;
  if (count == 0 || count > state->capacity) { return false; }

  auto const head = load_counter(&state->reservation_head);
  auto const tail = load_counter(&state->gpu_tail);
  if (head - tail > state->capacity - count) {
    reclaim_released(queue);
    return false;
  }
  if (atomicCAS(&state->reservation_head, head, head + count) != head) { return false; }

  first->sequence = head;
  first->slot     = static_cast<std::uint32_t>(head) & queue_mask;
  return true;
}

/** Try to reserve one queue slot without blocking the CTA indefinitely. */
RAFT_DEVICE_INLINE_FUNCTION auto try_reserve(queue_view queue, queue_reservation* reservation)
  -> bool
{
  return try_reserve_batch(queue, 1, reservation);
}

/** Atomically publish each complete request in a reserved batch. */
RAFT_DEVICE_INLINE_FUNCTION void submit_batch(queue_view queue,
                                              queue_reservation first,
                                              std::uint32_t const* row_ids,
                                              std::uint32_t count)
{
  // The atomically published word contains the complete request. No separate
  // payload needs a system fence. The host clears this word before publishing
  // the result; result acquisition and deferred reclamation gate slot reuse.
  for (std::uint32_t i = 0; i < count; ++i) {
    auto const reservation = reservation_at(first, i);
    cuda::atomic_ref<unsigned long long, cuda::thread_scope_system> command(
      queue.command_words[reservation.slot]);
    static_cast<void>(command.exchange(make_command(row_ids[i]), cuda::memory_order_relaxed));
  }
}

/** Publish one complete cross-graph row request atomically. */
RAFT_DEVICE_INLINE_FUNCTION void submit(queue_view queue,
                                        queue_reservation reservation,
                                        std::uint32_t row_id)
{
  submit_batch(queue, reservation, &row_id, 1);
}

RAFT_DEVICE_INLINE_FUNCTION auto result_ready(queue_view queue, queue_reservation reservation)
  -> bool
{
  return load_published_counter(&queue.result_sequence[reservation.slot]) ==
         reservation.sequence + 1;
}

RAFT_DEVICE_INLINE_FUNCTION auto result_row(queue_view queue, queue_reservation reservation)
  -> std::uint32_t const*
{
  return queue.rows + static_cast<std::size_t>(reservation.slot) * queue.state->row_length;
}

/**
 * Mark a consumed result as released and advance the global GPU tail over consecutive releases.
 */
RAFT_DEVICE_INLINE_FUNCTION void release(queue_view queue, queue_reservation reservation)
{
  atomicExch(&queue.released_sequence[reservation.slot], reservation.sequence + 1);

  while (true) {
    auto const tail = load_counter(&queue.state->gpu_tail);
    auto const slot = static_cast<std::uint32_t>(tail) & queue_mask;
    if (load_counter(&queue.released_sequence[slot]) != tail + 1) { break; }
    if (atomicCAS(&queue.state->gpu_tail, tail, tail + 1) != tail) { continue; }
  }
}

#endif  // __CUDACC__

#ifdef CUVS_ENABLE_FLOWANN_SEARCH
struct CUVS_EXPORT search_context_access {
  static auto device_queues(search_context const& context) noexcept -> queue_view*;
  static auto device_id(search_context const& context) noexcept -> int;
  static auto num_queues(search_context const& context) noexcept -> std::uint32_t;
  static auto row_count(search_context const& context) noexcept -> std::uint64_t;
  static auto row_length(search_context const& context) noexcept -> std::uint32_t;
  static void acquire_search(search_context& context);
  static void release_search(search_context& context);
  static auto is_paused(search_context const& context) noexcept -> bool;
  static auto pending_pause_requests(search_context const& context) noexcept -> std::size_t;
};
#endif

}  // namespace cuvs::neighbors::cagra::experimental::flowann::detail
