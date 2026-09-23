/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/flowann.hpp>

#include <neighbors/detail/flowann/queue.cuh>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_setter.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

namespace flowann = cuvs::neighbors::cagra::experimental::flowann;

namespace {

RAFT_KERNEL queue_round_trip(flowann::detail::queue_view* queues,
                             std::uint32_t num_queues,
                             std::uint32_t row_count,
                             std::uint32_t row_length,
                             std::uint32_t* output)
{
  if (threadIdx.x != 0) { return; }

  auto const command = blockIdx.x % row_count;
  auto queue         = queues[blockIdx.x % num_queues];
  flowann::detail::queue_reservation reservation{};
  while (!flowann::detail::try_reserve(queue, &reservation)) {}
  flowann::detail::submit(queue, reservation, command);
  while (!flowann::detail::result_ready(queue, reservation)) {}
  auto const* row = flowann::detail::result_row(queue, reservation);
  for (std::uint32_t i = 0; i < row_length; ++i) {
    output[static_cast<std::size_t>(blockIdx.x) * row_length + i] = row[i];
  }
  flowann::detail::release(queue, reservation);
}

RAFT_KERNEL mixed_release_queue_round_trip(flowann::detail::queue_view* queues,
                                           std::uint32_t num_queues,
                                           std::uint32_t row_count,
                                           std::uint32_t row_length,
                                           std::uint32_t round,
                                           std::uint32_t batch,
                                           std::uint32_t* output)
{
  auto queue = queues[blockIdx.x % num_queues];
  if (threadIdx.x == 0) {
    std::uint32_t ids[7];
    for (std::uint32_t i = 0; i < batch; ++i) {
      ids[i] = (blockIdx.x * batch + i + round) % row_count;
    }
    flowann::detail::queue_reservation first{};
    while (!flowann::detail::try_reserve_deferred(queue, batch, &first)) {}
    flowann::detail::submit_batch(queue, first, ids, batch);
    for (std::uint32_t i = 0; i < batch; ++i) {
      auto const reservation = flowann::detail::reservation_at(first, i);
      while (!flowann::detail::result_ready(queue, reservation)) {}
      auto const* row = flowann::detail::result_row(queue, reservation);
      for (std::uint32_t col = 0; col < row_length; ++col) {
        output[(static_cast<std::size_t>(blockIdx.x) * batch + i) * row_length + col] = row[col];
      }
      if (((blockIdx.x + i) & 1u) == 0) {
        flowann::detail::release_deferred(queue, reservation);
      } else {
        flowann::detail::release(queue, reservation);
      }
    }
  }
  __syncthreads();
  flowann::detail::reclaim_released_warp(queue);
}

TEST(FlowannQueue, CommandEncodingPreservesRowIdsAndRejectsPartialPublication)
{
  EXPECT_FALSE(flowann::detail::command_ready(flowann::detail::empty_command));
  for (std::uint32_t row : {0u, 1u, 65535u, 65536u, 0xabcdefu, 0x7fffffffu, 0xffffffffu}) {
    auto const word = flowann::detail::make_command(row);
    ASSERT_TRUE(flowann::detail::command_ready(word));
    EXPECT_EQ(static_cast<std::uint32_t>(word), row);
    // Reading either 32-bit half before the other is published must not
    // produce a ready word carrying a different request.
    for (auto partial : {word & 0xffffffff00000000ull, word & 0xffffffffull}) {
      if (flowann::detail::command_ready(partial)) { EXPECT_EQ(partial, word); }
    }
    for (unsigned bit = 0; bit < 64; ++bit) {
      EXPECT_FALSE(flowann::detail::command_ready(word ^ (1ull << bit)));
    }
  }
}

void check_round_trip_pause_resume(std::uint32_t num_queues)
{
  raft::resources res;
  constexpr std::uint32_t row_count  = 17;
  constexpr std::uint32_t row_length = 9;
  constexpr std::uint32_t commands   = 257;

  auto cross_graph = raft::make_host_matrix<std::uint32_t, int64_t>(row_count, row_length);
  for (std::uint32_t row = 0; row < row_count; ++row) {
    for (std::uint32_t col = 0; col < row_length; ++col) {
      cross_graph(row, col) = row * 1000 + col;
    }
  }

  flowann::queue_params params{};
  params.num_queues         = num_queues;
  params.empty_pause        = 8;
  params.collect_statistics = true;
  flowann::search_context context(raft::make_const_mdspan(cross_graph.view()), params);
  auto output = raft::make_device_matrix<std::uint32_t, int64_t>(res, commands, row_length);

  context.resume();
  queue_round_trip<<<commands, 1, 0, raft::resource::get_cuda_stream(res).get()>>>(
    flowann::detail::search_context_access::device_queues(context),
    context.num_queues(),
    row_count,
    row_length,
    output.data_handle());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  raft::resource::sync_stream(res);
  context.pause();

  auto host_output = raft::make_host_matrix<std::uint32_t, int64_t>(commands, row_length);
  raft::copy(host_output.data_handle(),
             output.data_handle(),
             output.size(),
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  for (std::uint32_t command = 0; command < commands; ++command) {
    auto const row = command % row_count;
    for (std::uint32_t col = 0; col < row_length; ++col) {
      EXPECT_EQ(host_output(command, col), cross_graph(row, col));
    }
  }
  EXPECT_EQ(context.statistics().processed_commands, commands);

  context.resume();
  context.pause();
}

TEST(FlowannQueue, RoundTripPauseResume) { check_round_trip_pause_resume(2); }

// Also run this test alone in a fresh process to exercise lazy kernel loading
// while many idle CUDA-copy pollers are already issuing transfers.
TEST(FlowannQueue, ColdLaunchWithManyPollers) { check_round_trip_pause_resume(44); }

TEST(FlowannQueue, PollersUseContextCudaDevice)
{
  if (raft::device_setter::get_device_count() < 2) {
    GTEST_SKIP() << "Requires at least two CUDA devices";
  }

  raft::device_setter device_scope(1);
  raft::resources res;
  constexpr std::uint32_t row_count  = 4;
  constexpr std::uint32_t row_length = 3;

  auto cross_graph = raft::make_host_matrix<std::uint32_t, int64_t>(row_count, row_length);
  for (std::uint32_t row = 0; row < row_count; ++row) {
    for (std::uint32_t col = 0; col < row_length; ++col) {
      cross_graph(row, col) = row * 10 + col;
    }
  }

  flowann::search_context context(raft::make_const_mdspan(cross_graph.view()));
  auto output = raft::make_device_matrix<std::uint32_t, int64_t>(res, row_count, row_length);
  context.resume();
  queue_round_trip<<<row_count, 1, 0, raft::resource::get_cuda_stream(res).get()>>>(
    flowann::detail::search_context_access::device_queues(context),
    context.num_queues(),
    row_count,
    row_length,
    output.data_handle());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  raft::resource::sync_stream(res);
  context.pause();

  auto host_output = raft::make_host_matrix<std::uint32_t, int64_t>(row_count, row_length);
  raft::copy(host_output.data_handle(),
             output.data_handle(),
             output.size(),
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  for (std::uint32_t row = 0; row < row_count; ++row) {
    for (std::uint32_t col = 0; col < row_length; ++col) {
      EXPECT_EQ(host_output(row, col), cross_graph(row, col));
    }
  }
}

TEST(FlowannQueue, RealPollersPreservePayloadsAcrossWrapWithMixedReleases)
{
  raft::resources res;
  auto const stream                  = raft::resource::get_cuda_stream(res);
  constexpr std::uint32_t row_count  = 17;
  constexpr std::uint32_t row_length = 9;
  constexpr std::uint32_t blocks     = 4099;
  constexpr std::uint32_t max_batch  = 7;
  constexpr std::uint32_t rounds     = 65;
  static_assert(static_cast<std::uint64_t>(blocks) * rounds >
                static_cast<std::uint64_t>(flowann::detail::queue_capacity) * 4);

  auto cross_graph = raft::make_host_matrix<std::uint32_t, int64_t>(row_count, row_length);
  for (std::uint32_t row = 0; row < row_count; ++row) {
    for (std::uint32_t col = 0; col < row_length; ++col) {
      cross_graph(row, col) = row * 1000 + col;
    }
  }
  auto output =
    raft::make_device_matrix<std::uint32_t, int64_t>(res, blocks * max_batch, row_length);
  auto host_output = raft::make_host_matrix<std::uint32_t, int64_t>(blocks * max_batch, row_length);

  for (std::uint32_t num_queues : {1u, 4u}) {
    for (std::uint32_t batch : {1u, max_batch}) {
      auto const commands = blocks * batch;
      flowann::queue_params params{};
      params.num_queues         = num_queues;
      params.empty_pause        = 0;
      params.collect_statistics = true;
      flowann::search_context context(raft::make_const_mdspan(cross_graph.view()), params);
      {
        flowann::search_session session(context);
        for (std::uint32_t round = 0; round < rounds; ++round) {
          mixed_release_queue_round_trip<<<blocks, 32, 0, stream.get()>>>(
            flowann::detail::search_context_access::device_queues(context),
            num_queues,
            row_count,
            row_length,
            round,
            batch,
            output.data_handle());
          RAFT_CUDA_TRY(cudaPeekAtLastError());
          // Finish the host-serviced kernel before copying into pageable host memory.
          // A pageable D2H copy can block inside the driver while waiting for this
          // kernel, preventing the CUDA-copy poller from submitting its transfers.
          raft::resource::sync_stream(res);
          raft::copy(host_output.data_handle(),
                     output.data_handle(),
                     static_cast<std::size_t>(commands) * row_length,
                     stream);
          raft::resource::sync_stream(res);
          for (std::uint32_t command = 0; command < commands; ++command) {
            auto const row = (command + round) % row_count;
            for (std::uint32_t col = 0; col < row_length; ++col) {
              if (host_output(command, col) != cross_graph(row, col)) {
                FAIL() << "payload mismatch with " << num_queues << " queues at round " << round
                       << ", command " << command << ", column " << col;
              }
            }
          }
        }
      }

      EXPECT_TRUE(flowann::detail::search_context_access::is_paused(context));
      EXPECT_EQ(context.statistics().processed_commands,
                static_cast<std::uint64_t>(commands) * rounds);
      std::vector<flowann::detail::queue_view> queues(num_queues);
      RAFT_CUDA_TRY(cudaMemcpy(queues.data(),
                               flowann::detail::search_context_access::device_queues(context),
                               queues.size() * sizeof(flowann::detail::queue_view),
                               cudaMemcpyDeviceToHost));
      for (auto const& queue : queues) {
        flowann::detail::queue_state state{};
        RAFT_CUDA_TRY(cudaMemcpy(&state, queue.state, sizeof(state), cudaMemcpyDeviceToHost));
        EXPECT_EQ(state.gpu_tail, state.reservation_head);
        EXPECT_EQ(state.cpu_tail, state.reservation_head);
      }
    }
  }
}

TEST(FlowannQueue, PollerStopsAtUnpublishedCommandsAndResumesFullBursts)
{
  constexpr std::uint32_t row_count  = 17;
  constexpr std::uint32_t row_length = 33;
  constexpr std::uint32_t commands   = 140;
  auto graph = raft::make_host_matrix<std::uint32_t, int64_t>(row_count, row_length);
  for (std::uint32_t row = 0; row < row_count; ++row) {
    for (std::uint32_t col = 0; col < row_length; ++col) {
      graph(row, col) = row * 1000 + col;
    }
  }

  // Exercise holes within the short prefix, at its boundary, and in a full batch.
  for (std::uint32_t hole : {5u, 8u, 127u}) {
    SCOPED_TRACE(hole);
    flowann::queue_params params{};
    params.num_queues         = 1;
    params.empty_pause        = 0;
    params.collect_statistics = true;
    flowann::search_context context(raft::make_const_mdspan(graph.view()), params);
    flowann::detail::queue_view queue{};
    RAFT_CUDA_TRY(cudaMemcpy(&queue,
                             flowann::detail::search_context_access::device_queues(context),
                             sizeof(queue),
                             cudaMemcpyDeviceToHost));
    unsigned long long const reserved = commands;
    RAFT_CUDA_TRY(cudaMemcpy(
      &queue.state->reservation_head, &reserved, sizeof(reserved), cudaMemcpyHostToDevice));
    std::vector<std::uint32_t> ids(commands);
    std::vector<unsigned long long> sequences(commands);
    for (std::uint32_t i = 0; i < commands; ++i) {
      ids[i] = i % row_count;
      sequences[i] =
        i == hole ? flowann::detail::empty_command : flowann::detail::make_command(ids[i]);
    }
    RAFT_CUDA_TRY(cudaMemcpy(queue.command_words,
                             sequences.data(),
                             sequences.size() * sizeof(sequences[0]),
                             cudaMemcpyHostToDevice));
    auto wait_for = [&](std::uint64_t expected) {
      auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
      while (context.statistics().processed_commands < expected &&
             std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
      }
      return context.statistics().processed_commands;
    };

    context.resume();
    auto const first_count = wait_for(hole);
    context.pause();
    ASSERT_EQ(first_count, hole);
    ASSERT_EQ(context.statistics().processed_commands, hole);
    flowann::detail::queue_state state{};
    RAFT_CUDA_TRY(cudaMemcpy(&state, queue.state, sizeof(state), cudaMemcpyDeviceToHost));
    EXPECT_EQ(state.cpu_tail, hole);
    std::vector<unsigned long long> published(commands);
    RAFT_CUDA_TRY(cudaMemcpy(published.data(),
                             queue.result_sequence,
                             published.size() * sizeof(published[0]),
                             cudaMemcpyDeviceToHost));
    for (std::uint32_t i = 0; i < commands; ++i) {
      EXPECT_EQ(published[i], i < hole ? i + 1ull : 0ull);
    }

    auto const ready = flowann::detail::make_command(ids[hole]);
    RAFT_CUDA_TRY(
      cudaMemcpy(queue.command_words + hole, &ready, sizeof(ready), cudaMemcpyHostToDevice));
    context.resume();
    auto const final_count = wait_for(commands);
    context.pause();
    ASSERT_EQ(final_count, commands);
    RAFT_CUDA_TRY(cudaMemcpy(&state, queue.state, sizeof(state), cudaMemcpyDeviceToHost));
    EXPECT_EQ(state.cpu_tail, commands);
    std::vector<unsigned long long> cleared(commands);
    RAFT_CUDA_TRY(cudaMemcpy(cleared.data(),
                             queue.command_words,
                             cleared.size() * sizeof(cleared[0]),
                             cudaMemcpyDeviceToHost));
    for (auto word : cleared) {
      EXPECT_EQ(word, flowann::detail::empty_command);
    }
    std::vector<std::uint32_t> returned(commands * row_length);
    RAFT_CUDA_TRY(cudaMemcpy(
      returned.data(), queue.rows, returned.size() * sizeof(returned[0]), cudaMemcpyDeviceToHost));
    for (std::uint32_t i = 0; i < commands; ++i) {
      for (std::uint32_t col = 0; col < row_length; ++col) {
        EXPECT_EQ(returned[i * row_length + col], graph(ids[i], col));
      }
    }
  }
}

RAFT_KERNEL deferred_capacity_reclaim(flowann::detail::queue_view q, unsigned* errors)
{
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  for (unsigned i = 0; i < flowann::detail::queue_capacity * 3 + 17; i++) {
    flowann::detail::queue_reservation r{};
    unsigned tries = 0;
    while (!flowann::detail::try_reserve_deferred(q, 1, &r)) {
      if (++tries > 4) {
        *errors = 1;
        return;
      }
    }
    if (r.sequence != i) {
      *errors = 2;
      return;
    }
    flowann::detail::release_deferred(q, r);
  }
  flowann::detail::reclaim_released(q);
  if (q.state->gpu_tail != q.state->reservation_head) *errors = 3;
}
RAFT_KERNEL deferred_release_hole(flowann::detail::queue_view q, unsigned* errors)
{
  if (threadIdx.x == 0) {
    q.state->gpu_tail         = 0;
    q.state->reservation_head = 3;
    q.released_sequence[0]    = 0;
    flowann::detail::release_deferred(q, {1, 1});
    flowann::detail::release_deferred(q, {2, 2});
  }
  __syncthreads();
  flowann::detail::reclaim_released_warp(q);
  if (threadIdx.x == 0 && q.state->gpu_tail != 0) *errors = 4;
  if (threadIdx.x == 0) flowann::detail::release_deferred(q, {0, 0});
  __syncthreads();
  flowann::detail::reclaim_released_warp(q);
  if (threadIdx.x == 0 && q.state->gpu_tail != 3) *errors = 5;
}

RAFT_KERNEL reserve_deferred_concurrently(flowann::detail::queue_view queue,
                                          std::uint64_t expected_base,
                                          std::uint32_t reservations_per_block,
                                          flowann::detail::queue_reservation* reservations,
                                          std::uint32_t* valid,
                                          std::uint32_t* sequence_owners,
                                          std::uint32_t* errors)
{
  auto const reservation_index = blockIdx.x;
  if (threadIdx.x == 0) {
    flowann::detail::queue_reservation first{};
    bool reserved = false;
    for (std::uint32_t attempt = 0; attempt < 4096 && !reserved; ++attempt) {
      reserved = flowann::detail::try_reserve_deferred(queue, reservations_per_block, &first);
    }
    reservations[reservation_index] = first;
    valid[reservation_index]        = reserved;
    if (!reserved) { atomicOr(errors, 1u); }
  }
  __syncthreads();

  if (valid[reservation_index] == 0) { return; }
  auto const first = reservations[reservation_index];
  for (std::uint32_t i = threadIdx.x; i < reservations_per_block; i += blockDim.x) {
    auto const reservation = flowann::detail::reservation_at(first, i);
    if (reservation.sequence < expected_base ||
        reservation.sequence >=
          expected_base + static_cast<std::uint64_t>(gridDim.x) * reservations_per_block) {
      atomicOr(errors, 2u);
      continue;
    }
    auto const offset = static_cast<std::uint32_t>(reservation.sequence - expected_base);
    if (atomicCAS(&sequence_owners[offset], ~std::uint32_t{0}, reservation_index) !=
        ~std::uint32_t{0}) {
      atomicOr(errors, 4u);
    }
  }
}

RAFT_KERNEL release_deferred_except_first(flowann::detail::queue_view queue,
                                          std::uint64_t first_sequence,
                                          std::uint32_t reservations_per_block,
                                          flowann::detail::queue_reservation const* reservations,
                                          std::uint32_t const* valid)
{
  if (valid[blockIdx.x] != 0) {
    auto const first = reservations[blockIdx.x];
    for (std::uint32_t i = threadIdx.x; i < reservations_per_block; i += blockDim.x) {
      auto const reservation = flowann::detail::reservation_at(first, i);
      if (reservation.sequence != first_sequence) {
        flowann::detail::release_deferred(queue, reservation);
      }
    }
  }
  __syncthreads();
  if (threadIdx.x < 32) { flowann::detail::reclaim_released_warp(queue); }
}

RAFT_KERNEL release_first_and_reclaim(flowann::detail::queue_view queue,
                                      std::uint64_t first_sequence,
                                      std::uint32_t reservations_per_block,
                                      flowann::detail::queue_reservation const* reservations,
                                      std::uint32_t const* valid)
{
  if (valid[blockIdx.x] != 0) {
    auto const first = reservations[blockIdx.x];
    if (first.sequence <= first_sequence &&
        first_sequence < first.sequence + reservations_per_block && threadIdx.x == 0) {
      flowann::detail::release_deferred(
        queue,
        flowann::detail::reservation_at(
          first, static_cast<std::uint32_t>(first_sequence - first.sequence)));
    }
  }
  __syncthreads();
  if (threadIdx.x < 32) { flowann::detail::reclaim_released_warp(queue); }
}

RAFT_KERNEL check_deferred_queue_counters(flowann::detail::queue_view queue,
                                          std::uint64_t expected_tail,
                                          std::uint32_t error_bit,
                                          std::uint32_t* errors)
{
  if (threadIdx.x == 0 && blockIdx.x == 0 &&
      (queue.state->gpu_tail != expected_tail ||
       (error_bit != 0 && queue.state->reservation_head != expected_tail))) {
    atomicOr(errors, error_bit == 0 ? 8u : error_bit);
  }
}

TEST(FlowannQueue, DeferredReclamationHandlesCapacityWrapAndUnreleasedHoles)
{
  raft::resources res;
  auto const stream = raft::resource::get_cuda_stream(res);
  auto cross        = raft::make_host_matrix<std::uint32_t, int64_t>(1, 2);
  cross(0, 0)       = 1;
  cross(0, 1)       = 0;
  flowann::search_context context(raft::make_const_mdspan(cross.view()));
  flowann::detail::queue_view queue;
  RAFT_CUDA_TRY(cudaMemcpy(&queue,
                           flowann::detail::search_context_access::device_queues(context),
                           sizeof(queue),
                           cudaMemcpyDeviceToHost));
  auto errors = raft::make_device_matrix<std::uint32_t, int64_t>(res, 1, 1);
  RAFT_CUDA_TRY(cudaMemsetAsync(errors.data_handle(), 0, sizeof(std::uint32_t), stream.get()));
  deferred_capacity_reclaim<<<1, 1, 0, stream.get()>>>(queue, errors.data_handle());
  deferred_release_hole<<<1, 32, 0, stream.get()>>>(queue, errors.data_handle());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  std::uint32_t host_error{};
  raft::copy(&host_error, errors.data_handle(), 1, stream);
  raft::resource::sync_stream(res);
  EXPECT_EQ(host_error, 0u);
}

TEST(FlowannQueue, ConcurrentDeferredReclamationPreservesOwnershipAcrossWrap)
{
  raft::resources res;
  auto const stream = raft::resource::get_cuda_stream(res);
  auto cross        = raft::make_host_matrix<std::uint32_t, int64_t>(1, 2);
  cross(0, 0)       = 1;
  cross(0, 1)       = 0;
  flowann::search_context context(raft::make_const_mdspan(cross.view()));
  flowann::detail::queue_view queue;
  RAFT_CUDA_TRY(cudaMemcpy(&queue,
                           flowann::detail::search_context_access::device_queues(context),
                           sizeof(queue),
                           cudaMemcpyDeviceToHost));

  constexpr std::uint32_t blocks                 = 255;
  constexpr std::uint32_t reservations_per_block = 257;
  constexpr std::uint32_t reservations_per_wave  = blocks * reservations_per_block;
  static_assert(reservations_per_wave == flowann::detail::queue_capacity - 1);

  auto reservations =
    raft::make_device_matrix<flowann::detail::queue_reservation, int64_t>(res, blocks, 1);
  auto valid  = raft::make_device_matrix<std::uint32_t, int64_t>(res, blocks, 1);
  auto owners = raft::make_device_matrix<std::uint32_t, int64_t>(res, reservations_per_wave, 1);
  auto errors = raft::make_device_matrix<std::uint32_t, int64_t>(res, 1, 1);
  RAFT_CUDA_TRY(cudaMemsetAsync(errors.data_handle(), 0, sizeof(std::uint32_t), stream.get()));

  std::uint64_t expected_base = 0;
  for (std::uint32_t wave = 0; wave < 3; ++wave) {
    RAFT_CUDA_TRY(
      cudaMemsetAsync(valid.data_handle(), 0, valid.size() * sizeof(std::uint32_t), stream.get()));
    RAFT_CUDA_TRY(cudaMemsetAsync(
      owners.data_handle(), 0xff, owners.size() * sizeof(std::uint32_t), stream.get()));
    reserve_deferred_concurrently<<<blocks, 256, 0, stream.get()>>>(queue,
                                                                    expected_base,
                                                                    reservations_per_block,
                                                                    reservations.data_handle(),
                                                                    valid.data_handle(),
                                                                    owners.data_handle(),
                                                                    errors.data_handle());
    release_deferred_except_first<<<blocks, 256, 0, stream.get()>>>(queue,
                                                                    expected_base,
                                                                    reservations_per_block,
                                                                    reservations.data_handle(),
                                                                    valid.data_handle());
    check_deferred_queue_counters<<<1, 1, 0, stream.get()>>>(
      queue, expected_base, 0, errors.data_handle());
    release_first_and_reclaim<<<blocks, 32, 0, stream.get()>>>(queue,
                                                               expected_base,
                                                               reservations_per_block,
                                                               reservations.data_handle(),
                                                               valid.data_handle());
    expected_base += reservations_per_wave;
    check_deferred_queue_counters<<<1, 1, 0, stream.get()>>>(
      queue, expected_base, 16u, errors.data_handle());
  }
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  std::uint32_t host_error{};
  raft::copy(&host_error, errors.data_handle(), 1, stream);
  raft::resource::sync_stream(res);
  EXPECT_EQ(host_error, 0u);
}

TEST(FlowannQueue, SearchLeaseConsumesRawResume)
{
  constexpr std::uint32_t row_count  = 4;
  constexpr std::uint32_t row_length = 3;
  auto cross_graph = raft::make_host_matrix<std::uint32_t, int64_t>(row_count, row_length);
  std::fill_n(cross_graph.data_handle(), cross_graph.size(), 0u);

  flowann::search_context context(raft::make_const_mdspan(cross_graph.view()));
  EXPECT_TRUE(flowann::detail::search_context_access::is_paused(context));

  context.resume();
  EXPECT_FALSE(flowann::detail::search_context_access::is_paused(context));
  flowann::detail::search_context_access::acquire_search(context);
  flowann::detail::search_context_access::release_search(context);
  EXPECT_TRUE(flowann::detail::search_context_access::is_paused(context));

  context.resume();
  context.pause();
  EXPECT_TRUE(flowann::detail::search_context_access::is_paused(context));

  flowann::detail::search_context_access::acquire_search(context);
  context.resume();
  flowann::detail::search_context_access::release_search(context);
  EXPECT_TRUE(flowann::detail::search_context_access::is_paused(context));
}

TEST(FlowannQueue, SearchSessionKeepsPollersRunningAcrossSearchLease)
{
  constexpr std::uint32_t row_count  = 4;
  constexpr std::uint32_t row_length = 3;
  auto cross_graph = raft::make_host_matrix<std::uint32_t, int64_t>(row_count, row_length);
  std::fill_n(cross_graph.data_handle(), cross_graph.size(), 0u);

  flowann::search_context context(raft::make_const_mdspan(cross_graph.view()));
  {
    flowann::search_session session(context);
    EXPECT_FALSE(flowann::detail::search_context_access::is_paused(context));
    flowann::detail::search_context_access::acquire_search(context);
    flowann::detail::search_context_access::release_search(context);
    EXPECT_FALSE(flowann::detail::search_context_access::is_paused(context));
  }
  EXPECT_TRUE(flowann::detail::search_context_access::is_paused(context));
}

TEST(FlowannQueue, PauseBlocksNewSearchesUntilPollersReachQuiescence)
{
  using namespace std::chrono_literals;
  constexpr std::uint32_t row_count  = 4;
  constexpr std::uint32_t row_length = 3;
  auto cross_graph = raft::make_host_matrix<std::uint32_t, int64_t>(row_count, row_length);
  std::fill_n(cross_graph.data_handle(), cross_graph.size(), 0u);

  flowann::search_context context(raft::make_const_mdspan(cross_graph.view()));
  flowann::detail::search_context_access::acquire_search(context);
  std::atomic<std::uint32_t> pauses_returned{};
  std::jthread first_pauser([&] {
    context.pause();
    pauses_returned.fetch_add(1, std::memory_order_release);
  });

  auto const deadline = std::chrono::steady_clock::now() + 2s;
  while (flowann::detail::search_context_access::pending_pause_requests(context) < 1 &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  std::jthread second_pauser([&] {
    context.pause();
    pauses_returned.fetch_add(1, std::memory_order_release);
  });
  auto const second_deadline = std::chrono::steady_clock::now() + 2s;
  while (flowann::detail::search_context_access::pending_pause_requests(context) < 2 &&
         std::chrono::steady_clock::now() < second_deadline) {
    std::this_thread::yield();
  }
  auto const pending_pauses =
    flowann::detail::search_context_access::pending_pause_requests(context);

  std::atomic<bool> later_search_started{};
  std::atomic<bool> later_search_acquired{};
  std::jthread later_search([&] {
    later_search_started.store(true, std::memory_order_release);
    flowann::detail::search_context_access::acquire_search(context);
    later_search_acquired.store(true, std::memory_order_release);
    flowann::detail::search_context_access::release_search(context);
  });
  while (!later_search_started.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  std::this_thread::sleep_for(20ms);
  if (pending_pauses == 2) { EXPECT_FALSE(later_search_acquired.load(std::memory_order_acquire)); }

  flowann::detail::search_context_access::release_search(context);
  first_pauser.join();
  second_pauser.join();
  later_search.join();
  EXPECT_EQ(pending_pauses, 2);
  EXPECT_EQ(pauses_returned.load(std::memory_order_acquire), 2);
  EXPECT_TRUE(later_search_acquired.load(std::memory_order_acquire));
  EXPECT_TRUE(flowann::detail::search_context_access::is_paused(context));
}

TEST(FlowannQueue, SearchLeasePausesAfterLastSearch)
{
  constexpr std::uint32_t row_count  = 4;
  constexpr std::uint32_t row_length = 3;
  auto cross_graph = raft::make_host_matrix<std::uint32_t, int64_t>(row_count, row_length);
  std::fill_n(cross_graph.data_handle(), cross_graph.size(), 0u);

  flowann::search_context context(raft::make_const_mdspan(cross_graph.view()));
  flowann::detail::search_context_access::acquire_search(context);
  flowann::detail::search_context_access::acquire_search(context);
  EXPECT_FALSE(flowann::detail::search_context_access::is_paused(context));

  flowann::detail::search_context_access::release_search(context);
  EXPECT_FALSE(flowann::detail::search_context_access::is_paused(context));
  flowann::detail::search_context_access::release_search(context);
  EXPECT_TRUE(flowann::detail::search_context_access::is_paused(context));
}

}  // namespace
