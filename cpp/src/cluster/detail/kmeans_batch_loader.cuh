/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/integer_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

namespace cuvs::cluster::kmeans::detail {

class kmeans_batch_descriptor {
 public:
  kmeans_batch_descriptor(std::size_t size, std::size_t offset, std::size_t partition)
    : size_(size), offset_(offset), partition_(partition)
  {
  }

  [[nodiscard]] auto size() const noexcept -> std::size_t { return size_; }
  [[nodiscard]] auto offset() const noexcept -> std::size_t { return offset_; }
  [[nodiscard]] auto partition() const noexcept -> std::size_t { return partition_; }

 private:
  std::size_t size_;
  std::size_t offset_;
  std::size_t partition_;
};

template <typename InputView>
struct kmeans_batch_source {
  InputView input;
  kmeans_batch_descriptor descriptor;
};

/** A contiguous KMeans input batch accessible from the main CUDA stream. */
template <typename DataT>
class kmeans_batch {
 public:
  [[nodiscard]] auto data() const noexcept -> DataT const* { return data_; }
  [[nodiscard]] auto descriptor() const noexcept -> kmeans_batch_descriptor const&
  {
    return descriptor_;
  }
  [[nodiscard]] auto size() const noexcept -> std::size_t { return descriptor_.size(); }
  [[nodiscard]] auto offset() const noexcept -> std::size_t { return descriptor_.offset(); }
  [[nodiscard]] auto partition() const noexcept -> std::size_t { return descriptor_.partition(); }

 private:
  template <typename, typename, bool>
  friend class kmeans_batch_loader;

  kmeans_batch(DataT const* data,
               kmeans_batch_descriptor const& descriptor,
               std::size_t position,
               int slot)
    : data_(data), descriptor_(descriptor), position_(position), slot_(slot)
  {
  }

  DataT const* data_;
  kmeans_batch_descriptor const& descriptor_;
  std::size_t position_;
  int slot_;
};

/**
 * Read-only batch loader used only by KMeans.
 *
 * The device specialization is a zero-copy view. The host specialization below owns the
 * two-buffer, cyclic H2D pipeline needed by out-of-core KMeans.
 */
template <typename DataT, typename IndexT, bool DataOnDevice>
class kmeans_batch_loader;

template <typename DataT, typename IndexT>
class kmeans_batch_loader<DataT, IndexT, true> {
 public:
  kmeans_batch_loader(raft::resources const& res,
                      raft::device_matrix_view<const DataT, IndexT> input,
                      IndexT batch_size,
                      cuda::stream_ref copy_stream,
                      rmm::device_async_resource_ref mr)
    : kmeans_batch_loader(res,
                          std::vector<raft::device_matrix_view<const DataT, IndexT>>{input},
                          batch_size,
                          copy_stream,
                          mr)
  {
  }

  kmeans_batch_loader(raft::resources const&,
                      std::vector<raft::device_matrix_view<const DataT, IndexT>> const& partitions,
                      IndexT batch_size,
                      cuda::stream_ref,
                      rmm::device_async_resource_ref)
    : batch_size_(std::max<std::size_t>(static_cast<std::size_t>(batch_size), 1))
  {
    for (std::size_t partition = 0; partition < partitions.size(); ++partition) {
      append_batches(partitions[partition], partition);
    }
  }

  [[nodiscard]] auto num_batches() const noexcept -> std::size_t { return batches_.size(); }
  void start() noexcept {}
  void prefetch(std::size_t) noexcept {}
  void recycle(kmeans_batch<DataT> const&, std::size_t) noexcept {}
  void release(kmeans_batch<DataT> const&) noexcept {}

  [[nodiscard]] auto acquire(std::size_t pos) const -> kmeans_batch<DataT>
  {
    RAFT_EXPECTS(pos < batches_.size(), "KMeans batch position is out of range");
    auto const& batch = batches_[pos];
    return {batch.input.data_handle() + batch.descriptor.offset() * batch.input.extent(1),
            batch.descriptor,
            pos,
            0};
  }

 private:
  void append_batches(raft::device_matrix_view<const DataT, IndexT> input, std::size_t partition)
  {
    if (input.extent(0) == 0) { return; }
    RAFT_EXPECTS(input.data_handle() != nullptr, "non-empty KMeans input partition cannot be null");
    for (std::size_t offset = 0; offset < static_cast<std::size_t>(input.extent(0));
         offset += batch_size_) {
      const auto size = std::min(batch_size_, static_cast<std::size_t>(input.extent(0)) - offset);
      batches_.push_back({input, {size, offset, partition}});
    }
  }

  std::size_t batch_size_ = 0;
  std::vector<kmeans_batch_source<raft::device_matrix_view<const DataT, IndexT>>> batches_;
};

template <typename DataT, typename IndexT>
class kmeans_batch_loader<DataT, IndexT, false> {
 public:
  kmeans_batch_loader(raft::resources const& res,
                      raft::host_matrix_view<const DataT, IndexT> input,
                      IndexT batch_size,
                      cuda::stream_ref copy_stream,
                      rmm::device_async_resource_ref mr)
    : kmeans_batch_loader(res,
                          std::vector<raft::host_matrix_view<const DataT, IndexT>>{input},
                          batch_size,
                          copy_stream,
                          mr)
  {
  }

  kmeans_batch_loader(raft::resources const& res,
                      std::vector<raft::host_matrix_view<const DataT, IndexT>> const& partitions,
                      IndexT batch_size,
                      cuda::stream_ref copy_stream,
                      rmm::device_async_resource_ref mr)
    : res_(&res),
      batch_size_(std::max<std::size_t>(static_cast<std::size_t>(batch_size), 1)),
      copy_stream_(copy_stream),
      buffer_0_(0, copy_stream, mr),
      buffer_1_(0, copy_stream, mr)
  {
    for (std::size_t partition = 0; partition < partitions.size(); ++partition) {
      append_batches(partitions[partition], partition);
    }
    if (batches_.empty()) { return; }

    std::size_t max_batch_elements = 0;
    for (auto const& batch : batches_) {
      max_batch_elements =
        std::max(max_batch_elements,
                 batch.descriptor.size() * static_cast<std::size_t>(batch.input.extent(1)));
    }
    buffer_0_.resize(max_batch_elements, copy_stream_);
    buffer_ptrs_[0] = buffer_0_.data();
    if (batches_.size() > 1) {
      buffer_1_.resize(max_batch_elements, copy_stream_);
      buffer_ptrs_[1] = buffer_1_.data();
    }
  }

  kmeans_batch_loader(kmeans_batch_loader const&)                    = delete;
  auto operator=(kmeans_batch_loader const&) -> kmeans_batch_loader& = delete;
  kmeans_batch_loader(kmeans_batch_loader&&)                         = delete;
  auto operator=(kmeans_batch_loader&&) -> kmeans_batch_loader&      = delete;

  ~kmeans_batch_loader() noexcept
  {
    if (!batches_.empty()) { raft::resource::sync_stream(*res_); }
    raft::resource::sync_stream(*res_, copy_stream_);
    for (auto event : events_) {
      if (event != nullptr) { RAFT_CUDA_TRY_NO_THROW(cudaEventDestroy(event)); }
    }
  }

  [[nodiscard]] auto num_batches() const noexcept -> std::size_t { return batches_.size(); }

  /** Start the pipeline by staging its first batch. */
  void start()
  {
    if (started_) { return; }
    if (!batches_.empty()) { prefetch(0); }
    started_ = true;
  }

  /** Stage a batch into an available slot; do nothing when both slots are occupied. */
  void prefetch(std::size_t pos)
  {
    RAFT_EXPECTS(pos < batches_.size(), "KMeans batch position is out of range");

    for (int slot = 0; slot < num_slots(); ++slot) {
      if (states_[slot] == slot_state::empty || states_[slot] == slot_state::reusable) {
        stage(slot, pos);
        return;
      }
    }
  }

  /** Make a prefetched batch visible to kernels on the main stream. */
  [[nodiscard]] auto acquire(std::size_t pos) -> kmeans_batch<DataT>
  {
    RAFT_EXPECTS(pos < batches_.size(), "KMeans batch position is out of range");
    for (int slot = 0; slot < num_slots(); ++slot) {
      if (states_[slot] == slot_state::staged && positions_[slot] == pos) {
        RAFT_CUDA_TRY(
          cudaStreamWaitEvent(raft::resource::get_cuda_stream(*res_).get(), ready_[slot], 0));
        states_[slot] = slot_state::acquired;

        auto const& batch = batches_[pos];
        return {buffer_ptrs_[slot], batch.descriptor, pos, slot};
      }
    }
    RAFT_FAIL("KMeans attempted to acquire a batch that was not prefetched");
  }

  /** Record completion of a batch, then refill the same slot with a future batch. */
  void recycle(kmeans_batch<DataT> const& batch, std::size_t next_pos)
  {
    RAFT_EXPECTS(next_pos < batches_.size(), "KMeans batch position is out of range");
    const int slot = validate_acquired(batch);

    // No transfer is needed when the requested future batch is already resident.
    if (positions_[slot] == next_pos) {
      states_[slot] = slot_state::staged;
      return;
    }

    mark_reusable(slot);
    stage(slot, next_pos);
  }

  /** Record completion without scheduling another transfer into the slot. */
  void release(kmeans_batch<DataT> const& batch)
  {
    const int slot = validate_acquired(batch);
    mark_reusable(slot);
  }

 private:
  enum class slot_state { empty, staged, acquired, reusable };

  [[nodiscard]] auto num_slots() const noexcept -> int { return batches_.size() > 1 ? 2 : 1; }

  void append_batches(raft::host_matrix_view<const DataT, IndexT> input, std::size_t partition)
  {
    if (input.extent(0) == 0) { return; }
    RAFT_EXPECTS(input.data_handle() != nullptr, "non-empty KMeans input partition cannot be null");
    for (std::size_t offset = 0; offset < static_cast<std::size_t>(input.extent(0));
         offset += batch_size_) {
      const auto size = std::min(batch_size_, static_cast<std::size_t>(input.extent(0)) - offset);
      batches_.push_back({input, {size, offset, partition}});
    }
  }

  [[nodiscard]] auto make_event() -> cudaEvent_t
  {
    cudaEvent_t event = nullptr;
    RAFT_CUDA_TRY(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    try {
      events_.push_back(event);
    } catch (...) {
      RAFT_CUDA_TRY_NO_THROW(cudaEventDestroy(event));
      throw;
    }
    return event;
  }

  void stage(int slot, std::size_t pos)
  {
    RAFT_EXPECTS(states_[slot] == slot_state::empty || states_[slot] == slot_state::reusable,
                 "KMeans attempted to overwrite an active batch buffer");
    if (states_[slot] == slot_state::reusable) {
      RAFT_CUDA_TRY(cudaStreamWaitEvent(copy_stream_.get(), reusable_[slot], 0));
    }
    queue_h2d(buffer_ptrs_[slot], pos);
    positions_[slot] = pos;
    if (ready_[slot] == nullptr) { ready_[slot] = make_event(); }
    // cudaStreamWaitEvent captures the latest record at the time the wait is submitted, so this
    // per-slot event can be reused after acquire() has enqueued that wait.
    RAFT_CUDA_TRY(cudaEventRecord(ready_[slot], copy_stream_.get()));
    states_[slot] = slot_state::staged;
  }

  void mark_reusable(int slot)
  {
    if (reusable_[slot] == nullptr) { reusable_[slot] = make_event(); }
    // The copy stream consumes this generation's record before the event is recorded again.
    RAFT_CUDA_TRY(cudaEventRecord(reusable_[slot], raft::resource::get_cuda_stream(*res_).get()));
    states_[slot] = slot_state::reusable;
  }

  [[nodiscard]] auto validate_acquired(kmeans_batch<DataT> const& batch) const -> int
  {
    const int slot = batch.slot_;
    RAFT_EXPECTS(slot >= 0 && slot < num_slots() && states_[slot] == slot_state::acquired &&
                   positions_[slot] == batch.position_ && buffer_ptrs_[slot] == batch.data(),
                 "KMeans attempted to release a batch that is not active");
    return slot;
  }

  void queue_h2d(DataT* dst, std::size_t pos)
  {
    auto const& batch = batches_[pos];
    raft::copy(dst,
               batch.input.data_handle() + batch.descriptor.offset() * batch.input.extent(1),
               batch.descriptor.size() * batch.input.extent(1),
               copy_stream_);
  }

  raft::resources const* res_ = nullptr;
  std::size_t batch_size_     = 0;
  std::vector<kmeans_batch_source<raft::host_matrix_view<const DataT, IndexT>>> batches_;
  cuda::stream_ref copy_stream_;
  rmm::device_uvector<DataT> buffer_0_;
  rmm::device_uvector<DataT> buffer_1_;
  DataT* buffer_ptrs_[2] = {nullptr, nullptr};
  std::optional<std::size_t> positions_[2];
  bool started_            = false;
  slot_state states_[2]    = {slot_state::empty, slot_state::empty};
  cudaEvent_t ready_[2]    = {nullptr, nullptr};
  cudaEvent_t reusable_[2] = {nullptr, nullptr};
  std::vector<cudaEvent_t> events_;
};

}  // namespace cuvs::cluster::kmeans::detail
