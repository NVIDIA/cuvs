/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/integer_utils.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

namespace cuvs::cluster::kmeans::detail {

/** A contiguous KMeans input batch accessible from the main CUDA stream. */
template <typename DataT>
class kmeans_batch {
 public:
  [[nodiscard]] auto data() const noexcept -> DataT const* { return data_; }
  [[nodiscard]] auto size() const noexcept -> std::size_t { return size_; }
  [[nodiscard]] auto offset() const noexcept -> std::size_t { return offset_; }

 private:
  template <typename, typename, bool>
  friend class kmeans_batch_loader;

  kmeans_batch(DataT const* data, std::size_t size, std::size_t offset)
    : data_(data), size_(size), offset_(offset)
  {
  }

  DataT const* data_  = nullptr;
  std::size_t size_   = 0;
  std::size_t offset_ = 0;
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
  kmeans_batch_loader(raft::resources const&,
                      DataT const* source,
                      IndexT n_rows,
                      IndexT row_width,
                      IndexT batch_size,
                      rmm::cuda_stream_view,
                      rmm::device_async_resource_ref)
    : source_(source),
      n_rows_(static_cast<std::size_t>(n_rows)),
      row_width_(static_cast<std::size_t>(row_width)),
      batch_size_(std::min<std::size_t>(static_cast<std::size_t>(batch_size),
                                        std::max<std::size_t>(n_rows_, 1))),
      n_batches_(n_rows_ == 0 ? 0 : raft::div_rounding_up_safe(n_rows_, batch_size_))
  {
  }

  [[nodiscard]] auto num_batches() const noexcept -> std::size_t { return n_batches_; }
  void prime() noexcept {}
  void prefetch(std::size_t) noexcept {}
  void prime_second_batch() noexcept {}

  [[nodiscard]] auto load(std::size_t pos) const -> kmeans_batch<DataT>
  {
    RAFT_EXPECTS(pos < n_batches_, "KMeans batch position is out of range");
    const auto offset = pos * batch_size_;
    const auto size   = std::min(batch_size_, n_rows_ - offset);
    return {source_ + offset * row_width_, size, offset};
  }

 private:
  DataT const* source_    = nullptr;
  std::size_t n_rows_     = 0;
  std::size_t row_width_  = 0;
  std::size_t batch_size_ = 0;
  std::size_t n_batches_  = 0;
};

template <typename DataT, typename IndexT>
class kmeans_batch_loader<DataT, IndexT, false> {
 public:
  kmeans_batch_loader(raft::resources const& res,
                      DataT const* source,
                      IndexT n_rows,
                      IndexT row_width,
                      IndexT batch_size,
                      rmm::cuda_stream_view copy_stream,
                      rmm::device_async_resource_ref mr)
    : res_(&res),
      source_(source),
      n_rows_(static_cast<std::size_t>(n_rows)),
      row_width_(static_cast<std::size_t>(row_width)),
      batch_size_(std::min<std::size_t>(static_cast<std::size_t>(batch_size),
                                        std::max<std::size_t>(n_rows_, 1))),
      n_batches_(n_rows_ == 0 ? 0 : raft::div_rounding_up_safe(n_rows_, batch_size_)),
      copy_stream_(copy_stream),
      buffer_0_(0, copy_stream, mr),
      buffer_1_(0, copy_stream, mr)
  {
    if (n_rows_ == 0 || source_ == nullptr) { return; }

    buffer_0_.resize(row_width_ * batch_size_, copy_stream_);
    current_ptr_ = buffer_0_.data();
    if (n_batches_ > 1) {
      buffer_1_.resize(row_width_ * batch_size_, copy_stream_);
      prefetch_ptr_ = buffer_1_.data();
    }
  }

  kmeans_batch_loader(kmeans_batch_loader const&)                    = delete;
  auto operator=(kmeans_batch_loader const&) -> kmeans_batch_loader& = delete;
  kmeans_batch_loader(kmeans_batch_loader&&)                         = delete;
  auto operator=(kmeans_batch_loader&&) -> kmeans_batch_loader&      = delete;

  ~kmeans_batch_loader() noexcept
  {
    if (source_ != nullptr) {
      RAFT_CUDA_TRY_NO_THROW(cudaStreamSynchronize(raft::resource::get_cuda_stream(*res_)));
    }
    RAFT_CUDA_TRY_NO_THROW(cudaStreamSynchronize(copy_stream_));
    for (auto event : events_) {
      if (event != nullptr) { RAFT_CUDA_TRY_NO_THROW(cudaEventDestroy(event)); }
    }
  }

  [[nodiscard]] auto num_batches() const noexcept -> std::size_t { return n_batches_; }

  /** Stage batch zero unless it is already active or staged by the previous pass. */
  void prime()
  {
    if (n_batches_ <= 1) { return; }
    const bool batch_zero_active = current_pos_.has_value() && *current_pos_ == 0;
    const bool batch_zero_staged = prefetch_pos_.has_value() && *prefetch_pos_ == 0;
    if (!batch_zero_active && !batch_zero_staged) { prefetch(0); }
  }

  /** Stage a future batch into the slot not currently consumed by KMeans. */
  void prefetch(std::size_t pos)
  {
    if (n_batches_ <= 1 || pos >= n_batches_ || source_ == nullptr) { return; }
    if (prefetch_pos_.has_value() && *prefetch_pos_ == pos) { return; }

    const int prefetch_slot = 1 - current_slot_;
    if (kernel_done_[prefetch_slot] != nullptr) {
      RAFT_CUDA_TRY(cudaStreamWaitEvent(copy_stream_, kernel_done_[prefetch_slot], 0));
    }

    queue_h2d(prefetch_ptr_, pos);
    prefetch_pos_            = pos;
    h2d_done_[prefetch_slot] = make_event();
    RAFT_CUDA_TRY(cudaEventRecord(h2d_done_[prefetch_slot], copy_stream_));
  }

  /**
   * Activate cyclic batch zero after pass-boundary kernels have been submitted, then stage batch
   * one into the slot retired by the previous pass's last batch.
   */
  void prime_second_batch()
  {
    prime();
    if (n_batches_ < 2) { return; }
    (void)load(0);
    prefetch(1);
  }

  /** Make a staged batch visible to kernels on the main stream. */
  [[nodiscard]] auto load(std::size_t pos) -> kmeans_batch<DataT>
  {
    RAFT_EXPECTS(pos < n_batches_, "KMeans batch position is out of range");
    if (!current_pos_.has_value() || *current_pos_ != pos) {
      if (prefetch_pos_.has_value() && *prefetch_pos_ == pos) {
        const int retired_slot = current_slot_;
        std::swap(current_ptr_, prefetch_ptr_);
        current_slot_ = 1 - current_slot_;
        prefetch_pos_.reset();

        kernel_done_[retired_slot] = make_event();
        RAFT_CUDA_TRY(
          cudaEventRecord(kernel_done_[retired_slot], raft::resource::get_cuda_stream(*res_)));
        RAFT_CUDA_TRY(
          cudaStreamWaitEvent(raft::resource::get_cuda_stream(*res_), h2d_done_[current_slot_], 0));
      } else {
        // A one-batch input has nothing to overlap. Stage it once, then reuse it for every pass.
        RAFT_EXPECTS(n_batches_ == 1, "KMeans attempted to load a batch that was not prefetched");
        queue_h2d(current_ptr_, pos);
        copy_stream_.synchronize();
      }
      current_pos_ = pos;
    }

    const auto offset = pos * batch_size_;
    const auto size   = std::min(batch_size_, n_rows_ - offset);
    return {current_ptr_, size, offset};
  }

 private:
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

  void queue_h2d(DataT* dst, std::size_t pos)
  {
    const auto offset = pos * batch_size_;
    const auto rows   = std::min(batch_size_, n_rows_ - offset);
    const auto bytes  = rows * row_width_ * sizeof(DataT);
    RAFT_CUDA_TRY(cudaMemcpyAsync(
      dst, source_ + offset * row_width_, bytes, cudaMemcpyHostToDevice, copy_stream_));
  }

  raft::resources const* res_ = nullptr;
  DataT const* source_        = nullptr;
  std::size_t n_rows_         = 0;
  std::size_t row_width_      = 0;
  std::size_t batch_size_     = 0;
  std::size_t n_batches_      = 0;
  rmm::cuda_stream_view copy_stream_;
  rmm::device_uvector<DataT> buffer_0_;
  rmm::device_uvector<DataT> buffer_1_;
  DataT* current_ptr_  = nullptr;
  DataT* prefetch_ptr_ = nullptr;
  int current_slot_    = 0;
  std::optional<std::size_t> current_pos_;
  std::optional<std::size_t> prefetch_pos_;
  cudaEvent_t h2d_done_[2]    = {nullptr, nullptr};
  cudaEvent_t kernel_done_[2] = {nullptr, nullptr};
  std::vector<cudaEvent_t> events_;
};

}  // namespace cuvs::cluster::kmeans::detail
