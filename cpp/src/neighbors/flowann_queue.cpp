/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/flowann.hpp>

#include "detail/flowann/queue.cuh"

#include <raft/core/error.hpp>
#include <raft/core/logger.hpp>
#include <raft/util/cudart_utils.hpp>

#ifdef CUVS_FLOWANN_USE_GDRCOPY
#include <cuda.h>
#include <gdrapi.h>
#endif

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

namespace cuvs::neighbors::cagra::experimental::flowann {
namespace {

constexpr auto align_up(std::size_t value, std::size_t alignment) -> std::size_t
{
  return (value + alignment - 1) & ~(alignment - 1);
}

struct queue_layout {
  std::size_t state;
  std::size_t rows;
  std::size_t command_words;
  std::size_t result_sequence;
  std::size_t released_sequence;
  std::size_t total_size;

  explicit queue_layout(std::uint32_t row_length)
  {
    constexpr std::size_t alignment = 128;
    state                           = 0;
    rows                            = align_up(sizeof(detail::queue_state), alignment);

    RAFT_EXPECTS(row_length <= std::numeric_limits<std::size_t>::max() / detail::queue_capacity /
                                 sizeof(std::uint32_t),
                 "FlowANN queue row storage size overflows");
    auto const row_bytes =
      static_cast<std::size_t>(detail::queue_capacity) * row_length * sizeof(std::uint32_t);
    command_words = align_up(rows + row_bytes, alignment);
    result_sequence =
      align_up(command_words + detail::queue_capacity * sizeof(unsigned long long), alignment);
    released_sequence =
      align_up(result_sequence + detail::queue_capacity * sizeof(unsigned long long), alignment);
    total_size =
      align_up(released_sequence + detail::queue_capacity * sizeof(unsigned long long), alignment);
  }
};

template <typename T>
auto add_bytes(void* base, std::size_t offset) -> T*
{
  return reinterpret_cast<T*>(static_cast<std::byte*>(base) + offset);
}

class cuda_allocation {
 public:
  cuda_allocation()                                          = default;
  cuda_allocation(cuda_allocation const&)                    = delete;
  auto operator=(cuda_allocation const&) -> cuda_allocation& = delete;

  ~cuda_allocation()
  {
    if (allocation_ != nullptr) { RAFT_CUDA_TRY_NO_THROW(cudaFree(allocation_)); }
  }

  void allocate(std::size_t size, std::size_t alignment = 1)
  {
    RAFT_EXPECTS(allocation_ == nullptr, "FlowANN queue allocation is already initialized");
    RAFT_EXPECTS(alignment > 0 && (alignment & (alignment - 1)) == 0,
                 "FlowANN queue allocation alignment must be a power of two");
    RAFT_EXPECTS(size <= std::numeric_limits<std::size_t>::max() - (alignment - 1),
                 "FlowANN queue allocation size overflows");
    RAFT_CUDA_TRY(cudaMalloc(&allocation_, size + alignment - 1));
    auto const address = reinterpret_cast<std::uintptr_t>(allocation_);
    ptr_               = reinterpret_cast<void*>(align_up(address, alignment));
  }

  [[nodiscard]] auto get() const noexcept -> void* { return ptr_; }

 private:
  void* allocation_{};
  void* ptr_{};
};

class cuda_stream {
 public:
  cuda_stream()                                      = default;
  cuda_stream(cuda_stream const&)                    = delete;
  auto operator=(cuda_stream const&) -> cuda_stream& = delete;

  ~cuda_stream()
  {
    if (stream_ != nullptr) { RAFT_CUDA_TRY_NO_THROW(cudaStreamDestroy(stream_)); }
  }

  void create()
  {
    RAFT_EXPECTS(stream_ == nullptr, "FlowANN queue stream is already initialized");
    RAFT_CUDA_TRY(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  }

  [[nodiscard]] auto get() const noexcept -> cudaStream_t { return stream_; }

 private:
  cudaStream_t stream_{};
};

#ifndef CUVS_FLOWANN_USE_GDRCOPY
class cuda_host_buffer {
 public:
  cuda_host_buffer()                                           = default;
  cuda_host_buffer(cuda_host_buffer const&)                    = delete;
  auto operator=(cuda_host_buffer const&) -> cuda_host_buffer& = delete;

  ~cuda_host_buffer()
  {
    if (ptr_ != nullptr) { RAFT_CUDA_TRY_NO_THROW(cudaFreeHost(ptr_)); }
  }

  void allocate(std::size_t size)
  {
    RAFT_EXPECTS(ptr_ == nullptr, "FlowANN host copy buffer is already initialized");
    RAFT_CUDA_TRY(cudaMallocHost(&ptr_, size));
    size_ = size;
  }

  [[nodiscard]] auto get(std::size_t size) const -> void*
  {
    RAFT_EXPECTS(size <= size_, "FlowANN transfer exceeds the host copy buffer");
    return ptr_;
  }

 private:
  void* ptr_{};
  std::size_t size_{};
};
#endif

#ifdef CUVS_FLOWANN_USE_GDRCOPY
class gdrcopy_mapping {
 public:
  gdrcopy_mapping(void* device_ptr, std::size_t size) : size_(size)
  {
    try {
      unsigned int sync_memops = 1;
      RAFT_EXPECTS(cuPointerSetAttribute(&sync_memops,
                                         CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                         reinterpret_cast<CUdeviceptr>(device_ptr)) == CUDA_SUCCESS,
                   "Failed to enable synchronous memory operations for a FlowANN GDRCopy queue");

      handle_ = gdr_open();
      RAFT_EXPECTS(handle_ != nullptr,
                   "Failed to open GDRCopy; ensure the gdrdrv kernel module is loaded");
      RAFT_EXPECTS(
        gdr_pin_buffer(
          handle_, reinterpret_cast<std::uintptr_t>(device_ptr), size_, 0, 0, &memory_handle_) == 0,
        "Failed to pin a FlowANN queue with GDRCopy");
      pinned_ = true;
      RAFT_EXPECTS(gdr_map(handle_, memory_handle_, &mapped_base_, size_) == 0,
                   "Failed to map a FlowANN queue with GDRCopy");
      mapped_ = true;

      gdr_info_t info{};
      RAFT_EXPECTS(gdr_get_info(handle_, memory_handle_, &info) == 0,
                   "Failed to query a FlowANN GDRCopy mapping");
      auto const device_address = reinterpret_cast<std::uintptr_t>(device_ptr);
      RAFT_EXPECTS(device_address >= info.va,
                   "GDRCopy returned an invalid FlowANN mapping address");
      auto const offset = device_address - info.va;
      RAFT_EXPECTS(offset <= size_, "GDRCopy returned an invalid FlowANN mapping offset");
      mapped_device_ptr_ = static_cast<std::byte*>(mapped_base_) + offset;
    } catch (...) {
      reset();
      throw;
    }
  }

  gdrcopy_mapping(gdrcopy_mapping const&)                    = delete;
  auto operator=(gdrcopy_mapping const&) -> gdrcopy_mapping& = delete;

  ~gdrcopy_mapping() { reset(); }

  void read(void* dst, std::size_t offset, std::size_t size)
  {
    RAFT_EXPECTS(offset <= size_ && size <= size_ - offset,
                 "FlowANN GDRCopy read exceeds its queue mapping");
    RAFT_EXPECTS(gdr_copy_from_mapping(memory_handle_, dst, mapped_device_ptr_ + offset, size) == 0,
                 "Failed to read a FlowANN GDRCopy queue");
  }

  void write(std::size_t offset, void const* src, std::size_t size)
  {
    RAFT_EXPECTS(offset <= size_ && size <= size_ - offset,
                 "FlowANN GDRCopy write exceeds its queue mapping");
    RAFT_EXPECTS(gdr_copy_to_mapping(memory_handle_, mapped_device_ptr_ + offset, src, size) == 0,
                 "Failed to write a FlowANN GDRCopy queue");
  }

 private:
  void reset() noexcept
  {
    if (mapped_) {
      static_cast<void>(gdr_unmap(handle_, memory_handle_, mapped_base_, size_));
      mapped_ = false;
    }
    if (pinned_) {
      static_cast<void>(gdr_unpin_buffer(handle_, memory_handle_));
      pinned_ = false;
    }
    if (handle_ != nullptr) {
      static_cast<void>(gdr_close(handle_));
      handle_ = nullptr;
    }
  }

  gdr_t handle_{};
  gdr_mh_t memory_handle_{};
  void* mapped_base_{};
  std::byte* mapped_device_ptr_{};
  std::size_t size_{};
  bool pinned_{};
  bool mapped_{};
};
#endif

#if defined(__linux__)
auto read_topology_value(int cpu, char const* name) -> int
{
  std::ifstream input("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/" + name);
  int value = -1;
  input >> value;
  return input ? value : -1;
}

auto physical_cpus() -> std::vector<int>
{
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) { return {}; }

  std::vector<std::tuple<int, int>> cores;
  std::vector<int> cpus;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (!CPU_ISSET(cpu, &allowed)) { continue; }
    auto package = read_topology_value(cpu, "physical_package_id");
    auto core    = read_topology_value(cpu, "core_id");
    if (package < 0 || core < 0) {
      package = 0;
      core    = cpu;
    }
    auto const identity = std::tuple{package, core};
    if (std::find(cores.begin(), cores.end(), identity) == cores.end()) {
      cores.push_back(identity);
      cpus.push_back(cpu);
    }
  }
  return cpus;
}

void bind_to_physical_cpu(std::uint32_t queue_id)
{
  static auto const cpus = physical_cpus();
  if (cpus.empty()) { return; }
  auto const cpu = cpus[queue_id % cpus.size()];
  cpu_set_t selected;
  CPU_ZERO(&selected);
  CPU_SET(cpu, &selected);
  auto const error = pthread_setaffinity_np(pthread_self(), sizeof(selected), &selected);
  if (error != 0) {
    RAFT_LOG_WARN(
      "FlowANN queue %u could not bind its poller to CPU %d (error=%d)", queue_id, cpu, error);
  }
  if (queue_id >= cpus.size()) {
    RAFT_LOG_WARN("FlowANN queue %u reuses physical CPU %d because only %zu are available",
                  queue_id,
                  cpu,
                  cpus.size());
  }
}
#else
void bind_to_physical_cpu(std::uint32_t) {}
#endif

class host_queue {
 public:
  static constexpr std::uint32_t max_batch_size = 128;

  host_queue(std::uint32_t const* cross_graph,
             std::uint64_t row_count,
             std::uint32_t row_length,
             bool collect_statistics)
    : cross_graph_(cross_graph),
      row_count_(row_count),
      row_length_(row_length),
      collect_statistics_(collect_statistics),
      layout_(row_length),
      empty_commands_(max_batch_size, detail::empty_command),
      command_staging_(max_batch_size),
      result_sequence_staging_(max_batch_size),
      row_staging_(static_cast<std::size_t>(max_batch_size) * row_length)
  {
#ifdef CUVS_FLOWANN_USE_GDRCOPY
    // gdrdrv rejects mappings whose device address is not aligned to its 64 KiB GPU page.
    // cudaMalloc can return a suballocation after the graph build, so align within an owning
    // allocation instead of relying on cudaMalloc's current address.
    storage_.allocate(layout_.total_size, 64u * 1024u);
#else
    storage_.allocate(layout_.total_size);
#endif
    RAFT_CUDA_TRY(cudaMemset(storage_.get(), 0, layout_.total_size));

    detail::queue_state state{};
    state.capacity   = detail::queue_capacity;
    state.row_length = row_length_;
    RAFT_CUDA_TRY(cudaMemcpy(add_bytes<void>(storage_.get(), layout_.state),
                             &state,
                             sizeof(state),
                             cudaMemcpyHostToDevice));

    view_.state           = add_bytes<detail::queue_state>(storage_.get(), layout_.state);
    view_.rows            = add_bytes<std::uint32_t>(storage_.get(), layout_.rows);
    view_.command_words   = add_bytes<unsigned long long>(storage_.get(), layout_.command_words);
    view_.result_sequence = add_bytes<unsigned long long>(storage_.get(), layout_.result_sequence);
    view_.released_sequence =
      add_bytes<unsigned long long>(storage_.get(), layout_.released_sequence);

#ifdef CUVS_FLOWANN_USE_GDRCOPY
    gdrcopy_.emplace(storage_.get(), layout_.total_size);
#else
    copy_stream_.create();
    // Pageable copies may synchronize inside the driver and contend with lazy
    // kernel initialization. Allocate once, before starting any pollers, and
    // use pinned memory for every transfer on the nonblocking copy stream.
    copy_staging_.allocate(std::max(row_staging_.size() * sizeof(std::uint32_t),
                                    command_staging_.size() * sizeof(unsigned long long)));
#endif
  }

  host_queue(host_queue const&)                    = delete;
  auto operator=(host_queue const&) -> host_queue& = delete;

  [[nodiscard]] auto view() const noexcept -> detail::queue_view { return view_; }

  [[nodiscard]] auto poll_batch() -> bool
  {
    if (collect_statistics_) { polls_.fetch_add(1, std::memory_order_relaxed); }
    auto const first_slot = static_cast<std::uint32_t>(cpu_tail_) & detail::queue_mask;
    // Sparse traffic needs only a short prefix. Read the rest only when the
    // whole prefix is ready, retaining full service batches under load.
    constexpr std::uint32_t prefix_size = 8;
    read_ring(command_staging_.data(), layout_.command_words, first_slot, prefix_size);
    if (!detail::command_ready(command_staging_[0])) {
      if (collect_statistics_) { empty_polls_.fetch_add(1, std::memory_order_relaxed); }
      return false;
    }

    std::uint32_t count = 1;
    while (count < prefix_size && detail::command_ready(command_staging_[count])) {
      ++count;
    }
    if (count == prefix_size) {
      auto const next_slot = (first_slot + prefix_size) & detail::queue_mask;
      read_ring(command_staging_.data() + prefix_size,
                layout_.command_words,
                next_slot,
                max_batch_size - prefix_size);
      while (count < max_batch_size && detail::command_ready(command_staging_[count])) {
        ++count;
      }
    }

    for (std::uint32_t i = 0; i < count; ++i) {
      auto const row_id = static_cast<std::uint32_t>(command_staging_[i]);
      RAFT_EXPECTS(
        row_id < row_count_, "FlowANN queue received out-of-range cross-graph row %u", row_id);
      std::copy_n(cross_graph_ + static_cast<std::size_t>(row_id) * row_length_,
                  row_length_,
                  row_staging_.data() + static_cast<std::size_t>(i) * row_length_);
      result_sequence_staging_[i] = cpu_tail_ + i + 1;
    }
    write_ring(row_staging_.data(), layout_.rows, first_slot, count, row_length_);
    // Clearing the request is ordered before result publication. A producer
    // cannot reuse this slot until it acquires and releases that result.
    write_ring(empty_commands_.data(), layout_.command_words, first_slot, count);
    std::atomic_thread_fence(std::memory_order_release);
    write_ring(result_sequence_staging_.data(), layout_.result_sequence, first_slot, count);

    cpu_tail_ += count;
    if (collect_statistics_) { processed_commands_.fetch_add(count, std::memory_order_relaxed); }
    return true;
  }

  void publish_progress()
  {
    // cpu_tail is an observational snapshot, not a GPU synchronization input:
    // consumers use result_sequence and producers reclaim capacity via gpu_tail.
    // Publish before acknowledging a pause instead of writing on every batch.
    write(layout_.state + offsetof(detail::queue_state, cpu_tail), &cpu_tail_, sizeof(cpu_tail_));
  }

  [[nodiscard]] auto statistics() const noexcept -> queue_statistics
  {
    return {polls_.load(std::memory_order_relaxed),
            empty_polls_.load(std::memory_order_relaxed),
            processed_commands_.load(std::memory_order_relaxed)};
  }

 private:
  template <typename T>
  void read_ring(T* destination,
                 std::size_t base_offset,
                 std::uint32_t first_slot,
                 std::uint32_t count)
  {
    auto const first_count = std::min(count, detail::queue_capacity - first_slot);
    read(destination,
         base_offset + static_cast<std::size_t>(first_slot) * sizeof(T),
         static_cast<std::size_t>(first_count) * sizeof(T));
    if (first_count < count) {
      read(destination + first_count,
           base_offset,
           static_cast<std::size_t>(count - first_count) * sizeof(T));
    }
  }

  template <typename T>
  void write_ring(T const* source,
                  std::size_t base_offset,
                  std::uint32_t first_slot,
                  std::uint32_t count,
                  std::uint32_t elements_per_slot = 1)
  {
    auto const first_count = std::min(count, detail::queue_capacity - first_slot);
    auto const slot_bytes  = static_cast<std::size_t>(elements_per_slot) * sizeof(T);
    write(base_offset + static_cast<std::size_t>(first_slot) * slot_bytes,
          source,
          static_cast<std::size_t>(first_count) * slot_bytes);
    if (first_count < count) {
      write(base_offset,
            source + static_cast<std::size_t>(first_count) * elements_per_slot,
            static_cast<std::size_t>(count - first_count) * slot_bytes);
    }
  }

  void read(void* dst, std::size_t offset, std::size_t size)
  {
#ifdef CUVS_FLOWANN_USE_GDRCOPY
    gdrcopy_->read(dst, offset, size);
#else
    auto* staging = copy_staging_.get(size);
    RAFT_CUDA_TRY(cudaMemcpyAsync(staging,
                                  add_bytes<void>(storage_.get(), offset),
                                  size,
                                  cudaMemcpyDeviceToHost,
                                  copy_stream_.get()));
    RAFT_CUDA_TRY(cudaStreamSynchronize(copy_stream_.get()));
    std::memcpy(dst, staging, size);
#endif
  }

  void write(std::size_t offset, void const* src, std::size_t size)
  {
#ifdef CUVS_FLOWANN_USE_GDRCOPY
    gdrcopy_->write(offset, src, size);
#else
    auto* staging = copy_staging_.get(size);
    std::memcpy(staging, src, size);
    RAFT_CUDA_TRY(cudaMemcpyAsync(add_bytes<void>(storage_.get(), offset),
                                  staging,
                                  size,
                                  cudaMemcpyHostToDevice,
                                  copy_stream_.get()));
    RAFT_CUDA_TRY(cudaStreamSynchronize(copy_stream_.get()));
#endif
  }

  std::uint32_t const* cross_graph_;
  std::uint64_t row_count_;
  std::uint32_t row_length_;
  bool collect_statistics_;
  queue_layout layout_;
  cuda_allocation storage_;
  detail::queue_view view_{};
  unsigned long long cpu_tail_{};
  std::atomic<std::uint64_t> polls_{};
  std::atomic<std::uint64_t> empty_polls_{};
  std::atomic<std::uint64_t> processed_commands_{};
  std::vector<unsigned long long> empty_commands_;
  std::vector<unsigned long long> command_staging_;
  std::vector<unsigned long long> result_sequence_staging_;
  std::vector<std::uint32_t> row_staging_;
#ifdef CUVS_FLOWANN_USE_GDRCOPY
  std::optional<gdrcopy_mapping> gdrcopy_;
#else
  cuda_stream copy_stream_;
  cuda_host_buffer copy_staging_;
#endif
};

}  // namespace

struct search_context::impl {
  explicit impl(raft::host_matrix_view<const std::uint32_t, int64_t, raft::row_major> cross_graph,
                queue_params params)
    : params_(params),
      row_count_(static_cast<std::uint64_t>(cross_graph.extent(0))),
      row_length_(static_cast<std::uint32_t>(cross_graph.extent(1)))
  {
    RAFT_CUDA_TRY(cudaGetDevice(&device_id_));
    RAFT_EXPECTS(cross_graph.extent(0) > 0, "FlowANN cross graph must not be empty");
    RAFT_EXPECTS(cross_graph.extent(1) > 0,
                 "FlowANN cross graph must contain a neighbor-count column");
    RAFT_EXPECTS(params.num_queues > 0, "FlowANN must use at least one command queue");
    RAFT_EXPECTS(cross_graph.extent(1) <= std::numeric_limits<std::uint32_t>::max(),
                 "FlowANN cross-graph row length exceeds uint32");

    queues_.reserve(params.num_queues);
    std::vector<detail::queue_view> views;
    views.reserve(params.num_queues);
    for (std::uint32_t i = 0; i < params.num_queues; ++i) {
      queues_.push_back(
        std::make_unique<host_queue>(cross_graph.data_handle(),
                                     static_cast<std::uint64_t>(cross_graph.extent(0)),
                                     static_cast<std::uint32_t>(cross_graph.extent(1)),
                                     params.collect_statistics));
      views.push_back(queues_.back()->view());
    }

    RAFT_CUDA_TRY(cudaMalloc(&device_views_, views.size() * sizeof(detail::queue_view)));
    try {
      RAFT_CUDA_TRY(cudaMemcpy(device_views_,
                               views.data(),
                               views.size() * sizeof(detail::queue_view),
                               cudaMemcpyHostToDevice));

      pollers_.reserve(queues_.size());
      for (std::uint32_t i = 0; i < queues_.size(); ++i) {
        pollers_.emplace_back([this, queue = queues_[i].get(), i](std::stop_token stop_token) {
          poll(stop_token, *queue, i);
        });
      }

      std::unique_lock lock(pause_mutex_);
      pause_cv_.wait(lock, [this] { return paused_pollers_ + failed_pollers_ == pollers_.size(); });
      auto error = poller_error_;
      lock.unlock();
      if (error != nullptr) { std::rethrow_exception(error); }
    } catch (...) {
      stop();
      throw;
    }
  }

  ~impl() { stop(); }

  void throw_if_failed()
  {
    if (poller_error_ != nullptr) { std::rethrow_exception(poller_error_); }
  }

  [[nodiscard]] auto should_run() const noexcept -> bool
  {
    return manually_resumed_ || active_sessions_ > 0 || active_searches_ > 0;
  }

  void update_pollers(std::unique_lock<std::mutex>& lock)
  {
    while (true) {
      throw_if_failed();
      if (should_run()) {
        if (paused_.load(std::memory_order_acquire)) {
          paused_.store(false, std::memory_order_release);
          pause_cv_.notify_all();
        }
        pause_cv_.wait(lock, [this] {
          return !should_run() || paused_pollers_ == 0 || poller_error_ != nullptr;
        });
        throw_if_failed();
        if (should_run() && paused_pollers_ == 0) { return; }
      } else {
        if (!paused_.load(std::memory_order_acquire)) {
          paused_.store(true, std::memory_order_release);
          pause_cv_.notify_all();
        }
        pause_cv_.wait(lock, [this] {
          return should_run() || paused_pollers_ + failed_pollers_ == pollers_.size() ||
                 poller_error_ != nullptr;
        });
        throw_if_failed();
        if (!should_run() && paused_pollers_ + failed_pollers_ == pollers_.size()) { return; }
      }
    }
  }

  void stop() noexcept
  {
    for (auto& poller : pollers_) {
      poller.request_stop();
    }
    {
      std::lock_guard lock(pause_mutex_);
      paused_.store(false, std::memory_order_release);
    }
    pause_cv_.notify_all();
    pollers_.clear();
    if (device_views_ != nullptr) {
      RAFT_CUDA_TRY_NO_THROW(cudaFree(device_views_));
      device_views_ = nullptr;
    }
  }

  void pause()
  {
    std::unique_lock lock(pause_mutex_);
    RAFT_EXPECTS(active_sessions_ == 0, "Cannot pause while a FlowANN search session is active");
    manually_resumed_ = false;
    ++pending_pause_requests_;
    try {
      pause_cv_.wait(lock, [this] { return active_searches_ == 0 || poller_error_ != nullptr; });
      throw_if_failed();
      update_pollers(lock);
    } catch (...) {
      --pending_pause_requests_;
      pause_cv_.notify_all();
      throw;
    }
    --pending_pause_requests_;
    pause_cv_.notify_all();
  }

  void resume()
  {
    std::unique_lock lock(pause_mutex_);
    pause_cv_.wait(lock,
                   [this] { return pending_pause_requests_ == 0 || poller_error_ != nullptr; });
    throw_if_failed();
    manually_resumed_ = true;
    update_pollers(lock);
  }

  void acquire_search()
  {
    std::unique_lock lock(pause_mutex_);
    pause_cv_.wait(lock,
                   [this] { return pending_pause_requests_ == 0 || poller_error_ != nullptr; });
    throw_if_failed();
    manually_resumed_ = false;
    ++active_searches_;
    try {
      update_pollers(lock);
    } catch (...) {
      --active_searches_;
      pause_cv_.notify_all();
      throw;
    }
  }

  void release_search()
  {
    std::unique_lock lock(pause_mutex_);
    RAFT_EXPECTS(active_searches_ > 0, "Cannot release an inactive FlowANN search lease");
    --active_searches_;
    if (active_searches_ == 0) {
      manually_resumed_ = false;
      pause_cv_.notify_all();
      update_pollers(lock);
    }
  }

  void begin_session()
  {
    std::unique_lock lock(pause_mutex_);
    pause_cv_.wait(lock,
                   [this] { return pending_pause_requests_ == 0 || poller_error_ != nullptr; });
    throw_if_failed();
    ++active_sessions_;
    try {
      update_pollers(lock);
    } catch (...) {
      --active_sessions_;
      throw;
    }
  }

  void end_session()
  {
    std::unique_lock lock(pause_mutex_);
    RAFT_EXPECTS(active_sessions_ > 0, "Cannot end an inactive FlowANN search session");
    --active_sessions_;
    update_pollers(lock);
  }

  void poll(std::stop_token stop_token, host_queue& queue, std::uint32_t queue_id) noexcept
  {
    try {
      RAFT_CUDA_TRY(cudaSetDevice(device_id_));
      bind_to_physical_cpu(queue_id);
      while (!stop_token.stop_requested()) {
        if (paused_.load(std::memory_order_acquire)) {
          queue.publish_progress();
          std::unique_lock lock(pause_mutex_);
          ++paused_pollers_;
          pause_cv_.notify_all();
          pause_cv_.wait(lock, [this, &stop_token] {
            return stop_token.stop_requested() || !paused_.load(std::memory_order_acquire);
          });
          --paused_pollers_;
          pause_cv_.notify_all();
          if (stop_token.stop_requested()) { break; }
        }

        auto const processed = queue.poll_batch();
        if (!processed) {
#if defined(__x86_64__) || defined(__i386__)
          for (std::uint32_t i = 0; i < params_.empty_pause; ++i) {
            __asm__ __volatile__("pause");
          }
#else
          std::this_thread::yield();
#endif
        }
      }
    } catch (...) {
      std::lock_guard lock(pause_mutex_);
      if (poller_error_ == nullptr) { poller_error_ = std::current_exception(); }
      ++failed_pollers_;
      pause_cv_.notify_all();
    }
  }

  [[nodiscard]] auto statistics() const noexcept -> queue_statistics
  {
    queue_statistics total{};
    for (auto const& queue : queues_) {
      auto const value = queue->statistics();
      total.polls += value.polls;
      total.empty_polls += value.empty_polls;
      total.processed_commands += value.processed_commands;
    }
    return total;
  }

  queue_params params_;
  int device_id_{};
  std::uint64_t row_count_;
  std::uint32_t row_length_;
  std::vector<std::unique_ptr<host_queue>> queues_;
  std::vector<std::jthread> pollers_;
  detail::queue_view* device_views_{};
  std::atomic<bool> paused_{true};
  mutable std::mutex pause_mutex_;
  std::condition_variable pause_cv_;
  std::size_t paused_pollers_{};
  std::size_t failed_pollers_{};
  std::size_t active_searches_{};
  std::size_t active_sessions_{};
  bool manually_resumed_{};
  std::size_t pending_pause_requests_{};
  std::exception_ptr poller_error_{};
};

search_context::search_context(
  raft::host_matrix_view<const std::uint32_t, int64_t, raft::row_major> cross_graph,
  queue_params params)
  : impl_(std::make_unique<impl>(cross_graph, params))
{
}

search_context::~search_context() = default;

search_context::search_context(search_context&&) noexcept = default;

auto search_context::operator=(search_context&&) noexcept -> search_context& = default;

search_session::search_session(search_context& context) : context_(&context)
{
  RAFT_EXPECTS(context.impl_ != nullptr, "Cannot start a session on a moved-from search context");
  context.impl_->begin_session();
}

search_session::~search_session() { reset(); }

search_session::search_session(search_session&& other) noexcept
  : context_(std::exchange(other.context_, nullptr))
{
}

auto search_session::operator=(search_session&& other) noexcept -> search_session&
{
  if (this != &other) {
    reset();
    context_ = std::exchange(other.context_, nullptr);
  }
  return *this;
}

void search_session::reset() noexcept
{
  if (context_ == nullptr) { return; }
  auto* context = std::exchange(context_, nullptr);
  try {
    context->impl_->end_session();
  } catch (...) {
  }
}

void search_context::pause()
{
  RAFT_EXPECTS(impl_ != nullptr, "Cannot pause a moved-from FlowANN search context");
  impl_->pause();
}

void search_context::resume()
{
  RAFT_EXPECTS(impl_ != nullptr, "Cannot resume a moved-from FlowANN search context");
  impl_->resume();
}

auto search_context::num_queues() const noexcept -> std::uint32_t
{
  return impl_ == nullptr ? 0 : static_cast<std::uint32_t>(impl_->queues_.size());
}

auto search_context::statistics() const noexcept -> queue_statistics
{
  return impl_ == nullptr ? queue_statistics{} : impl_->statistics();
}

auto detail::search_context_access::device_queues(search_context const& context) noexcept
  -> queue_view*
{
  return context.impl_ == nullptr ? nullptr : context.impl_->device_views_;
}

auto detail::search_context_access::device_id(search_context const& context) noexcept -> int
{
  return context.impl_ == nullptr ? -1 : context.impl_->device_id_;
}

auto detail::search_context_access::num_queues(search_context const& context) noexcept
  -> std::uint32_t
{
  return context.num_queues();
}

auto detail::search_context_access::row_count(search_context const& context) noexcept
  -> std::uint64_t
{
  return context.impl_ == nullptr ? 0 : context.impl_->row_count_;
}

auto detail::search_context_access::row_length(search_context const& context) noexcept
  -> std::uint32_t
{
  return context.impl_ == nullptr ? 0 : context.impl_->row_length_;
}

void detail::search_context_access::acquire_search(search_context& context)
{
  RAFT_EXPECTS(context.impl_ != nullptr, "Cannot search with a moved-from FlowANN search context");
  context.impl_->acquire_search();
}

void detail::search_context_access::release_search(search_context& context)
{
  RAFT_EXPECTS(context.impl_ != nullptr, "Cannot search with a moved-from FlowANN search context");
  context.impl_->release_search();
}

auto detail::search_context_access::is_paused(search_context const& context) noexcept -> bool
{
  return context.impl_ == nullptr || context.impl_->paused_.load(std::memory_order_acquire);
}

auto detail::search_context_access::pending_pause_requests(search_context const& context) noexcept
  -> std::size_t
{
  if (context.impl_ == nullptr) { return 0; }
  std::lock_guard lock(context.impl_->pause_mutex_);
  return context.impl_->pending_pause_requests_;
}

}  // namespace cuvs::neighbors::cagra::experimental::flowann
