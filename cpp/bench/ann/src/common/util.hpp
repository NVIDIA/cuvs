/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "ann_types.hpp"
#include "cuda_stub.hpp"  // cuda-related utils

#ifndef BUILD_CPU_ONLY
#include <rmm/cuda_stream.hpp>
#endif

#if __has_include(<nvtx3/nvToolsExt.h>)
#define ANN_BENCH_NVTX3_HEADERS_FOUND
#include <nvtx3/nvToolsExt.h>
#endif

#include <sched.h>  // sched_getaffinity
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace cuvs::bench {

/**
 * Current thread id as given by the benchmark State.
 * It's populated on every call of a benchmark case.
 * It's relevant in the 'throughput' mode of the search benchmarks,
 * where some algorithms might want to coordinate allocation of the resources.
 */
inline thread_local int benchmark_thread_id = 0;
/**
 * Total concurrent thread count as given by the benchmark State.
 * It's populated on every call of a benchmark case.
 * It's relevant in the 'throughput' mode of the search benchmarks,
 * where some algorithms might want to coordinate allocation of the resources.
 */
inline thread_local int benchmark_n_threads = 1;

struct cuda_timer {
 private:
  std::optional<cudaStream_t> stream_;
  cudaEvent_t start_{nullptr};
  cudaEvent_t stop_{nullptr};
  double total_time_{0};

  template <typename AnnT>
  static inline auto extract_stream(AnnT* algo) -> std::optional<cudaStream_t>
  {
    auto gpu_ann = dynamic_cast<algo_gpu*>(algo);
    if (gpu_ann != nullptr && gpu_ann->uses_stream()) {
      return std::make_optional(gpu_ann->get_sync_stream());
    }
    return std::nullopt;
  }

 public:
  struct cuda_lap {
   private:
    cudaStream_t stream_;
    cudaEvent_t start_;
    cudaEvent_t stop_;
    double& total_time_;

   public:
    cuda_lap(cudaStream_t stream, cudaEvent_t start, cudaEvent_t stop, double& total_time)
      : start_(start), stop_(stop), stream_(stream), total_time_(total_time)
    {
#ifndef BUILD_CPU_ONLY
      cudaEventRecord(start_, stream_);
#endif
    }
    cuda_lap() = delete;

    ~cuda_lap() noexcept
    {
#ifndef BUILD_CPU_ONLY
      cudaEventRecord(stop_, stream_);
      cudaEventSynchronize(stop_);
      float milliseconds = 0.0f;
      cudaEventElapsedTime(&milliseconds, start_, stop_);
      total_time_ += milliseconds / 1000.0;
#endif
    }
  };

  explicit cuda_timer(std::optional<cudaStream_t> stream) : stream_{stream}
  {
#ifndef BUILD_CPU_ONLY
    if (stream_.has_value()) {
      cudaEventCreate(&stop_);
      cudaEventCreate(&start_);
    }
#endif
  }

  template <typename AnnT>
  explicit cuda_timer(const std::unique_ptr<AnnT>& algo) : cuda_timer{extract_stream(algo.get())}
  {
  }

  ~cuda_timer() noexcept
  {
#ifndef BUILD_CPU_ONLY
    if (stream_.has_value()) {
      cudaStreamSynchronize(stream_.value());
      cudaEventDestroy(start_);
      cudaEventDestroy(stop_);
    }
#endif
  }

  cuda_timer()                                     = delete;
  cuda_timer(cuda_timer const&)                    = delete;
  cuda_timer(cuda_timer&&)                         = delete;
  auto operator=(cuda_timer const&) -> cuda_timer& = delete;
  auto operator=(cuda_timer&&) -> cuda_timer&      = delete;

  [[nodiscard]] auto stream() const -> std::optional<cudaStream_t> { return stream_; }

  [[nodiscard]] auto active() const -> bool { return stream_.has_value(); }

  [[nodiscard]] auto total_time() const -> double { return total_time_; }

  [[nodiscard]] auto lap(bool enabled = true) -> std::optional<cuda_timer::cuda_lap>
  {
    return enabled && stream_.has_value()
             ? std::make_optional<cuda_timer::cuda_lap>(stream_.value(), start_, stop_, total_time_)
             : std::nullopt;
  }
};

#ifndef BUILD_CPU_ONLY
namespace detail {
inline std::vector<rmm::cuda_stream> global_stream_pool(0);
inline std::mutex gsp_mutex;
}  // namespace detail
#endif

/**
 * Get a stream associated with the current benchmark thread.
 *
 * Note, the streams are reused between the benchmark cases.
 * This makes it easier to profile and analyse multiple benchmark cases in one timeline using tools
 * like nsys.
 */
inline auto get_stream_from_global_pool() -> cudaStream_t
{
#ifndef BUILD_CPU_ONLY
  std::lock_guard guard(detail::gsp_mutex);
  if (static_cast<int>(detail::global_stream_pool.size()) < benchmark_n_threads) {
    while (static_cast<int>(detail::global_stream_pool.size()) < benchmark_n_threads) {
      detail::global_stream_pool.emplace_back(rmm::cuda_stream::flags::non_blocking);
    }
  }
  return detail::global_stream_pool[benchmark_thread_id].view();
#else
  return nullptr;
#endif
}

/** The workspace buffer for use thread-locally. */
struct ws_buffer {
  explicit ws_buffer(size_t size, cudaStream_t stream) : size_{size}, stream_{stream} {}
  ws_buffer()                                    = delete;
  ws_buffer(ws_buffer&&)                         = delete;
  auto operator=(ws_buffer&&) -> ws_buffer&      = delete;
  ws_buffer(const ws_buffer&)                    = delete;
  auto operator=(const ws_buffer&) -> ws_buffer& = delete;
  ~ws_buffer() noexcept
  {
#ifndef BUILD_CPU_ONLY
    if (data_device_ != nullptr) {
      cudaFreeAsync(data_device_, stream_);
      cudaStreamSynchronize(stream_);
    }
    if (data_host_ != nullptr) { cudaFreeHost(data_host_); }
#else
    if (data_host_ != nullptr) { free(data_host_); }
#endif
  }

  [[nodiscard]] auto size() const noexcept { return size_; }
  [[nodiscard]] auto data(MemoryType loc) const noexcept -> void*
  {
    if (size_ == 0) { return nullptr; }
    switch (loc) {
#ifndef BUILD_CPU_ONLY
      case MemoryType::kDevice: {
        if (data_device_ == nullptr) {
          cudaMallocAsync(&data_device_, size_, stream_);
          cudaStreamSynchronize(stream_);
          needs_cleanup_device_ = false;
        } else if (needs_cleanup_device_) {
          cudaMemsetAsync(data_device_, 0, size_, stream_);
          cudaStreamSynchronize(stream_);
          needs_cleanup_device_ = false;
        }
        return data_device_;
      }
#endif
      default: {
        if (data_host_ == nullptr) {
#ifndef BUILD_CPU_ONLY
          cudaMallocHost(&data_host_, size_);
#else
          data_host_ = malloc(size_);
#endif
          needs_cleanup_host_ = false;
        } else if (needs_cleanup_host_) {
          memset(data_host_, 0, size_);
          needs_cleanup_host_ = false;
        }
        return data_host_;
      }
    }
  }

  void transfer_data(MemoryType dst, MemoryType src)
  {
    // The destination is overwritten and thus does not need cleanup
    if (dst == MemoryType::kDevice) {
      needs_cleanup_device_ = false;
    } else {
      needs_cleanup_host_ = false;
    }
    auto dst_ptr = data(dst);
    auto src_ptr = data(src);
    if (dst_ptr == src_ptr) { return; }
#ifndef BUILD_CPU_ONLY
    cudaMemcpyAsync(dst_ptr, src_ptr, size_, cudaMemcpyDefault, stream_);
    cudaStreamSynchronize(stream_);
#endif
  }

  /** Mark the buffer for reuse - it needs to be cleared to make sure the previous results are not
   * leaked to the new iteration. */
  void reuse()
  {
    needs_cleanup_host_   = true;
    needs_cleanup_device_ = true;
  }

 private:
  size_t size_{0};
  cudaStream_t stream_               = nullptr;
  mutable void* data_host_           = nullptr;
  mutable void* data_device_         = nullptr;
  mutable bool needs_cleanup_host_   = false;
  mutable bool needs_cleanup_device_ = false;
};

namespace detail {
inline std::vector<std::unique_ptr<ws_buffer>> global_result_buffer_pool(0);
inline std::mutex grp_mutex;
}  // namespace detail

/**
 * Get a result buffer associated with the current benchmark thread.
 *
 * Note, the allocations are reused between the benchmark cases.
 * This reduces the setup overhead and number of times the context is being blocked
 * (this is relevant if there is a persistent kernel running across multiples benchmark cases).
 */
inline auto get_result_buffer_from_global_pool(size_t size) -> ws_buffer&
{
  auto stream = get_stream_from_global_pool();
  auto& rb    = [stream, size]() -> ws_buffer& {
    std::lock_guard guard(detail::grp_mutex);
    if (static_cast<int>(detail::global_result_buffer_pool.size()) < benchmark_n_threads) {
      detail::global_result_buffer_pool.resize(benchmark_n_threads);
    }
    auto& rb = detail::global_result_buffer_pool[benchmark_thread_id];
    if (!rb || rb->size() < size) {
      rb = std::make_unique<ws_buffer>(size, stream);
    } else {
      rb->reuse();
    }
    return *rb;
  }();
  return rb;
}

namespace detail {
inline std::vector<std::unique_ptr<ws_buffer>> global_tmp_buffer_pool(0);
inline std::mutex gtp_mutex;
}  // namespace detail

/**
 * Global temporary buffer pool for use by algorithms.
 * In contrast to `get_result_buffer_from_global_pool`, the content of these buffers is never
 * initialized.
 */
inline auto get_tmp_buffer_from_global_pool(size_t size) -> ws_buffer&
{
  auto stream = get_stream_from_global_pool();
  auto& rb    = [stream, size]() -> ws_buffer& {
    std::lock_guard guard(detail::gtp_mutex);
    if (static_cast<int>(detail::global_tmp_buffer_pool.size()) < benchmark_n_threads) {
      detail::global_tmp_buffer_pool.resize(benchmark_n_threads);
    }
    auto& rb = detail::global_tmp_buffer_pool[benchmark_thread_id];
    if (!rb || rb->size() < size) { rb = std::make_unique<ws_buffer>(size, stream); }
    return *rb;
  }();
  return rb;
}

/**
 * Delete all streams and memory allocations in the global pool.
 * It's called at the end of the `main` function - before global/static variables and cuda context
 * is destroyed - to make sure they are destroyed gracefully and correctly seen by analysis tools
 * such as nsys.
 */
inline void reset_global_device_resources()
{
#ifndef BUILD_CPU_ONLY
  std::lock_guard guard(detail::gsp_mutex);
  detail::global_tmp_buffer_pool.resize(0);
  detail::global_result_buffer_pool.resize(0);
  detail::global_stream_pool.resize(0);
#endif
}

inline auto host_info()
{
  std::vector<std::tuple<std::string, std::string>> props;

  // Memory info
  auto page_size = sysconf(_SC_PAGE_SIZE);
  props.emplace_back("host_pagesize", std::to_string(page_size));

  struct sysinfo sys_info;
  if (sysinfo(&sys_info) != -1) {
    props.emplace_back("host_total_ram_size",
                       std::to_string(size_t(sys_info.totalram) * size_t(sys_info.mem_unit)));
    props.emplace_back("host_total_swap_size",
                       std::to_string(size_t(sys_info.totalswap) * size_t(sys_info.mem_unit)));
  }

  // CPU info
  int host_processors_configured = sysconf(_SC_NPROCESSORS_CONF);
  props.emplace_back("host_processors_sysconf", std::to_string(host_processors_configured));
  std::vector<uint8_t> affinity_mask_buf(CPU_ALLOC_SIZE(host_processors_configured));
  cpu_set_t* affinity_mask = reinterpret_cast<cpu_set_t*>(affinity_mask_buf.data());
  sched_getaffinity(0, affinity_mask_buf.size(), affinity_mask);
  uint64_t cpu_freq_min    = 0;
  uint64_t cpu_freq_max    = 0;
  int host_processors_used = 0;
  int host_cores_used      = 0;
  std::set<std::pair<int, int>> host_cores_selected{};  // pairs of (socket_id, core_id)
  for (int cpu_id = 0; cpu_id < host_processors_configured; cpu_id++) {
    if (CPU_ISSET_S(cpu_id, affinity_mask_buf.size(), affinity_mask) == 0) { continue; }
    host_processors_used++;
    std::string cpu_fpath = "/sys/devices/system/cpu/cpu" + std::to_string(cpu_id);
    if (!std::filesystem::exists(cpu_fpath)) { continue; }

    int this_cpu_core          = 0;
    int this_cpu_package       = 0;
    uint64_t this_cpu_freq_min = 0;
    uint64_t this_cpu_freq_max = 0;
    std::ifstream(cpu_fpath + "/topology/core_id") >> this_cpu_core;
    std::ifstream(cpu_fpath + "/topology/physical_package_id") >> this_cpu_package;
    std::ifstream(cpu_fpath + "/cpufreq/scaling_min_freq") >> this_cpu_freq_min;
    std::ifstream(cpu_fpath + "/cpufreq/scaling_max_freq") >> this_cpu_freq_max;
    host_cores_selected.insert(std::make_pair(this_cpu_package, this_cpu_core));
    cpu_freq_min = cpu_freq_min == 0
                     ? (this_cpu_freq_min * 1000ull)
                     : std::min<uint64_t>(this_cpu_freq_min * 1000ull, cpu_freq_min);
    cpu_freq_max = std::max<uint64_t>(this_cpu_freq_max * 1000ull, cpu_freq_max);
  }
  host_cores_used = host_cores_selected.size();
  if (host_processors_used != 0) {
    props.emplace_back("host_processors_used", std::to_string(host_processors_used));
  }
  if (host_cores_used != 0) {
    props.emplace_back("host_cores_used", std::to_string(host_cores_used));
  }
  if (cpu_freq_min != 0) { props.emplace_back("host_cpu_freq_min", std::to_string(cpu_freq_min)); }
  if (cpu_freq_max != 0) { props.emplace_back("host_cpu_freq_max", std::to_string(cpu_freq_max)); }

  return props;
}

inline auto cuda_info()
{
  std::vector<std::tuple<std::string, std::string>> props;
#ifndef BUILD_CPU_ONLY
  int dev, driver = 0, runtime = 0;
  cudaDriverGetVersion(&driver);
  cudaRuntimeGetVersion(&runtime);

  cudaDeviceProp device_prop;
  auto err_code = cudaGetDevice(&dev);
  if (err_code == cudaErrorNoDevice || err_code == cudaErrorInvalidDevice) {
    throw std::runtime_error{"cuda_info: call to cudaGetDevice failed with code " +
                             std::to_string(err_code) +
                             ". If you are running on a CPU-only machine, please "
                             "use the CPU package instead."};
  } else if (err_code != cudaSuccess) {
    throw std::runtime_error{"cuda_info: call to cudaGetDevice failed with code " +
                             std::to_string(err_code)};
  }
  err_code = cudaGetDeviceProperties(&device_prop, dev);
  if (err_code != cudaSuccess) {
    throw std::runtime_error{"cuda_info: call to cudaGetDeviceProperties failed with code " +
                             std::to_string(err_code)};
  }
  int clockRate       = 0;
  int memoryClockRate = 0;
  err_code            = cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, dev);
  err_code            = cudaDeviceGetAttribute(&memoryClockRate, cudaDevAttrMemoryClockRate, dev);

  props.emplace_back("gpu_name", std::string(device_prop.name));
  props.emplace_back("gpu_sm_count", std::to_string(device_prop.multiProcessorCount));
  props.emplace_back("gpu_sm_freq", std::to_string(clockRate * 1e3));
  props.emplace_back("gpu_mem_freq", std::to_string(memoryClockRate * 1e3));
  props.emplace_back("gpu_mem_bus_width", std::to_string(device_prop.memoryBusWidth));
  props.emplace_back("gpu_mem_global_size", std::to_string(device_prop.totalGlobalMem));
  props.emplace_back("gpu_mem_shared_size", std::to_string(device_prop.sharedMemPerMultiprocessor));
  props.emplace_back("gpu_driver_version",
                     std::to_string(driver / 1000) + "." + std::to_string((driver % 100) / 10));
  props.emplace_back("gpu_runtime_version",
                     std::to_string(runtime / 1000) + "." + std::to_string((runtime % 100) / 10));
  props.emplace_back("gpu_hostNativeAtomicSupported",
                     std::to_string(device_prop.hostNativeAtomicSupported));
  props.emplace_back("gpu_pageableMemoryAccess", std::to_string(device_prop.pageableMemoryAccess));
  props.emplace_back("gpu_pageableMemoryAccessUsesHostPageTables",
                     std::to_string(device_prop.pageableMemoryAccessUsesHostPageTables));
  props.emplace_back("gpu_gpuDirectRDMASupported",
                     std::to_string(device_prop.gpuDirectRDMASupported));
#endif
  return props;
}

struct nvtx_case {
#ifdef ANN_BENCH_NVTX3_HEADERS_FOUND
 private:
  std::string case_name_;
  std::array<char, 32> iter_name_{0};
  nvtxDomainHandle_t domain_;
  int64_t iteration_ = 0;
  nvtxEventAttributes_t case_attrib_{0};
  nvtxEventAttributes_t iter_attrib_{0};
#endif

 public:
  struct nvtx_lap {
#ifdef ANN_BENCH_NVTX3_HEADERS_FOUND
   private:
    nvtxDomainHandle_t domain_;

   public:
    nvtx_lap(nvtxDomainHandle_t domain, nvtxEventAttributes_t* attr) : domain_(domain)
    {
      nvtxDomainRangePushEx(domain_, attr);
    }
    nvtx_lap() = delete;
    ~nvtx_lap() noexcept { nvtxDomainRangePop(domain_); }
#endif
  };

#ifdef ANN_BENCH_NVTX3_HEADERS_FOUND
  explicit nvtx_case(std::string case_name)
    : case_name_(std::move(case_name)), domain_(nvtxDomainCreateA("algo benchmark"))
  {
    case_attrib_.version       = NVTX_VERSION;
    iter_attrib_.version       = NVTX_VERSION;
    case_attrib_.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    iter_attrib_.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    case_attrib_.colorType     = NVTX_COLOR_ARGB;
    iter_attrib_.colorType     = NVTX_COLOR_ARGB;
    case_attrib_.messageType   = NVTX_MESSAGE_TYPE_ASCII;
    iter_attrib_.messageType   = NVTX_MESSAGE_TYPE_ASCII;
    case_attrib_.message.ascii = case_name_.c_str();
    auto c                     = std::hash<std::string>{}(case_name_);
    case_attrib_.color         = c | 0xA0A0A0;
    nvtxDomainRangePushEx(domain_, &case_attrib_);
  }

  ~nvtx_case()
  {
    nvtxDomainRangePop(domain_);
    nvtxDomainDestroy(domain_);
  }
#else
  explicit nvtx_case(std::string) {}
#endif

  [[nodiscard]] auto lap() -> nvtx_case::nvtx_lap
  {
#ifdef ANN_BENCH_NVTX3_HEADERS_FOUND
    auto i     = iteration_++;
    uint32_t c = (i % 5);
    uint32_t r = 150 + c * 20;
    uint32_t g = 200 + c * 10;
    uint32_t b = 220 + c * 5;
    std::snprintf(iter_name_.data(), iter_name_.size(), "Lap %zd", i);
    iter_attrib_.message.ascii = iter_name_.data();
    iter_attrib_.color         = (r << 16) + (g << 8) + b;
    return nvtx_lap{domain_, &iter_attrib_};
#else
    return nvtx_lap{};
#endif
  }
};

/**
 * A progress tracker that allows syncing threads multiple times and resets the global
 * progress once the threads are done.
 */
struct progress_barrier {
  progress_barrier() = default;
  ~progress_barrier() noexcept
  {
    {
      // Lock makes sure the notified threads see the updates to `done_`.
      std::unique_lock lk(mutex_);
      done_.store(true, std::memory_order_relaxed);
      cv_.notify_all();
    }
    // This is the only place where the order of the updates to thread_progress_ and done_ is
    // important. They are not guarded by the mutex, and `done_` must not be reset to `true` by
    // other threads after the `total_progress_` is zero.
    // Hence the default memory order (std::memory_order_seq_cst).
    auto rem = total_progress_.fetch_sub(thread_progress_);
    if (rem == thread_progress_) {
      // the last thread to exit clears the progress state.
      done_.store(false);
    }
  }

  /**
   * Advance the progress counter by `n` and return the previous `progress` value.
   *
   * This can be used to track which thread arrives on the call site first.
   *
   * @return the previous progress counter value (before incrementing it by `n`).
   */
  auto arrive(int n)
  {
    thread_progress_ += n;
    // Lock makes sure the notified threads see the updates to `total_progress_`.
    std::unique_lock lk(mutex_);
    auto prev = total_progress_.fetch_add(n, std::memory_order_relaxed);
    cv_.notify_all();
    return prev;
  }

  /**
   * Wait till the progress counter reaches `n` or finishes abnormally.
   *
   * @return the latest observed value of the progress counter.
   */
  auto wait(int limit)
  {
    int cur = total_progress_.load(std::memory_order_relaxed);
    if (cur >= limit) { return cur; }
    auto done = done_.load(std::memory_order_relaxed);
    if (done) { return cur; }
    std::unique_lock lk(mutex_);
    while (cur < limit && !done) {
      using namespace std::chrono_literals;
      cv_.wait_for(lk, 10ms);
      cur  = total_progress_.load(std::memory_order_relaxed);
      done = done_.load(std::memory_order_relaxed);
    }
    return cur;
  }

 private:
  static inline std::atomic<int> total_progress_;
  static inline std::atomic<bool> done_;
  static inline std::mutex mutex_;
  static inline std::condition_variable cv_;
  int thread_progress_{0};
};

inline auto split(const std::string& s, char delimiter) -> std::vector<std::string>
{
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream iss(s);
  while (getline(iss, token, delimiter)) {
    if (!token.empty()) { tokens.push_back(token); }
  }
  return tokens;
}

inline auto file_exists(const std::string& filename) -> bool
{
  struct stat statbuf;
  if (stat(filename.c_str(), &statbuf) != 0) { return false; }
  return S_ISREG(statbuf.st_mode);
}

inline auto dir_exists(const std::string& dir) -> bool
{
  struct stat statbuf;
  if (stat(dir.c_str(), &statbuf) != 0) { return false; }
  return S_ISDIR(statbuf.st_mode);
}

inline auto create_dir(const std::string& dir) -> bool
{
  const auto path = split(dir, '/');

  std::string cwd;
  if (!dir.empty() && dir[0] == '/') { cwd += '/'; }

  for (const auto& p : path) {
    cwd += p + "/";
    if (!dir_exists(cwd)) {
      int ret = mkdir(cwd.c_str(), S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
      if (ret != 0) { return false; }
    }
  }
  return true;
}

inline void make_sure_parent_dir_exists(const std::string& file_path)
{
  const auto pos = file_path.rfind('/');
  if (pos != std::string::npos) {
    auto dir = file_path.substr(0, pos);
    if (!dir_exists(dir)) { create_dir(dir); }
  }
}

inline auto combine_path(const std::string& dir, const std::string& path)
{
  std::filesystem::path p_dir(dir);
  std::filesystem::path p_suf(path);
  return (p_dir / p_suf).string();
}

template <typename... Ts>
void log_with_level(const char* level, const Ts&... vs)
{
  char buf[20];
  auto now    = std::chrono::system_clock::now();
  auto now_tt = std::chrono::system_clock::to_time_t(now);
  size_t millis =
    std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count() %
    1000000ULL;
  std::strftime(buf, sizeof(buf), "%H:%M:%S", std::localtime(&now_tt));
  printf("[%s] [%s.%06zu] ", level, buf, millis);
  if constexpr (sizeof...(Ts) == 1) {
    printf("%s", vs...);
  } else {
    printf(vs...);
  }
  printf("\n");
  fflush(stdout);
}

template <typename... Ts>
void log_info(Ts&&... vs)
{
  log_with_level("I", std::forward<Ts>(vs)...);
}

template <typename... Ts>
void log_warn(Ts&&... vs)
{
  log_with_level("W", std::forward<Ts>(vs)...);
}

template <typename... Ts>
void log_error(Ts&&... vs)
{
  log_with_level("E", std::forward<Ts>(vs)...);
}

}  // namespace cuvs::bench
