/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cuda_stub.hpp"  // cudaStream_t

#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuvs::bench {

/** Benchmark mode: measuring latency vs throughput. */
enum class Mode {
  kThroughput,  // See how many vectors we can push through
  kLatency      // See how fast we can push a vector through
};

enum class MemoryType {
  kHost,
  kHostMmap,
  kHostPinned,
  kDevice,
  kManaged,
};

/** Request 2MB huge pages support for an allocation */
enum class HugePages {
  /** Don't use huge pages if possible. */
  kDisable = 0,
  /** Enable huge pages if possible, ignore otherwise. */
  kAsk = 1,
  /** Enable huge pages if possible, warn the user otherwise. */
  kRequire = 2,
  /** Force enable huge pages, throw an exception if not possible. */
  kDemand = 3
};

enum class Metric {
  kInnerProduct,
  kEuclidean,
};

inline auto parse_metric(const std::string& metric_str) -> Metric
{
  if (metric_str == "inner_product") {
    return cuvs::bench::Metric::kInnerProduct;
  } else if (metric_str == "euclidean") {
    return cuvs::bench::Metric::kEuclidean;
  } else {
    throw std::runtime_error("invalid metric: '" + metric_str + "'");
  }
}

inline auto parse_memory_type(const std::string& memory_type) -> MemoryType
{
  if (memory_type == "host") {
    return MemoryType::kHost;
  } else if (memory_type == "mmap") {
    return MemoryType::kHostMmap;
  } else if (memory_type == "pinned") {
    return MemoryType::kHostPinned;
  } else if (memory_type == "device") {
    return MemoryType::kDevice;
  } else if (memory_type == "managed") {
    return MemoryType::kManaged;
  } else {
    throw std::runtime_error("invalid memory type: '" + memory_type + "'");
  }
}

struct algo_property {
  MemoryType dataset_memory_type;
  // neighbors/distances should have same memory type as queries
  MemoryType query_memory_type;
};

class algo_base {
 public:
  using index_type = int64_t;

  inline algo_base(Metric metric, int dim) : metric_(metric), dim_(dim) {}
  virtual ~algo_base() noexcept = default;

 protected:
  Metric metric_;
  int dim_;
};

/**
 * The GPU-based algorithms, which do not perform CPU synchronization at the end of their build or
 * search methods, must implement this interface.
 *
 * The `cuda_timer` / `cuda_lap`  from `util.hpp` uses this stream to record GPU times with events
 * and, if necessary, also synchronize (via events) between iterations.
 *
 * If the algo does not implement this interface, GPU timings are disabled.
 */
class algo_gpu {
 public:
  /**
   * Return the main cuda stream for this algorithm.
   * If any work is done in multiple streams, they should synchornize with the main stream at the
   * end.
   */
  [[nodiscard]] virtual auto get_sync_stream() const noexcept -> cudaStream_t = 0;
  /**
   * By default a GPU algorithm uses a fixed stream to order GPU operations.
   * However, an algorithm may need to synchronize with the host at the end of its execution.
   * In that case, also synchronizing with a benchmark event would put it at disadvantage.
   *
   * We can disable event sync by passing `false` here
   *   - ONLY IF THE ALGORITHM HAS PRODUCED ITS OUTPUT BY THE TIME IT SYNCHRONIZES WITH CPU.
   */
  [[nodiscard]] virtual auto uses_stream() const noexcept -> bool { return true; }
  virtual ~algo_gpu() noexcept = default;
};

template <typename T>
class algo : public algo_base {
 public:
  struct search_param {
    virtual ~search_param() = default;
    [[nodiscard]] virtual auto needs_dataset() const -> bool { return false; };
  };

  inline algo(Metric metric, int dim) : algo_base(metric, dim) {}
  ~algo() noexcept override = default;

  virtual void build(const T* dataset, size_t nrow) = 0;

  virtual void set_search_param(const search_param& param, const void* filter_bitset) = 0;
  // TODO(snanditale): this assumes that an algorithm can always return k results.
  // This is not always possible.
  virtual void search(const T* queries,
                      int batch_size,
                      int k,
                      algo_base::index_type* neighbors,
                      float* distances) const = 0;

  virtual void save(const std::string& file) const = 0;
  virtual void load(const std::string& file)       = 0;

  [[nodiscard]] virtual auto get_preference() const -> algo_property = 0;

  // Some algorithms don't save the building dataset in their indices.
  // So they should be given the access to that dataset during searching.
  // The advantage of this way is that index has smaller size
  // and many indices can share one dataset.
  //
  // search_param::needs_dataset() of such algorithm should be true,
  // and set_search_dataset() should save the passed-in pointer somewhere.
  // The client code should call set_search_dataset() before searching,
  // and should not release dataset before searching is finished.
  virtual void set_search_dataset(const T* /*dataset*/, size_t /*nrow*/) {};

  /**
   * Make a shallow copy of the algo wrapper that shares the resources and ensures thread-safe
   * access to them. */
  virtual auto copy() -> std::unique_ptr<algo<T>> = 0;
};

}  // namespace cuvs::bench

#define REGISTER_ALGO_INSTANCE(DataT)                                                              \
  template auto cuvs::bench::create_algo<DataT>(                                                   \
    const std::string&, const std::string&, int, const nlohmann::json&)                            \
    -> std::unique_ptr<cuvs::bench::algo<DataT>>;                                                  \
  template auto cuvs::bench::create_search_param<DataT>(const std::string&, const nlohmann::json&) \
    -> std::unique_ptr<typename cuvs::bench::algo<DataT>::search_param>;
