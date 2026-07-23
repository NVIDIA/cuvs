/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../common/ann_types.hpp"
#include "../common/util.hpp"

#include <limits>

#include <disk_utils.h>
#include <index.h>
#include <linux_aligned_file_reader.h>
#include <omp.h>
#include <pq_flash_index.h>
#include <utils.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace cuvs::bench {

inline bool diskann_memory_artifact_exists(const std::string& file)
{
  std::error_code ec;
  return std::filesystem::exists(file, ec);
}

inline std::uintmax_t diskann_memory_artifact_size(const std::string& file)
{
  std::error_code ec;
  auto size = std::filesystem::file_size(file, ec);
  return ec ? 0 : size;
}

diskann::Metric parse_metric_to_diskann(cuvs::bench::Metric metric)
{
  if (metric == cuvs::bench::Metric::kInnerProduct) {
    return diskann::Metric::INNER_PRODUCT;
  } else if (metric == cuvs::bench::Metric::kEuclidean) {
    return diskann::Metric::L2;
  } else {
    throw std::runtime_error("currently only inner product and L2 supported for benchmarking");
  }
}

// TODO (tarangj): Remaining features are tracked at https://github.com/rapidsai/cuvs/issues/656
template <typename T>
class diskann_memory : public algo<T> {
 public:
  struct build_param {
    uint32_t R;
    uint32_t L_build;
    uint32_t build_pq_bytes = 0;
    float alpha             = 1.2;
    int num_threads         = omp_get_max_threads();
  };

  using search_param_base = typename algo<T>::search_param;
  struct search_param : public search_param_base {
    uint32_t L_search;
  };

  diskann_memory(Metric metric, int dim, const build_param& param);

  void build(const T* dataset, size_t nrow) override;

  void set_search_param(const search_param_base& param, const void* filter_bitset) override;

  void search(const T* queries,
              int batch_size,
              int k,
              algo_base::index_type* indices,
              float* distances) const override;

  void save(const std::string& index_file) const override;
  void load(const std::string& index_file) override;
  diskann_memory(const diskann_memory<T>& other) = default;
  std::unique_ptr<algo<T>> copy() override { return std::make_unique<diskann_memory<T>>(*this); }

  [[nodiscard]] auto get_preference() const -> algo_property override
  {
    algo_property property;
    property.dataset_memory_type = MemoryType::kHostMmap;
    property.query_memory_type   = MemoryType::kHost;
    return property;
  }

 private:
  std::shared_ptr<diskann::IndexWriteParameters> diskann_index_write_params_{nullptr};
  uint32_t max_points_;
  uint32_t build_pq_bytes_            = 0;
  uint32_t configured_build_pq_bytes_ = 0;
  uint32_t graph_degree_              = 0;
  uint32_t L_build_                   = 0;
  float alpha_                        = 0.0f;
  int num_threads_                    = 0;
  uint32_t L_search_                  = 0;
  Mode bench_mode_;
  std::string index_path_prefix_;
  std::shared_ptr<diskann::Index<T>> mem_index_{nullptr};
  void initialize_index_(size_t max_points);
};

template <typename T>
diskann_memory<T>::diskann_memory(Metric metric, int dim, const build_param& param)
  : algo<T>(metric, dim)
{
  assert(this->dim_ > 0);
  configured_build_pq_bytes_  = param.build_pq_bytes;
  graph_degree_               = param.R;
  L_build_                    = param.L_build;
  alpha_                      = param.alpha;
  num_threads_                = param.num_threads;
  diskann_index_write_params_ = std::make_shared<diskann::IndexWriteParameters>(
    diskann::IndexWriteParametersBuilder(param.L_build, param.R)
      .with_filter_list_size(0)
      .with_alpha(param.alpha)
      .with_saturate_graph(false)
      .with_num_threads(param.num_threads)
      .build());
}

template <typename T>
void diskann_memory<T>::initialize_index_(size_t max_points)
{
  this->mem_index_ = std::make_shared<diskann::Index<T>>(parse_metric_to_diskann(this->metric_),
                                                         this->dim_,
                                                         max_points,
                                                         diskann_index_write_params_,
                                                         nullptr,
                                                         0,
                                                         false,
                                                         false,
                                                         false,
                                                         build_pq_bytes_ > 0,
                                                         build_pq_bytes_,
                                                         false,
                                                         false);
}
template <typename T>
void diskann_memory<T>::build(const T* dataset, size_t nrow)
{
  log_info(
    "diskann_memory build start: nrow=%zu dim=%d graph_degree=%u L_build=%u alpha=%f "
    "configured_build_pq_bytes=%u build_pq_bytes=%u threads=%d",
    nrow,
    this->dim_,
    static_cast<unsigned>(graph_degree_),
    static_cast<unsigned>(L_build_),
    static_cast<double>(alpha_),
    static_cast<unsigned>(configured_build_pq_bytes_),
    static_cast<unsigned>(build_pq_bytes_),
    num_threads_);
  initialize_index_(nrow);
  try {
    mem_index_->build(dataset, nrow, std::vector<uint32_t>());
  } catch (const std::exception& e) {
    log_warn("diskann_memory build failed: nrow=%zu dim=%d error=%s", nrow, this->dim_, e.what());
    throw;
  }
  log_info("diskann_memory build finish: nrow=%zu dim=%d", nrow, this->dim_);
}

template <typename T>
void diskann_memory<T>::set_search_param(const search_param_base& param, const void* filter_bitset)
{
  if (filter_bitset != nullptr) { throw std::runtime_error("Filtering is not supported yet."); }
  auto sp   = dynamic_cast<const search_param&>(param);
  L_search_ = sp.L_search;
  log_info("diskann_memory set_search_param: L_search=%u dim=%d",
           static_cast<unsigned>(L_search_),
           this->dim_);
}

template <typename T>
void diskann_memory<T>::search(
  const T* queries, int batch_size, int k, algo_base::index_type* indices, float* distances) const
{
  static std::atomic<bool> search_logged{false};
  bool expected = false;
  if (search_logged.compare_exchange_strong(expected, true)) {
    log_info(
      "diskann_memory search first call: prefix=%s batch_size=%d k=%d L_search=%u dim=%d "
      "mode=%s benchmark_threads=%d",
      index_path_prefix_.c_str(),
      batch_size,
      k,
      static_cast<unsigned>(L_search_),
      this->dim_,
      bench_mode_ == Mode::kThroughput ? "throughput" : "latency",
      cuvs::bench::benchmark_n_threads);
  }

  for (int i = 0; i < batch_size; i++) {
    try {
      mem_index_->search(queries + i * this->dim_,
                         static_cast<size_t>(k),
                         L_search_,
                         reinterpret_cast<uint64_t*>(indices + i * k),
                         distances + i * k);
    } catch (const std::exception& e) {
      log_warn(
        "diskann_memory search failed: prefix=%s query_in_batch=%d batch_size=%d k=%d "
        "L_search=%u error=%s",
        index_path_prefix_.c_str(),
        i,
        batch_size,
        k,
        static_cast<unsigned>(L_search_),
        e.what());
      throw;
    }
  }
}

template <typename T>
void diskann_memory<T>::save(const std::string& index_file) const
{
  log_info("diskann_memory save start: file=%s dim=%d", index_file.c_str(), this->dim_);
  try {
    this->mem_index_->save(index_file.c_str());
  } catch (const std::exception& e) {
    log_warn("diskann_memory save failed: file=%s error=%s", index_file.c_str(), e.what());
    throw;
  }
  log_info("diskann_memory save finish: file=%s exists=%d bytes=%zu",
           index_file.c_str(),
           diskann_memory_artifact_exists(index_file) ? 1 : 0,
           static_cast<size_t>(diskann_memory_artifact_size(index_file)));
}

template <typename T>
void diskann_memory<T>::load(const std::string& index_file)
{
  index_path_prefix_ = index_file;

  bench_mode_ = (cuvs::bench::benchmark_n_threads > 1) ? Mode::kThroughput : Mode::kLatency;

  int load_threads = (bench_mode_ == Mode::kThroughput) ? cuvs::bench::benchmark_n_threads : 1;
  log_info(
    "diskann_memory load enter: prefix=%s dim=%d benchmark_threads=%d load_threads=%d "
    "mode=%s build_threads=%d",
    index_path_prefix_.c_str(),
    this->dim_,
    cuvs::bench::benchmark_n_threads,
    load_threads,
    bench_mode_ == Mode::kThroughput ? "throughput" : "latency",
    num_threads_);

  try {
    log_info("diskann_memory initialize_index start: prefix=%s max_points=0",
             index_path_prefix_.c_str());
    initialize_index_(0);
    log_info("diskann_memory initialize_index finish: prefix=%s", index_path_prefix_.c_str());
  } catch (const std::exception& e) {
    log_warn("diskann_memory initialize_index failed: prefix=%s error=%s",
             index_path_prefix_.c_str(),
             e.what());
    throw;
  } catch (...) {
    log_warn("diskann_memory initialize_index failed: prefix=%s unknown error",
             index_path_prefix_.c_str());
    throw;
  }

  log_info("diskann_memory load start: prefix=%s", index_path_prefix_.c_str());
  try {
    this->mem_index_->load(index_path_prefix_.c_str(), load_threads, 100);
  } catch (const std::exception& e) {
    log_warn(
      "diskann_memory load failed: prefix=%s error=%s", index_path_prefix_.c_str(), e.what());
    throw;
  } catch (...) {
    log_warn("diskann_memory load failed: prefix=%s unknown error", index_path_prefix_.c_str());
    throw;
  }
  log_info("diskann_memory load finish: prefix=%s", index_path_prefix_.c_str());
}

template <typename T>
class diskann_ssd : public algo<T> {
 public:
  struct build_param {
    uint32_t R;
    uint32_t L_build;
    uint32_t build_pq_bytes               = 0;
    float alpha                           = 1.2;
    int num_threads                       = omp_get_max_threads();
    uint32_t QD                           = 192;
    std::string dataset_base_file         = "";
    std::string index_file                = "";
    uint32_t build_dram_budget_megabytes  = std::numeric_limits<uint32_t>::max();
    uint32_t search_dram_budget_megabytes = std::numeric_limits<uint32_t>::max();
  };
  using search_param_base = typename algo<T>::search_param;

  struct search_param : public search_param_base {
    uint32_t L_search;
    uint32_t num_nodes_to_cache = 10000;
    int beam_width              = 2;
  };

  diskann_ssd(Metric metric, int dim, const build_param& param);

  void build(const T* dataset, size_t nrow) override;

  void set_search_param(const search_param_base& param, const void* filter_bitset) override;

  void search(const T* queries,
              int batch_size,
              int k,
              algo_base::index_type* neighbors,
              float* distances) const override;

  void save(const std::string& index_file) const override;
  void load(const std::string& index_file) override;
  diskann_ssd(const diskann_ssd<T>& other) = default;
  std::unique_ptr<algo<T>> copy() override { return std::make_unique<diskann_ssd<T>>(*this); }

  [[nodiscard]] auto get_preference() const -> algo_property override
  {
    algo_property property;
    property.dataset_memory_type = MemoryType::kHostMmap;
    property.query_memory_type   = MemoryType::kHost;
    return property;
  }

 private:
  std::string index_build_params_str;
  int beam_width_;
  uint32_t num_nodes_to_cache_;

  // in-memory index params
  uint32_t build_pq_bytes_ = 0;
  uint32_t max_points_;
  uint32_t L_search_ = 384;
  Mode bench_mode_;
  std::string base_file_;
  std::string index_path_prefix_;
  std::shared_ptr<AlignedFileReader> reader = nullptr;
  std::shared_ptr<diskann::PQFlashIndex<T, uint32_t>> p_flash_index_;
};

template <typename T>
diskann_ssd<T>::diskann_ssd(Metric metric, int dim, const build_param& param) : algo<T>(metric, dim)
{
  // Currently set the indexing RAM budget and the search RAM budget to max value to avoid sharding
  float build_dram_budget  = static_cast<float>(param.build_dram_budget_megabytes) / 1024.0f;
  float search_dram_budget = static_cast<float>(param.search_dram_budget_megabytes) / 1024.0f;
  char search_buf[16];
  char build_buf[16];
  std::snprintf(search_buf, sizeof(search_buf), "%.2f", search_dram_budget);
  std::snprintf(build_buf, sizeof(build_buf), "%.2f", build_dram_budget);
  const std::string search_dram_budget_gb(search_buf);
  const std::string build_dram_budget_gb(build_buf);
  index_build_params_str =
    std::string(std::to_string(param.R)) + " " + std::string(std::to_string(param.L_build)) + " " +
    search_dram_budget_gb + " " + build_dram_budget_gb + " " +
    std::string(std::to_string(param.num_threads)) + " " + std::string(std::to_string(false)) +
    " " + std::string(std::to_string(false)) + " " + std::string(std::to_string(0)) + " " +
    std::string(std::to_string(param.QD));
  base_file_         = param.dataset_base_file;
  index_path_prefix_ = param.index_file;
}

template <typename T>
void diskann_ssd<T>::build(const T* dataset, size_t nrow)
{
  diskann::build_disk_index<float>(base_file_.c_str(),
                                   index_path_prefix_.c_str(),
                                   index_build_params_str.c_str(),
                                   parse_metric_to_diskann(this->metric_),
                                   false,
                                   std::string(""),
                                   false,
                                   std::string(""),
                                   std::string(""),
                                   0,
                                   0);
}

template <typename T>
void diskann_ssd<T>::set_search_param(const search_param_base& param, const void* filter_bitset)
{
  if (filter_bitset != nullptr) { throw std::runtime_error("Filtering is not supported yet."); }
  auto sp             = dynamic_cast<const search_param&>(param);
  L_search_           = sp.L_search;
  num_nodes_to_cache_ = sp.num_nodes_to_cache;
  beam_width_         = sp.beam_width;
}

template <typename T>
void diskann_ssd<T>::search(
  const T* queries, int batch_size, int k, algo_base::index_type* neighbors, float* distances) const
{
  for (int64_t i = 0; i < (int64_t)batch_size; i++) {
    p_flash_index_->cached_beam_search(queries + (i * this->dim_),
                                       static_cast<size_t>(k),
                                       L_search_,
                                       reinterpret_cast<uint64_t*>(neighbors + i * k),
                                       distances + i * k,
                                       beam_width_,
                                       false,
                                       nullptr);
  }
}

template <typename T>
void diskann_ssd<T>::save(const std::string& index_file) const
{
  // Nothing to do here. Index already saved in build stage, but an empty file has to be created
  // with the index filename.
  std::ofstream of(index_file);
  of.close();
}

template <typename T>
void diskann_ssd<T>::load(const std::string& index_file)
{
  index_path_prefix_ = index_file;

  bench_mode_ = (cuvs::bench::benchmark_n_threads > 1) ? Mode::kThroughput : Mode::kLatency;

  int load_threads = (bench_mode_ == Mode::kThroughput) ? cuvs::bench::benchmark_n_threads : 1;

  reader.reset(new LinuxAlignedFileReader());
  p_flash_index_ =
    std::make_shared<diskann::PQFlashIndex<T>>(reader, parse_metric_to_diskann(this->metric_));
  int result = p_flash_index_->load(load_threads, index_path_prefix_.c_str());
  std::vector<uint32_t> node_list;
  p_flash_index_->cache_bfs_levels(num_nodes_to_cache_, node_list);
  p_flash_index_->load_cache_list(node_list);
  node_list.clear();
  node_list.shrink_to_fit();
}
};  // namespace cuvs::bench
