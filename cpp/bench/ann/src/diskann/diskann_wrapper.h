/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

namespace cuvs::bench {

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

// TODO (tarangj): Remaining features are tracked at https://github.com/nvidia/cuvs/issues/656
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
  uint32_t build_pq_bytes_ = 0;
  int num_threads_;
  uint32_t L_search_;
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
  initialize_index_(nrow);
  mem_index_->build(dataset, nrow, std::vector<uint32_t>());
}

template <typename T>
void diskann_memory<T>::set_search_param(const search_param_base& param, const void* filter_bitset)
{
  if (filter_bitset != nullptr) { throw std::runtime_error("Filtering is not supported yet."); }
  auto sp   = dynamic_cast<const search_param&>(param);
  L_search_ = sp.L_search;
}

template <typename T>
void diskann_memory<T>::search(
  const T* queries, int batch_size, int k, algo_base::index_type* indices, float* distances) const
{
  for (int i = 0; i < batch_size; i++) {
    mem_index_->search(queries + i * this->dim_,
                       static_cast<size_t>(k),
                       L_search_,
                       reinterpret_cast<uint64_t*>(indices + i * k),
                       distances + i * k);
  }
}

template <typename T>
void diskann_memory<T>::save(const std::string& index_file) const
{
  this->mem_index_->save(index_file.c_str());
}

template <typename T>
void diskann_memory<T>::load(const std::string& index_file)
{
  index_path_prefix_ = index_file;

  bench_mode_ = (cuvs::bench::benchmark_n_threads > 1) ? Mode::kThroughput : Mode::kLatency;

  initialize_index_(0);

  int load_threads = (bench_mode_ == Mode::kThroughput) ? cuvs::bench::benchmark_n_threads : 1;
  this->mem_index_->load(index_path_prefix_.c_str(), load_threads, 2000);
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
