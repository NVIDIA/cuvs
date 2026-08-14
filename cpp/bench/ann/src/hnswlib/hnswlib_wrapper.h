/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../common/ann_types.hpp"
#include "../common/thread_pool.hpp"
#include "../common/util.hpp"

#include <hnswlib/hnswlib.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <ctime>
#include <future>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace cuvs::bench {

template <typename T>
struct hnsw_dist_t {
  using type = void;
};

template <>
struct hnsw_dist_t<float> {
  using type = float;
};

template <>
struct hnsw_dist_t<half> {
  using type = float;
};

template <>
struct hnsw_dist_t<uint8_t> {
  using type = int;
};

template <>
struct hnsw_dist_t<int8_t> {
  using type = int;
};

template <typename T>
class hnsw_lib : public algo<T> {
 public:
  // https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
  struct build_param {
    int m;
    int ef_construction;
    int num_threads = omp_get_num_procs();
  };

  using search_param_base = typename algo<T>::search_param;
  struct search_param : public search_param_base {
    int ef;
    int num_threads = 1;
  };

  hnsw_lib(Metric metric, int dim, const build_param& param);

  void build(const T* dataset, size_t nrow) override;

  void set_search_param(const search_param_base& param, const void* filter_bitset) override;
  void search(const T* query,
              int batch_size,
              int k,
              algo_base::index_type* indices,
              float* distances) const override;

  void save(const std::string& path_to_index) const override;
  void load(const std::string& path_to_index) override;
  auto copy() -> std::unique_ptr<algo<T>> override { return std::make_unique<hnsw_lib<T>>(*this); };

  [[nodiscard]] auto get_preference() const -> algo_property override
  {
    algo_property property;
    property.dataset_memory_type = MemoryType::kHost;
    property.query_memory_type   = MemoryType::kHost;
    return property;
  }

  void set_base_layer_only() { appr_alg_->base_layer_only = true; }

 private:
  void get_search_knn_results(const T* query,
                              int k,
                              algo_base::index_type* indices,
                              float* distances) const;

  std::shared_ptr<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type>> appr_alg_;
  std::shared_ptr<hnswlib::SpaceInterface<typename hnsw_dist_t<T>::type>> space_;

  using algo<T>::metric_;
  using algo<T>::dim_;
  int ef_construction_;
  int m_;
  int num_threads_;
  std::shared_ptr<fixed_thread_pool> thread_pool_;
  Mode bench_mode_;
};

template <typename T>
hnsw_lib<T>::hnsw_lib(Metric metric, int dim, const build_param& param) : algo<T>(metric, dim)
{
  assert(dim_ > 0);
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, half> || std::is_same_v<T, int8_t> ||
                  std::is_same_v<T, uint8_t>,
                "Only float, half, uint8, and int8 are supported");

  ef_construction_ = param.ef_construction;
  m_               = param.m;
  num_threads_     = param.num_threads;
}

template <typename T>
void hnsw_lib<T>::build(const T* dataset, size_t nrow)
{
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, half> || std::is_same_v<T, int8_t> ||
                  std::is_same_v<T, uint8_t>,
                "Only float, half, uint8, and int8 are supported");
  if (metric_ == Metric::kInnerProduct) {
    space_ = std::make_shared<hnswlib::InnerProductSpace<T, typename hnsw_dist_t<T>::type>>(dim_);
  } else {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
      space_ = std::make_shared<hnswlib::L2Space<T, typename hnsw_dist_t<T>::type>>(dim_);
    } else {
      space_ = std::make_shared<hnswlib::L2SpaceI<T>>(dim_);
    }
  }

  appr_alg_ = std::make_shared<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type>>(
    space_.get(), nrow, m_, ef_construction_);

  thread_pool_                  = std::make_shared<fixed_thread_pool>(num_threads_);
  const size_t items_per_thread = nrow / (num_threads_ + 1);

  thread_pool_->submit(
    [&](size_t i) {
      if (i < items_per_thread && i % 10000 == 0) {
        char buf[20];
        std::time_t now = std::time(nullptr);
        std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
        printf("%s building %zu / %zu\n", buf, i, items_per_thread);
        fflush(stdout);
      }

      appr_alg_->addPoint(dataset + i * dim_, i);
    },
    nrow);
}

template <typename T>
void hnsw_lib<T>::set_search_param(const search_param_base& param_, const void* filter_bitset)
{
  if (filter_bitset != nullptr) { throw std::runtime_error("Filtering is not supported yet."); }
  auto param     = dynamic_cast<const search_param&>(param_);
  appr_alg_->ef_ = param.ef;
  num_threads_   = param.num_threads;
  if (cuvs::bench::benchmark_n_threads > 1) {
    bench_mode_  = Mode::kThroughput;
    num_threads_ = 1;  // Prevent nested parallelism (gbench threads + batch threads).
  } else {
    bench_mode_ = Mode::kLatency;
  }

  // Create a pool if multiple query threads have been set and the pool hasn't been created already
  bool create_pool = (bench_mode_ == Mode::kLatency && num_threads_ > 1 && !thread_pool_);
  if (create_pool) { thread_pool_ = std::make_shared<fixed_thread_pool>(num_threads_); }
}

template <typename T>
void hnsw_lib<T>::search(
  const T* query, int batch_size, int k, algo_base::index_type* indices, float* distances) const
{
  if (batch_size == 1) {
    get_search_knn_results(query, k, indices, distances);
    return;
  }

  auto f = [&](int i) {
    // hnsw can only handle a single vector at a time.
    get_search_knn_results(query + i * dim_, k, indices + i * k, distances + i * k);
  };
  if (bench_mode_ == Mode::kLatency && num_threads_ > 1) {
    thread_pool_->submit(f, batch_size);
  } else {
    for (int i = 0; i < batch_size; i++) {
      f(i);
    }
  }
}

template <typename T>
void hnsw_lib<T>::save(const std::string& path_to_index) const
{
  appr_alg_->saveIndex(std::string(path_to_index));
}

template <typename T>
void hnsw_lib<T>::load(const std::string& path_to_index)
{
  if (metric_ == Metric::kInnerProduct) {
    space_ = std::make_shared<hnswlib::InnerProductSpace<T, typename hnsw_dist_t<T>::type>>(dim_);
  } else {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
      space_ = std::make_shared<hnswlib::L2Space<T, typename hnsw_dist_t<T>::type>>(dim_);
    } else {
      space_ = std::make_shared<hnswlib::L2SpaceI<T>>(dim_);
    }
  }

  appr_alg_ = std::make_shared<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type>>(
    space_.get(), path_to_index);
}

template <typename T>
void hnsw_lib<T>::get_search_knn_results(const T* query,
                                         int k,
                                         algo_base::index_type* indices,
                                         float* distances) const
{
  auto result = appr_alg_->searchKnn(query, k);
  assert(result.size() >= static_cast<size_t>(k));

  for (int i = k - 1; i >= 0; --i) {
    indices[i]   = result.top().second;
    distances[i] = result.top().first;
    result.pop();
  }
}

};  // namespace cuvs::bench
