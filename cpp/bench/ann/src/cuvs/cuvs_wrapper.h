/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../common/ann_types.hpp"
#include "cuvs_ann_bench_utils.h"

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <raft/core/device_resources.hpp>

#include <cassert>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <type_traits>

namespace raft_temp {

inline auto parse_metric_type(cuvs::bench::Metric metric) -> cuvs::distance::DistanceType
{
  switch (metric) {
    case cuvs::bench::Metric::kInnerProduct: return cuvs::distance::DistanceType::InnerProduct;
    case cuvs::bench::Metric::kEuclidean: return cuvs::distance::DistanceType::L2Expanded;
    default: throw std::runtime_error("raft supports only metric type of inner product and L2");
  }
}
}  // namespace raft_temp

namespace cuvs::bench {

// brute force KNN - RAFT
template <typename T>
class cuvs_gpu : public algo<T>, public algo_gpu {
 public:
  using search_param_base = typename algo<T>::search_param;

  struct search_param : public search_param_base {
    [[nodiscard]] auto needs_dataset() const -> bool override { return true; }
  };

  cuvs_gpu(Metric metric, int dim);

  void build(const T*, size_t) final;

  void set_search_param(const search_param_base& param, const void* filter_bitset) override;

  void search(const T* queries,
              int batch_size,
              int k,
              algo_base::index_type* neighbors,
              float* distances) const final;

  // to enable dataset access from GPU memory
  [[nodiscard]] auto get_preference() const -> algo_property override
  {
    algo_property property;
    property.dataset_memory_type = MemoryType::kDevice;
    property.query_memory_type   = MemoryType::kDevice;
    return property;
  }
  [[nodiscard]] auto get_sync_stream() const noexcept -> cudaStream_t override
  {
    return handle_.get_sync_stream().get();
  }
  void set_search_dataset(const T* dataset, size_t nrow) override;
  void save(const std::string& file) const override;
  void load(const std::string&) override;
  std::unique_ptr<algo<T>> copy() override;

 protected:
  // handle_ must go first to make sure it dies last and all memory allocated in pool
  configured_raft_resources handle_{};
  std::shared_ptr<cuvs::neighbors::brute_force::index<T>> index_;
  cuvs::distance::DistanceType metric_type_;
  int device_;
  const T* dataset_;
  size_t nrow_;

  std::shared_ptr<cuvs::neighbors::filtering::base_filter> filter_;
};

template <typename T>
cuvs_gpu<T>::cuvs_gpu(Metric metric, int dim)
  : algo<T>(metric, dim), metric_type_(raft_temp::parse_metric_type(metric))
{
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "raft bfknn only supports float/double");
  RAFT_CUDA_TRY(cudaGetDevice(&device_));
}

template <typename T>
void cuvs_gpu<T>::build(const T* dataset, size_t nrow)
{
  auto dataset_view = raft::make_device_matrix_view<const T, int64_t>(dataset, nrow, this->dim_);
  index_            = std::make_shared<cuvs::neighbors::brute_force::index<T>>(
    std::move(cuvs::neighbors::brute_force::build(handle_, dataset_view, metric_type_)));
}

template <typename T>
void cuvs_gpu<T>::set_search_param(const search_param_base&, const void* filter_bitset)
{
  filter_ = make_cuvs_filter(filter_bitset, index_->size());
}

template <typename T>
void cuvs_gpu<T>::set_search_dataset(const T* dataset, size_t nrow)
{
  dataset_ = dataset;
  nrow_    = nrow;
  // Wrap the dataset with an index.
  auto dataset_view = raft::make_device_matrix_view<const T, int64_t>(dataset, nrow, this->dim_);
  index_            = std::make_shared<cuvs::neighbors::brute_force::index<T>>(
    std::move(cuvs::neighbors::brute_force::build(handle_, dataset_view, metric_type_)));
}

template <typename T>
void cuvs_gpu<T>::save(const std::string& file) const
{
  // The index is just the dataset with metadata (shape). The dataset already exist on disk,
  // therefore we do not need to save it here.
  // We create an empty file because the benchmark logic requires an index file to be created.
  std::ofstream of(file);
  of.close();
}

template <typename T>
void cuvs_gpu<T>::load(const std::string& file)
{
  // We do not have serialization of brute force index. We can simply wrap the
  // dataset into a brute force index, like it is done in set_search_dataset.
}

template <typename T>
void cuvs_gpu<T>::search(
  const T* queries, int batch_size, int k, algo_base::index_type* neighbors, float* distances) const
{
  auto queries_view =
    raft::make_device_matrix_view<const T, int64_t>(queries, batch_size, this->dim_);

  auto neighbors_view =
    raft::make_device_matrix_view<algo_base::index_type, int64_t>(neighbors, batch_size, k);
  auto distances_view = raft::make_device_matrix_view<float, int64_t>(distances, batch_size, k);

  cuvs::neighbors::brute_force::search(
    handle_, *index_, queries_view, neighbors_view, distances_view, *filter_);
}

template <typename T>
std::unique_ptr<algo<T>> cuvs_gpu<T>::copy()
{
  return std::make_unique<cuvs_gpu<T>>(*this);  // use copy constructor
}

}  // namespace cuvs::bench
