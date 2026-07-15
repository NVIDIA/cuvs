/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../common/ann_types.hpp"
#include "cuvs_ann_bench_utils.h"

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/ivf_sq.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/cuda_stream_pool.hpp>

#include <cassert>
#include <memory>
#include <string>
#include <type_traits>

namespace cuvs::bench {

template <typename T>
class cuvs_ivf_sq : public algo<T>, public algo_gpu {
 public:
  using search_param_base = typename algo<T>::search_param;

  struct search_param : public search_param_base {
    cuvs::neighbors::ivf_sq::search_params ivf_sq_params;
  };

  using build_param = cuvs::neighbors::ivf_sq::index_params;

  cuvs_ivf_sq(Metric metric, int dim, const build_param& param)
    : algo<T>(metric, dim), index_params_(param), dimension_(dim)
  {
    index_params_.metric                         = parse_metric_type(metric);
    index_params_.conservative_memory_allocation = true;
    RAFT_CUDA_TRY(cudaGetDevice(&device_));
  }

  void build(const T* dataset, size_t nrow) final;

  void set_search_param(const search_param_base& param, const void* filter_bitset) override;

  void search(const T* queries,
              int batch_size,
              int k,
              algo_base::index_type* neighbors,
              float* distances) const override;

  [[nodiscard]] auto get_sync_stream() const noexcept -> cudaStream_t override
  {
    return handle_.get_sync_stream();
  }

  [[nodiscard]] auto get_preference() const -> algo_property override
  {
    algo_property property;
    property.dataset_memory_type = MemoryType::kHostMmap;
    property.query_memory_type   = MemoryType::kDevice;
    return property;
  }

  void save(const std::string& file) const override;
  void load(const std::string&) override;
  std::unique_ptr<algo<T>> copy() override;

 private:
  configured_raft_resources handle_{};
  build_param index_params_;
  cuvs::neighbors::ivf_sq::search_params search_params_;
  std::shared_ptr<cuvs::neighbors::ivf_sq::index<uint8_t>> index_;
  int device_;
  int dimension_;

  std::shared_ptr<cuvs::neighbors::filtering::base_filter> filter_;
};

template <typename T>
void cuvs_ivf_sq<T>::build(const T* dataset, size_t nrow)
{
  size_t n_streams = 1;
  raft::resource::set_cuda_stream_pool(handle_, std::make_shared<rmm::cuda_stream_pool>(n_streams));
  index_ = std::make_shared<cuvs::neighbors::ivf_sq::index<uint8_t>>(
    std::move(cuvs::neighbors::ivf_sq::build(
      handle_,
      index_params_,
      raft::make_host_matrix_view<const T, int64_t>(dataset, nrow, dimension_))));
}

template <typename T>
void cuvs_ivf_sq<T>::set_search_param(const search_param_base& param, const void* filter_bitset)
{
  filter_        = make_cuvs_filter(filter_bitset, index_->size());
  auto sp        = dynamic_cast<const search_param&>(param);
  search_params_ = sp.ivf_sq_params;
  assert(search_params_.n_probes <= index_params_.n_lists);
}

template <typename T>
void cuvs_ivf_sq<T>::save(const std::string& file) const
{
  cuvs::neighbors::ivf_sq::serialize(handle_, file, *index_);
}

template <typename T>
void cuvs_ivf_sq<T>::load(const std::string& file)
{
  index_ =
    std::make_shared<cuvs::neighbors::ivf_sq::index<uint8_t>>(handle_, index_params_, this->dim_);
  cuvs::neighbors::ivf_sq::deserialize(handle_, file, index_.get());
}

template <typename T>
std::unique_ptr<algo<T>> cuvs_ivf_sq<T>::copy()
{
  return std::make_unique<cuvs_ivf_sq<T>>(*this);
}

template <typename T>
void cuvs_ivf_sq<T>::search(
  const T* queries, int batch_size, int k, algo_base::index_type* neighbors, float* distances) const
{
  static_assert(sizeof(algo_base::index_type) == sizeof(int64_t));

  cuvs::neighbors::ivf_sq::search(
    handle_,
    search_params_,
    *index_,
    raft::make_device_matrix_view<const T, int64_t>(queries, batch_size, index_->dim()),
    raft::make_device_matrix_view<int64_t, int64_t>(
      reinterpret_cast<int64_t*>(neighbors), batch_size, k),
    raft::make_device_matrix_view<float, int64_t>(distances, batch_size, k),
    *filter_);
}

}  // namespace cuvs::bench
