/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../common/ann_types.hpp"
#include "cuvs_ann_bench_utils.h"

#include <cuvs/neighbors/scann.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>

#include <filesystem>
#include <memory>
#include <stdexcept>

namespace cuvs::bench {

template <typename T, typename IdxT>
class cuvs_scann : public algo<T>, public algo_gpu {
 public:
  using search_param_base = typename algo<T>::search_param;
  using algo<T>::dim_;

  struct search_param : public search_param_base {};

  using build_param = cuvs::neighbors::scann::index_params;

  cuvs_scann(Metric metric, int dim, const build_param& param)
    : algo<T>(metric, dim), index_params_(param)
  {
    index_params_.metric = parse_metric_type(metric);
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
    return handle_.get_sync_stream().get();
  }

  // to enable dataset access from GPU memory
  [[nodiscard]] auto get_preference() const -> algo_property override
  {
    algo_property property;
    property.dataset_memory_type = MemoryType::kDevice;
    property.query_memory_type   = MemoryType::kHost;
    return property;
  }
  void save(const std::string& file) const override;
  void load(const std::string&) override;
  std::unique_ptr<algo<T>> copy() override;

 private:
  // handle_ must go first to make sure it dies last and all memory allocated in pool
  configured_raft_resources handle_{};
  build_param index_params_;
  std::shared_ptr<cuvs::neighbors::scann::index<T, IdxT>> index_;
};

template <typename T, typename IdxT>
void cuvs_scann<T, IdxT>::build(const T* dataset, size_t nrow)
{
  auto dataset_v = raft::make_device_matrix_view<const T, IdxT>(dataset, IdxT(nrow), dim_);
  std::make_shared<cuvs::neighbors::scann::index<T, IdxT>>(
    std::move(cuvs::neighbors::scann::build(handle_, index_params_, dataset_v)))
    .swap(index_);
}

template <typename T, typename IdxT>
std::unique_ptr<algo<T>> cuvs_scann<T, IdxT>::copy()
{
  return std::make_unique<cuvs_scann<T, IdxT>>(*this);  // use copy constructor
}

template <typename T, typename IdxT>
void cuvs_scann<T, IdxT>::set_search_param(const search_param_base&, const void*)
{
}

template <typename T, typename IdxT>
void cuvs_scann<T, IdxT>::save(const std::string& file) const
{
  // ScaNN serializes a bundle of assets into a directory, so make sure it exists
  std::filesystem::create_directories(file);
  cuvs::neighbors::scann::serialize(handle_, file, *index_);
}

template <typename T, typename IdxT>
void cuvs_scann<T, IdxT>::load(const std::string&)
{
  throw std::runtime_error("ScaNN benchmark supports build only; load is not implemented");
}

template <typename T, typename IdxT>
void cuvs_scann<T, IdxT>::search(
  const T*, int, int, algo_base::index_type*, float*) const
{
  throw std::runtime_error("ScaNN benchmark supports build only; search is not implemented");
}

}  // namespace cuvs::bench
