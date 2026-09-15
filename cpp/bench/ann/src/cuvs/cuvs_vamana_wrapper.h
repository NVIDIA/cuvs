/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../common/ann_types.hpp"
#include "../diskann/diskann_wrapper.h"
#include "cuvs_ann_bench_utils.h"
#include <cuvs/neighbors/vamana.hpp>
#include <utils.h>

#include <memory>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/mdspan.hpp>

namespace cuvs::bench {

template <typename T, typename IdxT>
class cuvs_vamana : public algo<T>, public algo_gpu {
 public:
  using build_param       = cuvs::neighbors::vamana::index_params;
  using search_param_base = typename algo<T>::search_param;
  using search_param      = typename diskann_memory<T>::search_param;

  cuvs_vamana(Metric metric, int dim, const build_param& param);

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
  std::unique_ptr<algo<T>> copy() override { return std::make_unique<cuvs_vamana<T, IdxT>>(*this); }

 private:
  configured_raft_resources handle_{};
  build_param vamana_index_params_;
  std::shared_ptr<cuvs::neighbors::vamana::index<T, IdxT>> vamana_index_;
  std::shared_ptr<diskann_memory<T>> diskann_memory_search_;
};

template <typename T, typename IdxT>
cuvs_vamana<T, IdxT>::cuvs_vamana(Metric metric, int dim, const build_param& param)
  : algo<T>(metric, dim)
{
  this->vamana_index_params_ = param;
  diskann_memory_search_     = std::make_shared<cuvs::bench::diskann_memory<T>>(
    metric, dim, typename diskann_memory<T>::build_param{param.graph_degree, param.visited_size});
}

template <typename T, typename IdxT>
void cuvs_vamana<T, IdxT>::build(const T* dataset, size_t nrow)
{
  auto dataset_view_host = raft::make_mdspan<const T, int64_t, raft::row_major, true, false>(
    dataset, raft::make_extents<int64_t>(nrow, this->dim_));
  auto dataset_view_device = raft::make_mdspan<const T, int64_t, raft::row_major, false, true>(
    dataset, raft::make_extents<int64_t>(nrow, this->dim_));
  bool dataset_is_on_host = raft::get_device_for_address(dataset) == -1;

  vamana_index_ = std::make_shared<cuvs::neighbors::vamana::index<T, uint32_t>>(std::move(
    dataset_is_on_host
      ? cuvs::neighbors::vamana::build(handle_, vamana_index_params_, dataset_view_host)
      : cuvs::neighbors::vamana::build(handle_, vamana_index_params_, dataset_view_device)));
}

template <typename T, typename IdxT>
void cuvs_vamana<T, IdxT>::set_search_param(const search_param_base& param,
                                            const void* filter_bitset)
{
  if (filter_bitset != nullptr) { throw std::runtime_error("Filtering is not supported yet."); }
  diskann_memory_search_->set_search_param(param, nullptr);
}

template <typename T, typename IdxT>
void cuvs_vamana<T, IdxT>::save(const std::string& file) const
{
  cuvs::neighbors::vamana::serialize(handle_, file, *vamana_index_);
}

template <typename T, typename IdxT>
void cuvs_vamana<T, IdxT>::load(const std::string& file)
{
  diskann_memory_search_->load(file);
}

template <typename T, typename IdxT>
void cuvs_vamana<T, IdxT>::search(
  const T* queries, int batch_size, int k, algo_base::index_type* neighbors, float* distances) const
{
  diskann_memory_search_->search(queries, batch_size, k, neighbors, distances);
}

}  // namespace cuvs::bench
