/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../common/ann_types.hpp"
#include "cuvs_ann_bench_utils.h"

#include <cuvs/neighbors/ivf_binary.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>

#include <memory>
#include <string>

namespace cuvs::bench {

/**
 * IVF over bit-packed codes: each uint8_t row holds 8 dimensions, so the bench dimension is the
 * number of bytes per row and the index dimension is 8 times that. Queries must be packed the same
 * way; the codes are used as is (zero threshold).
 */
class cuvs_ivf_binary : public algo<uint8_t>, public algo_gpu {
 public:
  using search_param_base = typename algo<uint8_t>::search_param;

  struct search_param : public search_param_base {
    cuvs::neighbors::ivf_binary::search_params ivf_binary_params;
  };

  using build_param = cuvs::neighbors::ivf_binary::index_params;

  cuvs_ivf_binary(Metric metric, int dim, const build_param& param)
    : algo<uint8_t>(metric, dim), index_params_(param), dimension_(dim)
  {
    index_params_.conservative_memory_allocation = true;
  }

  void build(const uint8_t* dataset, size_t nrow) final
  {
    auto codes = raft::make_device_matrix_view<const uint8_t, int64_t>(dataset, nrow, dimension_);
    cuvs::device_binary_dataset_view<float, int64_t> data{
      codes,
      cuvs::binary_dictionary_view<float>{
        raft::make_device_vector_view<const float, int64_t>(nullptr, 0)},
      uint32_t(dimension_) * 8};
    index_ = std::make_shared<cuvs::neighbors::ivf_binary::index<int64_t>>(
      cuvs::neighbors::ivf_binary::build(handle_, index_params_, data));
  }

  void set_search_param(const search_param_base& param, const void* filter_bitset) override
  {
    filter_        = make_cuvs_filter(filter_bitset, index_->size());
    search_params_ = dynamic_cast<const search_param&>(param).ivf_binary_params;
  }

  void search(const uint8_t* queries,
              int batch_size,
              int k,
              algo_base::index_type* neighbors,
              float* distances) const override
  {
    static_assert(sizeof(algo_base::index_type) == sizeof(int64_t));
    cuvs::neighbors::ivf_binary::search(
      handle_,
      search_params_,
      *index_,
      raft::make_device_matrix_view<const uint8_t, int64_t>(queries, batch_size, dimension_),
      raft::make_device_matrix_view<int64_t, int64_t>(
        reinterpret_cast<int64_t*>(neighbors), batch_size, k),
      raft::make_device_matrix_view<float, int64_t>(distances, batch_size, k),
      *filter_);
  }

  [[nodiscard]] auto get_sync_stream() const noexcept -> cudaStream_t override
  {
    return handle_.get_sync_stream().get();
  }

  [[nodiscard]] auto get_preference() const -> algo_property override
  {
    algo_property property;
    property.dataset_memory_type = MemoryType::kDevice;
    property.query_memory_type   = MemoryType::kDevice;
    return property;
  }

  void save(const std::string& file) const override
  {
    cuvs::neighbors::ivf_binary::serialize(handle_, file, *index_);
  }

  void load(const std::string& file) override
  {
    index_ = std::make_shared<cuvs::neighbors::ivf_binary::index<int64_t>>(handle_);
    cuvs::neighbors::ivf_binary::deserialize(handle_, file, index_.get());
  }

  std::unique_ptr<algo<uint8_t>> copy() override
  {
    return std::make_unique<cuvs_ivf_binary>(*this);
  }

 private:
  // handle_ must go first to make sure it dies last and all memory allocated in pool
  configured_raft_resources handle_{};
  build_param index_params_;
  cuvs::neighbors::ivf_binary::search_params search_params_;
  std::shared_ptr<cuvs::neighbors::ivf_binary::index<int64_t>> index_;
  int dimension_;

  std::shared_ptr<cuvs::neighbors::filtering::base_filter> filter_;
};

}  // namespace cuvs::bench
