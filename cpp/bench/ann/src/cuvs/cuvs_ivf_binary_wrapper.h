/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../common/ann_types.hpp"
#include "cuvs_ann_bench_utils.h"

#include <cuvs/neighbors/ivf_binary.hpp>

#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace cuvs::bench {

class cuvs_ivf_binary : public algo<uint8_t>, public algo_gpu {
 public:
  using search_param_base = algo<uint8_t>::search_param;
  using build_param       = cuvs::neighbors::ivf_binary::index_params;

  struct search_param : public search_param_base {
    cuvs::neighbors::ivf_binary::search_params ivf_binary_params;
  };

  // The benchmark dimension counts packed bytes; IVF-Binary's dimension counts bits.
  cuvs_ivf_binary(Metric metric, int dim, const build_param& param)
    : algo<uint8_t>(metric, dim), index_params_(param)
  {
    if (metric != Metric::kBitwiseHamming) {
      throw std::invalid_argument("cuvs_ivf_binary requires bitwise_hamming distance");
    }
    if (dim <= 0 || uint64_t(dim) > std::numeric_limits<uint32_t>::max() / 8) {
      throw std::invalid_argument("invalid packed binary dimension");
    }
    index_params_.conservative_memory_allocation = true;
  }

  void build(const uint8_t* dataset, size_t nrow) override
  {
    // Inputs are already packed codes. An empty dictionary records a zero threshold.
    cuvs::preprocessing::quantize::binary::quantizer<float> quantizer(handle_);
    auto view = cuvs::make_device_binary_dataset_view<int64_t>(
      raft::make_device_matrix_view<const uint8_t, int64_t>(dataset, nrow, dim_),
      quantizer,
      uint32_t(dim_) * 8);
    index_ = std::make_shared<cuvs::neighbors::ivf_binary::index<int64_t>>(
      cuvs::neighbors::ivf_binary::build(handle_, index_params_, view));
  }

  void set_search_param(const search_param_base& param, const void* filter_bitset) override
  {
    search_params_ = dynamic_cast<const search_param&>(param).ivf_binary_params;
    filter_        = make_cuvs_filter(filter_bitset, index_->size());
  }

  void search(const uint8_t* queries,
              int batch_size,
              int k,
              algo_base::index_type* neighbors,
              float* distances) const override
  {
    cuvs::neighbors::ivf_binary::search(
      handle_,
      search_params_,
      *index_,
      raft::make_device_matrix_view<const uint8_t, int64_t>(
        queries, batch_size, index_->storage_dim()),
      raft::make_device_matrix_view<int64_t, int64_t>(neighbors, batch_size, k),
      raft::make_device_matrix_view<float, int64_t>(distances, batch_size, k),
      *filter_);
  }

  [[nodiscard]] auto get_sync_stream() const noexcept -> cudaStream_t override
  {
    return handle_.get_sync_stream().get();
  }

  [[nodiscard]] auto get_preference() const -> algo_property override
  {
    return {MemoryType::kDevice, MemoryType::kDevice};
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
  // Keep the resources alive until the index and filter have been destroyed.
  configured_raft_resources handle_{};
  build_param index_params_;
  cuvs::neighbors::ivf_binary::search_params search_params_;
  std::shared_ptr<cuvs::neighbors::ivf_binary::index<int64_t>> index_;
  std::shared_ptr<cuvs::neighbors::filtering::base_filter> filter_;
};

}  // namespace cuvs::bench
