/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../common/ann_types.hpp"
#include "../diskann/diskann_wrapper.h"
#include "cuvs_ann_bench_utils.h"
#include <cuvs/neighbors/vamana.hpp>
#include <utils.h>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/mdspan.hpp>
#include <stdexcept>
#include <string>
#include <system_error>

namespace cuvs::bench {

inline bool cuvs_vamana_artifact_exists(const std::string& file)
{
  std::error_code ec;
  return std::filesystem::exists(file, ec);
}

inline std::uintmax_t cuvs_vamana_artifact_size(const std::string& file)
{
  std::error_code ec;
  auto size = std::filesystem::file_size(file, ec);
  return ec ? 0 : size;
}

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
    return handle_.get_sync_stream();
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

  log_info(
    "cuvs_vamana build start: nrow=%zu dim=%d graph_degree=%u visited_size=%u alpha=%f "
    "dataset_memory=%s",
    nrow,
    this->dim_,
    static_cast<unsigned>(vamana_index_params_.graph_degree),
    static_cast<unsigned>(vamana_index_params_.visited_size),
    static_cast<double>(vamana_index_params_.alpha),
    dataset_is_on_host ? "host" : "device");

  vamana_index_ = std::make_shared<cuvs::neighbors::vamana::index<T, uint32_t>>(std::move(
    dataset_is_on_host
      ? cuvs::neighbors::vamana::build(handle_, vamana_index_params_, dataset_view_host)
      : cuvs::neighbors::vamana::build(handle_, vamana_index_params_, dataset_view_device)));

  log_info("cuvs_vamana build finish: index_size=%zu index_dim=%u graph_degree=%u",
           static_cast<size_t>(vamana_index_->size()),
           static_cast<unsigned>(vamana_index_->dim()),
           static_cast<unsigned>(vamana_index_->graph_degree()));

  if (vamana_index_->size() == 0 || vamana_index_->dim() == 0) {
    throw std::runtime_error("cuvs_vamana build produced an empty index: size=" +
                             std::to_string(static_cast<size_t>(vamana_index_->size())) +
                             ", dim=" + std::to_string(vamana_index_->dim()));
  }
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
  if (!vamana_index_) { throw std::runtime_error("cuvs_vamana save called before build"); }

  log_info("cuvs_vamana save start: file=%s index_size=%zu index_dim=%u graph_degree=%u",
           file.c_str(),
           static_cast<size_t>(vamana_index_->size()),
           static_cast<unsigned>(vamana_index_->dim()),
           static_cast<unsigned>(vamana_index_->graph_degree()));

  cuvs::neighbors::vamana::serialize(handle_, file, *vamana_index_);

  const auto data_file    = file + ".data";
  const auto index_bytes  = cuvs_vamana_artifact_size(file);
  const auto data_bytes   = cuvs_vamana_artifact_size(data_file);
  const auto index_exists = cuvs_vamana_artifact_exists(file);
  const auto data_exists  = cuvs_vamana_artifact_exists(data_file);

  log_info(
    "cuvs_vamana save artifacts: index_file=%s exists=%d bytes=%zu data_file=%s "
    "exists=%d bytes=%zu",
    file.c_str(),
    index_exists ? 1 : 0,
    static_cast<size_t>(index_bytes),
    data_file.c_str(),
    data_exists ? 1 : 0,
    static_cast<size_t>(data_bytes));

  if (!index_exists) {
    throw std::runtime_error("cuvs_vamana serialize did not write index file: " + file);
  }
  if (!data_exists) {
    throw std::runtime_error("cuvs_vamana serialize did not write dataset sidecar: " + data_file);
  }
  if (index_bytes == 0 || data_bytes == 0) {
    throw std::runtime_error("cuvs_vamana serialize wrote empty artifact(s): index_file=" + file +
                             ", data_file=" + data_file);
  }
}

template <typename T, typename IdxT>
void cuvs_vamana<T, IdxT>::load(const std::string& file)
{
  const auto data_file   = file + ".data";
  const auto index_bytes = cuvs_vamana_artifact_size(file);
  const auto data_bytes  = cuvs_vamana_artifact_size(data_file);

  log_info(
    "cuvs_vamana load start: index_file=%s exists=%d bytes=%zu data_file=%s exists=%d "
    "bytes=%zu",
    file.c_str(),
    cuvs_vamana_artifact_exists(file) ? 1 : 0,
    static_cast<size_t>(index_bytes),
    data_file.c_str(),
    cuvs_vamana_artifact_exists(data_file) ? 1 : 0,
    static_cast<size_t>(data_bytes));

  try {
    log_info("cuvs_vamana load delegating to diskann_memory: file=%s", file.c_str());
    diskann_memory_search_->load(file);
  } catch (const std::exception& e) {
    log_warn("cuvs_vamana load failed while loading diskann_memory: file=%s error=%s",
             file.c_str(),
             e.what());
    throw;
  } catch (...) {
    log_warn("cuvs_vamana load failed while loading diskann_memory: file=%s unknown error",
             file.c_str());
    throw;
  }
  log_info("cuvs_vamana load finish: file=%s", file.c_str());
}

template <typename T, typename IdxT>
void cuvs_vamana<T, IdxT>::search(
  const T* queries, int batch_size, int k, algo_base::index_type* neighbors, float* distances) const
{
  static std::atomic<bool> search_logged{false};
  bool expected = false;
  if (search_logged.compare_exchange_strong(expected, true)) {
    log_info("cuvs_vamana search first call: batch_size=%d k=%d query_ptr=%p",
             batch_size,
             k,
             static_cast<const void*>(queries));
  }
  diskann_memory_search_->search(queries, batch_size, k, neighbors, distances);
}

}  // namespace cuvs::bench
