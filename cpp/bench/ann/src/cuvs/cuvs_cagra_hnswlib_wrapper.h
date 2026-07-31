/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "cuvs_cagra_wrapper.h"
#include <cuvs/neighbors/hnsw.hpp>
#include <raft/core/logger.hpp>

#include <chrono>
#include <memory>

namespace cuvs::bench {

template <typename T, typename IdxT>
class cuvs_cagra_hnswlib : public algo<T>, public algo_gpu {
 public:
  using search_param_base = typename algo<T>::search_param;

  struct build_param {
    using cagra_wrapper_params = typename cuvs_cagra<T, IdxT>::build_param;
    cagra_wrapper_params cagra_build_params;
    cuvs::neighbors::hnsw::index_params hnsw_index_params;
  };

  struct search_param : public search_param_base {
    cuvs::neighbors::hnsw::search_params hnsw_search_param;
  };

  cuvs_cagra_hnswlib(Metric metric, int dim, const build_param& param, int concurrent_searches = 1)
    : algo<T>(metric, dim), build_param_{param}
  {
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

  [[nodiscard]] auto uses_stream() const noexcept -> bool override
  {
    // there's no need to synchronize with the GPU neither on build nor on search
    return false;
  }

  // to enable dataset access from GPU memory
  [[nodiscard]] auto get_preference() const -> algo_property override
  {
    algo_property property;
    property.dataset_memory_type = MemoryType::kHostMmap;
    property.query_memory_type   = MemoryType::kHost;
    return property;
  }

  void save(const std::string& file) const override;
  void load(const std::string&) override;
  std::unique_ptr<algo<T>> copy() override
  {
    return std::make_unique<cuvs_cagra_hnswlib<T, IdxT>>(*this);
  }

 private:
  configured_raft_resources handle_{};
  build_param build_param_;
  search_param search_param_;
  std::shared_ptr<cuvs::neighbors::hnsw::index<T>> hnsw_index_;

  bool cagra_ace_build_ = false;
};

template <typename T, typename IdxT>
void cuvs_cagra_hnswlib<T, IdxT>::build(const T* dataset, size_t nrow)
{
  // when the data set is on host, we can pass it directly to HNSW
  bool dataset_is_on_host = raft::get_device_for_address(dataset) == -1;

  auto dataset_view = raft::make_host_matrix_view<const T, int64_t>(dataset, nrow, this->dim_);
  // convert the index to HNSW format
  hnsw_index_ = cuvs::neighbors::hnsw::build(handle_, build_param_.hnsw_index_params, dataset_view);
}

template <typename T, typename IdxT>
void cuvs_cagra_hnswlib<T, IdxT>::set_search_param(const search_param_base& param_,
                                                   const void* filter_bitset)
{
  if (filter_bitset != nullptr) { throw std::runtime_error("Filtering is not supported yet."); }
  search_param_ = dynamic_cast<const search_param&>(param_);
}

template <typename T, typename IdxT>
void cuvs_cagra_hnswlib<T, IdxT>::save(const std::string& file) const
{
  if (cagra_ace_build_) {
    std::string index_filename = hnsw_index_->file_path();
    RAFT_EXPECTS(!index_filename.empty(), "HNSW index file path is not available.");
    RAFT_EXPECTS(std::filesystem::exists(index_filename),
                 "Index file '%s' does not exist.",
                 index_filename.c_str());
    if (std::filesystem::exists(file)) { std::filesystem::remove(file); }
    // might fail when using 2 different filesystems
    std::error_code ec;
    std::filesystem::rename(index_filename, file, ec);
    RAFT_EXPECTS(
      !ec, "Failed to rename index file '%s' to '%s'.", index_filename.c_str(), file.c_str());
  } else {
    cuvs::neighbors::hnsw::serialize(handle_, file, *(hnsw_index_.get()));
  }
}

template <typename T, typename IdxT>
void cuvs_cagra_hnswlib<T, IdxT>::load(const std::string& file)
{
  cuvs::neighbors::hnsw::index<T>* idx = nullptr;
  cuvs::neighbors::hnsw::deserialize(handle_,
                                     build_param_.hnsw_index_params,
                                     file,
                                     this->dim_,
                                     parse_metric_type(this->metric_),
                                     &idx);
  hnsw_index_ = std::shared_ptr<cuvs::neighbors::hnsw::index<T>>(idx);
}

template <typename T, typename IdxT>
void cuvs_cagra_hnswlib<T, IdxT>::search(
  const T* queries, int batch_size, int k, algo_base::index_type* neighbors, float* distances) const
{
  // Only Latency mode is supported for now
  auto queries_view =
    raft::make_host_matrix_view<const T, int64_t>(queries, batch_size, this->dim_);
  auto neighbors_view = raft::make_host_matrix_view<uint64_t, int64_t>(
    reinterpret_cast<uint64_t*>(neighbors), batch_size, k);
  auto distances_view = raft::make_host_matrix_view<float, int64_t>(distances, batch_size, k);

  cuvs::neighbors::hnsw::search(handle_,
                                search_param_.hnsw_search_param,
                                *(hnsw_index_.get()),
                                queries_view,
                                neighbors_view,
                                distances_view);
}

}  // namespace cuvs::bench
