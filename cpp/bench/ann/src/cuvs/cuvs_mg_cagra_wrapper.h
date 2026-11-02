/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "cuvs_ann_bench_utils.h"
#include "cuvs_cagra_wrapper.h"
#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/device_resources_snmg.hpp>

#include <algorithm>
#include <vector>

namespace cuvs::bench {
using namespace cuvs::neighbors;

enum class AllocatorType;
enum class CagraBuildAlgo;

template <typename T, typename IdxT>
class cuvs_mg_cagra : public algo<T>, public algo_gpu {
 public:
  using search_param_base = typename algo<T>::search_param;
  using algo<T>::dim_;
  using algo<T>::metric_;

  struct build_param : public cuvs::bench::cuvs_cagra<T, IdxT>::build_param {
    cuvs::neighbors::distribution_mode mode;
  };

  struct search_param : public cuvs::bench::cuvs_cagra<T, IdxT>::search_param {
    cuvs::neighbors::sharded_merge_mode merge_mode;
  };

  cuvs_mg_cagra(Metric metric, int dim, const build_param& param, int concurrent_searches = 1)
    : algo<T>(metric, dim), index_params_(param), clique_(), query_dim_(dim)
  {
    clique_.set_memory_pool(80);
  }

  void build(const T* dataset, size_t nrow) final;

  void set_search_param(const search_param_base& param, const void* filter_bitset) override;

  void set_search_dataset(const T* dataset, size_t nrow) override;

  void search(const T* queries,
              int batch_size,
              int k,
              algo_base::index_type* neighbors,
              float* distances) const override;
  void search_base(const T* queries,
                   int batch_size,
                   int k,
                   algo_base::index_type* neighbors,
                   float* distances) const;

  [[nodiscard]] auto get_sync_stream() const noexcept -> cudaStream_t override
  {
    auto stream = raft::resource::get_cuda_stream(clique_);
    return stream.get();
  }

  // to enable dataset access from GPU memory
  [[nodiscard]] auto get_preference() const -> algo_property override
  {
    algo_property property;
    property.dataset_memory_type = MemoryType::kHost;
    property.query_memory_type   = MemoryType::kHost;
    return property;
  }
  void save(const std::string& file) const override;
  void load(const std::string&) override;
  void save_to_hnswlib(const std::string& file) const;
  std::unique_ptr<algo<T>> copy() override;

 private:
  raft::device_resources_snmg clique_;
  float refine_ratio_;
  build_param index_params_;
  cuvs::neighbors::mg_search_params<cagra::search_params> search_params_;
  // The width of the rows the index was built on, which is the width queries must have. An MG index
  // built from a padded host dataset takes the row stride as its dimension, so for a dimension that
  // is not aligned this is the padded width and the queries are zero-padded to it.
  uint32_t query_dim_;
  std::shared_ptr<
    cuvs::neighbors::mg_index<cuvs::neighbors::cagra::device_padded_index<T, IdxT>, T, IdxT>>
    index_;
};

template <typename T, typename IdxT>
void cuvs_mg_cagra<T, IdxT>::build(const T* dataset, size_t nrow)
{
  auto dataset_extents = raft::make_extents<IdxT>(nrow, dim_);
  auto params          = index_params_.cagra_params(dataset_extents, parse_metric_type(metric_));

  cuvs::neighbors::mg_index_params<cagra::index_params> build_params = params;
  build_params.mode                                                  = index_params_.mode;

  auto dataset_mds =
    raft::make_host_matrix_view<const T, int64_t, raft::row_major>(dataset, nrow, dim_);
  auto idx = [&] {
    if (cuvs::neighbors::matrix_row_width_matches_cagra_required(dataset_mds)) {
      // The rows are aligned already: view the input without a copy.
      query_dim_ = static_cast<uint32_t>(dim_);
      return cuvs::neighbors::cagra::build(
        clique_, build_params, cuvs::neighbors::make_host_padded_dataset_view(dataset_mds));
    }
    // A padded view cannot wrap rows that are not aligned, so the host rows get padded in a copy.
    // The copy is only needed while building. The index takes the padded row width as its
    // dimension; the padding is zero, which leaves the distances unchanged.
    auto dataset_padded     = cuvs::neighbors::make_host_padded_dataset(clique_, dataset_mds);
    auto const& padded_view = std::as_const(*dataset_padded).view();
    query_dim_              = static_cast<uint32_t>(padded_view.data_const_view().stride(0));
    return cuvs::neighbors::cagra::build(clique_, build_params, padded_view);
  }();
  index_ = std::make_shared<
    cuvs::neighbors::mg_index<cuvs::neighbors::cagra::device_padded_index<T, IdxT>, T, IdxT>>(
    std::move(idx));
}

inline auto allocator_to_string(AllocatorType mem_type) -> std::string;

template <typename T, typename IdxT>
void cuvs_mg_cagra<T, IdxT>::set_search_param(const search_param_base& param,
                                              const void* filter_bitset)
{
  if (filter_bitset != nullptr) { throw std::runtime_error("Filtering is not supported yet."); }
  auto sp                                  = dynamic_cast<const search_param&>(param);
  cagra::search_params* search_params_ptr_ = static_cast<cagra::search_params*>(&search_params_);
  *search_params_ptr_                      = sp.p;
  search_params_.merge_mode                = sp.merge_mode;
  refine_ratio_                            = sp.refine_ratio;
}

template <typename T, typename IdxT>
void cuvs_mg_cagra<T, IdxT>::set_search_dataset(const T* dataset, size_t nrow)
{
}

template <typename T, typename IdxT>
void cuvs_mg_cagra<T, IdxT>::save(const std::string& file) const
{
  cuvs::neighbors::cagra::serialize(clique_, *index_, file);
}

template <typename T, typename IdxT>
void cuvs_mg_cagra<T, IdxT>::load(const std::string& file)
{
  index_ = std::make_shared<
    cuvs::neighbors::mg_index<cuvs::neighbors::cagra::device_padded_index<T, IdxT>, T, IdxT>>(
    clique_, index_params_.mode);
  cuvs::neighbors::cagra::deserialize<T, IdxT>(clique_, file, index_.get());
  query_dim_ = static_cast<uint32_t>(dim_);
  if (!index_->ann_interfaces_.empty() && index_->ann_interfaces_[0].index_.has_value()) {
    query_dim_ = static_cast<uint32_t>(index_->ann_interfaces_[0].index_->dim());
  }
}

template <typename T, typename IdxT>
std::unique_ptr<algo<T>> cuvs_mg_cagra<T, IdxT>::copy()
{
  return std::make_unique<cuvs_mg_cagra<T, IdxT>>(*this);  // use copy constructor
}

template <typename T, typename IdxT>
void cuvs_mg_cagra<T, IdxT>::search_base(
  const T* queries, int batch_size, int k, algo_base::index_type* neighbors, float* distances) const
{
  static_assert(std::is_integral_v<algo_base::index_type>);
  static_assert(std::is_integral_v<IdxT>);

  // The queries have to be as wide as the index rows; pad them with zeros when the index is wider.
  thread_local std::vector<T> padded_queries;
  if (query_dim_ != static_cast<uint32_t>(dim_)) {
    padded_queries.assign(static_cast<size_t>(batch_size) * query_dim_, T{});
    for (int i = 0; i < batch_size; i++) {
      std::copy_n(queries + static_cast<size_t>(i) * dim_,
                  dim_,
                  padded_queries.data() + static_cast<size_t>(i) * query_dim_);
    }
    queries = padded_queries.data();
  }
  auto queries_view =
    raft::make_host_matrix_view<const T, int64_t, raft::row_major>(queries, batch_size, query_dim_);
  auto neighbors_view =
    raft::make_host_matrix_view<int64_t, int64_t, raft::row_major>(neighbors, batch_size, k);
  auto distances_view =
    raft::make_host_matrix_view<float, int64_t, raft::row_major>(distances, batch_size, k);

  cuvs::neighbors::cagra::search(
    clique_, *index_, search_params_, queries_view, neighbors_view, distances_view);
}

template <typename T, typename IdxT>
void cuvs_mg_cagra<T, IdxT>::search(
  const T* queries, int batch_size, int k, algo_base::index_type* neighbors, float* distances) const
{
  auto k0                       = static_cast<size_t>(refine_ratio_ * k);
  const bool disable_refinement = k0 <= static_cast<size_t>(k);

  if (disable_refinement) {
    search_base(queries, batch_size, k, neighbors, distances);
  } else {
    throw std::runtime_error("refinement not supported");
  }
}
}  // namespace cuvs::bench
