/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cuvs_ann_bench_utils.h"
#include "cuvs_ivf_flat_wrapper.h"
#include <cuvs/neighbors/ivf_flat.hpp>
#include <raft/core/device_resources_snmg.hpp>

namespace cuvs::bench {
using namespace cuvs::neighbors;

template <typename T, typename IdxT>
class cuvs_mg_ivf_flat : public algo<T>, public algo_gpu {
 public:
  using search_param_base = typename algo<T>::search_param;
  using algo<T>::dim_;

  using build_param = cuvs::neighbors::mg_index_params<ivf_flat::index_params>;

  struct search_param : public cuvs::bench::cuvs_ivf_flat<T, IdxT>::search_param {
    cuvs::neighbors::sharded_merge_mode merge_mode;
  };

  cuvs_mg_ivf_flat(Metric metric, int dim, const build_param& param)
    : algo<T>(metric, dim), index_params_(param), clique_()
  {
    index_params_.metric = parse_metric_type(metric);

    clique_.set_memory_pool(80);
  }

  void build(const T* dataset, size_t nrow) final;
  void set_search_param(const search_param_base& param, const void* filter_bitset) override;
  void search(const T* queries,
              int batch_size,
              int k,
              algo_base::index_type* neighbors,
              float* distances) const override;

  [[nodiscard]] auto get_preference() const -> algo_property override
  {
    algo_property property;
    property.dataset_memory_type = MemoryType::kHost;
    property.query_memory_type   = MemoryType::kHost;
    return property;
  }

  [[nodiscard]] auto get_sync_stream() const noexcept -> cudaStream_t override
  {
    auto stream = raft::resource::get_cuda_stream(clique_);
    return stream.get();
  }

  [[nodiscard]] auto uses_stream() const noexcept -> bool override { return false; }

  void save(const std::string& file) const override;
  void load(const std::string&) override;
  std::unique_ptr<algo<T>> copy() override;

 private:
  raft::device_resources_snmg clique_;
  build_param index_params_;
  cuvs::neighbors::mg_search_params<ivf_flat::search_params> search_params_;
  std::shared_ptr<cuvs::neighbors::mg_index<cuvs::neighbors::ivf_flat::index<T, IdxT>, T, IdxT>>
    index_;
};

template <typename T, typename IdxT>
void cuvs_mg_ivf_flat<T, IdxT>::build(const T* dataset, size_t nrow)
{
  auto dataset_view =
    raft::make_host_matrix_view<const T, int64_t, raft::row_major>(dataset, IdxT(nrow), IdxT(dim_));
  auto idx = cuvs::neighbors::ivf_flat::build(clique_, index_params_, dataset_view);
  index_ =
    std::make_shared<cuvs::neighbors::mg_index<cuvs::neighbors::ivf_flat::index<T, IdxT>, T, IdxT>>(
      std::move(idx));
}

template <typename T, typename IdxT>
void cuvs_mg_ivf_flat<T, IdxT>::set_search_param(const search_param_base& param,
                                                 const void* filter_bitset)
{
  if (filter_bitset != nullptr) { throw std::runtime_error("Filtering is not supported yet."); }
  auto sp = dynamic_cast<const search_param&>(param);
  // search_params_ = sp.ivf_flat_params;
  ivf_flat::search_params* search_params_ptr_ =
    static_cast<ivf_flat::search_params*>(&search_params_);
  *search_params_ptr_       = sp.ivf_flat_params;
  search_params_.merge_mode = sp.merge_mode;
  assert(search_params_.n_probes <= index_params_.n_lists);
}

template <typename T, typename IdxT>
void cuvs_mg_ivf_flat<T, IdxT>::save(const std::string& file) const
{
  cuvs::neighbors::ivf_flat::serialize(clique_, *index_, file);
}

template <typename T, typename IdxT>
void cuvs_mg_ivf_flat<T, IdxT>::load(const std::string& file)
{
  index_ =
    std::make_shared<cuvs::neighbors::mg_index<cuvs::neighbors::ivf_flat::index<T, IdxT>, T, IdxT>>(
      std::move(cuvs::neighbors::ivf_flat::deserialize<T, IdxT>(clique_, file)));
}

template <typename T, typename IdxT>
std::unique_ptr<algo<T>> cuvs_mg_ivf_flat<T, IdxT>::copy()
{
  return std::make_unique<cuvs_mg_ivf_flat<T, IdxT>>(*this);  // use copy constructor
}

template <typename T, typename IdxT>
void cuvs_mg_ivf_flat<T, IdxT>::search(
  const T* queries, int batch_size, int k, algo_base::index_type* neighbors, float* distances) const
{
  auto queries_view = raft::make_host_matrix_view<const T, int64_t, raft::row_major>(
    queries, IdxT(batch_size), IdxT(dim_));
  auto neighbors_view = raft::make_host_matrix_view<IdxT, int64_t, raft::row_major>(
    neighbors, IdxT(batch_size), IdxT(k));
  auto distances_view = raft::make_host_matrix_view<float, int64_t, raft::row_major>(
    distances, IdxT(batch_size), IdxT(k));

  cuvs::neighbors::ivf_flat::search(
    clique_, *index_, search_params_, queries_view, neighbors_view, distances_view);
}

}  // namespace cuvs::bench
