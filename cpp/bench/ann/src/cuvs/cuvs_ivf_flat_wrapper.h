/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../common/ann_types.hpp"
#include "cuvs_ann_bench_utils.h"

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/util/cudart_utils.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace cuvs::bench {

template <typename T, typename IdxT>
class cuvs_ivf_flat : public algo<T>, public algo_gpu {
 public:
  using search_param_base = typename algo<T>::search_param;

  struct search_param : public search_param_base {
    cuvs::neighbors::ivf_flat::search_params ivf_flat_params;
  };

  using build_param = cuvs::neighbors::ivf_flat::index_params;

  cuvs_ivf_flat(Metric metric, int dim, const build_param& param)
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
    return handle_.get_sync_stream().get();
  }

  // to enable dataset access from GPU memory
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
  [[nodiscard]] auto estimate_work(int n_queries, int k) const -> work_estimate override;

 private:
  // handle_ must go first to make sure it dies last and all memory allocated in pool
  configured_raft_resources handle_{};
  build_param index_params_;
  cuvs::neighbors::ivf_flat::search_params search_params_;
  std::shared_ptr<cuvs::neighbors::ivf_flat::index<T, IdxT>> index_;
  int device_;
  int dimension_;

  std::shared_ptr<cuvs::neighbors::filtering::base_filter> filter_;
};

template <typename T, typename IdxT>
void cuvs_ivf_flat<T, IdxT>::build(const T* dataset, size_t nrow)
{
  index_ = std::make_shared<cuvs::neighbors::ivf_flat::index<T, IdxT>>(
    std::move(cuvs::neighbors::ivf_flat::build(
      handle_,
      index_params_,
      raft::make_host_matrix_view<const T, int64_t>(dataset, nrow, dimension_))));
  // Note: internally the IVF-Flat build works with simple pointers, and accepts both host and
  // device pointer. Therefore, although we provide here a host_mdspan, this works with device
  // pointer too.
}

template <typename T, typename IdxT>
void cuvs_ivf_flat<T, IdxT>::set_search_param(const search_param_base& param,
                                              const void* filter_bitset)
{
  filter_        = make_cuvs_filter(filter_bitset, index_->size());
  auto sp        = dynamic_cast<const search_param&>(param);
  search_params_ = sp.ivf_flat_params;
  assert(search_params_.n_probes <= index_params_.n_lists);
}

template <typename T, typename IdxT>
void cuvs_ivf_flat<T, IdxT>::save(const std::string& file) const
{
  cuvs::neighbors::ivf_flat::serialize(handle_, file, *index_);
  return;
}

template <typename T, typename IdxT>
void cuvs_ivf_flat<T, IdxT>::load(const std::string& file)
{
  index_ =
    std::make_shared<cuvs::neighbors::ivf_flat::index<T, IdxT>>(handle_, index_params_, this->dim_);

  cuvs::neighbors::ivf_flat::deserialize(handle_, file, index_.get());
  return;
}

template <typename T, typename IdxT>
std::unique_ptr<algo<T>> cuvs_ivf_flat<T, IdxT>::copy()
{
  return std::make_unique<cuvs_ivf_flat<T, IdxT>>(*this);  // use copy constructor
}

template <typename T, typename IdxT>
auto cuvs_ivf_flat<T, IdxT>::estimate_work(int n_queries, int k) const -> work_estimate
{
  // No estimate for filtered searches: how much data they load is not known.
  if (filter_->get_filter_type() != cuvs::neighbors::filtering::FilterType::None) { return {}; }

  const double nlist  = index_->n_lists();
  const double nprobe = std::min<double>(search_params_.n_probes, nlist);
  const double n_rows = index_->size();
  const double dim    = dimension_;

  // Average number of vectors in one list (the lists are assumed to be equal-sized).
  const double average_cluster_size = n_rows / nlist;

  // Distance comparisons per query in the fine search: one per vector in the probed clusters.
  const double comparisons_per_query = nprobe * average_cluster_size;

  // Coarse search over the centroids + fine search over the probed lists.
  const double flops = 2.0 * n_queries * dim * (nlist + comparisons_per_query);

  // Expected number of distinct clusters probed in the batch if each query picks nprobe clusters
  // uniformly at random. We assume that each of them is read once per batch. This is the ideal
  // case: unlike IVF-PQ, the IVF-Flat search does not sort the probes by cluster ID, so queries
  // that share a cluster may read it more than once, and the real traffic may be higher. We also
  // assume that the centroids are read once per batch.
  const double distinct_clusters_probed = nlist * (1.0 - std::pow(1.0 - nprobe / nlist, n_queries));
  const size_t bytes =
    sizeof(float) * nlist * dim +
    sizeof(T) * (distinct_clusters_probed * average_cluster_size * dim + n_queries * dim) +
    static_cast<size_t>(n_queries) * k * (sizeof(float) + sizeof(algo_base::index_type));
  return work_estimate{flops, bytes};
}

template <typename T, typename IdxT>
void cuvs_ivf_flat<T, IdxT>::search(
  const T* queries, int batch_size, int k, algo_base::index_type* neighbors, float* distances) const
{
  static_assert(std::is_integral_v<algo_base::index_type>);
  static_assert(std::is_integral_v<IdxT>);

  IdxT* neighbors_idx_t;
  std::optional<rmm::device_uvector<IdxT>> neighbors_storage{std::nullopt};
  if constexpr (sizeof(IdxT) == sizeof(algo_base::index_type)) {
    neighbors_idx_t = reinterpret_cast<IdxT*>(neighbors);
  } else {
    neighbors_storage.emplace(batch_size * k, raft::resource::get_cuda_stream(handle_));
    neighbors_idx_t = neighbors_storage->data();
  }
  cuvs::neighbors::ivf_flat::search(
    handle_,
    search_params_,
    *index_,
    raft::make_device_matrix_view<const T, int64_t>(queries, batch_size, index_->dim()),
    raft::make_device_matrix_view<IdxT, int64_t>(neighbors_idx_t, batch_size, k),
    raft::make_device_matrix_view<float, int64_t>(distances, batch_size, k),
    *filter_);
  if constexpr (sizeof(IdxT) != sizeof(algo_base::index_type)) {
    raft::linalg::unaryOp(neighbors,
                          neighbors_idx_t,
                          batch_size * k,
                          raft::cast_op<algo_base::index_type>(),
                          raft::resource::get_cuda_stream(handle_));
  }
}
}  // namespace cuvs::bench
