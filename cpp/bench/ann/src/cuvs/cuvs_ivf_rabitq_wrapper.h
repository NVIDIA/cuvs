/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../common/ann_types.hpp"
#include "cuvs_ann_bench_utils.h"

#include <cuvs/neighbors/ivf_rabitq.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/util/cudart_utils.hpp>
#include <rmm/cuda_stream_pool.hpp>

#include <type_traits>

namespace cuvs::bench {

template <typename T, typename IdxT>
class cuvs_ivf_rabitq : public algo<T>, public algo_gpu {
 public:
  using search_param_base = typename algo<T>::search_param;
  using algo<T>::dim_;

  struct search_param : public search_param_base {
    cuvs::neighbors::ivf_rabitq::search_params rabitq_param;
    float refine_ratio = 1.0f;
    [[nodiscard]] auto needs_dataset() const -> bool override { return refine_ratio > 1.0f; }
  };

  using build_param = cuvs::neighbors::ivf_rabitq::index_params;

  cuvs_ivf_rabitq(Metric metric, int dim, const build_param& param)
    : algo<T>(metric, dim), index_params_(param), dimension_(dim)
  {
  }

  void build(const T* dataset, size_t nrow) final;

  void set_search_param(const search_param_base& param, const void* filter_bitset) override;
  void set_search_dataset(const T* dataset, size_t nrow) override;

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
    property.dataset_memory_type = MemoryType::kHost;
    property.query_memory_type   = MemoryType::kDevice;
    return property;
  }
  void save(const std::string& file) const override;
  void load(const std::string&) override;
  std::unique_ptr<algo<T>> copy() override;

 private:
  // handle_ must go first to make sure it dies last and all memory allocated in pool
  configured_raft_resources handle_{};
  build_param index_params_;
  cuvs::neighbors::ivf_rabitq::search_params search_params_;
  std::shared_ptr<cuvs::neighbors::ivf_rabitq::index<IdxT>> index_;
  int dimension_;
  float refine_ratio_ = 1.0;
  raft::device_matrix_view<const T, IdxT> dataset_;
};

template <typename T, typename IdxT>
void cuvs_ivf_rabitq<T, IdxT>::save(const std::string& file) const
{
  cuvs::neighbors::ivf_rabitq::serialize(handle_, file, *index_);
}

template <typename T, typename IdxT>
void cuvs_ivf_rabitq<T, IdxT>::load(const std::string& file)
{
  index_ = std::make_shared<cuvs::neighbors::ivf_rabitq::index<IdxT>>(handle_);
  cuvs::neighbors::ivf_rabitq::deserialize(handle_, file, index_.get());
}

template <typename T, typename IdxT>
void cuvs_ivf_rabitq<T, IdxT>::build(const T* dataset, size_t nrow)
{
  // Create a CUDA stream pool with 1 stream (besides main stream) for kernel/copy overlapping.
  size_t n_streams = 1;
  raft::resource::set_cuda_stream_pool(handle_, std::make_shared<rmm::cuda_stream_pool>(n_streams));
  auto dataset_v = raft::make_device_matrix_view<const T, IdxT>(dataset, IdxT(nrow), dim_);
  std::make_shared<cuvs::neighbors::ivf_rabitq::index<IdxT>>(
    std::move(cuvs::neighbors::ivf_rabitq::build(handle_, index_params_, dataset_v)))
    .swap(index_);
  // Note: internally the IVF-RaBitQ build works with simple pointers, and accepts both host and
  // device pointer. Therefore, although we provide here a device_mdspan, this works with host
  // pointer too.
}

template <typename T, typename IdxT>
std::unique_ptr<algo<T>> cuvs_ivf_rabitq<T, IdxT>::copy()
{
  return std::make_unique<cuvs_ivf_rabitq<T, IdxT>>(*this);  // use copy constructor
}

template <typename T, typename IdxT>
void cuvs_ivf_rabitq<T, IdxT>::set_search_param(const search_param_base& param, const void*)
{
  auto sp        = dynamic_cast<const search_param&>(param);
  search_params_ = sp.rabitq_param;
  refine_ratio_  = sp.refine_ratio;
  assert(search_params_.n_probes <= index_params_.n_lists);
}

template <typename T, typename IdxT>
void cuvs_ivf_rabitq<T, IdxT>::set_search_dataset(const T* dataset, size_t nrow)
{
  dataset_ = raft::make_device_matrix_view<const T, IdxT>(dataset, nrow, index_->dim());
}

template <typename T, typename IdxT>
void cuvs_ivf_rabitq<T, IdxT>::search(
  const T* queries, int batch_size, int k, algo_base::index_type* neighbors, float* distances) const
{
  static_assert(std::is_integral_v<algo_base::index_type>);
  static_assert(std::is_integral_v<IdxT>);

  IdxT* neighbors_idx;
  std::optional<rmm::device_uvector<IdxT>> neighbors_storage{std::nullopt};
  if constexpr (sizeof(IdxT) == sizeof(algo_base::index_type)) {
    neighbors_idx = reinterpret_cast<IdxT*>(neighbors);
  } else {
    neighbors_storage.emplace(batch_size * k, raft::resource::get_cuda_stream(handle_));
    neighbors_idx = neighbors_storage->data();
  }

  auto queries_view =
    raft::make_device_matrix_view<const T, int64_t>(queries, batch_size, dimension_);
  auto neighbors_view = raft::make_device_matrix_view<IdxT, int64_t>(neighbors_idx, batch_size, k);
  auto distances_view = raft::make_device_matrix_view<float, int64_t>(distances, batch_size, k);

  cuvs::neighbors::ivf_rabitq::search(
    handle_, search_params_, *index_, queries_view, neighbors_view, distances_view);

  if constexpr (sizeof(IdxT) != sizeof(algo_base::index_type)) {
    raft::linalg::unaryOp(neighbors,
                          neighbors_idx,
                          batch_size * k,
                          raft::cast_op<algo_base::index_type>(),
                          raft::resource::get_cuda_stream(handle_));
  }
}
}  // namespace cuvs::bench
