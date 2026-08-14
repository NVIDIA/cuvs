/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../common/ann_types.hpp"
#include "../common/util.hpp"
#include <cuvs/neighbors/refine.hpp>

#include <cuvs/distance/distance.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/failure_callback_resource_adaptor.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <type_traits>

namespace cuvs::bench {

inline auto parse_metric_type(cuvs::bench::Metric metric) -> cuvs::distance::DistanceType
{
  if (metric == cuvs::bench::Metric::kInnerProduct) {
    return cuvs::distance::DistanceType::InnerProduct;
  } else if (metric == cuvs::bench::Metric::kEuclidean) {
    // Even for L2 expanded RAFT IVF Flat uses unexpanded formula
    return cuvs::distance::DistanceType::L2Expanded;
  } else {
    throw std::runtime_error("raft supports only metric type of inner product and L2");
  }
}

/** Report a more verbose error with a backtrace when OOM occurs on RMM side. */
inline auto rmm_oom_callback(std::size_t bytes, void*) -> bool
{
  auto cuda_status = cudaGetLastError();
  size_t free      = 0;
  size_t total     = 0;
  RAFT_CUDA_TRY_NO_THROW(cudaMemGetInfo(&free, &total));
  RAFT_FAIL(
    "Failed to allocate %zu bytes using RMM memory resource. "
    "NB: latest cuda status = %s, free memory = %zu, total memory = %zu.",
    bytes,
    cudaGetErrorName(cuda_status),
    free,
    total);
}

/**
 * This container keeps the part of raft state that should be shared among multiple copies of raft
 * handles (in different CPU threads).
 * An example of this is an RMM memory resource: if we had an RMM memory pool per thread, we'd
 * quickly run out of memory.
 */
class shared_raft_resources {
 public:
  using large_mr_type = rmm::mr::managed_memory_resource;

  shared_raft_resources()
  try : large_mr_() {
    orig_resource_ =
      rmm::mr::set_current_device_resource(rmm::mr::failure_callback_resource_adaptor<>{
        rmm::mr::pool_memory_resource{rmm::mr::get_current_device_resource_ref(),
                                      1024 * 1024 * 1024ull},
        rmm_oom_callback,
        nullptr});
  } catch (const std::exception& e) {
    auto cuda_status = cudaGetLastError();
    size_t free      = 0;
    size_t total     = 0;
    RAFT_CUDA_TRY_NO_THROW(cudaMemGetInfo(&free, &total));
    RAFT_FAIL(
      "Failed to initialize shared raft resources (NB: latest cuda status = %s, free memory = %zu, "
      "total memory = %zu): %s",
      cudaGetErrorName(cuda_status),
      free,
      total,
      e.what());
  }

  shared_raft_resources(shared_raft_resources&&)                               = delete;
  auto operator=(shared_raft_resources&&) -> shared_raft_resources&            = delete;
  shared_raft_resources(const shared_raft_resources& res)                      = delete;
  auto operator=(const shared_raft_resources& other) -> shared_raft_resources& = delete;

  ~shared_raft_resources() noexcept { rmm::mr::set_current_device_resource(orig_resource_); }

  auto get_large_memory_resource() noexcept -> rmm::device_async_resource_ref { return large_mr_; }

 private:
  cuda::mr::any_resource<cuda::mr::device_accessible> orig_resource_;
  large_mr_type large_mr_;
};

/**
 * This struct is used by multiple raft benchmark wrappers. It serves as a thread-safe keeper of
 * shared and private GPU resources (see below).
 *
 * - Accessing the same `configured_raft_resources` from concurrent threads is not safe.
 * - Accessing the copies of `configured_raft_resources` from concurrent threads is safe.
 * - There must be at most one "original" `configured_raft_resources` at any time, but as many
 *   copies of it as needed (modifies the program static state).
 */
class configured_raft_resources {
 public:
  /**
   * This constructor has the shared state passed unmodified but creates the local state anew.
   * It's used by the copy constructor.
   */
  explicit configured_raft_resources(const std::shared_ptr<shared_raft_resources>& shared_res)
    : shared_res_{shared_res},
      res_{std::make_unique<raft::device_resources>(
        rmm::cuda_stream_view(get_stream_from_global_pool()))}
  {
    raft::resource::set_large_workspace_resource(
      *res_, raft::mr::device_resource{shared_res_->get_large_memory_resource()});
  }

  /** Default constructor creates all resources anew. */
  configured_raft_resources() : configured_raft_resources{std::make_shared<shared_raft_resources>()}
  {
  }

  configured_raft_resources(configured_raft_resources&&);
  auto operator=(configured_raft_resources&&) -> configured_raft_resources&;
  ~configured_raft_resources() = default;
  configured_raft_resources(const configured_raft_resources& res)
    : configured_raft_resources{res.shared_res_}
  {
  }
  auto operator=(const configured_raft_resources& other) -> configured_raft_resources&
  {
    this->shared_res_ = other.shared_res_;
    return *this;
  }

  operator raft::resources&() noexcept { return *res_; }              // NOLINT
  operator const raft::resources&() const noexcept { return *res_; }  // NOLINT

  /** Get the main stream */
  [[nodiscard]] auto get_sync_stream() const noexcept
  {
    return raft::resource::get_cuda_stream(*res_);
  }

 private:
  /** The resources shared among multiple raft handles / threads. */
  std::shared_ptr<shared_raft_resources> shared_res_;
  /**
   * Until we make the use of copies of raft::resources thread-safe, each benchmark wrapper must
   * have its own copy of it.
   */
  std::unique_ptr<raft::device_resources> res_ = std::make_unique<raft::device_resources>();
};

inline configured_raft_resources::configured_raft_resources(configured_raft_resources&&) = default;
inline auto configured_raft_resources::operator=(configured_raft_resources&&)
  -> configured_raft_resources& = default;

/** A helper to refine the neighbors when the data is on device or on host. */
template <typename DatasetT, typename QueriesT, typename CandidatesT>
void refine_helper(const raft::resources& res,
                   const DatasetT& dataset,
                   const QueriesT& queries,
                   const CandidatesT& candidates,
                   int k,
                   algo_base::index_type* neighbors,
                   float* distances,
                   cuvs::distance::DistanceType metric)
{
  using data_type    = typename DatasetT::value_type;
  using index_type   = algo_base::index_type;
  using extents_type = int64_t;  // device-side refine requires this

  static_assert(std::is_same_v<data_type, typename QueriesT::value_type>);
  static_assert(std::is_same_v<data_type, typename DatasetT::value_type>);
  static_assert(std::is_same_v<index_type, typename CandidatesT::value_type>);

  extents_type batch_size = queries.extent(0);
  extents_type dim        = queries.extent(1);
  extents_type k0         = candidates.extent(1);

  if (raft::get_device_for_address(dataset.data_handle()) >= 0) {
    auto dataset_device = raft::make_device_matrix_view<const data_type, extents_type>(
      dataset.data_handle(), dataset.extent(0), dataset.extent(1));
    auto queries_device = raft::make_device_matrix_view<const data_type, extents_type>(
      queries.data_handle(), batch_size, dim);
    auto candidates_device = raft::make_device_matrix_view<const index_type, extents_type>(
      candidates.data_handle(), batch_size, k0);
    auto neighbors_device =
      raft::make_device_matrix_view<index_type, extents_type>(neighbors, batch_size, k);
    auto distances_device =
      raft::make_device_matrix_view<float, extents_type>(distances, batch_size, k);

    cuvs::neighbors::refine(res,
                            dataset_device,
                            queries_device,
                            candidates_device,
                            neighbors_device,
                            distances_device,
                            metric);
  } else {
    auto dataset_host = raft::make_host_matrix_view<const data_type, extents_type>(
      dataset.data_handle(), dataset.extent(0), dataset.extent(1));
    if (raft::get_device_for_address(queries.data_handle()) >= 0) {
      // Queries & results are on the device

      auto queries_host    = raft::make_host_matrix<data_type, extents_type>(batch_size, dim);
      auto candidates_host = raft::make_host_matrix<index_type, extents_type>(batch_size, k0);
      auto neighbors_host  = raft::make_host_matrix<index_type, extents_type>(batch_size, k);
      auto distances_host  = raft::make_host_matrix<float, extents_type>(batch_size, k);

      auto stream = raft::resource::get_cuda_stream(res);
      raft::copy(queries_host.data_handle(), queries.data_handle(), queries_host.size(), stream);
      raft::copy(
        candidates_host.data_handle(), candidates.data_handle(), candidates_host.size(), stream);

      raft::resource::sync_stream(res);  // wait for the queries and candidates
      cuvs::neighbors::refine(res,
                              dataset_host,
                              queries_host.view(),
                              candidates_host.view(),
                              neighbors_host.view(),
                              distances_host.view(),
                              metric);

      raft::copy(neighbors, neighbors_host.data_handle(), neighbors_host.size(), stream);
      raft::copy(distances, distances_host.data_handle(), distances_host.size(), stream);

    } else {
      // Queries & results are on the host - no device sync / copy needed

      auto queries_host = raft::make_host_matrix_view<const data_type, extents_type>(
        queries.data_handle(), batch_size, dim);
      auto candidates_host = raft::make_host_matrix_view<const index_type, extents_type>(
        candidates.data_handle(), batch_size, k0);
      auto neighbors_host =
        raft::make_host_matrix_view<index_type, extents_type>(neighbors, batch_size, k);
      auto distances_host =
        raft::make_host_matrix_view<float, extents_type>(distances, batch_size, k);

      cuvs::neighbors::refine(
        res, dataset_host, queries_host, candidates_host, neighbors_host, distances_host, metric);
    }
  }
}

/**
 * Construct a cuVS-compatible bitset filter object from a raw pointer to the bitset.
 *
 * @param[in] filter_bitset a pointer to a pre-generated bitset
 * @param[in] n_rows the number of elements in the bitset / dataset.
 * @return a shared pointer to the filter object (doesn't own the bitset data!)
 */
inline auto make_cuvs_filter(const void* filter_bitset, int64_t n_rows)
  -> std::shared_ptr<cuvs::neighbors::filtering::base_filter>
{
  if (filter_bitset != nullptr) {
    return std::make_shared<cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>>(
      raft::core::bitset_view<uint32_t, int64_t>(
        const_cast<uint32_t*>(reinterpret_cast<const uint32_t*>(filter_bitset)), n_rows));
  } else {
    return std::make_shared<cuvs::neighbors::filtering::none_sample_filter>();
  }
}

}  // namespace cuvs::bench
