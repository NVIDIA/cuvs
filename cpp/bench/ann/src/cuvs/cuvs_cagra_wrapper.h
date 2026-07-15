/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../../../../src/neighbors/detail/cagra/utils.hpp"
#include "../common/ann_types.hpp"
#include "../common/cuda_huge_page_resource.hpp"
#include "cuvs_ann_bench_utils.h"
#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/composite/index.hpp>
#include <cuvs/neighbors/dynamic_batching.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/nn_descent.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/operators.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <atomic>
#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <raft/util/integer_utils.hpp>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace cuvs::bench {

namespace detail {

/** If persistent CAGRA search uses few benchmark threads, log a throughput hint once per process.
 */
inline void maybe_log_cagra_persistent_concurrency_hint(bool persistent_search)
{
  if (!persistent_search) { return; }

  const unsigned hc = std::max(1u, static_cast<unsigned>(std::thread::hardware_concurrency()));
  const unsigned bn = static_cast<unsigned>(std::max(0, benchmark_n_threads));
  if (bn >= 2u * hc) { return; }

  static std::atomic<bool> logged{false};
  bool expected = false;
  if (!logged.compare_exchange_strong(expected, true)) { return; }

  const unsigned threads_rec = 16u * hc;
  RAFT_LOG_INFO(
    "CAGRA persistent search benefits from high client concurrency (try `--mode=throughput "
    "--threads=1:%u`).",
    threads_rec);
}

}  // namespace detail

enum class AllocatorType { kHostPinned, kHostHugePage, kDevice };
enum class CagraBuildAlgo { kAuto, kIvfPq, kNnDescent };
enum class CagraMergeType { kPhysical, kLogical };

template <typename T, typename IdxT>
class cuvs_cagra : public algo<T>, public algo_gpu {
 public:
  using search_param_base = typename algo<T>::search_param;
  using algo<T>::dim_;
  using algo<T>::metric_;

  struct search_param : public search_param_base {
    cuvs::neighbors::cagra::search_params p;
    float refine_ratio;
    AllocatorType graph_mem   = AllocatorType::kDevice;
    AllocatorType dataset_mem = AllocatorType::kDevice;
    [[nodiscard]] auto needs_dataset() const -> bool override { return true; }
    /* Dynamic batching */
    bool dynamic_batching = false;
    int64_t dynamic_batching_k;
    int64_t dynamic_batching_max_batch_size     = 4;
    double dynamic_batching_dispatch_timeout_ms = 0.01;
    size_t dynamic_batching_n_queues            = 8;
    bool dynamic_batching_conservative_dispatch = false;
  };

  struct build_param {
    // The optimal defaults depend on the dataset shape and thus only available once the build
    // function is called.
    using dataset_dependent_params = std::function<cuvs::neighbors::cagra::index_params(
      raft::matrix_extent<int64_t>, cuvs::distance::DistanceType)>;
    dataset_dependent_params cagra_params;
    size_t num_dataset_splits = 1;
    CagraMergeType merge_type = CagraMergeType::kPhysical;
  };

  cuvs_cagra(Metric metric, int dim, const build_param& param, int concurrent_searches = 1)
    : algo<T>(metric, dim),
      index_params_(param),

      dataset_(std::make_shared<raft::device_matrix<T, int64_t, raft::row_major>>(
        std::move(raft::make_device_matrix<T, int64_t>(handle_, 0, 0)))),
      graph_(std::make_shared<raft::device_matrix<IdxT, int64_t, raft::row_major>>(
        std::move(raft::make_device_matrix<IdxT, int64_t>(handle_, 0, 0)))),
      input_dataset_v_(
        std::make_shared<raft::device_matrix_view<const T, int64_t, raft::row_major>>(
          nullptr, 0, 0))

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
  void search_base(const T* queries,
                   int batch_size,
                   int k,
                   algo_base::index_type* neighbors,
                   float* distances) const;

  [[nodiscard]] auto get_sync_stream() const noexcept -> cudaStream_t override
  {
    return handle_.get_sync_stream();
  }

  [[nodiscard]] auto uses_stream() const noexcept -> bool override
  {
    // If the algorithm uses persistent kernel, the CPU has to synchronize by the end of computing
    // the result. Hence it guarantees the benchmark CUDA stream is empty by the end of the
    // execution. Hence we inform the benchmark to not waste the time on recording & synchronizing
    // the event.
    return !search_params_.persistent;
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
  void save_to_hnswlib(const std::string& file) const;
  std::unique_ptr<algo<T>> copy() override;

  auto get_index() const -> const cuvs::neighbors::cagra::index<T, IdxT>* { return index_.get(); }

 private:
  // handle_ must go first to make sure it dies last and all memory allocated in pool
  configured_raft_resources handle_{};
  rmm::mr::pinned_host_memory_resource mr_pinned_;
  raft::mr::cuda_huge_page_resource mr_huge_page_;
  AllocatorType graph_mem_{AllocatorType::kDevice};
  AllocatorType dataset_mem_{AllocatorType::kDevice};
  float refine_ratio_;
  build_param index_params_;
  bool need_dataset_update_{true};
  cuvs::neighbors::cagra::search_params search_params_;
  std::shared_ptr<cuvs::neighbors::cagra::index<T, IdxT>> index_;
  std::shared_ptr<raft::device_matrix<IdxT, int64_t, raft::row_major>> graph_;
  std::shared_ptr<raft::device_matrix<T, int64_t, raft::row_major>> dataset_;
  std::shared_ptr<raft::device_matrix_view<const T, int64_t, raft::row_major>> input_dataset_v_;

  std::shared_ptr<cuvs::neighbors::dynamic_batching::index<T, algo_base::index_type>>
    dynamic_batcher_;
  cuvs::neighbors::dynamic_batching::search_params dynamic_batcher_sp_{};
  int64_t dynamic_batching_max_batch_size_;
  size_t dynamic_batching_n_queues_;
  bool dynamic_batching_conservative_dispatch_;

  std::shared_ptr<cuvs::neighbors::filtering::base_filter> filter_;
  std::vector<std::shared_ptr<cuvs::neighbors::cagra::index<T, IdxT>>> sub_indices_;

  inline rmm::device_async_resource_ref get_mr(AllocatorType mem_type)
  {
    switch (mem_type) {
      case (AllocatorType::kHostPinned): return mr_pinned_;
      case (AllocatorType::kHostHugePage): return mr_huge_page_;
      default: return rmm::mr::get_current_device_resource_ref();
    }
  }
};

template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::build(const T* dataset, size_t nrow)
{
  auto dataset_extents = raft::make_extents<IdxT>(nrow, dim_);
  auto params          = index_params_.cagra_params(dataset_extents, parse_metric_type(metric_));

  auto dataset_view_host =
    raft::make_mdspan<const T, IdxT, raft::row_major, true, false>(dataset, dataset_extents);
  auto dataset_view_device =
    raft::make_mdspan<const T, IdxT, raft::row_major, false, true>(dataset, dataset_extents);
  bool dataset_is_on_host = raft::get_device_for_address(dataset) == -1;
  if (index_params_.num_dataset_splits <= 1) {
    index_ = std::make_shared<cuvs::neighbors::cagra::index<T, IdxT>>(std::move(
      dataset_is_on_host ? cuvs::neighbors::cagra::build(handle_, params, dataset_view_host)
                         : cuvs::neighbors::cagra::build(handle_, params, dataset_view_device)));
  } else {
    IdxT rows_per_split =
      raft::ceildiv<IdxT>(nrow, static_cast<IdxT>(index_params_.num_dataset_splits));
    for (size_t i = 0; i < index_params_.num_dataset_splits; ++i) {
      IdxT start = static_cast<IdxT>(i * rows_per_split);
      if (start >= nrow) break;
      IdxT rows        = std::min(rows_per_split, static_cast<IdxT>(nrow) - start);
      const T* sub_ptr = dataset + static_cast<size_t>(start) * dim_;
      auto sub_host =
        raft::make_host_matrix_view<const T, int64_t, raft::row_major>(sub_ptr, rows, dim_);
      auto sub_dev =
        raft::make_device_matrix_view<const T, int64_t, raft::row_major>(sub_ptr, rows, dim_);

      auto sub_index = cuvs::neighbors::cagra::index<T, IdxT>(handle_, params.metric);
      if (index_params_.merge_type == CagraMergeType::kPhysical) {
        if (dataset_is_on_host) {
          sub_index.update_dataset(handle_, sub_host);
        } else {
          sub_index.update_dataset(handle_, sub_dev);
        }
      }
      if (index_params_.merge_type == CagraMergeType::kLogical) {
        if (dataset_is_on_host) {
          sub_index = cuvs::neighbors::cagra::build(handle_, params, sub_host);
        } else {
          sub_index = cuvs::neighbors::cagra::build(handle_, params, sub_dev);
        }
      }
      auto sub_index_shared =
        std::make_shared<cuvs::neighbors::cagra::index<T, IdxT>>(std::move(sub_index));
      sub_indices_.push_back(std::move(sub_index_shared));
    }
    if (index_params_.merge_type == CagraMergeType::kPhysical) {
      std::vector<cuvs::neighbors::cagra::index<T, IdxT>*> indices;
      indices.reserve(sub_indices_.size());
      for (auto& ptr : sub_indices_) {
        indices.push_back(ptr.get());
      }

      index_ = std::make_shared<cuvs::neighbors::cagra::index<T, IdxT>>(
        std::move(cuvs::neighbors::cagra::merge(handle_, params, indices)));
    }
  }
}

inline auto allocator_to_string(AllocatorType mem_type) -> std::string
{
  if (mem_type == AllocatorType::kDevice) {
    return "device";
  } else if (mem_type == AllocatorType::kHostPinned) {
    return "host_pinned";
  } else if (mem_type == AllocatorType::kHostHugePage) {
    return "host_huge_page";
  }
  return "<invalid allocator type>";
}

template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::set_search_param(const search_param_base& param,
                                           const void* filter_bitset)
{
  if (index_) { filter_ = make_cuvs_filter(filter_bitset, index_->size()); }
  auto sp = dynamic_cast<const search_param&>(param);
  bool needs_dynamic_batcher_update =
    (dynamic_batching_max_batch_size_ != sp.dynamic_batching_max_batch_size) ||
    (dynamic_batching_n_queues_ != sp.dynamic_batching_n_queues) ||
    (dynamic_batching_conservative_dispatch_ != sp.dynamic_batching_conservative_dispatch);
  dynamic_batching_max_batch_size_        = sp.dynamic_batching_max_batch_size;
  dynamic_batching_n_queues_              = sp.dynamic_batching_n_queues;
  dynamic_batching_conservative_dispatch_ = sp.dynamic_batching_conservative_dispatch;
  search_params_                          = sp.p;
  refine_ratio_                           = sp.refine_ratio;
  if (sp.graph_mem != graph_mem_) {
    // Move graph to correct memory space
    graph_mem_ = sp.graph_mem;
    RAFT_LOG_DEBUG("moving graph to new memory space: %s", allocator_to_string(graph_mem_).c_str());
    // We create a new graph and copy to it from existing graph
    auto mr = get_mr(graph_mem_);

    // Create a new graph, then copy, and __only then__ replace the shared pointer.
    auto old_graph =
      index_->graph();  // view of graph_ if it exists, of an internal index member otherwise
    auto new_graph = raft::make_device_mdarray<IdxT, int64_t>(handle_, mr, old_graph.extents());
    raft::copy(new_graph.data_handle(),
               old_graph.data_handle(),
               old_graph.size(),
               raft::resource::get_cuda_stream(handle_));
    raft::resource::sync_stream(handle_);
    *graph_ = std::move(new_graph);

    // NB: update_graph() only stores a view in the index. We need to keep the graph object alive.
    index_->update_graph(handle_, make_const_mdspan(graph_->view()));
    needs_dynamic_batcher_update = true;
  }

  if (sp.dataset_mem != dataset_mem_ || need_dataset_update_) {
    dataset_mem_ = sp.dataset_mem;

    // First free up existing memory
    *dataset_ = raft::make_device_matrix<T, int64_t>(handle_, 0, 0);
    index_->update_dataset(handle_, make_const_mdspan(dataset_->view()));

    // Allocate space using the correct memory resource.
    RAFT_LOG_DEBUG("moving dataset to new memory space: %s",
                   allocator_to_string(dataset_mem_).c_str());

    auto mr = get_mr(dataset_mem_);
    cuvs::neighbors::cagra::detail::copy_with_padding(handle_, *dataset_, *input_dataset_v_, mr);

    auto dataset_view = raft::make_device_strided_matrix_view<const T, int64_t>(
      dataset_->data_handle(), dataset_->extent(0), this->dim_, dataset_->extent(1));
    index_->update_dataset(handle_, dataset_view);

    need_dataset_update_         = false;
    needs_dynamic_batcher_update = true;
  }

  // dynamic batching
  if (sp.dynamic_batching) {
    if (!dynamic_batcher_ || needs_dynamic_batcher_update) {
      dynamic_batcher_ =
        std::make_shared<cuvs::neighbors::dynamic_batching::index<T, algo_base::index_type>>(
          handle_,
          cuvs::neighbors::dynamic_batching::index_params{
            {},
            sp.dynamic_batching_k,
            sp.dynamic_batching_max_batch_size,
            sp.dynamic_batching_n_queues,
            sp.dynamic_batching_conservative_dispatch},
          *index_,
          search_params_,
          filter_.get());
    }
    dynamic_batcher_sp_.dispatch_timeout_ms = sp.dynamic_batching_dispatch_timeout_ms;
  } else {
    if (dynamic_batcher_) { dynamic_batcher_.reset(); }
  }

  detail::maybe_log_cagra_persistent_concurrency_hint(search_params_.persistent);
}

template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::set_search_dataset(const T* dataset, size_t nrow)
{
  if (index_params_.num_dataset_splits > 1 &&
      index_params_.merge_type == CagraMergeType::kLogical) {
    bool dataset_is_on_host = raft::get_device_for_address(dataset) == -1;
    IdxT rows_per_split =
      raft::ceildiv<IdxT>(nrow, static_cast<IdxT>(index_params_.num_dataset_splits));
    for (size_t i = 0; i < sub_indices_.size(); ++i) {
      IdxT start = static_cast<IdxT>(i * rows_per_split);
      if (start >= nrow) break;
      IdxT rows        = std::min(rows_per_split, static_cast<IdxT>(nrow) - start);
      const T* sub_ptr = dataset + static_cast<size_t>(start) * dim_;
      auto sub_host =
        raft::make_host_matrix_view<const T, int64_t, raft::row_major>(sub_ptr, rows, dim_);
      auto sub_dev =
        raft::make_device_matrix_view<const T, int64_t, raft::row_major>(sub_ptr, rows, dim_);
      auto sub_index = sub_indices_[i].get();
      if (index_params_.merge_type == CagraMergeType::kLogical) {
        if (dataset_is_on_host) {
          sub_index->update_dataset(handle_, sub_host);
        } else {
          sub_index->update_dataset(handle_, sub_dev);
        }
      }
    }
    need_dataset_update_ = false;
  } else {
    using ds_idx_type = decltype(index_->data().n_rows());
    bool is_vpq =
      dynamic_cast<const cuvs::neighbors::vpq_dataset<half, ds_idx_type>*>(&index_->data()) ||
      dynamic_cast<const cuvs::neighbors::vpq_dataset<float, ds_idx_type>*>(&index_->data());
    // It can happen that we are re-using a previous algo object which already has
    // the dataset set. Check if we need update.
    if (static_cast<size_t>(input_dataset_v_->extent(0)) != nrow ||
        input_dataset_v_->data_handle() != dataset) {
      *input_dataset_v_ =
        raft::make_device_matrix_view<const T, int64_t>(dataset, nrow, this->dim_);
      need_dataset_update_ = !is_vpq;  // ignore update if this is a VPQ dataset.
    }
  }
}

template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::save(const std::string& file) const
{
  if (index_params_.num_dataset_splits > 1 &&
      index_params_.merge_type == CagraMergeType::kLogical) {
    for (size_t i = 0; i < sub_indices_.size(); ++i) {
      std::string subfile = file + (i == 0 ? "" : ".subidx." + std::to_string(i));
      cuvs::neighbors::cagra::serialize(handle_, subfile, *sub_indices_[i], false);
    }
    std::ofstream f(file + ".submeta", std::ios::out);
    f << sub_indices_.size();
    f.close();
  } else {
    using ds_idx_type = decltype(index_->data().n_rows());
    bool is_vpq =
      dynamic_cast<const cuvs::neighbors::vpq_dataset<half, ds_idx_type>*>(&index_->data()) ||
      dynamic_cast<const cuvs::neighbors::vpq_dataset<float, ds_idx_type>*>(&index_->data());
    cuvs::neighbors::cagra::serialize(handle_, file, *index_, is_vpq);
  }
}

template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::save_to_hnswlib(const std::string& file) const
{
  cuvs::neighbors::cagra::serialize_to_hnswlib(handle_, file, *index_);
}

template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::load(const std::string& file)
{
  std::ifstream meta(file + ".submeta", std::ios::in);
  if (index_params_.num_dataset_splits > 1 &&
      index_params_.merge_type == CagraMergeType::kLogical && meta.good()) {
    // Load multiple sub-indices for logical merge
    size_t count;
    meta >> count;
    meta.close();
    sub_indices_.clear();
    for (size_t i = 0; i < count; ++i) {
      std::string subfile = file + (i == 0 ? "" : ".subidx." + std::to_string(i));
      auto sub_index      = std::make_shared<cuvs::neighbors::cagra::index<T, IdxT>>(handle_);
      cuvs::neighbors::cagra::deserialize(handle_, subfile, sub_index.get());
      sub_indices_.push_back(std::move(sub_index));
    }
  } else {
    index_ = std::make_shared<cuvs::neighbors::cagra::index<T, IdxT>>(handle_);
    cuvs::neighbors::cagra::deserialize(handle_, file, index_.get());
  }
}

template <typename T, typename IdxT>
std::unique_ptr<algo<T>> cuvs_cagra<T, IdxT>::copy()
{
  return std::make_unique<cuvs_cagra<T, IdxT>>(std::cref(*this));  // use copy constructor
}

template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::search_base(
  const T* queries, int batch_size, int k, algo_base::index_type* neighbors, float* distances) const
{
  static_assert(std::is_integral_v<algo_base::index_type>);
  static_assert(std::is_integral_v<IdxT>);

  auto queries_view = raft::make_device_matrix_view<const T, int64_t>(queries, batch_size, dim_);
  auto neighbors_view =
    raft::make_device_matrix_view<algo_base::index_type, int64_t>(neighbors, batch_size, k);
  auto distances_view = raft::make_device_matrix_view<float, int64_t>(distances, batch_size, k);

  if (dynamic_batcher_) {
    cuvs::neighbors::dynamic_batching::search(handle_,
                                              dynamic_batcher_sp_,
                                              *dynamic_batcher_,
                                              queries_view,
                                              neighbors_view,
                                              distances_view);
  } else {
    if (index_params_.num_dataset_splits <= 1 ||
        index_params_.merge_type == CagraMergeType::kPhysical) {
      cuvs::neighbors::cagra::search(
        handle_, search_params_, *index_, queries_view, neighbors_view, distances_view, *filter_);
    } else {
      if (index_params_.merge_type == CagraMergeType::kLogical) {
        // TODO: index merge must happen outside of search, otherwise what are we benchmarking?
        std::vector<cuvs::neighbors::cagra::index<T, IdxT>*> cagra_indices;
        cagra_indices.reserve(sub_indices_.size());
        for (auto& ptr : sub_indices_) {
          cagra_indices.push_back(ptr.get());
        }

        raft::resources composite_handle(handle_);
        size_t n_streams = cagra_indices.size();
        raft::resource::set_cuda_stream_pool(composite_handle,
                                             std::make_shared<rmm::cuda_stream_pool>(n_streams));

        cuvs::neighbors::composite::composite_index<T, IdxT, algo_base::index_type> composite(
          cagra_indices);
        composite.search(
          composite_handle, search_params_, queries_view, neighbors_view, distances_view);
      }
    }
  }
}

template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::search(
  const T* queries, int batch_size, int k, algo_base::index_type* neighbors, float* distances) const
{
  static_assert(std::is_integral_v<algo_base::index_type>);
  static_assert(std::is_integral_v<IdxT>);

  auto k0                       = static_cast<size_t>(refine_ratio_ * k);
  const bool disable_refinement = k0 <= static_cast<size_t>(k);
  const raft::resources& res    = handle_;
  // NOTE: caching mem_type to reduce mutex locks
  // raft::get_device_for_address call cuda API to get the pointer properties,
  // this means it locks the context mutex for a very small amount of time.
  // In the event of thread contention (such as thousands threads), this time can actually increase.
  // Hence we try to bypass this check for repeated search calls.
  thread_local MemoryType mem_type                   = MemoryType::kDevice;
  thread_local algo_base::index_type* prev_neighbors = nullptr;
  if (prev_neighbors != neighbors) {
    prev_neighbors = neighbors;
    mem_type =
      raft::get_device_for_address(neighbors) >= 0 ? MemoryType::kDevice : MemoryType::kHostPinned;
  }

  // If dynamic batching is used and there's no sync between benchmark laps, multiple sequential
  // requests can group together. The data is copied asynchronously, and if the same intermediate
  // buffer is used for multiple requests, they can override each other's data. Hence, we need to
  // allocate as much space as required by the maximum number of sequential requests.
  auto max_dyn_grouping = dynamic_batcher_ ? raft::div_rounding_up_safe<int64_t>(
                                               dynamic_batching_max_batch_size_, batch_size) *
                                               dynamic_batching_n_queues_
                                           : 1;
  auto tmp_buf_size =
    ((disable_refinement ? 0 : (sizeof(float) + sizeof(algo_base::index_type)))) * batch_size * k0;
  auto& tmp_buf = get_tmp_buffer_from_global_pool(tmp_buf_size * max_dyn_grouping);
  thread_local static int64_t group_id = 0;
  auto* candidates_ptr                 = reinterpret_cast<algo_base::index_type*>(
    reinterpret_cast<uint8_t*>(tmp_buf.data(mem_type)) + tmp_buf_size * group_id);
  group_id = (group_id + 1) % max_dyn_grouping;
  auto* candidate_dists_ptr =
    reinterpret_cast<float*>(candidates_ptr + (disable_refinement ? 0 : batch_size * k0));

  if (disable_refinement) {
    search_base(queries, batch_size, k, neighbors, distances);
  } else {
    search_base(queries, batch_size, k0, candidates_ptr, candidate_dists_ptr);

    if (mem_type == MemoryType::kHostPinned && uses_stream()) {
      // If the algorithm uses a stream to synchronize (non-persistent kernel), but the data is in
      // the pinned host memory, we need to synchronize before the refinement operation to wait for
      // the data being available for the host.
      raft::resource::sync_stream(res);
    }

    auto candidate_ixs =
      raft::make_device_matrix_view<const algo_base::index_type, algo_base::index_type>(
        candidates_ptr, batch_size, k0);
    auto queries_v =
      raft::make_device_matrix_view<const T, algo_base::index_type>(queries, batch_size, dim_);
    refine_helper(
      res, *input_dataset_v_, queries_v, candidate_ixs, k, neighbors, distances, index_->metric());
  }
}
}  // namespace cuvs::bench
