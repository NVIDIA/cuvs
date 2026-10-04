/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/cagra.hpp>
#include <neighbors/detail/flowann/flowann_serialize.hpp>
#ifdef CUVS_ENABLE_FLOWANN_BUILD
#include <neighbors/detail/flowann/flowann_build.hpp>
#endif
#include "detail/flowann/cagra_adapter.hpp"
#include <mutex>

namespace cuvs::neighbors::cagra::detail {
namespace flowann = cuvs::neighbors::cagra::experimental::flowann;

// This owner is stable across moves of the public index. The context and session are destroyed
// before the graph/dataset bundle, so no polling thread can reference released graph storage.
template <typename T, typename Bundle>
class tiered_state final
  : public tiered_index_state<T, uint32_t, typename decltype(Bundle::index)::dataset_view_type> {
  using view_type = typename decltype(Bundle::index)::dataset_view_type;

 public:
  explicit tiered_state(Bundle&& bundle) : bundle_(std::move(bundle))
  {
    RAFT_CUDA_TRY(cudaGetDevice(&device_));
  }
  auto size() const noexcept -> uint32_t override { return bundle_.index.size(); }
  auto graph_degree() const noexcept -> uint32_t override { return bundle_.index.graph_degree(); }
  auto dataset() const noexcept -> view_type override { return bundle_.index.dataset(); }
  auto source_indices() const noexcept
    -> std::optional<raft::device_vector_view<const uint32_t, int64_t>> override
  {
    return bundle_.index.source_indices();
  }
  void serialize(raft::resources const& res, std::ostream& os, bool include_dataset) const override
  {
    RAFT_EXPECTS(include_dataset, "Tiered serialization currently requires include_dataset=true");
    std::lock_guard lock(mutex_);
    int device;
    RAFT_CUDA_TRY(cudaGetDevice(&device));
    RAFT_EXPECTS(device == device_, "Tiered serialization must use the index CUDA device");
    os.write("CTIR", 4);
    raft::serialize_scalar(res, os, uint32_t{1});
    flowann::serialize(res, os, bundle_.index);
  }
  auto statistics() const -> flowann::queue_statistics
  {
    std::lock_guard lock(mutex_);
    return context_ ? context_->statistics() : flowann::queue_statistics{};
  }
  template <typename OutputIdxT>
  void run(raft::resources const& res,
           search_params const& params,
           raft::device_matrix_view<const T, int64_t, raft::row_major> queries,
           raft::device_matrix_view<OutputIdxT, int64_t, raft::row_major> neighbors,
           raft::device_matrix_view<float, int64_t, raft::row_major> distances,
           cuvs::neighbors::filtering::base_filter const& filter) const
  {
    std::lock_guard lock(mutex_);
    int device;
    RAFT_CUDA_TRY(cudaGetDevice(&device));
    RAFT_EXPECTS(device == device_, "Tiered search must use the index's CUDA device");
    auto const options = params.tiered.value_or(tiered_search_params{});
    RAFT_EXPECTS(options.num_queues > 0, "Tiered search requires at least one queue");
    if (!context_ || options.num_queues != queue_.num_queues ||
        options.empty_pause != queue_.empty_pause ||
        options.collect_statistics != queue_.collect_statistics) {
      session_.reset();
      context_.reset();
      flowann::queue_params queue{
        options.num_queues, options.empty_pause, options.collect_statistics};
      context_ = std::make_unique<flowann::search_context>(bundle_.index.cross_graph(), queue);
      queue_   = options;
    }
    if (!options.keep_pollers_running) {
      session_.reset();
    } else if (!session_) {
      session_ = std::make_unique<flowann::search_session>(*context_);
    }
    flowann::search_params effective;
    static_cast<search_params&>(effective) = params;
    effective.tiered.reset();
    effective.num_seeds           = options.num_seeds;
    effective.sync_window_scale   = options.sync_window_scale;
    effective.sync_drop_threshold = options.sync_drop_threshold;
    flowann::search(
      res, effective, bundle_.index, *context_, queries, neighbors, distances, filter);
  }
#define SEARCH_OVERRIDE(I)                                                          \
  void search(raft::resources const& res,                                           \
              search_params const& params,                                          \
              raft::device_matrix_view<const T, int64_t, raft::row_major> queries,  \
              raft::device_matrix_view<I, int64_t, raft::row_major> neighbors,      \
              raft::device_matrix_view<float, int64_t, raft::row_major> distances,  \
              cuvs::neighbors::filtering::base_filter const& filter) const override \
  {                                                                                 \
    run(res, params, queries, neighbors, distances, filter);                        \
  }
  SEARCH_OVERRIDE(uint32_t)
  SEARCH_OVERRIDE(int64_t)
#undef SEARCH_OVERRIDE
 private:
  Bundle bundle_;
  int device_{};
  mutable std::mutex mutex_;
  mutable tiered_search_params queue_;
  mutable std::unique_ptr<flowann::search_context> context_;
  mutable std::unique_ptr<flowann::search_session> session_;
};

template <typename T, typename Bundle>
auto wrap_tiered(raft::resources const& res, Bundle&& bundle)
{
  using view_type = typename decltype(Bundle::index)::dataset_view_type;
  index<T, uint32_t, view_type> result(res, bundle.index.metric());
  auto state = std::make_shared<tiered_state<T, Bundle>>(std::move(bundle));
  tiered_index_access::install(result, std::move(state));
  return result;
}

#define DEFINE_ADOPT_TIERED(T)                                                                   \
  auto adopt_tiered(raft::resources const& res, flowann::device_padded_index_bundle<T>&& bundle) \
    -> device_padded_index<T>                                                                    \
  {                                                                                              \
    return wrap_tiered<T>(res, std::move(bundle));                                               \
  }                                                                                              \
  auto adopt_tiered(raft::resources const& res, flowann::vpq_f16_index_bundle<T>&& bundle)       \
    -> device_pq_index<T>                                                                        \
  {                                                                                              \
    return wrap_tiered<T>(res, std::move(bundle));                                               \
  }
DEFINE_ADOPT_TIERED(float)
DEFINE_ADOPT_TIERED(half)
DEFINE_ADOPT_TIERED(int8_t)
DEFINE_ADOPT_TIERED(uint8_t)
#undef DEFINE_ADOPT_TIERED

auto tiered_statistics(device_pq_index<float> const& idx) -> flowann::queue_statistics
{
  auto* state = dynamic_cast<tiered_state<float, flowann::vpq_f16_index_bundle<float>>*>(
    tiered_index_access::get(idx));
  RAFT_EXPECTS(state != nullptr, "Expected an owning tiered VPQ index");
  return state->statistics();
}

template <typename T, typename View>
void read_tiered(raft::resources const& res, std::istream& is, index<T, uint32_t, View>* idx)
{
  auto const version = raft::deserialize_scalar<uint32_t>(res, is);
  RAFT_EXPECTS(version == 1, "Unsupported CAGRA tiered serialization version");
  if constexpr (is_device_padded_dataset_view_v<View>) {
    auto bundle = flowann::deserialize_device_padded<T>(res, is);
    *idx        = wrap_tiered<T>(res, std::move(bundle));
  } else {
    auto bundle = flowann::deserialize_vpq_f16<T>(res, is);
    *idx        = wrap_tiered<T>(res, std::move(bundle));
  }
}

#define DEFINE_TIERED_IO(T)                                                                \
  void deserialize_tiered(                                                                 \
    raft::resources const& res, std::istream& is, device_padded_index<T>* idx)             \
  {                                                                                        \
    read_tiered(res, is, idx);                                                             \
  }                                                                                        \
  void deserialize_tiered(                                                                 \
    raft::resources const& res, std::istream& is, device_pq_index<T, uint32_t, half>* idx) \
  {                                                                                        \
    read_tiered(res, is, idx);                                                             \
  }
DEFINE_TIERED_IO(float)
DEFINE_TIERED_IO(half)
DEFINE_TIERED_IO(int8_t)
DEFINE_TIERED_IO(uint8_t)
#undef DEFINE_TIERED_IO

#ifdef CUVS_ENABLE_FLOWANN_BUILD
inline auto flowann_build_params(index_params const& params)
{
  auto const options = params.tiered.value_or(tiered_graph_params{});
  flowann::index_params out;
  out.cagra_params               = params;
  out.cagra_params.graph_storage = graph_storage_kind::device;
  out.cagra_params.tiered.reset();
  out.device_graph_budget_bytes      = options.device_graph_budget_bytes;
  out.node_per_cacheline             = options.node_per_cacheline;
  out.grouping.enabled               = options.grouping_enabled;
  out.grouping.n_groups              = options.n_groups;
  out.grouping.n_bits                = options.n_bits;
  out.grouping.balance_tolerance     = options.balance_tolerance;
  out.grouping.training_rows         = options.training_rows;
  out.grouping.assignment_batch_rows = options.assignment_batch_rows;
  out.grouping.kmeans_n_iters        = options.kmeans_n_iters;
  out.grouping.validate              = options.validate;
  out.num_seeds                      = options.num_seeds;
  out.seed_training_rows             = options.seed_training_rows;
  out.seed                           = options.seed;
  return out;
}
#define DEFINE_TIERED_BUILD(T)                                                \
  auto build_tiered(raft::resources const& res,                               \
                    index_params const& params,                               \
                    device_padded_dataset_view<T, int64_t> const& dataset)    \
    -> device_padded_index<T>                                                 \
  {                                                                           \
    auto bundle = flowann::build(res, flowann_build_params(params), dataset); \
    return wrap_tiered<T>(res, std::move(bundle));                            \
  }
DEFINE_TIERED_BUILD(float)
DEFINE_TIERED_BUILD(half)
DEFINE_TIERED_BUILD(int8_t)
DEFINE_TIERED_BUILD(uint8_t)
#undef DEFINE_TIERED_BUILD
#endif
}  // namespace cuvs::neighbors::cagra::detail

#ifdef CUVS_ENABLE_FLOWANN_BUILD
namespace cuvs::neighbors::cagra {
#define DEFINE_TIERED_PQ_BUILD(T)                                                              \
  auto build(raft::resources const& res,                                                       \
             index_params const& params,                                                       \
             cuvs::neighbors::vpq_params const& compression,                                   \
             host_standard_dataset_view<T, int64_t> const& dataset) -> device_pq_index<T>      \
  {                                                                                            \
    RAFT_EXPECTS(params.graph_storage == graph_storage_kind::tiered,                           \
                 "This compressed build overload requires a tiered graph");                    \
    RAFT_EXPECTS(params.attach_dataset_on_build, "Tiered builds must attach their dataset");   \
    auto bundle =                                                                              \
      detail::flowann::build(res, detail::flowann_build_params(params), compression, dataset); \
    return detail::wrap_tiered<T>(res, std::move(bundle));                                     \
  }
DEFINE_TIERED_PQ_BUILD(float)
DEFINE_TIERED_PQ_BUILD(half)
DEFINE_TIERED_PQ_BUILD(int8_t)
DEFINE_TIERED_PQ_BUILD(uint8_t)
#undef DEFINE_TIERED_PQ_BUILD
}  // namespace cuvs::neighbors::cagra
#endif
