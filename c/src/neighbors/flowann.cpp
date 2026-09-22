/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/flowann.h>

#if defined(CUVS_ENABLE_FLOWANN_SEARCH) && defined(CUVS_ENABLE_FLOWANN_BUILD)

#include "../core/exceptions.hpp"
#include "cagra.hpp"

#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/flowann.hpp>
#include <cuvs/neighbors/flowann_build.hpp>
#include <cuvs/neighbors/flowann_serialize.hpp>

#include <raft/core/resources.hpp>

#include "../core/interop.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>

namespace {

namespace flowann = cuvs::neighbors::cagra::experimental::flowann;

struct flowann_index_iface {
  virtual ~flowann_index_iface()                                  = default;
  virtual auto size() const noexcept -> int64_t                    = 0;
  virtual auto dim() const noexcept -> int64_t                     = 0;
  virtual auto graph_degree() const noexcept -> int64_t            = 0;
  virtual void serialize(raft::resources const&, std::string const&) const = 0;
  virtual void search(raft::resources const&,
                      cuvsFlowannSearchParams const&,
                      DLManagedTensor*,
                      DLManagedTensor*,
                      DLManagedTensor*) = 0;
};

template <typename T>
struct flowann_index_holder final : flowann_index_iface {
  using bundle_type = flowann::device_padded_index_bundle<T, std::uint32_t>;

  explicit flowann_index_holder(bundle_type&& value) : bundle(std::move(value)) {}

  auto size() const noexcept -> int64_t override { return bundle.index.size(); }
  auto dim() const noexcept -> int64_t override { return bundle.index.dim(); }
  auto graph_degree() const noexcept -> int64_t override { return bundle.index.graph_degree(); }

  void serialize(raft::resources const& res, std::string const& filename) const override
  {
    flowann::serialize(res, filename, bundle.index);
  }

  void search(raft::resources const& res,
              cuvsFlowannSearchParams const& params,
              DLManagedTensor* queries,
              DLManagedTensor* neighbors,
              DLManagedTensor* distances) override
  {
    using query_view = raft::device_matrix_view<const T, int64_t, raft::row_major>;
    using neighbor_view = raft::device_matrix_view<std::uint32_t, int64_t, raft::row_major>;
    using distance_view = raft::device_matrix_view<float, int64_t, raft::row_major>;

    auto query_matrix    = cuvs::core::from_dlpack<query_view>(queries);
    auto neighbor_matrix = cuvs::core::from_dlpack<neighbor_view>(neighbors);
    auto distance_matrix = cuvs::core::from_dlpack<distance_view>(distances);
    RAFT_EXPECTS(query_matrix.extent(1) == bundle.index.dim(),
                 "FlowANN query dimension does not match the index");
    RAFT_EXPECTS(neighbor_matrix.extent(0) == query_matrix.extent(0) &&
                   distance_matrix.extent(0) == query_matrix.extent(0),
                 "FlowANN output row count does not match the query count");
    RAFT_EXPECTS(neighbor_matrix.extent(1) == distance_matrix.extent(1),
                 "FlowANN neighbor and distance output widths differ");

    auto queue = flowann::queue_params{params.num_queues,
                                       params.empty_pause,
                                       params.collect_statistics};
    if (context == nullptr || queue.num_queues != queue_params.num_queues ||
        queue.empty_pause != queue_params.empty_pause ||
        queue.collect_statistics != queue_params.collect_statistics) {
      context.reset();
      context     = std::make_unique<flowann::search_context>(bundle.index.cross_graph(), queue);
      queue_params = queue;
    }

    flowann::search_params cpp_params;
    cuvs::neighbors::cagra::convert_c_search_params(params.cagra, &cpp_params);
    cpp_params.num_seeds            = params.num_seeds;
    cpp_params.sync_window_scale    = params.sync_window_scale;
    cpp_params.sync_drop_threshold  = params.sync_drop_threshold;
    flowann::search(res,
                    cpp_params,
                    bundle.index,
                    *context,
                    query_matrix,
                    neighbor_matrix,
                    distance_matrix);
  }

  bundle_type bundle;
  std::unique_ptr<flowann::search_context> context;
  flowann::queue_params queue_params{};
};

inline auto checked_holder(cuvsFlowannIndex_t index) -> flowann_index_iface*
{
  RAFT_EXPECTS(index != nullptr, "FlowANN index handle is null");
  RAFT_EXPECTS(index->addr != 0, "FlowANN index is empty");
  return reinterpret_cast<flowann_index_iface*>(index->addr);
}

inline void destroy_index(cuvsFlowannIndex_t index)
{
  if (index != nullptr && index->addr != 0) {
    delete reinterpret_cast<flowann_index_iface*>(index->addr);
    index->addr = 0;
  }
}

template <typename OwnerT, typename ViewT, typename Fn>
void with_dataset_view(cuvsDataset_t dataset, Fn&& fn)
{
  if (dataset->is_owning) {
    fn(reinterpret_cast<OwnerT*>(dataset->addr)->as_dataset_view());
  } else {
    fn(*reinterpret_cast<ViewT*>(dataset->addr));
  }
}

inline auto make_cpp_params(cuvsFlowannIndexParams const& params, int64_t rows, int64_t dim)
  -> flowann::index_params
{
  flowann::index_params out;
  cuvsCagraIndexParams cagra_params{params.metric,
                                    params.intermediate_graph_degree,
                                    params.graph_degree,
                                    params.build_algo,
                                    params.nn_descent_niter,
                                    nullptr};
  cuvs::neighbors::cagra::convert_c_index_params(cagra_params, rows, dim, &out.cagra_params);
  out.device_graph_budget_bytes       = params.device_graph_budget_bytes;
  out.node_per_cacheline              = params.node_per_cacheline;
  out.grouping.enabled                = params.grouping_enabled;
  out.grouping.n_groups               = params.n_groups;
  out.grouping.n_bits                 = params.n_bits;
  out.grouping.balance_tolerance      = params.balance_tolerance;
  out.grouping.training_rows          = params.training_rows;
  out.grouping.assignment_batch_rows  = params.assignment_batch_rows;
  out.grouping.kmeans_n_iters         = params.kmeans_n_iters;
  out.grouping.validate               = params.validate;
  out.num_seeds                       = params.num_seeds;
  out.seed_training_rows              = params.seed_training_rows;
  out.seed                            = params.seed;
  return out;
}

template <typename T>
void build_typed(raft::resources const& res,
                 cuvsFlowannIndexParams const& params,
                 cuvsDataset_t dataset,
                 cuvsFlowannIndex_t output)
{
  using owner_type = cuvs::neighbors::device_padded_dataset<T, int64_t>;
  using view_type  = cuvs::neighbors::device_padded_dataset_view<T, int64_t>;
  with_dataset_view<owner_type, view_type>(dataset, [&](auto const& view) {
    auto cpp_params = make_cpp_params(params, view.n_rows(), view.dim());
    auto bundle     = flowann::build(res, cpp_params, view);
    output->addr = reinterpret_cast<uintptr_t>(
      new flowann_index_holder<T>(std::move(bundle)));
  });
}

template <typename T>
void deserialize_typed(raft::resources const& res,
                       std::string const& filename,
                       cuvsFlowannIndex_t output)
{
  auto bundle = flowann::deserialize_device_padded<T, std::uint32_t>(res, filename);
  output->addr = reinterpret_cast<uintptr_t>(new flowann_index_holder<T>(std::move(bundle)));
}

template <typename Fn>
void dispatch_dtype(DLDataType dtype, Fn&& fn)
{
  if (dtype.code == kDLFloat && dtype.bits == 32) {
    fn.template operator()<float>();
  } else if (dtype.code == kDLInt && dtype.bits == 8) {
    fn.template operator()<std::int8_t>();
  } else if (dtype.code == kDLUInt && dtype.bits == 8) {
    fn.template operator()<std::uint8_t>();
  } else {
    RAFT_FAIL("FlowANN supports float32, int8, and uint8 datasets");
  }
}

}  // namespace

extern "C" cuvsError_t cuvsFlowannIndexParamsCreate(cuvsFlowannIndexParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(params != nullptr, "FlowANN index params output is null");
    auto defaults = flowann::index_params{};
    *params = new cuvsFlowannIndexParams{
      static_cast<cuvsDistanceType>(defaults.cagra_params.metric),
      defaults.cagra_params.intermediate_graph_degree,
      defaults.cagra_params.graph_degree,
      AUTO_SELECT,
      20,
      defaults.device_graph_budget_bytes,
      defaults.node_per_cacheline,
      defaults.grouping.enabled,
      defaults.grouping.n_groups,
      defaults.grouping.n_bits,
      defaults.grouping.balance_tolerance,
      defaults.grouping.training_rows,
      defaults.grouping.assignment_batch_rows,
      defaults.grouping.kmeans_n_iters,
      defaults.grouping.validate,
      defaults.num_seeds,
      defaults.seed_training_rows,
      defaults.seed};
  });
}

extern "C" cuvsError_t cuvsFlowannIndexParamsDestroy(cuvsFlowannIndexParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsFlowannSearchParamsCreate(cuvsFlowannSearchParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(params != nullptr, "FlowANN search params output is null");
    cuvsCagraSearchParams_t cagra = nullptr;
    auto status = cuvsCagraSearchParamsCreate(&cagra);
    RAFT_EXPECTS(status == CUVS_SUCCESS, "Failed to create default CAGRA search parameters");
    auto defaults = flowann::search_params{};
    auto queue    = flowann::queue_params{};
    *params = new cuvsFlowannSearchParams{*cagra,
                                          defaults.num_seeds,
                                          defaults.sync_window_scale,
                                          defaults.sync_drop_threshold,
                                          queue.num_queues,
                                          queue.empty_pause,
                                          queue.collect_statistics};
    cuvsCagraSearchParamsDestroy(cagra);
  });
}

extern "C" cuvsError_t cuvsFlowannSearchParamsDestroy(cuvsFlowannSearchParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsFlowannIndexCreate(cuvsFlowannIndex_t* index)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(index != nullptr, "FlowANN index output is null");
    *index = new cuvsFlowannIndex{0, {}};
  });
}

extern "C" cuvsError_t cuvsFlowannIndexDestroy(cuvsFlowannIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    destroy_index(index);
    delete index;
  });
}

extern "C" cuvsError_t cuvsFlowannIndexGetDims(cuvsFlowannIndex_t index, int64_t* dim)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(dim != nullptr, "FlowANN dimension output is null");
    *dim = checked_holder(index)->dim();
  });
}

extern "C" cuvsError_t cuvsFlowannIndexGetSize(cuvsFlowannIndex_t index, int64_t* size)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(size != nullptr, "FlowANN size output is null");
    *size = checked_holder(index)->size();
  });
}

extern "C" cuvsError_t cuvsFlowannIndexGetGraphDegree(cuvsFlowannIndex_t index, int64_t* degree)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(degree != nullptr, "FlowANN graph-degree output is null");
    *degree = checked_holder(index)->graph_degree();
  });
}

extern "C" cuvsError_t cuvsFlowannBuild(cuvsResources_t res,
                                         cuvsFlowannIndexParams_t params,
                                         cuvsDataset_t dataset,
                                         cuvsFlowannIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(res != 0 && params != nullptr && dataset != nullptr && index != nullptr,
                 "FlowANN build received a null argument");
    RAFT_EXPECTS(dataset->addr != 0, "FlowANN build dataset is empty");
    RAFT_EXPECTS(dataset->mem_type == CUVS_DATASET_MEM_TYPE_DEVICE &&
                   dataset->layout == CUVS_DATASET_LAYOUT_PADDED,
                 "FlowANN build requires a device-padded dataset");
    RAFT_EXPECTS(params->build_algo != ACE, "FlowANN integrated build does not support ACE");
    destroy_index(index);
    index->dtype = dataset->dtype;
    auto& resources = *reinterpret_cast<raft::resources*>(res);
    dispatch_dtype(dataset->dtype, [&]<typename T>() { build_typed<T>(resources, *params, dataset, index); });
  });
}

extern "C" cuvsError_t cuvsFlowannSearch(cuvsResources_t res,
                                          cuvsFlowannSearchParams_t params,
                                          cuvsFlowannIndex_t index,
                                          DLManagedTensor* queries,
                                          DLManagedTensor* neighbors,
                                          DLManagedTensor* distances)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(res != 0 && params != nullptr && index != nullptr && queries != nullptr &&
                   neighbors != nullptr && distances != nullptr,
                 "FlowANN search received a null argument");
    RAFT_EXPECTS(queries->dl_tensor.dtype.code == index->dtype.code &&
                   queries->dl_tensor.dtype.bits == index->dtype.bits,
                 "FlowANN query dtype does not match the index");
    checked_holder(index)->search(*reinterpret_cast<raft::resources*>(res),
                                  *params,
                                  queries,
                                  neighbors,
                                  distances);
  });
}

extern "C" cuvsError_t cuvsFlowannSerialize(cuvsResources_t res,
                                             const char* filename,
                                             cuvsFlowannIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(res != 0 && filename != nullptr, "FlowANN serialize received a null argument");
    checked_holder(index)->serialize(*reinterpret_cast<raft::resources*>(res), filename);
  });
}

extern "C" cuvsError_t cuvsFlowannDeserialize(cuvsResources_t res,
                                               const char* filename,
                                               DLDataType dtype,
                                               cuvsFlowannIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(res != 0 && filename != nullptr && index != nullptr,
                 "FlowANN deserialize received a null argument");
    destroy_index(index);
    index->dtype = dtype;
    auto& resources = *reinterpret_cast<raft::resources*>(res);
    dispatch_dtype(dtype, [&]<typename T>() { deserialize_typed<T>(resources, filename, index); });
  });
}

#endif  // CUVS_ENABLE_FLOWANN_SEARCH && CUVS_ENABLE_FLOWANN_BUILD
