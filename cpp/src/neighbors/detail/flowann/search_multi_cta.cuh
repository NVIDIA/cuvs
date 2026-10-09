/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "search_single_cta.cuh"

#include <neighbors/detail/cagra/set_value_batch.cuh>
#include <neighbors/detail/cagra/topk_for_cagra/topk.h>

#include <raft/linalg/map.cuh>

#include <rmm/device_uvector.hpp>

namespace cuvs::neighbors::cagra::experimental::flowann::detail {

template <typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SampleFilterT,
          typename OutputIndexT>
struct multi_cta_plan
  : cuvs::neighbors::cagra::detail::
      search_plan_impl<DataT, IndexT, DistanceT, SampleFilterT, IndexT, OutputIndexT> {
  using base_type = cuvs::neighbors::cagra::detail::
    search_plan_impl<DataT, IndexT, DistanceT, SampleFilterT, IndexT, OutputIndexT>;

  static constexpr bool needs_index_copy = sizeof(IndexT) != sizeof(OutputIndexT);

  std::uint32_t graph_degree;
  std::uint32_t num_cta_per_query;
  rmm::device_uvector<IndexT> intermediate_indices;
  rmm::device_uvector<DistanceT> intermediate_distances;
  rmm::device_uvector<std::uint32_t> topk_workspace;
  std::size_t topk_workspace_size{};

  static auto planning_params(flowann::search_params const& input)
    -> cuvs::neighbors::cagra::search_params
  {
    cuvs::neighbors::cagra::search_params result = input;
    result.algo                                  = cuvs::neighbors::cagra::search_algo::MULTI_CTA;
    result.persistent                            = false;
    return result;
  }

  multi_cta_plan(
    raft::resources const& res,
    flowann::search_params const& params,
    const cuvs::neighbors::cagra::detail::dataset_descriptor_host<DataT, IndexT, DistanceT>&
      dataset_desc,
    int64_t dim,
    int64_t dataset_size,
    std::uint32_t input_graph_degree,
    std::uint32_t topk)
    : base_type(res,
                planning_params(params),
                dataset_desc,
                dim,
                dataset_size,
                (1 + additional_buffer_width) * input_graph_degree,
                topk),
      graph_degree(input_graph_degree),
      num_cta_per_query(0),
      intermediate_indices(0, raft::resource::get_cuda_stream(res)),
      intermediate_distances(0, raft::resource::get_cuda_stream(res)),
      topk_workspace(0, raft::resource::get_cuda_stream(res))
  {
    auto const global_itopk             = this->itopk_size;
    constexpr std::uint32_t local_itopk = 32;
    this->itopk_size                    = local_itopk;
    this->search_width                  = 1;
    num_cta_per_query                   = static_cast<std::uint32_t>(
      std::max(params.search_width, raft::ceildiv(global_itopk, std::size_t{local_itopk})));
    auto const base_result_buffer_size =
      local_itopk + (1 + additional_buffer_width) * input_graph_degree;
    this->result_buffer_size = std::max(base_result_buffer_size, params.num_seeds);
    auto const result_buffer_size_32 =
      raft::round_up_safe<std::uint32_t>(this->result_buffer_size, 32);
    RAFT_EXPECTS(result_buffer_size_32 <= 256,
                 "FlowANN multi-CTA result buffer cannot exceed 256 elements");
    this->smem_size =
      this->dataset_desc.smem_ws_size_in_bytes +
      (sizeof(IndexT) + sizeof(DistanceT)) * result_buffer_size_32 +
      sizeof(IndexT) * cuvs::neighbors::cagra::detail::hashmap::get_size(this->small_hash_bitlen) +
      sizeof(IndexT) + sizeof(int);

    constexpr std::uint32_t min_block_size = 64;
    constexpr std::uint32_t max_block_size = 1024;
    auto block_size                        = static_cast<std::uint32_t>(this->thread_block_size);
    if (block_size == 0) {
      block_size                                  = min_block_size;
      constexpr std::uint32_t smem_per_warp_limit = 4096;
      while (block_size<max_block_size&& this->smem_size> smem_per_warp_limit / 32 * block_size) {
        block_size *= 2;
      }
      auto const properties = raft::resource::get_device_properties(res);
      while (block_size < max_block_size &&
             input_graph_degree * this->team_size >= block_size * 2 &&
             num_cta_per_query * this->max_queries <=
               (1024 / (block_size * 2)) * properties.multiProcessorCount) {
        block_size *= 2;
      }
    }
    RAFT_EXPECTS(block_size >= min_block_size && block_size <= max_block_size,
                 "FlowANN multi-CTA thread_block_size must be in [64, 1024]");
    this->thread_block_size = block_size;

    auto const intermediate_count = num_cta_per_query * local_itopk;
    intermediate_indices.resize(
      intermediate_count * this->max_queries + (needs_index_copy ? topk * this->max_queries : 0),
      stream(res));
    intermediate_distances.resize(intermediate_count * this->max_queries, stream(res));
    this->hashmap.resize(this->hashmap_size, stream(res));
    topk_workspace_size = cuvs::neighbors::cagra::detail::_cuann_find_topk_bufferSize(
      topk, this->max_queries, intermediate_count, CUDA_R_32F);
    topk_workspace.resize(topk_workspace_size, stream(res));
  }

 private:
  static auto stream(raft::resources const& res) -> rmm::cuda_stream_view
  {
    return raft::resource::get_cuda_stream(res);
  }
};

template <typename DataT,
          typename OutputIndexT,
          typename SampleFilterT,
          ann_dataset_view DatasetViewT>
void search_multi_cta(raft::resources const& res,
                      flowann::search_params params,
                      flowann::index<DataT, std::uint32_t, DatasetViewT> const& index,
                      search_context& context,
                      raft::device_matrix_view<const DataT, int64_t, raft::row_major> queries,
                      raft::device_matrix_view<OutputIndexT, int64_t, raft::row_major> neighbors,
                      raft::device_matrix_view<float, int64_t, raft::row_major> distances,
                      SampleFilterT sample_filter)
{
  using data_t         = DataT;
  using index_t        = std::uint32_t;
  using distance_t     = float;
  using source_index_t = std::uint32_t;
  using filter_t =
    typename cuvs::neighbors::cagra::detail::CagraSampleFilterT_Selector<SampleFilterT>::type;
  static_assert(
    std::is_same_v<OutputIndexT, std::uint32_t> || std::is_same_v<OutputIndexT, std::int64_t>,
    "FlowANN output indices must be uint32_t or int64_t");
  RAFT_EXPECTS(index.size() > 0 && index.graph_degree() > 0,
               "FlowANN search requires a non-empty tiered graph");
  RAFT_EXPECTS(index.dataset().n_rows() == static_cast<int64_t>(index.size()),
               "FlowANN search requires a dataset row for every graph node");
  RAFT_EXPECTS(index.subgraph_offsets().has_value() && index.subgraph_ids().has_value(),
               "FlowANN search requires subgraph metadata");
  RAFT_EXPECTS(queries.extent(1) == index.dim(),
               "FlowANN query dimension must match the index dimension");
  RAFT_EXPECTS(neighbors.extents() == distances.extents() &&
                 neighbors.extent(0) == queries.extent(0) && neighbors.extent(1) > 0,
               "FlowANN outputs must have shape [n_queries, k]");
  RAFT_EXPECTS(queries.extent(0) <= std::numeric_limits<std::uint32_t>::max(),
               "FlowANN multi-CTA query count exceeds uint32");
  RAFT_EXPECTS(params.search_width > 0 && params.sync_window_scale >= 0.0f,
               "FlowANN multi-CTA parameters are invalid");
  RAFT_EXPECTS(search_context_access::row_count(context) == index.size() &&
                 search_context_access::row_length(context) == index.cross_degree() + 1,
               "FlowANN search context does not match the index cross graph");

  auto const index_seeds        = index.seeds();
  auto const seed_cross_graph   = index.seed_cross_graph();
  auto const seed_lookup_keys   = index.seed_lookup_keys();
  auto const seed_lookup_values = index.seed_lookup_values();
  RAFT_EXPECTS(params.num_seeds == 0 || index_seeds.has_value(),
               "FlowANN num_seeds requires preloaded index seeds");
  RAFT_EXPECTS(!index_seeds.has_value() || params.num_seeds <= index_seeds->extent(0),
               "FlowANN num_seeds exceeds the preloaded seed count");
  RAFT_EXPECTS(params.num_seeds == 0 || seed_cross_graph.has_value(),
               "FlowANN num_seeds requires cached seed cross-graph rows");
  RAFT_EXPECTS(
    params.num_seeds == 0 || (seed_lookup_keys.has_value() && seed_lookup_values.has_value()),
    "FlowANN num_seeds requires a prebuilt seed lookup");
  auto const* seeds          = params.num_seeds == 0 ? nullptr : index_seeds->data_handle();
  auto const* seed_cross_ptr = params.num_seeds == 0 ? nullptr : seed_cross_graph->data_handle();
  auto const* seed_lookup_keys_ptr =
    params.num_seeds == 0 ? nullptr : seed_lookup_keys->data_handle();
  auto const* seed_lookup_values_ptr =
    params.num_seeds == 0 ? nullptr : seed_lookup_values->data_handle();
  auto const seed_lookup_capacity =
    params.num_seeds == 0 ? 0u : static_cast<std::uint32_t>(seed_lookup_keys->extent(0));
  auto const properties = raft::resource::get_device_properties(res);
  if (params.max_queries == 0) {
    params.max_queries = std::min<std::size_t>(queries.extent(0), properties.maxGridSize[1]);
  }

  auto descriptor = make_dataset_descriptor<data_t, index_t, distance_t>(res, params, index);
  auto const topk = static_cast<std::uint32_t>(neighbors.extent(1));
  multi_cta_plan<data_t, index_t, distance_t, filter_t, OutputIndexT> plan(
    res, params, descriptor, index.dim(), index.size(), index.graph_degree(), topk);
  RAFT_EXPECTS(plan.num_cta_per_query * plan.itopk_size >= topk,
               "FlowANN multi-CTA candidate count is smaller than k");
  auto launcher_filter = cuvs::neighbors::cagra::detail::set_offset(sample_filter, 0);
  auto launcher        = make_multi_cta_launcher<
           data_t,
           index_t,
           distance_t,
           source_index_t,
           cuvs::neighbors::cagra::detail::sample_filter_jit_tag_t<decltype(launcher_filter)>>(
    descriptor,
    cuvs::neighbors::cagra::detail::make_cagra_sample_filter_udf_fragment<source_index_t>(
      launcher_filter));
  RAFT_EXPECTS(launcher != nullptr, "Failed to create the FlowANN multi-CTA JIT launcher");

  auto const query_dim = static_cast<std::uint32_t>(queries.extent(1));
  std::unique_ptr<cuvs::neighbors::device_padded_dataset<data_t, int64_t>> padded_queries;
  data_t const* query_data{};
  std::uint32_t query_stride{};
  if (cuvs::neighbors::matrix_row_width_matches_cagra_required(queries)) {
    auto view    = cuvs::neighbors::make_device_padded_dataset_view(res, queries);
    query_data   = view.view().data_handle();
    query_stride = view.stride();
  } else {
    padded_queries = cuvs::neighbors::make_device_padded_dataset(res, queries);
    auto view      = padded_queries->as_dataset_view();
    query_data     = view.view().data_handle();
    query_stride   = view.stride();
  }
  auto const max_batch_queries = query_stride == query_dim ? plan.max_queries : std::size_t{1};
  auto const stream            = raft::resource::get_cuda_stream(res).get();
  auto const* descriptor_ptr   = descriptor.dev_ptr(stream);
  auto const inner_graph       = index.inner_graph();
  auto const subgraph_offsets  = index.subgraph_offsets().value();
  auto const subgraph_ids      = index.subgraph_ids().value();
  auto const source_indices    = index.source_indices();
  auto const* source_ptr = source_indices.has_value() ? source_indices->data_handle() : nullptr;
  using kernel_type = search_multi_cta_kernel_func_t<data_t, index_t, distance_t, source_index_t>;
  auto const max_elements       = raft::round_up_safe<std::uint32_t>(plan.result_buffer_size, 32);
  auto const intermediate_count = plan.num_cta_per_query * plan.itopk_size;
  auto const num_queries        = static_cast<std::uint32_t>(queries.extent(0));

  context_run_guard running(context);
  for (std::uint32_t offset = 0; offset < num_queries;
       offset += static_cast<std::uint32_t>(max_batch_queries)) {
    auto const count =
      static_cast<std::uint32_t>(std::min<std::size_t>(max_batch_queries, num_queries - offset));
    auto const* query_ptr = query_data + static_cast<std::size_t>(offset) * query_stride;
    auto batch_filter     = cuvs::neighbors::cagra::detail::set_offset(sample_filter, offset);
    auto filter_payload =
      cuvs::neighbors::cagra::detail::extract_cagra_sample_filter<source_index_t>(batch_filter,
                                                                                  stream);
    auto const traversed_hash_size =
      cuvs::neighbors::cagra::detail::hashmap::get_size(plan.hash_bitlen);
    cuvs::neighbors::cagra::detail::set_value_batch(
      plan.hashmap.data(), traversed_hash_size, ~index_t{0}, traversed_hash_size, count, stream);
    dim3 grid(plan.num_cta_per_query, count, 1);
    dim3 block(static_cast<std::uint32_t>(plan.thread_block_size), 1, 1);
    auto kernel_launcher = [&]() {
      launcher->template dispatch<kernel_type>(
        stream,
        grid,
        block,
        static_cast<std::size_t>(plan.smem_size),
        plan.intermediate_indices.data(),
        plan.intermediate_distances.data(),
        descriptor_ptr,
        query_ptr,
        inner_graph.data_handle(),
        static_cast<std::uint32_t>(inner_graph.extent(1)),
        static_cast<std::uint32_t>(index.node_per_cacheline()),
        static_cast<std::uint32_t>(index.n_bits()),
        subgraph_offsets.data_handle(),
        subgraph_ids.data_handle(),
        search_context_access::device_queues(context),
        search_context_access::num_queues(context),
        max_elements,
        index.graph_degree(),
        index.cross_degree(),
        source_ptr,
        static_cast<unsigned>(plan.num_random_samplings),
        plan.rand_xor_mask,
        seeds,
        params.num_seeds,
        seed_cross_ptr,
        seed_lookup_keys_ptr,
        seed_lookup_values_ptr,
        seed_lookup_capacity,
        static_cast<std::uint32_t>(plan.small_hash_bitlen),
        plan.hashmap.data(),
        static_cast<std::uint32_t>(plan.hash_bitlen),
        static_cast<std::uint32_t>(plan.itopk_size),
        static_cast<std::uint32_t>(plan.min_iterations),
        static_cast<std::uint32_t>(plan.max_iterations),
        static_cast<std::uint32_t*>(nullptr),
        index.size(),
        filter_payload.query_id_offset,
        params.sync_window_scale,
        params.sync_drop_threshold,
        filter_payload);
    };
    cuvs::neighbors::detail::safely_launch_kernel_with_smem_size<kernel_type>(
      plan.smem_size, kernel_launcher, launcher->get_kernel());
    RAFT_CUDA_TRY(cudaPeekAtLastError());

#ifndef CUVS_FLOWANN_USE_GDRCOPY
    // Lazy loading the top-k kernel can wait for active kernels while holding
    // a driver lock needed by the CUDA-copy pollers. Finish the host-serviced
    // search before launching kernels from another module.
    raft::resource::sync_stream(res);
#endif

    auto* output_indices =
      plan.needs_index_copy
        ? plan.intermediate_indices.data() + intermediate_count * plan.max_queries
        : reinterpret_cast<index_t*>(neighbors.data_handle() +
                                     static_cast<std::size_t>(offset) * topk);
    cuvs::neighbors::cagra::detail::_cuann_find_topk(
      topk,
      count,
      intermediate_count,
      plan.intermediate_distances.data(),
      intermediate_count,
      plan.intermediate_indices.data(),
      intermediate_count,
      distances.data_handle() + static_cast<std::size_t>(offset) * topk,
      topk,
      output_indices,
      topk,
      plan.topk_workspace.data(),
      true,
      nullptr,
      stream);
    if (source_ptr != nullptr) {
      auto output = raft::make_device_matrix_view<OutputIndexT, int64_t>(
        neighbors.data_handle() + static_cast<std::size_t>(offset) * topk, count, topk);
      raft::linalg::map(
        res,
        output,
        [source_ptr] __device__(index_t value) -> OutputIndexT {
          return value == ~index_t{0} ? ~OutputIndexT{0}
                                      : static_cast<OutputIndexT>(source_ptr[value]);
        },
        raft::make_device_matrix_view<const index_t, int64_t>(output_indices, count, topk));
    } else if constexpr (sizeof(index_t) != sizeof(OutputIndexT)) {
      raft::linalg::map(
        res,
        raft::make_device_matrix_view<OutputIndexT, int64_t>(
          neighbors.data_handle() + static_cast<std::size_t>(offset) * topk, count, topk),
        [] __device__(index_t value) -> OutputIndexT {
          return value == ~index_t{0} ? ~OutputIndexT{0} : static_cast<OutputIndexT>(value);
        },
        raft::make_device_matrix_view<const index_t, int64_t>(output_indices, count, topk));
    }
  }
  raft::resource::sync_stream(res);
  running.pause();
  postprocess_distances(res, index.metric(), queries, distances);
}

template <typename DataT,
          typename OutputIndexT,
          typename SampleFilterT,
          ann_dataset_view DatasetViewT>
void search_dispatch(raft::resources const& res,
                     flowann::search_params params,
                     flowann::index<DataT, std::uint32_t, DatasetViewT> const& index,
                     search_context& context,
                     raft::device_matrix_view<const DataT, int64_t, raft::row_major> queries,
                     raft::device_matrix_view<OutputIndexT, int64_t, raft::row_major> neighbors,
                     raft::device_matrix_view<float, int64_t, raft::row_major> distances,
                     SampleFilterT sample_filter)
{
  int current_device{};
  RAFT_CUDA_TRY(cudaGetDevice(&current_device));
  RAFT_EXPECTS(current_device == search_context_access::device_id(context),
               "FlowANN search context must be used on the CUDA device where it was created");
  RAFT_EXPECTS(!params.persistent,
               "FlowANN persistent search is not available for the host-serviced graph provider");
  auto algo = params.algo;
  if (algo == cuvs::neighbors::cagra::search_algo::AUTO) {
    auto const properties = raft::resource::get_device_properties(res);
    auto const effective_max_queries =
      params.max_queries == 0 ? static_cast<std::size_t>(queries.extent(0)) : params.max_queries;
    algo = params.itopk_size <= 512 &&
               effective_max_queries >= static_cast<std::size_t>(properties.multiProcessorCount) * 2
             ? cuvs::neighbors::cagra::search_algo::SINGLE_CTA
             : cuvs::neighbors::cagra::search_algo::MULTI_CTA;
  }
  if (algo == cuvs::neighbors::cagra::search_algo::SINGLE_CTA) {
    params.algo = algo;
    search_single_cta(res, params, index, context, queries, neighbors, distances, sample_filter);
    return;
  }
  RAFT_EXPECTS(algo == cuvs::neighbors::cagra::search_algo::MULTI_CTA,
               "FlowANN search supports SINGLE_CTA, MULTI_CTA, or AUTO");
  params.algo = algo;
  search_multi_cta(res, params, index, context, queries, neighbors, distances, sample_filter);
}

}  // namespace cuvs::neighbors::cagra::experimental::flowann::detail
