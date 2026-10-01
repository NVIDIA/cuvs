/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "launcher.hpp"
#include "queue.cuh"

#include <neighbors/detail/cagra/factory.cuh>
#include <neighbors/detail/cagra/sample_filter_utils.cuh>
#include <neighbors/detail/cagra/search_plan.cuh>
#include <neighbors/detail/cagra/search_single_cta_kernel_launcher_common.cuh>
#include <neighbors/detail/cagra/topk_by_radix.cuh>
#include <neighbors/detail/smem_utils.cuh>
#include <neighbors/ivf_common.cuh>

#include <cuvs/neighbors/flowann.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_properties.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/reduce.cuh>
#include <raft/util/pow2_utils.cuh>

#include <cub/warp/warp_scan.cuh>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <type_traits>

namespace cuvs::neighbors::cagra::experimental::flowann::detail {

inline auto automatic_max_iterations(std::uint32_t itopk_size,
                                     std::uint32_t search_width,
                                     std::uint64_t dataset_size,
                                     std::uint32_t graph_degree,
                                     std::uint32_t min_iterations) -> std::uint32_t
{
  auto iterations         = itopk_size / search_width;
  std::uint64_t reachable = 1;
  while (reachable < dataset_size) {
    auto const expansion = std::max<std::uint64_t>(2, graph_degree / 2);
    if (reachable > std::numeric_limits<std::uint64_t>::max() / expansion) { break; }
    reachable *= expansion;
    ++iterations;
  }
  return std::max(iterations, min_iterations);
}

template <typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SampleFilterT,
          typename OutputIndexT>
struct single_cta_plan
  : cuvs::neighbors::cagra::detail::
      search_plan_impl<DataT, IndexT, DistanceT, SampleFilterT, IndexT, OutputIndexT> {
  using base_type = cuvs::neighbors::cagra::detail::
    search_plan_impl<DataT, IndexT, DistanceT, SampleFilterT, IndexT, OutputIndexT>;

  std::uint32_t num_itopk_candidates{};

  static auto planning_params(flowann::search_params const& input)
    -> cuvs::neighbors::cagra::search_params
  {
    cuvs::neighbors::cagra::search_params result = input;
    result.search_width += additional_buffer_width;
    result.algo       = cuvs::neighbors::cagra::search_algo::SINGLE_CTA;
    result.persistent = false;
    return result;
  }

  single_cta_plan(
    raft::resources const& res,
    flowann::search_params const& params,
    const cuvs::neighbors::cagra::detail::dataset_descriptor_host<DataT, IndexT, DistanceT>&
      dataset_desc,
    int64_t dim,
    int64_t dataset_size,
    int64_t graph_degree,
    std::uint32_t topk)
    : base_type(res, planning_params(params), dataset_desc, dim, dataset_size, graph_degree, topk)
  {
    // The inherited CAGRA plan sizes hashmaps and candidate storage for two additional graph rows,
    // while the actual traversal width remains the user-requested value.
    this->search_width = params.search_width;
    if (params.max_iterations == 0) {
      this->max_iterations = automatic_max_iterations(static_cast<std::uint32_t>(this->itopk_size),
                                                      params.search_width,
                                                      dataset_size,
                                                      graph_degree,
                                                      params.min_iterations);
    } else {
      this->max_iterations = std::max<std::size_t>(params.max_iterations, params.min_iterations);
    }
    set_launch_params(res, params.num_seeds);
  }

  void set_launch_params(raft::resources const& res, std::uint32_t num_seeds)
  {
    auto const expansion_candidates =
      (static_cast<std::uint32_t>(this->search_width) + additional_buffer_width) *
      static_cast<std::uint32_t>(this->graph_degree);
    auto const base_result_buffer_size =
      static_cast<std::uint32_t>(this->itopk_size) + expansion_candidates;
    this->result_buffer_size = std::max(base_result_buffer_size, num_seeds);
    num_itopk_candidates = this->result_buffer_size - static_cast<std::uint32_t>(this->itopk_size);
    auto const result_buffer_size_32 =
      raft::round_up_safe<std::uint32_t>(this->result_buffer_size, 32);
    constexpr std::uint32_t max_itopk = 512;
    RAFT_EXPECTS(
      this->itopk_size <= max_itopk, "FlowANN itopk_size cannot be larger than %u", max_itopk);

    auto const base_smem_size =
      this->dataset_desc.smem_ws_size_in_bytes +
      (sizeof(IndexT) + sizeof(DistanceT)) * result_buffer_size_32 +
      sizeof(IndexT) * cuvs::neighbors::cagra::detail::hashmap::get_size(this->small_hash_bitlen) +
      sizeof(IndexT) * this->search_width + 4 * sizeof(std::uint32_t);
    auto additional_smem_size = std::uint32_t{0};
    if (num_itopk_candidates > 256) {
      auto const radix_itopk = this->itopk_size <= 256 ? 256u : 512u;
      additional_smem_size =
        cuvs::neighbors::cagra::detail::single_cta_search::topk_by_radix_sort<IndexT>::smem_size(
          radix_itopk) *
        sizeof(std::uint32_t);
    }
    if constexpr (!std::is_same_v<SampleFilterT, cuvs::neighbors::filtering::none_sample_filter>) {
      using scan_op_t = cub::WarpScan<unsigned>;
      additional_smem_size =
        std::max<std::uint32_t>(additional_smem_size, sizeof(typename scan_op_t::TempStorage));
    }
    this->smem_size = base_smem_size + additional_smem_size;

    constexpr std::uint32_t min_block_size       = 64;
    constexpr std::uint32_t min_radix_block_size = 256;
    constexpr std::uint32_t max_block_size       = 1024;
    auto block_size = static_cast<std::uint32_t>(this->thread_block_size);
    if (block_size == 0) {
      block_size = num_itopk_candidates > 256 ? min_radix_block_size : min_block_size;
      while (num_itopk_candidates > 256 && block_size < max_block_size &&
             max_itopk / block_size > 4) {
        block_size *= 2;
      }
      constexpr std::uint32_t smem_per_warp_limit = 4096;
      while (block_size<max_block_size&& this->smem_size> smem_per_warp_limit / 32 * block_size) {
        block_size *= 2;
      }
      auto const properties = raft::resource::get_device_properties(res);
      while (block_size < max_block_size &&
             this->graph_degree * this->search_width * this->team_size >= block_size * 2 &&
             this->max_queries <= (1024 / (block_size * 2)) * properties.multiProcessorCount) {
        block_size *= 2;
      }
    }
    RAFT_EXPECTS(block_size >= min_block_size && block_size <= max_block_size,
                 "FlowANN thread_block_size must be in [64, 1024]");
    this->thread_block_size = block_size;

    this->hashmap_size = 0;
    if (this->small_hash_bitlen == 0) {
      this->hashmap_size =
        this->max_queries * cuvs::neighbors::cagra::detail::hashmap::get_size(this->hash_bitlen);
      this->hashmap.resize(this->hashmap_size, raft::resource::get_cuda_stream(res));
    }
  }
};

template <typename DataT, typename IndexT, typename DistanceT, ann_dataset_view DatasetViewT>
auto make_dataset_descriptor(raft::resources const& res,
                             flowann::search_params& params,
                             flowann::index<DataT, std::uint32_t, DatasetViewT> const& index)
{
  if constexpr (cuvs::neighbors::is_device_vpq_f16_dataset_view_v<DatasetViewT>) {
    if (params.smem_dtype == cuvs::neighbors::cagra::internal_dtype::E5M2 &&
        raft::getComputeCapability().first < 9) {
      RAFT_LOG_WARN(
        "FlowANN VPQ E5M2 smem_dtype requires native FP8 support on SM90+. Falling back to F16.");
      params.smem_dtype = cuvs::neighbors::cagra::internal_dtype::F16;
    }
    return cuvs::neighbors::cagra::detail::
      dataset_descriptor_init_with_cache<DataT, IndexT, DistanceT>(
        res, params, index.dataset().dset(), index.metric(), nullptr);
  } else {
    static_assert(cuvs::neighbors::is_device_padded_dataset_view_v<DatasetViewT>,
                  "FlowANN search supports device-padded or float16 VPQ datasets");
    if (params.smem_dtype != cuvs::neighbors::cagra::internal_dtype::F16) {
      RAFT_LOG_WARN("FlowANN dense search supports only F16 smem_dtype. Falling back to F16.");
      params.smem_dtype = cuvs::neighbors::cagra::internal_dtype::F16;
    }
    auto const dataset_norms = index.dataset_norms();
    RAFT_EXPECTS(
      index.metric() != cuvs::distance::DistanceType::CosineExpanded || dataset_norms.has_value(),
      "FlowANN dataset norms must be available for CosineExpanded metric");
    auto const* dataset_norms_ptr =
      dataset_norms.has_value() ? dataset_norms->data_handle() : nullptr;
    return cuvs::neighbors::cagra::detail::
      dataset_descriptor_init_with_cache<DataT, IndexT, DistanceT>(
        res, params, index.dataset(), index.metric(), dataset_norms_ptr);
  }
}

template <typename DataT, typename DistanceT>
void postprocess_distances(raft::resources const& res,
                           cuvs::distance::DistanceType metric,
                           raft::device_matrix_view<const DataT, int64_t, raft::row_major> queries,
                           raft::device_matrix_view<DistanceT, int64_t, raft::row_major> distances)
{
  static_assert(std::is_same_v<DistanceT, float>,
                "FlowANN currently supports only float distances");
  constexpr float scale = cuvs::spatial::knn::detail::utils::config<DataT>::kDivisor /
                          cuvs::spatial::knn::detail::utils::config<DistanceT>::kDivisor;

  if (metric == cuvs::distance::DistanceType::CosineExpanded) {
    auto query_norms  = raft::make_device_vector<DistanceT, int64_t>(res, queries.extent(0));
    auto scaled_sq_op = raft::compose_op(
      raft::sq_op{}, raft::div_const_op<DistanceT>{DistanceT(scale)}, raft::cast_op<DistanceT>());
    raft::linalg::reduce<raft::Apply::ALONG_ROWS>(res,
                                                  queries,
                                                  query_norms.view(),
                                                  DistanceT{0},
                                                  false,
                                                  scaled_sq_op,
                                                  raft::add_op(),
                                                  raft::sqrt_op{});
    raft::linalg::matrix_vector_op<raft::Apply::ALONG_COLUMNS>(
      res,
      raft::make_const_mdspan(distances),
      raft::make_const_mdspan(query_norms.view()),
      distances,
      raft::compose_op(raft::add_const_op<DistanceT>{DistanceT{1}}, raft::div_checkzero_op{}));
  } else {
    cuvs::neighbors::ivf::detail::postprocess_distances(res,
                                                        distances.data_handle(),
                                                        distances.data_handle(),
                                                        metric,
                                                        distances.extent(0),
                                                        distances.extent(1),
                                                        scale,
                                                        true);
  }
}

class context_run_guard {
 public:
  explicit context_run_guard(search_context& context) : context_(context)
  {
    search_context_access::acquire_search(context_);
  }
  context_run_guard(context_run_guard const&)                    = delete;
  auto operator=(context_run_guard const&) -> context_run_guard& = delete;

  ~context_run_guard()
  {
    if (active_) {
      try {
        active_ = false;
        search_context_access::release_search(context_);
      } catch (...) {
      }
    }
  }

  void pause()
  {
    active_ = false;
    search_context_access::release_search(context_);
  }

 private:
  search_context& context_;
  bool active_{true};
};

template <typename DataT,
          typename OutputIndexT,
          typename SampleFilterT,
          ann_dataset_view DatasetViewT>
void search_single_cta(raft::resources const& res,
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
               "FlowANN search requires subgraph offsets and per-node subgraph IDs");
  RAFT_EXPECTS(queries.extent(1) == index.dim(),
               "FlowANN query dimension must match the index dimension");
  RAFT_EXPECTS(neighbors.extent(0) == queries.extent(0) && distances.extent(0) == queries.extent(0),
               "FlowANN output row count must match the number of queries");
  RAFT_EXPECTS(neighbors.extent(1) == distances.extent(1) && neighbors.extent(1) > 0,
               "FlowANN neighbor and distance outputs must have the same positive k");
  RAFT_EXPECTS(queries.extent(0) <= std::numeric_limits<std::uint32_t>::max(),
               "FlowANN single-CTA query count exceeds uint32");
  RAFT_EXPECTS(params.search_width > 0 && params.search_width <= max_search_width,
               "FlowANN search_width must be in [1, %u]",
               max_search_width);
  RAFT_EXPECTS(params.sync_window_scale >= 0.0f, "FlowANN sync_window_scale must not be negative");
  RAFT_EXPECTS(params.algo == cuvs::neighbors::cagra::search_algo::AUTO ||
                 params.algo == cuvs::neighbors::cagra::search_algo::SINGLE_CTA,
               "This FlowANN path supports SINGLE_CTA or AUTO");
  RAFT_EXPECTS(search_context_access::row_count(context) == index.size() &&
                 search_context_access::row_length(context) == index.cross_degree() + 1,
               "FlowANN search context does not match the index cross graph");

  auto const requested_seeds    = params.num_seeds;
  auto const index_seeds        = index.seeds();
  auto const seed_cross_graph   = index.seed_cross_graph();
  auto const seed_lookup_keys   = index.seed_lookup_keys();
  auto const seed_lookup_values = index.seed_lookup_values();
  RAFT_EXPECTS(requested_seeds == 0 || index_seeds.has_value(),
               "FlowANN num_seeds requires preloaded index seeds");
  RAFT_EXPECTS(!index_seeds.has_value() || requested_seeds <= index_seeds->extent(0),
               "FlowANN num_seeds exceeds the number of preloaded seeds");
  RAFT_EXPECTS(requested_seeds == 0 || seed_cross_graph.has_value(),
               "FlowANN num_seeds requires cached seed cross-graph rows");
  RAFT_EXPECTS(
    requested_seeds == 0 || (seed_lookup_keys.has_value() && seed_lookup_values.has_value()),
    "FlowANN num_seeds requires a prebuilt seed lookup");
  auto const* seeds          = requested_seeds == 0 ? nullptr : index_seeds->data_handle();
  auto const* seed_cross_ptr = requested_seeds == 0 ? nullptr : seed_cross_graph->data_handle();
  auto const* seed_lookup_keys_ptr =
    requested_seeds == 0 ? nullptr : seed_lookup_keys->data_handle();
  auto const* seed_lookup_values_ptr =
    requested_seeds == 0 ? nullptr : seed_lookup_values->data_handle();
  auto const seed_lookup_capacity =
    requested_seeds == 0 ? 0u : static_cast<std::uint32_t>(seed_lookup_keys->extent(0));

  auto device_properties = raft::resource::get_device_properties(res);
  if (params.max_queries == 0) {
    params.max_queries = std::min<std::size_t>(queries.extent(0), device_properties.maxGridSize[1]);
  }
  RAFT_EXPECTS(params.max_queries > 0, "FlowANN max_queries must be positive");

  auto descriptor = make_dataset_descriptor<data_t, index_t, distance_t>(res, params, index);
  auto const topk = static_cast<std::uint32_t>(neighbors.extent(1));
  single_cta_plan<data_t, index_t, distance_t, filter_t, OutputIndexT> plan(
    res, params, descriptor, index.dim(), index.size(), index.graph_degree(), topk);
  plan.check(topk);
  plan.num_seeds = requested_seeds;

  auto launch_config = cuvs::neighbors::cagra::detail::single_cta_search::compute_launch_config(
    plan.num_itopk_candidates,
    static_cast<std::uint32_t>(plan.itopk_size),
    static_cast<std::uint32_t>(plan.thread_block_size));
  auto launcher_filter = cuvs::neighbors::cagra::detail::set_offset(sample_filter, 0);
  auto launcher        = make_single_cta_launcher<
           data_t,
           index_t,
           distance_t,
           source_index_t,
           cuvs::neighbors::cagra::detail::sample_filter_jit_tag_t<decltype(launcher_filter)>>(
    descriptor,
    launch_config.topk_by_bitonic_sort,
    launch_config.bitonic_sort_and_merge_multi_warps,
    cuvs::neighbors::cagra::detail::make_cagra_sample_filter_udf_fragment<source_index_t>(
      launcher_filter));
  RAFT_EXPECTS(launcher != nullptr, "Failed to create the FlowANN single-CTA JIT launcher");

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
  constexpr uintptr_t output_tag = raft::Pow2<sizeof(OutputIndexT)>::Log2;
  static_assert(output_tag <= 3, "FlowANN output index type cannot exceed 8 bytes");
  auto const num_queries = static_cast<std::uint32_t>(queries.extent(0));
  using kernel_type = search_single_cta_kernel_func_t<data_t, index_t, distance_t, source_index_t>;

  context_run_guard running(context);
  for (std::uint32_t offset = 0; offset < num_queries;
       offset += static_cast<std::uint32_t>(max_batch_queries)) {
    auto const count =
      static_cast<std::uint32_t>(std::min<std::size_t>(max_batch_queries, num_queries - offset));
    auto const result_ptr = reinterpret_cast<uintptr_t>(neighbors.data_handle() +
                                                        static_cast<std::size_t>(offset) * topk);
    if constexpr (output_tag <= 1) {
      RAFT_EXPECTS((result_ptr & 0x3) == 0,
                   "FlowANN result indices must be at least 4-byte aligned");
    }
    auto const* query_ptr = query_data + static_cast<std::size_t>(offset) * query_stride;
    auto batch_filter     = cuvs::neighbors::cagra::detail::set_offset(sample_filter, offset);
    auto filter_payload =
      cuvs::neighbors::cagra::detail::extract_cagra_sample_filter<source_index_t>(batch_filter,
                                                                                  stream);
    dim3 grid(1, count, 1);
    dim3 block(static_cast<std::uint32_t>(plan.thread_block_size), 1, 1);

    auto kernel_launcher = [&]() {
      launcher->template dispatch<kernel_type>(
        stream,
        grid,
        block,
        static_cast<std::size_t>(plan.smem_size),
        result_ptr | output_tag,
        distances.data_handle() + static_cast<std::size_t>(offset) * topk,
        topk,
        query_ptr,
        inner_graph.data_handle(),
        static_cast<std::uint32_t>(inner_graph.extent(1)),
        static_cast<std::uint32_t>(index.node_per_cacheline()),
        static_cast<std::uint32_t>(index.n_bits()),
        subgraph_offsets.data_handle(),
        subgraph_ids.data_handle(),
        search_context_access::device_queues(context),
        search_context_access::num_queues(context),
        index.graph_degree(),
        index.cross_degree(),
        source_ptr,
        static_cast<unsigned>(plan.num_random_samplings),
        plan.rand_xor_mask,
        seeds,
        requested_seeds,
        seed_cross_ptr,
        seed_lookup_keys_ptr,
        seed_lookup_values_ptr,
        seed_lookup_capacity,
        plan.hashmap.data(),
        launch_config.max_candidates,
        launch_config.max_itopk,
        static_cast<std::uint32_t>(plan.itopk_size),
        static_cast<std::uint32_t>(plan.search_width),
        static_cast<std::uint32_t>(plan.min_iterations),
        static_cast<std::uint32_t>(plan.max_iterations),
        static_cast<std::uint32_t*>(nullptr),
        static_cast<std::uint32_t>(plan.hash_bitlen),
        static_cast<std::uint32_t>(plan.small_hash_bitlen),
        static_cast<std::uint32_t>(plan.small_hash_reset_interval),
        filter_payload.query_id_offset,
        descriptor_ptr,
        index.size(),
        params.sync_window_scale,
        params.sync_drop_threshold,
        filter_payload);
    };
    cuvs::neighbors::detail::safely_launch_kernel_with_smem_size<kernel_type>(
      plan.smem_size, kernel_launcher, launcher->get_kernel());
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
  raft::resource::sync_stream(res);
  running.pause();

  postprocess_distances(res, index.metric(), queries, distances);
}

}  // namespace cuvs::neighbors::cagra::experimental::flowann::detail
