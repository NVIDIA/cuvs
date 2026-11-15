/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Tags header should be included before this header (at file scope, not inside functions)
// to avoid namespace definition errors when this header is included inside function bodies

#include "compute_distance.hpp"  // For dataset_descriptor_host
#include "jit_lto_kernels/cagra_jit_launcher_factory.hpp"
#include "jit_lto_kernels/kernel_def.hpp"
#include "jit_lto_kernels/search_multi_kernel_planner.hpp"
#include "search_plan.cuh"          // For search_params
#include "shared_launcher_jit.hpp"  // sample-filter payload helpers and JIT tags
#include <cuvs/distance/distance.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/util/kernel_launch.hpp>
#include <rtcx/algorithm_launcher.hpp>

#include <cstddef>
#include <cuda_runtime.h>
#include <type_traits>
// - The launcher doesn't need the kernel function definitions
// - The kernel is dispatched via the JIT LTO launcher system
// - Including it would pull in impl files that cause namespace issues

namespace cuvs::neighbors::cagra::detail::multi_kernel_search {

// JIT version of random_pickup
template <typename DataT, typename IndexT, typename DistanceT>
void random_pickup_jit(const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
                       const DataT* queries_ptr,  // [num_queries, dataset_dim]
                       std::size_t num_queries,
                       std::size_t num_pickup,
                       unsigned num_distilation,
                       uint64_t rand_xor_mask,
                       const IndexT* seed_ptr,  // [num_queries, num_seeds]
                       uint32_t num_seeds,
                       IndexT* result_indices_ptr,       // [num_queries, ldr]
                       DistanceT* result_distances_ptr,  // [num_queries, ldr]
                       std::size_t ldr,                  // (*) ldr >= num_pickup
                       IndexT* visited_hashmap_ptr,      // [num_queries, 1 << bitlen]
                       std::uint32_t hash_bitlen,
                       cudaStream_t cuda_stream,
                       IndexT graph_size)
{
  std::shared_ptr<rtcx::algorithm_launcher> launcher =
    make_cagra_multi_kernel_jit_launcher<DataT, IndexT, DistanceT, IndexT>(dataset_desc,
                                                                           "random_pickup");

  const auto block_size                = 256u;
  const auto num_teams_per_threadblock = block_size / dataset_desc.team_size;
  const dim3 grid_size((num_pickup + num_teams_per_threadblock - 1) / num_teams_per_threadblock,
                       num_queries);

  // Get the device descriptor pointer
  const auto* dev_desc = dataset_desc.dev_ptr(cuda_stream);

  // `ldr` is wider than the kernel parameter; the cast records the intentional narrowing.
  const uint32_t ldr_u32 = static_cast<uint32_t>(ldr);

  raft::launch_kernel(
    {cuda_stream, dataset_desc.smem_ws_size_in_bytes},
    grid_size,
    dim3(block_size, 1, 1),
    raft::kernel_ref<random_pickup_kernel_func_t<DataT, IndexT, DistanceT>>{launcher->get_kernel()},
    dev_desc,
    queries_ptr,
    num_pickup,
    num_distilation,
    rand_xor_mask,
    seed_ptr,
    num_seeds,
    result_indices_ptr,
    result_distances_ptr,
    ldr_u32,
    visited_hashmap_ptr,
    hash_bitlen,
    graph_size);
}

// JIT version of compute_distance_to_child_nodes
template <typename DataT,
          typename IndexT,
          typename DistanceT,
          class SourceIndexT,
          class SAMPLE_FILTER_T>
void compute_distance_to_child_nodes_jit(
  const IndexT* parent_node_list,        // [num_queries, search_width]
  IndexT* const parent_candidates_ptr,   // [num_queries, search_width]
  DistanceT* const parent_distance_ptr,  // [num_queries, search_width]
  std::size_t lds,
  uint32_t search_width,
  const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
  const IndexT* neighbor_graph_ptr,  // [dataset_size, graph_degree]
  std::uint32_t graph_degree,
  const SourceIndexT* source_indices_ptr,
  const DataT* query_ptr,  // [num_queries, data_dim]
  std::uint32_t num_queries,
  IndexT* visited_hashmap_ptr,  // [num_queries, 1 << hash_bitlen]
  std::uint32_t hash_bitlen,
  IndexT* result_indices_ptr,       // [num_queries, ldd]
  DistanceT* result_distances_ptr,  // [num_queries, ldd]
  std::uint32_t ldd,                // (*) ldd >= search_width * graph_degree
  SAMPLE_FILTER_T sample_filter,
  cudaStream_t cuda_stream,
  std::shared_ptr<rtcx::algorithm_launcher> const& launcher)
{
  const auto filter_payload = extract_cagra_sample_filter<SourceIndexT>(sample_filter, cuda_stream);

  const auto block_size      = 128;
  const auto teams_per_block = block_size / dataset_desc.team_size;
  const dim3 grid_size((search_width * graph_degree + teams_per_block - 1) / teams_per_block,
                       num_queries);

  // Get the device descriptor pointer
  const auto* dev_desc = dataset_desc.dev_ptr(cuda_stream);

  raft::launch_kernel(
    {cuda_stream, dataset_desc.smem_ws_size_in_bytes},
    grid_size,
    dim3(block_size, 1, 1),
    raft::kernel_ref<
      compute_distance_to_child_nodes_kernel_func_t<DataT, IndexT, DistanceT, SourceIndexT>>{
      launcher->get_kernel()},
    parent_node_list,
    parent_candidates_ptr,
    parent_distance_ptr,
    lds,
    search_width,
    dev_desc,
    neighbor_graph_ptr,
    graph_degree,
    source_indices_ptr,
    query_ptr,
    visited_hashmap_ptr,
    hash_bitlen,
    result_indices_ptr,
    result_distances_ptr,
    ldd,
    filter_payload);
}

// JIT version of apply_filter
template <class INDEX_T, class DISTANCE_T, class SourceIndexT, class SAMPLE_FILTER_T>
void apply_filter_jit(const SourceIndexT* source_indices_ptr,
                      INDEX_T* const result_indices_ptr,
                      DISTANCE_T* const result_distances_ptr,
                      const std::size_t lds,
                      const std::uint32_t result_buffer_size,
                      const std::uint32_t num_queries,
                      const std::uint32_t query_id_offset,
                      SAMPLE_FILTER_T sample_filter,
                      cudaStream_t cuda_stream,
                      std::shared_ptr<rtcx::algorithm_launcher> const& launcher)
{
  const auto filter_payload = extract_cagra_sample_filter<SourceIndexT>(sample_filter, cuda_stream);
  const auto effective_query_id_offset = query_id_offset + filter_payload.query_id_offset;

  const std::uint32_t block_size = 256;
  const std::uint32_t grid_size  = raft::ceildiv(num_queries * result_buffer_size, block_size);

  raft::launch_kernel(
    cuda_stream,
    dim3(grid_size, 1, 1),
    dim3(block_size, 1, 1),
    raft::kernel_ref<apply_filter_kernel_func_t<INDEX_T, DISTANCE_T, SourceIndexT>>{
      launcher->get_kernel()},
    source_indices_ptr,
    result_indices_ptr,
    result_distances_ptr,
    lds,
    result_buffer_size,
    num_queries,
    effective_query_id_offset,
    filter_payload);
}

}  // namespace cuvs::neighbors::cagra::detail::multi_kernel_search
