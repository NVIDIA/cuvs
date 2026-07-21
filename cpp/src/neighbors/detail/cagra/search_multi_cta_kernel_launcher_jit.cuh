/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../smem_utils.cuh"

// Include tags header before any other includes that might open namespaces
#include <cuvs/detail/jit_lto/cagra/cagra_fragments.hpp>

#include "compute_distance.hpp"  // For dataset_descriptor_host
#include "jit_lto_kernels/cagra_jit_launcher_factory.hpp"
#include "jit_lto_kernels/kernel_def.hpp"
#include "multi_partition_desc.hpp"
#include "sample_filter_utils.cuh"  // For CagraSampleFilterWithQueryIdOffset
#include "search_plan.cuh"          // For search_params
#include "set_value_batch.cuh"      // For set_value_batch
#include "shared_launcher_jit.hpp"  // For shared JIT helper functions
#include <cuvs/detail/jit_lto/AlgorithmLauncher.hpp>
#include <cuvs/distance/distance.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger.hpp>

#include <cstddef>
#include <cuda_runtime.h>
#include <type_traits>

namespace cuvs::neighbors::cagra::detail::multi_cta_search {

template <typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT,
          typename SampleFilterT>
void select_and_run(const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
                    raft::device_matrix_view<const IndexT, int64_t, raft::row_major> graph,
                    const SourceIndexT* source_indices_ptr,
                    IndexT* topk_indices_ptr,       // [num_queries, num_cta_per_query, itopk_size]
                    DistanceT* topk_distances_ptr,  // [num_queries, num_cta_per_query, itopk_size]
                    const DataT* queries_ptr,       // [num_queries, dataset_dim]
                    uint32_t num_queries,
                    const IndexT* dev_seed_ptr,         // [num_queries, num_seeds]
                    uint32_t* num_executed_iterations,  // [num_queries,]
                    const search_params& ps,
                    uint32_t topk,
                    // multi_cta_search (params struct)
                    uint32_t block_size,  //
                    uint32_t result_buffer_size,
                    uint32_t smem_size,
                    uint32_t visited_hash_bitlen,
                    int64_t traversed_hash_bitlen,
                    IndexT* traversed_hashmap_ptr,
                    uint32_t num_cta_per_query,
                    uint32_t num_seeds,
                    SampleFilterT sample_filter,
                    cudaStream_t stream)
{
  const auto filter_payload      = extract_cagra_sample_filter<SourceIndexT>(sample_filter, stream);
  const uint32_t query_id_offset = filter_payload.query_id_offset;

  std::shared_ptr<AlgorithmLauncher> launcher =
    make_cagra_multi_cta_jit_launcher<DataT,
                                      IndexT,
                                      DistanceT,
                                      SourceIndexT,
                                      sample_filter_jit_tag_t<SampleFilterT>>(
      dataset_desc, make_cagra_sample_filter_udf_fragment<SourceIndexT>(sample_filter));

  if (!launcher) { RAFT_FAIL("Failed to get JIT launcher"); }

  uint32_t max_elements{};
  if (result_buffer_size <= 64) {
    max_elements = 64;
  } else if (result_buffer_size <= 128) {
    max_elements = 128;
  } else if (result_buffer_size <= 256) {
    max_elements = 256;
  } else {
    THROW("Result buffer size %u larger than max buffer size %u", result_buffer_size, 256);
  }

  // Initialize hash table
  const uint32_t traversed_hash_size = hashmap::get_size(traversed_hash_bitlen);
  set_value_batch(traversed_hashmap_ptr,
                  traversed_hash_size,
                  ~static_cast<IndexT>(0),
                  traversed_hash_size,
                  num_queries,
                  stream);

  dim3 block_dims(block_size, 1, 1);
  dim3 grid_dims(num_cta_per_query, num_queries, 1);

  // Get the device descriptor pointer
  const dataset_descriptor_base_t<DataT, IndexT, DistanceT>* dev_desc_base =
    dataset_desc.dev_ptr(stream);
  const auto* dev_desc = dev_desc_base;

  // Note: dataset_desc is passed by const reference, so it stays alive for the duration of this
  // function The descriptor's state is managed by a shared_ptr internally, so no need to explicitly
  // keep it alive

  // Cast size_t/int64_t parameters to match kernel signature exactly
  // The dispatch mechanism uses void* pointers, so parameter sizes must match exactly
  // graph.extent(1) returns int64_t but kernel expects uint32_t
  // traversed_hash_bitlen is int64_t but kernel expects uint32_t
  // ps.itopk_size, ps.min_iterations, ps.max_iterations are size_t (8 bytes) but kernel expects
  // uint32_t (4 bytes) ps.num_random_samplings is uint32_t but kernel expects unsigned - cast for
  // consistency
  const uint32_t graph_degree_u32          = static_cast<uint32_t>(graph.extent(1));
  const uint32_t traversed_hash_bitlen_u32 = static_cast<uint32_t>(traversed_hash_bitlen);
  const uint32_t itopk_size_u32            = static_cast<uint32_t>(ps.itopk_size);
  const uint32_t min_iterations_u32        = static_cast<uint32_t>(ps.min_iterations);
  const uint32_t max_iterations_u32        = static_cast<uint32_t>(ps.max_iterations);
  const unsigned num_random_samplings_u    = static_cast<unsigned>(ps.num_random_samplings);

  auto kernel_launcher = [&]() -> void {
    launcher->dispatch<
      multi_cta_search::search_multi_cta_kernel_func_t<DataT, IndexT, DistanceT, SourceIndexT>>(
      stream,
      grid_dims,
      block_dims,
      smem_size,
      topk_indices_ptr,
      topk_distances_ptr,
      dev_desc,
      queries_ptr,
      graph.data_handle(),
      max_elements,
      graph_degree_u32,
      source_indices_ptr,
      num_random_samplings_u,
      ps.rand_xor_mask,
      dev_seed_ptr,
      num_seeds,
      visited_hash_bitlen,
      traversed_hashmap_ptr,
      traversed_hash_bitlen_u32,
      itopk_size_u32,
      min_iterations_u32,
      max_iterations_u32,
      num_executed_iterations,
      static_cast<IndexT>(graph.extent(0)),
      query_id_offset,
      filter_payload);
  };
  cuvs::neighbors::detail::safely_launch_kernel_with_smem_size<
    multi_cta_search::search_multi_cta_kernel_func_t<DataT, IndexT, DistanceT, SourceIndexT>>(
    smem_size, kernel_launcher, launcher->get_kernel());

  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

// Multi-partition launcher. Drives `search_multi_cta_mp` with a 3D grid
// (num_cta_per_query, num_queries, num_partitions). `ref_dataset_desc` is used only for JIT tag
// dispatch (metric / vpq / team_size / block_dim) and must be representative of every
// partition's descriptor. Per-partition device descriptors are read from `partition_descs` by
// the kernel itself.
template <typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT,
          typename SampleFilterT>
void select_and_run_mp(const dataset_descriptor_host<DataT, IndexT, DistanceT>& ref_dataset_desc,
                       const multi_partition_desc_t<DataT, IndexT, DistanceT>* partition_descs,
                       uint32_t num_partitions,
                       uint32_t max_graph_degree,
                       IndexT* intermediate_indices_ptr,
                       DistanceT* intermediate_distances_ptr,
                       const DataT* queries_ptr,
                       uint32_t num_queries,
                       const search_params& ps,
                       uint32_t block_size,
                       uint32_t result_buffer_size,
                       uint32_t smem_size,
                       uint32_t visited_hash_bitlen,
                       int64_t traversed_hash_bitlen,
                       IndexT* traversed_hashmap_ptr,
                       uint32_t num_cta_per_query,
                       SampleFilterT sample_filter,
                       cudaStream_t stream)
{
  // The per-partition bitset views live in the partition descriptors (filled in cagra_search.cuh).
  const uint32_t query_id_offset = cagra_filter_query_id_offset(sample_filter);

  std::shared_ptr<AlgorithmLauncher> launcher =
    make_cagra_multi_cta_mp_jit_launcher<DataT,
                                         IndexT,
                                         DistanceT,
                                         SourceIndexT,
                                         sample_filter_jit_tag_t<SampleFilterT>>(ref_dataset_desc);

  if (!launcher) { RAFT_FAIL("Failed to get JIT launcher"); }

  uint32_t max_elements{};
  if (result_buffer_size <= 64) {
    max_elements = 64;
  } else if (result_buffer_size <= 128) {
    max_elements = 128;
  } else if (result_buffer_size <= 256) {
    max_elements = 256;
  } else {
    THROW("Result buffer size %u larger than max buffer size %u", result_buffer_size, 256);
  }

  const uint32_t traversed_hash_size = hashmap::get_size(traversed_hash_bitlen);
  set_value_batch(traversed_hashmap_ptr,
                  traversed_hash_size,
                  ~static_cast<IndexT>(0),
                  traversed_hash_size,
                  static_cast<std::size_t>(num_queries) * num_partitions,
                  stream);

  dim3 block_dims(block_size, 1, 1);
  dim3 grid_dims(num_cta_per_query, num_queries, num_partitions);

  const uint32_t max_graph_degree_u32      = static_cast<uint32_t>(max_graph_degree);
  const uint32_t traversed_hash_bitlen_u32 = static_cast<uint32_t>(traversed_hash_bitlen);
  const uint32_t itopk_size_u32            = static_cast<uint32_t>(ps.itopk_size);
  const uint32_t min_iterations_u32        = static_cast<uint32_t>(ps.min_iterations);
  const uint32_t max_iterations_u32        = static_cast<uint32_t>(ps.max_iterations);
  const unsigned num_random_samplings_u    = static_cast<unsigned>(ps.num_random_samplings);

  auto kernel_launcher = [&]() -> void {
    launcher->dispatch<
      multi_cta_search::search_multi_cta_mp_kernel_func_t<DataT, IndexT, DistanceT, SourceIndexT>>(
      stream,
      grid_dims,
      block_dims,
      smem_size,
      partition_descs,
      intermediate_indices_ptr,
      intermediate_distances_ptr,
      queries_ptr,
      max_elements,
      max_graph_degree_u32,
      num_random_samplings_u,
      ps.rand_xor_mask,
      visited_hash_bitlen,
      traversed_hashmap_ptr,
      traversed_hash_bitlen_u32,
      itopk_size_u32,
      min_iterations_u32,
      max_iterations_u32,
      query_id_offset);
  };
  cuvs::neighbors::detail::safely_launch_kernel_with_smem_size<
    multi_cta_search::search_multi_cta_mp_kernel_func_t<DataT, IndexT, DistanceT, SourceIndexT>>(
    smem_size, kernel_launcher, launcher->get_kernel());

  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace cuvs::neighbors::cagra::detail::multi_cta_search
