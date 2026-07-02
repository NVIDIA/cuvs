/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <neighbors/detail/cagra/compute_distance-ext.cuh>
#include <neighbors/detail/cagra/multi_partition_desc.hpp>

#include <cuvs/neighbors/cagra.hpp>

namespace cuvs::neighbors::cagra::detail::single_cta_search {

template <typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT,
          typename SampleFilterT>
void select_and_run(
  const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
  raft::device_matrix_view<const IndexT, int64_t, raft::row_major> graph,
  std::optional<raft::device_vector_view<const SourceIndexT, int64_t>> source_indices,
  uintptr_t topk_indices_ptr,     // [num_queries, topk]
  DistanceT* topk_distances_ptr,  // [num_queries, topk]
  const DataT* queries_ptr,       // [num_queries, dataset_dim]
  uint32_t num_queries,
  const IndexT* dev_seed_ptr,         // [num_queries, num_seeds]
  uint32_t* num_executed_iterations,  // [num_queries,]
  const search_params& ps,
  uint32_t topk,
  uint32_t num_itopk_candidates,
  uint32_t block_size,  //
  uint32_t smem_size,
  int64_t hash_bitlen,
  IndexT* hashmap_ptr,
  size_t small_hash_bitlen,
  size_t small_hash_reset_interval,
  uint32_t num_seeds,
  SampleFilterT sample_filter,
  cudaStream_t stream);

template <typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT,
          typename SampleFilterT>
void select_and_run_multi_partition(
  const dataset_descriptor_host<DataT, IndexT, DistanceT>& ref_dataset_desc,
  const multi_partition_desc_t<DataT, IndexT, DistanceT>* partition_descs,
  uint32_t num_partitions,
  const DataT* queries_ptr,
  uint32_t num_queries,
  IndexT* intermediate_neighbors_ptr,
  DistanceT* intermediate_distances_ptr,
  const search_params& ps,
  uint32_t topk,
  uint32_t num_itopk_candidates,
  uint32_t block_size,
  uint32_t smem_size,
  int64_t hash_bitlen,
  IndexT* hashmap_ptr,
  size_t small_hash_bitlen,
  size_t small_hash_reset_interval,
  SampleFilterT sample_filter,
  cudaStream_t stream);

}  // namespace cuvs::neighbors::cagra::detail::single_cta_search
