/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <neighbors/detail/cagra/compute_distance-ext.cuh>
#include <neighbors/detail/cagra/multi_partition_desc.hpp>

#include <cuvs/neighbors/cagra.hpp>

namespace cuvs::neighbors::cagra::detail::multi_cta_search {

template <typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT,
          typename SampleFilterT>
void select_and_run(const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
                    raft::device_matrix_view<const IndexT, int64_t, raft::row_major> graph,
                    const SourceIndexT* source_indices_ptr,
                    IndexT* topk_indices_ptr,       // [num_queries, topk]
                    DistanceT* topk_distances_ptr,  // [num_queries, topk]
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
                    cudaStream_t stream);

/**
 * Multi-partition launcher. Drives `search_kernel_mp` with a 3D grid
 * (num_cta_per_query, num_queries, num_partitions). Per-(query, partition) outputs are written
 * into the intermediate buffer in partition-major layout
 * [num_partitions, num_queries, num_cta_per_query * itopk_size]. Each partition's data
 * (dataset_desc, graph, graph_degree) is read by the kernel from partition_descs[blockIdx.z];
 * smem and the result buffer are sized for the max graph_degree across partitions.
 */
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
                       cudaStream_t stream);

}  // namespace cuvs::neighbors::cagra::detail::multi_cta_search
