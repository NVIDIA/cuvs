/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <raft/core/device_mdspan.hpp>
#include <vector_types.h>

namespace cuvs::neighbors::vamana::detail {
template <typename IdxT, typename accT>
struct QueryCandidates;
template <typename IdxT, typename accT>
struct DistPair;
template <typename IdxT, typename accT>
void launch_init_query_candidate_list(dim3 grid,
                                      dim3 block,
                                      size_t shared,
                                      cudaStream_t stream,
                                      QueryCandidates<IdxT, accT>* query_list,
                                      IdxT* visited_id_ptr,
                                      accT* visited_dist_ptr,
                                      int num_queries,
                                      int maxSize,
                                      int extra_queries_in_list = 0);
template <typename IdxT, typename accT>
void launch_set_query_ids(dim3 grid,
                          dim3 block,
                          size_t shared,
                          cudaStream_t stream,
                          void* query_list_ptr,
                          IdxT* d_query_ids,
                          int step_size);
template <typename accT, typename IdxT>
void launch_write_graph_edges_kernel(dim3 grid,
                                     dim3 block,
                                     size_t shared,
                                     cudaStream_t stream,
                                     raft::device_matrix_view<IdxT, int64_t> graph,
                                     void* query_list_ptr,
                                     int degree,
                                     int num_queries);
template <typename accT, typename IdxT>
void launch_create_reverse_edge_list(dim3 grid,
                                     dim3 block,
                                     size_t shared,
                                     cudaStream_t stream,
                                     void* query_list_ptr,
                                     int num_queries,
                                     int degree,
                                     IdxT* edge_src,
                                     DistPair<IdxT, accT>* edge_dest);
template <typename accT, typename IdxT>
void launch_scatter_prefix_offsets(dim3 grid,
                                   dim3 block,
                                   size_t shared,
                                   cudaStream_t stream,
                                   QueryCandidates<IdxT, accT>* query_list,
                                   const int* edge_offsets,
                                   int count);

template <typename Key, typename Value, typename Count, typename Compare>
cudaError_t sort_reverse_pairs(void* storage,
                               size_t& bytes,
                               Key* keys,
                               Value* values,
                               Count count,
                               Compare compare,
                               cudaStream_t stream);
template <typename Policy, typename Key, typename Value>
void unique_reverse_keys(Policy&& policy, Key begin, Key end, Value values);
}  // namespace cuvs::neighbors::vamana::detail
