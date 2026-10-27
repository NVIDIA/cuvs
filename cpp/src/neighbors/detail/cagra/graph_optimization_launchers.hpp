/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/resources.hpp>

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace cuvs::neighbors::cagra::detail::graph {

double optimization_time();

template <typename IdxT, typename AccessorOutputGraph>
void check_duplicates_and_out_of_range(
  raft::resources const& res,
  raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, AccessorOutputGraph>
    output_graph);

template <typename IdxT, typename AccessorOutputGraph>
void log_incoming_edges_histogram(
  raft::resources const& res,
  raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, AccessorOutputGraph>
    output_graph);

template <typename IdxT, typename AccessorOutputGraph>
void make_reverse_graph_gpu(
  raft::resources const& res,
  raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, AccessorOutputGraph>
    output_graph,
  raft::device_matrix_view<IdxT, int64_t> d_rev_graph,
  raft::device_vector_view<uint32_t, int64_t> d_rev_graph_count);

template <typename IdxT, typename AccessorOutputGraph>
void merge_graph_gpu(
  raft::resources const& res,
  raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, AccessorOutputGraph>
    output_graph,
  raft::device_matrix_view<IdxT, int64_t> d_rev_graph,
  raft::device_vector_view<uint32_t, int64_t> d_rev_graph_count,
  raft::host_matrix_view<IdxT, int64_t> mst_graph,
  raft::host_vector_view<uint32_t, int64_t> mst_graph_num_edges,
  bool guarantee_connectivity);

template <typename IdxT>
void mst_opt_update_graph(IdxT* mst_graph_ptr,
                          IdxT* candidate_edges_ptr,
                          IdxT* outgoing_num_edges_ptr,
                          IdxT* incoming_num_edges_ptr,
                          IdxT* outgoing_max_edges_ptr,
                          IdxT* incoming_max_edges_ptr,
                          IdxT* label_ptr,
                          IdxT graph_size,
                          uint32_t mst_graph_degree,
                          uint64_t k,
                          int& num_direct,
                          int& num_alternate,
                          int& num_failure);

template <typename IdxT, typename AccessorKnnGraph>
void mst_optimization(
  raft::resources const& res,
  raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, AccessorKnnGraph> input_graph,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> output_graph,
  raft::host_vector_view<uint32_t, int64_t> mst_graph_num_edges,
  bool use_gpu);

template <typename IdxT, typename AccessorKnnGraph, typename AccessorOutputGraph>
void optimize(
  raft::resources const& res_const,
  raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, AccessorKnnGraph> knn_graph,
  raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, AccessorOutputGraph> new_graph,
  const bool guarantee_connectivity,
  const bool use_gpu_for_mst_optimization);

template <typename IdxT, typename AccessorKnnGraph, typename AccessorOutputGraph>
void prune_graph_gpu(
  raft::resources const& res,
  raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, AccessorKnnGraph> knn_graph,
  raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, AccessorOutputGraph>
    output_graph);

template <typename IdxT>
void launch_merge_graph(dim3 grid,
                        dim3 block,
                        size_t shared_bytes,
                        cudaStream_t stream,
                        raft::device_matrix_view<IdxT, int64_t> output_graph,
                        raft::device_matrix_view<IdxT, int64_t> rev_graph,
                        raft::device_vector_view<uint32_t, int64_t> rev_graph_count,
                        raft::device_matrix_view<IdxT, int64_t> mst_graph,
                        raft::device_matrix_view<IdxT, int64_t> mst_graph_num_edges,
                        const uint32_t batch_size,
                        const uint32_t batch_id,
                        bool guarantee_connectivity,
                        uint32_t* check_num_protected_edges);

template <typename IdxT>
void launch_make_rev_graph_k(dim3 grid,
                             dim3 block,
                             size_t shared_bytes,
                             cudaStream_t stream,
                             raft::device_matrix_view<IdxT, int64_t> output_graph,
                             raft::device_matrix_view<IdxT, int64_t> rev_graph,
                             raft::device_vector_view<uint32_t, int64_t> rev_graph_count,
                             uint64_t k);

template <typename IdxT>
void launch_fused_prune(dim3 grid,
                        dim3 block,
                        size_t shared_bytes,
                        cudaStream_t stream,
                        raft::device_matrix_view<IdxT, int64_t> knn_graph,
                        raft::device_matrix_view<IdxT, int64_t> output_graph,
                        const uint32_t batch_size,
                        const uint32_t batch_id,
                        uint32_t* const d_invalid_neighbor_list,
                        uint64_t* const stats);

template <typename IdxT>
void launch_mst_opt_update_graph(dim3 grid,
                                 dim3 block,
                                 size_t shared_bytes,
                                 cudaStream_t stream,
                                 IdxT* mst_graph,
                                 const IdxT* candidate_edges,
                                 IdxT* outgoing_num_edges,
                                 IdxT* incoming_num_edges,
                                 const IdxT* outgoing_max_edges,
                                 const IdxT* incoming_max_edges,
                                 const IdxT* label,
                                 const uint32_t graph_size,
                                 const uint32_t graph_degree,
                                 uint64_t* stats);

template <typename IdxT>
void launch_mst_opt_labeling(dim3 grid,
                             dim3 block,
                             size_t shared_bytes,
                             cudaStream_t stream,
                             IdxT* label,
                             const IdxT* mst_graph,
                             const uint32_t graph_size,
                             const uint32_t graph_degree,
                             uint64_t* stats);

template <typename IdxT>
void launch_mst_opt_cluster_size(dim3 grid,
                                 dim3 block,
                                 size_t shared_bytes,
                                 cudaStream_t stream,
                                 IdxT* cluster_size,
                                 const IdxT* label,
                                 const uint32_t graph_size,
                                 uint64_t* stats);

template <typename IdxT>
void launch_mst_opt_postprocessing(dim3 grid,
                                   dim3 block,
                                   size_t shared_bytes,
                                   cudaStream_t stream,
                                   IdxT* outgoing_num_edges,
                                   IdxT* incoming_num_edges,
                                   IdxT* outgoing_max_edges,
                                   IdxT* incoming_max_edges,
                                   const IdxT* cluster_size,
                                   const uint32_t graph_size,
                                   const uint32_t graph_degree,
                                   uint64_t* stats);

}  // namespace cuvs::neighbors::cagra::detail::graph
