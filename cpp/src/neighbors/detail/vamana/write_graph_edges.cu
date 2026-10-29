/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "build_helpers.hpp"
#include "vamana_structs.cuh"

namespace cuvs::neighbors::vamana::detail {

namespace {

// Update the graph from the results of the query list (or reverse edge list)
template <typename accT, typename IdxT = uint32_t>
__global__ void write_graph_edges_kernel(raft::device_matrix_view<IdxT, int64_t> graph,
                                         void* query_list_ptr,
                                         int degree,
                                         int num_queries)
{
  QueryCandidates<IdxT, accT>* query_list =
    static_cast<QueryCandidates<IdxT, accT>*>(query_list_ptr);

  for (int i = blockIdx.x; i < num_queries; i += gridDim.x) {
    for (int j = threadIdx.x; j < query_list[i].size; j += blockDim.x) {
      graph(query_list[i].queryId, j) = query_list[i].ids[j];
    }
  }
}

}  // namespace

template <typename accT, typename IdxT>
void launch_write_graph_edges_kernel(dim3 grid,
                                     dim3 block,
                                     size_t shared,
                                     cudaStream_t stream,
                                     raft::device_matrix_view<IdxT, int64_t> graph,
                                     void* query_list_ptr,
                                     int degree,
                                     int num_queries)
{
  write_graph_edges_kernel<accT, IdxT>
    <<<grid, block, shared, stream>>>(graph, query_list_ptr, degree, num_queries);
}

template void cuvs::neighbors::vamana::detail::launch_write_graph_edges_kernel<float, unsigned int>(
  dim3,
  dim3,
  unsigned long,
  cudaStream_t,
  ::cuda::std::mdspan<
    unsigned int,
    ::cuda::std::extents<long, 18446744073709551615ul, 18446744073709551615ul>,
    ::cuda::std::layout_right,
    raft::host_device_accessor<::cuda::std::default_accessor<unsigned int>, (raft::memory_type)2>>,
  void*,
  int,
  int);

}  // namespace cuvs::neighbors::vamana::detail
