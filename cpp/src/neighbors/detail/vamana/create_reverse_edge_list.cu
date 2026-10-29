/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "build_helpers.hpp"
#include "vamana_structs.cuh"

namespace cuvs::neighbors::vamana::detail {

namespace {

// Create src and dest edge lists used to sort and create reverse edges
template <typename accT, typename IdxT = uint32_t>
__global__ void create_reverse_edge_list(void* query_list_ptr,
                                         int num_queries,
                                         int degree,
                                         IdxT* edge_src,
                                         DistPair<IdxT, accT>* edge_dest)
{
  QueryCandidates<IdxT, accT>* query_list =
    static_cast<QueryCandidates<IdxT, accT>*>(query_list_ptr);

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_queries;
       i += blockDim.x * gridDim.x) {
    int cand_count = query_list[i + 1].size - query_list[i].size;

    for (int j = 0; j < cand_count; j++) {
      edge_src[query_list[i].size + j]       = query_list[i].queryId;
      edge_dest[query_list[i].size + j].idx  = query_list[i].ids[j];
      edge_dest[query_list[i].size + j].dist = query_list[i].dists[j];
    }
  }
}

}  // namespace

template <typename accT, typename IdxT>
void launch_create_reverse_edge_list(dim3 grid,
                                     dim3 block,
                                     size_t shared,
                                     cudaStream_t stream,
                                     void* query_list_ptr,
                                     int num_queries,
                                     int degree,
                                     IdxT* edge_src,
                                     DistPair<IdxT, accT>* edge_dest)
{
  create_reverse_edge_list<accT, IdxT>
    <<<grid, block, shared, stream>>>(query_list_ptr, num_queries, degree, edge_src, edge_dest);
}

template void cuvs::neighbors::vamana::detail::launch_create_reverse_edge_list<float, unsigned int>(
  dim3,
  dim3,
  unsigned long,
  cudaStream_t,
  void*,
  int,
  int,
  unsigned int*,
  cuvs::neighbors::vamana::detail::DistPair<unsigned int, float>*);

}  // namespace cuvs::neighbors::vamana::detail
