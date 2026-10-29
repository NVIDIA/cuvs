/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "build_helpers.hpp"
#include "vamana_structs.cuh"

namespace cuvs::neighbors::vamana::detail {

namespace {

// Initialize a list of QueryCandidates objects: assign memory to mpointers and initialize values
template <typename IdxT, typename accT>
__global__ void init_query_candidate_list(QueryCandidates<IdxT, accT>* query_list,
                                          IdxT* visited_id_ptr,
                                          accT* visited_dist_ptr,
                                          int num_queries,
                                          int maxSize,
                                          int extra_queries_in_list = 0)
{
  IdxT* ids_ptr  = static_cast<IdxT*>(visited_id_ptr);
  accT* dist_ptr = static_cast<accT*>(visited_dist_ptr);

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_queries * maxSize;
       i += blockDim.x * gridDim.x) {
    ids_ptr[i]  = raft::upper_bound<IdxT>();
    dist_ptr[i] = raft::upper_bound<accT>();
  }

  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_queries + extra_queries_in_list;
       i += blockDim.x * gridDim.x) {
    query_list[i].maxSize = maxSize;
    query_list[i].size    = 0;
    query_list[i].ids     = &ids_ptr[i * (size_t)(maxSize)];
    query_list[i].dists   = &dist_ptr[i * (size_t)(maxSize)];
  }
}

}  // namespace

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
                                      int extra_queries_in_list)
{
  init_query_candidate_list<IdxT, accT><<<grid, block, shared, stream>>>(
    query_list, visited_id_ptr, visited_dist_ptr, num_queries, maxSize, extra_queries_in_list);
}

template void
cuvs::neighbors::vamana::detail::launch_init_query_candidate_list<unsigned int, float>(
  dim3,
  dim3,
  unsigned long,
  cudaStream_t,
  cuvs::neighbors::vamana::detail::QueryCandidates<unsigned int, float>*,
  unsigned int*,
  float*,
  int,
  int,
  int);

}  // namespace cuvs::neighbors::vamana::detail
