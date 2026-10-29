/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "build_helpers.hpp"
#include "vamana_structs.cuh"

namespace cuvs::neighbors::vamana::detail {

namespace {

// Copy query ID values from input array
template <typename IdxT, typename accT>
__global__ void set_query_ids(void* query_list_ptr, IdxT* d_query_ids, int step_size)
{
  QueryCandidates<IdxT, accT>* query_list =
    static_cast<QueryCandidates<IdxT, accT>*>(query_list_ptr);

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < step_size; i += blockDim.x * gridDim.x) {
    query_list[i].queryId = d_query_ids[i];
    query_list[i].size    = 0;
  }
}

}  // namespace

template <typename IdxT, typename accT>
void launch_set_query_ids(dim3 grid,
                          dim3 block,
                          size_t shared,
                          cudaStream_t stream,
                          void* query_list_ptr,
                          IdxT* d_query_ids,
                          int step_size)
{
  set_query_ids<IdxT, accT>
    <<<grid, block, shared, stream>>>(query_list_ptr, d_query_ids, step_size);
}

template void cuvs::neighbors::vamana::detail::launch_set_query_ids<unsigned int, float>(
  dim3, dim3, unsigned long, cudaStream_t, void*, unsigned int*, int);

}  // namespace cuvs::neighbors::vamana::detail
