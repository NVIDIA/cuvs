/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "build_helpers.hpp"
#include "vamana_structs.cuh"

namespace cuvs::neighbors::vamana::detail {

template <typename accT, typename IdxT>
__global__ void scatter_prefix_offsets(QueryCandidates<IdxT, accT>* query_list,
                                       const int* edge_offsets,
                                       int count)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
    query_list[i].size = edge_offsets[i];
  }
}

template <typename accT, typename IdxT>
void launch_scatter_prefix_offsets(dim3 grid,
                                   dim3 block,
                                   size_t shared,
                                   cudaStream_t stream,
                                   QueryCandidates<IdxT, accT>* query_list,
                                   const int* edge_offsets,
                                   int count)
{
  scatter_prefix_offsets<accT, IdxT>
    <<<grid, block, shared, stream>>>(query_list, edge_offsets, count);
}

template void cuvs::neighbors::vamana::detail::launch_scatter_prefix_offsets<float, unsigned int>(
  dim3,
  dim3,
  unsigned long,
  cudaStream_t,
  cuvs::neighbors::vamana::detail::QueryCandidates<unsigned int, float>*,
  int const*,
  int);

}  // namespace cuvs::neighbors::vamana::detail
