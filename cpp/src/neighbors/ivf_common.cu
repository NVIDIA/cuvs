/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ivf_common.cuh"

#include <raft/util/cudart_utils.hpp>
#include <raft/util/pow2_utils.cuh>

#include <cub/block/block_scan.cuh>
#include <cub/device/device_radix_sort.cuh>

namespace cuvs::neighbors::ivf::detail {

/**
 * For each query, we calculate a cumulative sum of the cluster sizes that we probe, and return that
 * in chunk_indices. Essentially this is a segmented inclusive scan of the cluster sizes. The total
 * number of samples per query (sum of the cluster sizes that we probe) is returned in n_samples.
 */
template <int BlockDim>
__launch_bounds__(BlockDim) RAFT_KERNEL
  calc_chunk_indices_kernel(uint32_t n_probes,
                            const uint32_t* cluster_sizes,      // [n_clusters]
                            const uint32_t* clusters_to_probe,  // [n_queries, n_probes]
                            uint32_t* chunk_indices,            // [n_queries, n_probes]
                            uint32_t* n_samples                 // [n_queries]
  )
{
  using block_scan = cub::BlockScan<uint32_t, BlockDim>;
  __shared__ typename block_scan::TempStorage shm;

  // locate the query data
  clusters_to_probe += n_probes * blockIdx.x;
  chunk_indices += n_probes * blockIdx.x;

  // block scan
  const uint32_t n_probes_aligned = raft::Pow2<BlockDim>::roundUp(n_probes);
  uint32_t total                  = 0;
  for (uint32_t probe_ix = threadIdx.x; probe_ix < n_probes_aligned; probe_ix += BlockDim) {
    auto label = probe_ix < n_probes ? clusters_to_probe[probe_ix] : 0u;
    auto chunk = probe_ix < n_probes ? cluster_sizes[label] : 0u;
    if (threadIdx.x == 0) { chunk += total; }
    block_scan(shm).InclusiveSum(chunk, chunk, total);
    __syncthreads();
    if (probe_ix < n_probes) { chunk_indices[probe_ix] = chunk; }
  }
  // save the total size
  if (threadIdx.x == 0) { n_samples[blockIdx.x] = total; }
}

void calc_chunk_indices::configured::operator()(const uint32_t* cluster_sizes,
                                                const uint32_t* clusters_to_probe,
                                                uint32_t* chunk_indices,
                                                uint32_t* n_samples,
                                                rmm::cuda_stream_view stream)
{
  void* kernel = nullptr;
  switch (block_dim.x) {
    case 32: kernel = reinterpret_cast<void*>(calc_chunk_indices_kernel<32>); break;
    case 64: kernel = reinterpret_cast<void*>(calc_chunk_indices_kernel<64>); break;
    case 128: kernel = reinterpret_cast<void*>(calc_chunk_indices_kernel<128>); break;
    case 256: kernel = reinterpret_cast<void*>(calc_chunk_indices_kernel<256>); break;
    case 512: kernel = reinterpret_cast<void*>(calc_chunk_indices_kernel<512>); break;
    case 1024: kernel = reinterpret_cast<void*>(calc_chunk_indices_kernel<1024>); break;
    default:
      RAFT_FAIL("Unsupported block dimension for calc_chunk_indices::configured() : %d",
                block_dim.x);
  }

  void* args[] =  // NOLINT
    {&n_probes, &cluster_sizes, &clusters_to_probe, &chunk_indices, &n_samples};
  RAFT_CUDA_TRY(cudaLaunchKernel(kernel, grid_dim, block_dim, args, 0, stream));
}

// Helper function to sort cluster sizes using CUB, extracted from template to avoid
// including cub/device/* in the header file
void sort_cluster_sizes_descending(uint32_t* input,
                                   uint32_t* output,
                                   uint32_t n_lists,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref tmp_res)
{
  int begin_bit             = 0;
  int end_bit               = sizeof(uint32_t) * 8;
  size_t cub_workspace_size = 0;
  cub::DeviceRadixSort::SortKeysDescending(
    nullptr, cub_workspace_size, input, output, n_lists, begin_bit, end_bit, stream);
  rmm::device_buffer cub_workspace(cub_workspace_size, stream, tmp_res);
  cub::DeviceRadixSort::SortKeysDescending(
    cub_workspace.data(), cub_workspace_size, input, output, n_lists, begin_bit, end_bit, stream);
}

}  // namespace cuvs::neighbors::ivf::detail
