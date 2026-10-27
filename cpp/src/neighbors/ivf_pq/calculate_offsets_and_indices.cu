/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <core/cub_runtime_launchers.hpp>
#include <core/prefix_scans_launchers.hpp>

#include <raft/core/error.hpp>
#include <raft/core/logger.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/device_atomics.cuh>
#include <raft/util/integer_utils.hpp>

#include <cuda/stream>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <thrust/functional.h>

#include <cstdint>

namespace cuvs::neighbors::ivf_pq::detail {

template <uint32_t BlockDim, typename IdxT>
__launch_bounds__(BlockDim) static __global__ void fill_indices_kernel(IdxT n_rows,
                                                                       IdxT* data_indices,
                                                                       IdxT* data_offsets,
                                                                       const uint32_t* labels)
{
  const auto i = IdxT(BlockDim) * IdxT(blockIdx.x) + IdxT(threadIdx.x);
  if (i >= n_rows) { return; }
  data_indices[atomicAdd<IdxT>(data_offsets + labels[i], 1)] = i;
}

template <typename IdxT>
auto calculate_offsets_and_indices(IdxT n_rows,
                                   uint32_t n_lists,
                                   const uint32_t* labels,
                                   const uint32_t* cluster_sizes,
                                   IdxT* cluster_offsets,
                                   IdxT* data_indices,
                                   cuda::stream_ref stream) -> uint32_t
{
  auto exec_policy = rmm::exec_policy(stream);
  // Calculate the offsets
  IdxT cumsum = 0;
  raft::update_device(cluster_offsets, &cumsum, 1, stream);
  cuvs::detail::thrust_inclusive_scan(
    exec_policy, cluster_sizes, cluster_sizes + n_lists, cluster_offsets + 1, thrust::plus<>{});
  raft::update_host(&cumsum, cluster_offsets + n_lists, 1, stream);
  uint32_t max_cluster_size =
    *cuvs::detail::thrust_max_element(exec_policy, cluster_sizes, cluster_sizes + n_lists);
  stream.sync();
  RAFT_EXPECTS(cumsum == n_rows, "cluster sizes do not add up.");
  RAFT_LOG_DEBUG("Max cluster size %d", max_cluster_size);
  rmm::device_uvector<IdxT> data_offsets_buf(n_lists, stream);
  auto data_offsets = data_offsets_buf.data();
  raft::copy(data_offsets, cluster_offsets, n_lists, stream);
  constexpr uint32_t n_threads = 128;  // NOLINT
  const IdxT n_blocks          = raft::div_rounding_up_unsafe(n_rows, n_threads);
  fill_indices_kernel<n_threads>
    <<<n_blocks, n_threads, 0, stream.get()>>>(n_rows, data_indices, data_offsets, labels);
  return max_cluster_size;
}

template uint32_t calculate_offsets_and_indices<long>(
  long, uint32_t, const uint32_t*, const uint32_t*, long*, long*, cuda::stream_ref);

}  // namespace cuvs::neighbors::ivf_pq::detail
