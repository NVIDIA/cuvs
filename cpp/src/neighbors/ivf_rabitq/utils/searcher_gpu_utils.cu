/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "searcher_gpu_utils.hpp"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>

#include <cuda_runtime.h>

namespace cuvs::neighbors::ivf_rabitq::detail {

// finds the maximum cluster size among clusters to be probed and, optionally, the maximum number of
// vectors probed for any single query
void get_max_probed_cluster_size_and_vectors_count(
  raft::resources const& handle,
  const ClusterQueryPair* d_cluster_query_pairs,
  const size_t num_pairs,
  const IVFGPU::GPUClusterMeta* d_cluster_meta,  // actually just need the cluster sizes
  const size_t num_queries,
  uint32_t& max_probed_cluster_size,
  std::optional<size_t>& max_probed_vectors_count)
{
  auto stream = raft::resource::get_cuda_stream(handle);

  // check if max_probed_vectors_count is requested for evaluation
  const bool get_max_probed_vectors_count = max_probed_vectors_count.has_value();

  auto d_max_probed_cluster_size = raft::make_device_scalar<uint32_t>(handle, 0);
  auto d_probed_vectors_count    = raft::make_device_vector<unsigned long long, int64_t>(
    handle, get_max_probed_vectors_count ? num_queries : 0);
  // raw pointers for passing by value to device lambda
  auto d_max_probed_cluster_size_ptr = d_max_probed_cluster_size.data_handle();
  auto d_probed_vectors_count_ptr    = d_probed_vectors_count.data_handle();
  if (get_max_probed_vectors_count) {
    RAFT_CUDA_TRY(cudaMemsetAsync(
      d_probed_vectors_count_ptr, 0, num_queries * sizeof(size_t), stream));  // Initialize to 0
  }

  auto count = thrust::make_counting_iterator<int64_t>(0);
  thrust::for_each(
    raft::resource::get_thrust_policy(handle), count, count + num_pairs, [=] __device__(int64_t i) {
      auto [cluster_idx, query_idx] = d_cluster_query_pairs[i];
      auto cluster_size             = d_cluster_meta[cluster_idx].num;
      atomicMax(d_max_probed_cluster_size_ptr, cluster_size);
      if (get_max_probed_vectors_count)
        atomicAdd(&d_probed_vectors_count_ptr[query_idx],
                  static_cast<unsigned long long>(cluster_size));
    });
  raft::copy(&max_probed_cluster_size, d_max_probed_cluster_size_ptr, 1, stream);
  if (get_max_probed_vectors_count) {
    max_probed_vectors_count = thrust::reduce(raft::resource::get_thrust_policy(handle),
                                              d_probed_vectors_count_ptr,
                                              d_probed_vectors_count_ptr + num_queries,
                                              0,
                                              thrust::maximum<size_t>());
  }
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
