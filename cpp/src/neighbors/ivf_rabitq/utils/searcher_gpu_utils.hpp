/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../gpu_index/ivf_gpu.cuh"

#include <raft/core/resources.hpp>

#include <optional>
#include <stdint.h>

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
  std::optional<size_t>& max_probed_vectors_count);

}  // namespace cuvs::neighbors::ivf_rabitq::detail
