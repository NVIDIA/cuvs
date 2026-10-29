/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cub/device/dispatch/dispatch_common.cuh>
#include <cuvs/core/bitset.hpp>
#include <raft/core/kvp.hpp>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace cuvs::detail {
template <cub::SelectImpl Selection,
          class Input,
          class Flags,
          class Output,
          class Count,
          class Select,
          class Equal,
          class Offset,
          class Policy>
cudaError_t cub_select_dispatch(
  void*, size_t&, Input, Flags, Output, Count, Select, Equal, Offset, cudaStream_t, Policy);
template <class T, class Index, class Bits, class Distance>
struct bitset_row_filter {
  cuvs::core::bitset_view<Bits, Index> filter_view;
  __device__ bool operator()(Index index) const;
};
template <class IndexIt>
struct container_start {
  IndexIt indices;
  long const* num_indices;
  __device__ bool operator()(long index) const;
};
}  // namespace cuvs::detail
namespace cuco::experimental::detail {
template <class IndexIt>
struct is_container_start;
}

namespace cuvs::cluster::kmeans::detail {
template <typename DataT, typename IndexT>
struct SamplingOp {
  DataT* rnd;
  uint8_t* flag;
  DataT cluster_cost;
  double oversampling_factor;
  IndexT n_clusters;

  __host__ __device__ SamplingOp(DataT c, double l, IndexT k, DataT* rand, uint8_t* ptr);
  __host__ __device__ bool operator()(const raft::KeyValuePair<ptrdiff_t, DataT>& a) const;
};
}  // namespace cuvs::cluster::kmeans::detail
