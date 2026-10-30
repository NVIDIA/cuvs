/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../compute_distance_standard-impl.cuh"

#include <raft/util/reduction.cuh>

namespace cuvs::neighbors::cagra::detail {

// Count binary set members once per workspace setup, then scale to match the dot product.
template <typename DataT, typename IndexT, typename DistanceT, typename QueryT>
__device__ void setup_query_norm_jaccard_impl(
  typename dataset_descriptor_base_t<DataT, IndexT, DistanceT>::args_t& args, const DataT* query)
{
  DistanceT count = 0;
  for (uint32_t i = threadIdx.x; i < args.dim; i += blockDim.x) {
    count += static_cast<DistanceT>(query[i]);
  }
  __shared__ DistanceT warp_sums[32];
  count = raft::blockReduce(count, reinterpret_cast<char*>(warp_sums));
  // extra_word2 stores this block's scaled query count in its shared descriptor.
  if (threadIdx.x == 0) {
    constexpr DistanceT kScale = cuvs::spatial::knn::detail::utils::config<DataT>::kDivisor /
                                 cuvs::spatial::knn::detail::utils::config<QueryT>::kDivisor;
    args.extra_word2 = __float_as_uint(count / (kScale * kScale));
  }
  __syncthreads();
}

template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename QueryT>
__device__ DistanceT apply_normalization_standard_jaccard_impl(
  DistanceT distance,
  const typename dataset_descriptor_base_t<DataT, IndexT, DistanceT>::args_t args,
  IndexT dataset_index)
{
  // InnerProduct accumulates -dot(query, dataset) using scaled coordinates. Expansion needs
  // the complete dot product. Some callers invoke this only for valid teams, so name this
  // team's lanes explicitly rather than using a full-warp or instantaneous mask.
  const auto team_lane = (threadIdx.x % 32) / TeamSize * TeamSize;
  const auto mask      = (0xffffffffu >> (32 - TeamSize)) << team_lane;
#pragma unroll
  for (uint32_t offset = TeamSize / 2; offset > 0; offset /= 2) {
    distance += __shfl_xor_sync(mask, distance, offset, TeamSize);
  }
  const auto* dataset_norms =
    standard_dataset_descriptor_t<TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT, QueryT>::
      dataset_norms_ptr(args);
  auto query_norm = __uint_as_float(args.extra_word2);
  auto set_union  = query_norm + dataset_norms[dataset_index] + distance;
  auto result     = set_union > 0 ? DistanceT(1) + distance / set_union : DistanceT(0);
  // Preserve the per-thread interface: the caller sums these identical shares.
  return result / DistanceT(TeamSize);
}

}  // namespace cuvs::neighbors::cagra::detail
