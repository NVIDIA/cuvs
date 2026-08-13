/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../compute_distance_standard-impl.cuh"
#include "extern_device_functions.cuh"

namespace cuvs::neighbors::cagra::detail {

template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename QueryT>
__device__ DistanceT apply_normalization_standard_noop_impl(
  DistanceT distance,
  const typename dataset_descriptor_base_t<DataT, IndexT, DistanceT>::args_t args,
  IndexT dataset_index)
{
  (void)args;
  (void)dataset_index;
  return distance;
}

template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename QueryT>
__device__ DistanceT apply_normalization_standard_cosine_impl(
  DistanceT distance,
  const typename dataset_descriptor_base_t<DataT, IndexT, DistanceT>::args_t args,
  IndexT dataset_index)
{
  const auto* dataset_norms =
    standard_dataset_descriptor_t<TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT, QueryT>::
      dataset_norms_ptr(args);
  auto norm = dataset_norms[dataset_index];
  if (norm > 0) { distance = distance / norm; }
  return distance;
}

}  // namespace cuvs::neighbors::cagra::detail
