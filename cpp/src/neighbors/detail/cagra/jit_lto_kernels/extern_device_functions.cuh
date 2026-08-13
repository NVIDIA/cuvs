/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#include "../compute_distance.hpp"
#include <cuvs/distance/distance.hpp>

namespace cuvs::neighbors::cagra::detail {

template <typename QUERY_T, typename DISTANCE_T>
extern __device__ DISTANCE_T dist_op(QUERY_T a, QUERY_T b);

template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename QueryT>
extern __device__ DistanceT apply_normalization_standard(
  DistanceT distance,
  const typename dataset_descriptor_base_t<DataT, IndexT, DistanceT>::args_t args,
  IndexT dataset_index);

template <typename DataT, typename IndexT, typename DistanceT>
extern __device__ const dataset_descriptor_base_t<DataT, IndexT, DistanceT>* setup_workspace(
  const dataset_descriptor_base_t<DataT, IndexT, DistanceT>*, void*, const DataT*, uint32_t);

template <typename DataT, typename IndexT, typename DistanceT>
extern __device__ DistanceT
compute_distance(const typename dataset_descriptor_base_t<DataT, IndexT, DistanceT>::args_t args,
                 IndexT dataset_index,
                 bool valid,
                 uint32_t team_size_bits);

template <typename DataT, typename IndexT, typename DistanceT>
extern __device__ DistanceT compute_distance_per_thread(
  const typename dataset_descriptor_base_t<DataT, IndexT, DistanceT>::args_t, IndexT);
}  // namespace cuvs::neighbors::cagra::detail

namespace cuvs::neighbors::detail {

template <typename SourceIndexT>
extern __device__ bool sample_filter(uint32_t query_id, SourceIndexT node_id, void* filter_data);

}  // namespace cuvs::neighbors::detail
