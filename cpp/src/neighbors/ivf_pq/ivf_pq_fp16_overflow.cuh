/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../detail/ann_utils.cuh"  // cuvs::spatial::knn::detail::utils::mapping

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map_reduce.cuh>
#include <raft/linalg/reduce.cuh>
#include <raft/util/cuda_dev_essentials.cuh>
#include <raft/util/cudart_utils.hpp>

#include <thrust/logical.h>

#include <algorithm>
#include <cstdint>

namespace cuvs::neighbors::ivf_pq::helpers {

/**
 * @brief Detect whether FP16 internal distance dtypes overflow for this dataset during search.
 *
 * Runs a small probe search against an already-built IVF-PQ index with current distance types,
 * and reports whether any returned distance is non-finite (inf/NaN).
 */
template <typename DataT, typename Accessor>
bool detect_fp16_overflow(
  raft::resources const& handle,
  const cuvs::neighbors::ivf_pq::index<int64_t>& index,
  cuvs::neighbors::ivf_pq::search_params search_params,
  raft::mdspan<const DataT, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset)
{
  const int64_t n_rows = dataset.extent(0);
  if (n_rows == 0) { return false; }

  auto stream       = raft::resource::get_cuda_stream(handle);
  const int64_t dim = dataset.extent(1);

  constexpr int64_t kMaxSampleQueries = 128;
  constexpr uint32_t kProbeTopK       = 32;
  const int64_t n_sample              = std::min<int64_t>(n_rows, kMaxSampleQueries);
  const uint32_t top_k = std::min<uint32_t>(static_cast<uint32_t>(n_rows), kProbeTopK);

  auto mr = raft::resource::get_workspace_resource_ref(handle);
  auto queries =
    raft::make_device_mdarray<DataT>(handle, mr, raft::make_extents<int64_t>(n_sample, dim));
  raft::copy(queries.data_handle(), dataset.data_handle(), n_sample * dim, stream);

  auto neighbors =
    raft::make_device_mdarray<int64_t>(handle, mr, raft::make_extents<int64_t>(n_sample, top_k));
  auto distances =
    raft::make_device_mdarray<float>(handle, mr, raft::make_extents<int64_t>(n_sample, top_k));

  cuvs::neighbors::ivf_pq::search(handle,
                                  search_params,
                                  index,
                                  raft::make_const_mdspan(queries.view()),
                                  neighbors.view(),
                                  distances.view());

  const int64_t count       = n_sample * static_cast<int64_t>(top_k);
  auto is_non_finite_op     = [] __device__(float v) { return isnan(v) || isinf(v); };
  const bool any_non_finite = thrust::any_of(raft::resource::get_thrust_policy(handle),
                                             distances.data_handle(),
                                             distances.data_handle() + count,
                                             is_non_finite_op);
  raft::resource::sync_stream(handle);
  return any_non_finite;
}

}  // namespace cuvs::neighbors::ivf_pq::helpers
