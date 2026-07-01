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

namespace cuvs::neighbors::ivf_pq::detail {

struct is_non_finite_op {
  __device__ auto operator()(float v) const -> bool { return isnan(v) || isinf(v); }
};

/**
 * Estimate max_i ||x_i||^2 over the dataset.
 */
template <typename DataT, typename Accessor>
float estimate_max_squared_norm(
  raft::resources const& handle,
  raft::mdspan<const DataT, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset)
{
  common::nvtx::range<common::nvtx::domain::cuvs> r("estimate_max_squared_norm");
  auto stream          = raft::resource::get_cuda_stream(handle);
  const int64_t n_rows = dataset.extent(0);
  const int64_t dim    = dataset.extent(1);

  int64_t n_sample = std::min<int64_t>(n_rows, 20000);

  auto mr = raft::resource::get_workspace_resource_ref(handle);
  auto sample =
    raft::make_device_mdarray<DataT>(handle, mr, raft::make_extents<int64_t>(n_sample, dim));
  raft::copy(sample.data_handle(),
             dataset.data_handle(),
             n_sample * dim,
             raft::resource::get_cuda_stream(handle));

  // Compute float-mapped squared norm
  auto d_map_sq_norm = raft::make_device_vector<float, int64_t>(handle, n_sample);
  raft::linalg::reduce<raft::Apply::ALONG_ROWS>(
    handle,
    raft::make_const_mdspan(sample.view()),
    d_map_sq_norm.view(),
    0.0f,
    false,
    [] __device__(DataT v, auto) -> float {
      float e = cuvs::spatial::knn::detail::utils::mapping<float>{}(v);
      return e * e;
    },
    raft::add_op(),
    raft::identity_op());
  // Compute max of squared norm vector
  auto d_max_sq = raft::make_device_scalar<float>(handle, 0.0f);
  raft::linalg::map_reduce(handle,
                           raft::make_const_mdspan(d_map_sq_norm.view()),
                           d_max_sq.view(),
                           0.0f,
                           raft::identity_op(),
                           raft::max_op());

  float max_sq = 0.0f;
  raft::update_host(&max_sq, d_max_sq.data_handle(), 1, stream);
  raft::resource::sync_stream(handle);

  return max_sq;
}

}  // namespace cuvs::neighbors::ivf_pq::detail

namespace cuvs::neighbors::ivf_pq::helpers {

/**
 * @brief Estimate whether FP16 is insufficient for IVF-PQ's full-magnitude distance
 * computations on this dataset (i.e. `internal_distance_dtype` and `coarse_search_dtype`).
 *
 * We bound the largest achievable score from the dataset's vector norms. With R = max_i ||x_i||
 * (estimated from a fraction of the dataset):
 *   - L2Expanded:     ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x,y> <= (||x|| + ||y||)^2 <= 4 * R^2
 *   - InnerProduct:   |<x, y>|    <= ||x|| * ||y||                                  <=     R^2
 *   - CosineExpanded: data is L2-normalized, so |score| <= 1 and overflow is impossible.
 */
template <typename DataT, typename Accessor>
bool estimate_fp16_overflow(
  raft::resources const& handle,
  raft::mdspan<const DataT, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
  cuvs::distance::DistanceType metric)
{
  common::nvtx::range<common::nvtx::domain::cuvs> r("estimate_fp16_overflow");
  if (dataset.extent(0) == 0) { return false; }

  float dist_factor = 1.0f;
  switch (metric) {
    case cuvs::distance::DistanceType::L2Expanded: dist_factor = 4.0f; break;
    case cuvs::distance::DistanceType::CosineExpanded:
      // Cosine similarity scores does normalization itself, so overflow won't happen
      return false;
    case cuvs::distance::DistanceType::InnerProduct: dist_factor = 1.0f; break;
    default: RAFT_FAIL("Unsupported distance type for IVF-PQ search %d.", int(metric));
  }

  const float max_vector_sq_norm =
    cuvs::neighbors::ivf_pq::detail::estimate_max_squared_norm(handle, dataset);
  const float max_distance_sq_norm = dist_factor * max_vector_sq_norm;

  constexpr float kFp16Max = 65504.0f;
  return max_distance_sq_norm > kFp16Max;
}

/**
 * @brief Estimate whether FP16 is insufficient, using the IVF-PQ cluster centers.
 */
inline bool estimate_fp16_overflow_centers(raft::resources const& handle,
                                           const cuvs::neighbors::ivf_pq::index<int64_t>& index,
                                           cuvs::distance::DistanceType metric)
{
  common::nvtx::range<common::nvtx::domain::cuvs> r("estimate_fp16_overflow_centers");
  const int64_t n_lists = index.n_lists();
  const int64_t dim     = index.dim();
  if (n_lists == 0 || dim == 0) { return false; }

  float dist_factor = 1.0f;
  switch (metric) {
    case cuvs::distance::DistanceType::L2Expanded: dist_factor = 4.0f; break;
    case cuvs::distance::DistanceType::CosineExpanded:
      // Cosine similarity scores does normalization itself, so overflow won't happen
      return false;
    case cuvs::distance::DistanceType::InnerProduct: dist_factor = 1.0f; break;
    default: RAFT_FAIL("Unsupported distance type for IVF-PQ search %d.", int(metric));
  }

  const int64_t dim_ext = index.dim_ext();
  auto stream           = raft::resource::get_cuda_stream(handle);
  auto mr               = raft::resource::get_workspace_resource_ref(handle);
  auto centers_contig =
    raft::make_device_mdarray<float>(handle, mr, raft::make_extents<int64_t>(n_lists, dim));
  RAFT_CUDA_TRY(cudaMemcpy2DAsync(centers_contig.data_handle(),
                                  dim * sizeof(float),
                                  index.centers().data_handle(),
                                  dim_ext * sizeof(float),
                                  dim * sizeof(float),
                                  n_lists,
                                  cudaMemcpyDeviceToDevice,
                                  stream));

  const float max_vector_sq_norm = cuvs::neighbors::ivf_pq::detail::estimate_max_squared_norm(
    handle, raft::make_const_mdspan(centers_contig.view()));
  const float max_distance_sq_norm = dist_factor * max_vector_sq_norm;

  constexpr float kFp16Max = 65504.0f;
  return max_distance_sq_norm > kFp16Max;
}

/**
 * @brief Detect whether FP16 internal distance dtypes overflow for this dataset during search.
 *
 * Runs a small probe search against an already-built IVF-PQ index with FP16 internal/coarse
 * distance dtypes, and reports whether any returned distance is non-finite (inf/NaN).
 */
template <typename DataT, typename Accessor>
bool detect_fp16_overflow(
  raft::resources const& handle,
  const cuvs::neighbors::ivf_pq::index<int64_t>& index,
  cuvs::neighbors::ivf_pq::search_params search_params,
  raft::mdspan<const DataT, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset)
{
  common::nvtx::range<common::nvtx::domain::cuvs> r("detect_fp16_overflow");
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
  const bool any_non_finite = thrust::any_of(raft::resource::get_thrust_policy(handle),
                                             distances.data_handle(),
                                             distances.data_handle() + count,
                                             cuvs::neighbors::ivf_pq::detail::is_non_finite_op{});
  raft::resource::sync_stream(handle);
  return any_non_finite;
}

}  // namespace cuvs::neighbors::ivf_pq::helpers
