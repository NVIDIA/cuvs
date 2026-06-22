/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../detail/ann_utils.cuh"  // cuvs::spatial::knn::detail::utils::mapping

#include <cuvs/distance/distance.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map_reduce.cuh>
#include <raft/linalg/reduce.cuh>
#include <raft/matrix/sample_rows.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cstdint>

namespace cuvs::neighbors::ivf_pq::detail {

/**
 * Estimate max_i ||x_i||^2 over the dataset by uniformly sampling a fraction from it.
 *
 * Decision: Uniform sampling is selected as it is sufficient to detect FP16 overflow in
 * the datasets, where overflow-causing large vectors are frequent (e.g. SIFT 1M). For 
 * dataset with rare large outliers, we might preferably sample biasedly towards large vectors,
 * e.g. via top-k selection over the vectors with largest L_inf norm.
 */
template <typename DataT, typename Accessor>
float estimate_max_squared_norm(
  raft::resources const& handle,
  raft::mdspan<const DataT, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset)
{
  auto stream          = raft::resource::get_cuda_stream(handle);
  const int64_t n_rows = dataset.extent(0);
  const int64_t dim    = dataset.extent(1);

  // Determine sample size based on a smooth saturation equation. The equation satisfies:
  // - n_sample is always less than or equal to n_rows
  // - n_sample saturates to kSaturation when n_rows is inf
  // - n_sample increases fast for small n_rows and slow to saturation for large n_rows
  // Idea: we sample most of the dataset when it is small-sized, and only a small fraction
  // (up to a maximum/saturation number) when the dataset size grows large.
  // kSaturation and kDelay are selected as a compromise between runtime and outlier recall.
  constexpr int64_t kSaturation = 20000;
  constexpr int64_t kDelay      = kSaturation * 10;
  RAFT_EXPECTS(kDelay >= kSaturation,
               "kDelay must not be smaller than kSaturation so that n_sample is always less than "
               "or equal to n_rows");
  int64_t n_sample = (n_rows * kSaturation + (n_rows + kDelay - 1)) / (n_rows + kDelay);

  // Sample from the dataset
  auto mr = raft::resource::get_workspace_resource_ref(handle);
  auto sample =
    raft::make_device_mdarray<DataT>(handle, mr, raft::make_extents<int64_t>(n_sample, dim));
  raft::matrix::sample_rows<DataT, int64_t>(
    handle, raft::random::RngState{137}, dataset, sample.view());

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
 * @brief Estimate whether FP16 is likely insufficient for IVF-PQ's full-magnitude distance
 * computations on this dataset (i.e. `internal_distance_dtype` and `coarse_search_dtype`).
 *
 * We bound the largest achievable score from the dataset's vector norms. With R = max_i ||x_i||
 * (estimated from a random sample of the dataset):
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
  if (dataset.extent(0) == 0) { return false; }

  // Cosine similarity scores does normalization itself, so overflow won't happen
  if (metric == cuvs::distance::DistanceType::CosineExpanded) { return false; }

  // FP16 largest finite value, with a defensive margin to also avoid precision loss near the limit.
  constexpr float kFp16Max              = 65504.0f;
  constexpr float kFp16DefensiveMargin  = 0.25f;
  const float overflow_detect_threshold = kFp16DefensiveMargin * kFp16Max;

  const float max_vector_sq_norm =
    cuvs::neighbors::ivf_pq::detail::estimate_max_squared_norm(handle, dataset);

  const float max_distance_sq_norm = metric == cuvs::distance::DistanceType::L2Expanded
                                       ? 4.0f * max_vector_sq_norm
                                       : max_vector_sq_norm;

  return max_distance_sq_norm > overflow_detect_threshold;
}

}  // namespace cuvs::neighbors::ivf_pq::helpers
