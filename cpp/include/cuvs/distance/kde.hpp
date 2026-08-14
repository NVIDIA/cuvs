/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/distance/distance.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <cuvs/core/export.hpp>
#include <optional>

namespace CUVS_EXPORT cuvs {
namespace distance {

/**
 * @brief Compute log-density estimates for query points using kernel density estimation.
 *
 * Fuses pairwise distance computation, kernel evaluation, logsumexp reduction,
 * and normalization into a single CUDA kernel pass. O(N+M) memory usage —
 * the full N×M pairwise distance matrix is never materialised.
 *
 * Supports 13 distance metrics (all expressible as per-feature accumulation),
 * 6 density kernel functions, float32 and float64, and both uniform and
 * weighted training sets.
 *
 * When the query count is small relative to the number of GPU SMs, the
 * training set is automatically split across a 2D grid (multi-pass mode) to
 * keep the GPU fully utilised. Partial logsumexp results are merged by a
 * reduction kernel.
 *
 * @tparam T  float or double
 *
 * @param[in]  handle      RAFT resources handle for stream management
 * @param[in]  query       Query points, row-major (n_query × n_features)
 * @param[in]  train       Training points, row-major (n_train × n_features)
 * @param[in]  weights     Per-training-point weights (n_train,), or nullopt for uniform
 * @param[out] output      Log-density estimates (n_query,)
 * @param[in]  bandwidth   Kernel bandwidth (must be > 0)
 * @param[in]  sum_weights Sum of sample weights (or n_train if uniform)
 * @param[in]  kernel      Density kernel function
 * @param[in]  metric      Distance metric
 * @param[in]  metric_arg  Metric parameter (e.g. p for Minkowski; ignored otherwise)
 */
template <typename T>
void kde(raft::resources const& handle,
         raft::device_matrix_view<const T, std::int64_t, raft::layout_c_contiguous> query,
         raft::device_matrix_view<const T, std::int64_t, raft::layout_c_contiguous> train,
         std::optional<raft::device_vector_view<const T, std::int64_t>> weights,
         raft::device_vector_view<T, std::int64_t> output,
         T bandwidth,
         T sum_weights,
         DensityKernelType kernel,
         cuvs::distance::DistanceType metric,
         T metric_arg);

extern template void kde<float>(
  raft::resources const&,
  raft::device_matrix_view<const float, std::int64_t, raft::layout_c_contiguous>,
  raft::device_matrix_view<const float, std::int64_t, raft::layout_c_contiguous>,
  std::optional<raft::device_vector_view<const float, std::int64_t>>,
  raft::device_vector_view<float, std::int64_t>,
  float,
  float,
  DensityKernelType,
  cuvs::distance::DistanceType,
  float);

extern template void kde<double>(
  raft::resources const&,
  raft::device_matrix_view<const double, std::int64_t, raft::layout_c_contiguous>,
  raft::device_matrix_view<const double, std::int64_t, raft::layout_c_contiguous>,
  std::optional<raft::device_vector_view<const double, std::int64_t>>,
  raft::device_vector_view<double, std::int64_t>,
  double,
  double,
  DensityKernelType,
  cuvs::distance::DistanceType,
  double);

}  // namespace distance
}  // namespace CUVS_EXPORT cuvs
