/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuvs/core/export.hpp>
#include <cuvs/distance/distance.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

namespace CUVS_EXPORT cuvs {
namespace stats {

/**
 * @defgroup stats_silhouette_score Silhouette Score
 * @{
 */
/**
 * @brief main function that returns the average silhouette score for a given set of data and its
 * clusterings
 * @param[in]  handle: raft handle for managing expensive resources
 * @param[in]  X_in: input matrix Data in row-major format (nRows x nCols)
 * @param[in]  labels: the pointer to the array containing labels for every data sample (length:
 * nRows)
 * @param[out] silhouette_score_per_sample: optional array populated with the silhouette score
 * for every sample (length: nRows)
 * @param[in]  n_unique_labels: number of unique labels in the labels array
 * @param[in]  metric: Distance metric to use. Euclidean (L2) is used by default
 * @return: The silhouette score.
 */
float silhouette_score(
  raft::resources const& handle,
  raft::device_matrix_view<const float, int64_t, raft::row_major> X_in,
  raft::device_vector_view<const int, int64_t> labels,
  std::optional<raft::device_vector_view<float, int64_t>> silhouette_score_per_sample,
  int64_t n_unique_labels,
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded);

/**
 * @brief function that returns the average silhouette score for a given set of data and its
 * clusterings
 * @param[in]  handle: raft handle for managing expensive resources
 * @param[in]  X: input matrix Data in row-major format (nRows x nCols)
 * @param[in]  labels: the pointer to the array containing labels for every data sample (length:
 * nRows)
 * @param[out] silhouette_score_per_sample: optional array populated with the silhouette score
 * for every sample (length: nRows)
 * @param[in]  n_unique_labels: number of unique labels in the labels array
 * @param[in]  batch_size: number of samples per batch
 * @param[in]  metric: the numerical value that maps to the type of distance metric to be used in
 * the calculations
 * @return: The silhouette score.
 */
float silhouette_score_batched(
  raft::resources const& handle,
  raft::device_matrix_view<const float, int64_t, raft::row_major> X,
  raft::device_vector_view<const int, int64_t> labels,
  std::optional<raft::device_vector_view<float, int64_t>> silhouette_score_per_sample,
  int64_t n_unique_labels,
  int64_t batch_size,
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded);

/**
 * @brief main function that returns the average silhouette score for a given set of data and its
 * clusterings
 * @param[in]  handle: raft handle for managing expensive resources
 * @param[in]  X_in: input matrix Data in row-major format (nRows x nCols)
 * @param[in]  labels: the pointer to the array containing labels for every data sample (length:
 * nRows)
 * @param[out] silhouette_score_per_sample: optional array populated with the silhouette score
 * for every sample (length: nRows)
 * @param[in]  n_unique_labels: number of unique labels in the labels array
 * @param[in]  metric: the numerical value that maps to the type of distance metric to be used in
 * the calculations
 * @return: The silhouette score.
 */
double silhouette_score(
  raft::resources const& handle,
  raft::device_matrix_view<const double, int64_t, raft::row_major> X_in,
  raft::device_vector_view<const int, int64_t> labels,
  std::optional<raft::device_vector_view<double, int64_t>> silhouette_score_per_sample,
  int64_t n_unique_labels,
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded);

/**
 * @brief function that returns the average silhouette score for a given set of data and its
 * clusterings
 * @param[in]  handle: raft handle for managing expensive resources
 * @param[in]  X: input matrix Data in row-major format (nRows x nCols)
 * @param[in]  labels: the pointer to the array containing labels for every data sample (length:
 * nRows)
 * @param[out] silhouette_score_per_sample: optional array populated with the silhouette score
 * for every sample (length: nRows)
 * @param[in]  n_unique_labels: number of unique labels in the labels array
 * @param[in]  batch_size: number of samples per batch
 * @param[in]  metric: the numerical value that maps to the type of distance metric to be used in
 * the calculations
 * @return: The silhouette score.
 */
double silhouette_score_batched(
  raft::resources const& handle,
  raft::device_matrix_view<const double, int64_t, raft::row_major> X,
  raft::device_vector_view<const int, int64_t> labels,
  std::optional<raft::device_vector_view<double, int64_t>> silhouette_score_per_sample,
  int64_t n_unique_labels,
  int64_t batch_size,
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded);

}  // namespace stats
}  // namespace CUVS_EXPORT cuvs
