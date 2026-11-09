/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cuvs/distance/distance.hpp>
#include <raft/core/device_mdarray.hpp>  // raft::make_device_matrix
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/copy.cuh>
#include <raft/matrix/detail/select_k.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include "naive_knn.cuh"

#include "../test_utils.cuh"
#include <atomic>
#include <cstdio>
#include <filesystem>
#include <gtest/gtest.h>
#include <iostream>
#include <limits>

namespace cuvs::neighbors {

/** Compute capability of the current device as major * 10 + minor. */
inline auto device_compute_capability() -> int
{
  auto [major, minor] = raft::getComputeCapability();
  return major * 10 + minor;
}

struct print_dtype {
  cudaDataType_t value;
};

inline auto operator<<(std::ostream& os, const print_dtype& p) -> std::ostream&
{
  switch (p.value) {
    case CUDA_R_16F: os << "CUDA_R_16F"; break;
    case CUDA_C_16F: os << "CUDA_C_16F"; break;
    case CUDA_R_16BF: os << "CUDA_R_16BF"; break;
    case CUDA_C_16BF: os << "CUDA_C_16BF"; break;
    case CUDA_R_32F: os << "CUDA_R_32F"; break;
    case CUDA_C_32F: os << "CUDA_C_32F"; break;
    case CUDA_R_64F: os << "CUDA_R_64F"; break;
    case CUDA_C_64F: os << "CUDA_C_64F"; break;
    case CUDA_R_4I: os << "CUDA_R_4I"; break;
    case CUDA_C_4I: os << "CUDA_C_4I"; break;
    case CUDA_R_4U: os << "CUDA_R_4U"; break;
    case CUDA_C_4U: os << "CUDA_C_4U"; break;
    case CUDA_R_8I: os << "CUDA_R_8I"; break;
    case CUDA_C_8I: os << "CUDA_C_8I"; break;
    case CUDA_R_8U: os << "CUDA_R_8U"; break;
    case CUDA_C_8U: os << "CUDA_C_8U"; break;
    case CUDA_R_16I: os << "CUDA_R_16I"; break;
    case CUDA_C_16I: os << "CUDA_C_16I"; break;
    case CUDA_R_16U: os << "CUDA_R_16U"; break;
    case CUDA_C_16U: os << "CUDA_C_16U"; break;
    case CUDA_R_32I: os << "CUDA_R_32I"; break;
    case CUDA_C_32I: os << "CUDA_C_32I"; break;
    case CUDA_R_32U: os << "CUDA_R_32U"; break;
    case CUDA_C_32U: os << "CUDA_C_32U"; break;
    case CUDA_R_64I: os << "CUDA_R_64I"; break;
    case CUDA_C_64I: os << "CUDA_C_64I"; break;
    case CUDA_R_64U: os << "CUDA_R_64U"; break;
    case CUDA_C_64U: os << "CUDA_C_64U"; break;
    default: RAFT_FAIL("unreachable code");
  }
  return os;
}

struct print_metric {
  cuvs::distance::DistanceType value;
};

inline auto operator<<(std::ostream& os, const print_metric& p) -> std::ostream&
{
  switch (p.value) {
    case cuvs::distance::DistanceType::L2Expanded: os << "distance::L2Expanded"; break;
    case cuvs::distance::DistanceType::L2SqrtExpanded: os << "distance::L2SqrtExpanded"; break;
    case cuvs::distance::DistanceType::CosineExpanded: os << "distance::CosineExpanded"; break;
    case cuvs::distance::DistanceType::L1: os << "distance::L1"; break;
    case cuvs::distance::DistanceType::L2Unexpanded: os << "distance::L2Unexpanded"; break;
    case cuvs::distance::DistanceType::L2SqrtUnexpanded: os << "distance::L2SqrtUnexpanded"; break;
    case cuvs::distance::DistanceType::InnerProduct: os << "distance::InnerProduct"; break;
    case cuvs::distance::DistanceType::Linf: os << "distance::Linf"; break;
    case cuvs::distance::DistanceType::Canberra: os << "distance::Canberra"; break;
    case cuvs::distance::DistanceType::LpUnexpanded: os << "distance::LpUnexpanded"; break;
    case cuvs::distance::DistanceType::CorrelationExpanded:
      os << "distance::CorrelationExpanded";
      break;
    case cuvs::distance::DistanceType::JaccardExpanded: os << "distance::JaccardExpanded"; break;
    case cuvs::distance::DistanceType::HellingerExpanded:
      os << "distance::HellingerExpanded";
      break;
    case cuvs::distance::DistanceType::Haversine: os << "distance::Haversine"; break;
    case cuvs::distance::DistanceType::BrayCurtis: os << "distance::BrayCurtis"; break;
    case cuvs::distance::DistanceType::JensenShannon: os << "distance::JensenShannon"; break;
    case cuvs::distance::DistanceType::HammingUnexpanded:
      os << "distance::HammingUnexpanded";
      break;
    case cuvs::distance::DistanceType::KLDivergence: os << "distance::KLDivergence"; break;
    case cuvs::distance::DistanceType::RusselRaoExpanded:
      os << "distance::RusselRaoExpanded";
      break;
    case cuvs::distance::DistanceType::DiceExpanded: os << "distance::DiceExpanded"; break;
    case cuvs::distance::DistanceType::Precomputed: os << "distance::Precomputed"; break;
    case cuvs::distance::DistanceType::BitwiseHamming: os << "distance::BitwiseHamming"; break;
    default: RAFT_FAIL("unreachable code");
  }
  return os;
}

template <typename IdxT, typename DistT, typename CompareDist>
struct idx_dist_pair {
  IdxT idx;
  DistT dist;
  CompareDist eq_compare;
  auto operator==(const idx_dist_pair<IdxT, DistT, CompareDist>& a) const -> bool
  {
    if (idx == a.idx) return true;
    if (eq_compare(dist, a.dist)) return true;
    return false;
  }
  idx_dist_pair(IdxT x, DistT y, CompareDist op) : idx(x), dist(y), eq_compare(op) {}
};

/** Calculate recall value using only neighbor indices
 */
template <typename T>
auto calc_recall(const std::vector<T>& expected_idx,
                 const std::vector<T>& actual_idx,
                 size_t rows,
                 size_t cols)
{
  size_t match_count = 0;
  size_t total_count = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  for (size_t i = 0; i < rows; ++i) {
    for (size_t k = 0; k < cols; ++k) {
      size_t idx_k = i * cols + k;  // row major assumption!
      auto act_idx = actual_idx[idx_k];
      for (size_t j = 0; j < cols; ++j) {
        size_t idx   = i * cols + j;  // row major assumption!
        auto exp_idx = expected_idx[idx];
        if (act_idx == exp_idx) {
          match_count++;
          break;
        }
      }
    }
  }
  return std::make_tuple(
    static_cast<double>(match_count) / static_cast<double>(total_count), match_count, total_count);
}

/** check uniqueness of indices
 */
template <typename T>
auto check_unique_indices(const std::vector<T>& actual_idx,
                          size_t rows,
                          size_t cols,
                          size_t max_duplicates = 0)
{
  size_t max_count;
  size_t dup_count = 0lu;

  std::set<T> unique_indices;
  for (size_t i = 0; i < rows; ++i) {
    unique_indices.clear();
    max_count = 0;
    for (size_t k = 0; k < cols; ++k) {
      size_t idx_k = i * cols + k;  // row major assumption!
      auto act_idx = actual_idx[idx_k];
      if (act_idx == std::numeric_limits<T>::max()) {
        max_count++;
      } else if (unique_indices.find(act_idx) == unique_indices.end()) {
        unique_indices.insert(act_idx);
      } else {
        dup_count++;
        if (dup_count > max_duplicates) {
          return testing::AssertionFailure()
                 << "Duplicated index " << act_idx << " at k " << k << " for query " << i << "! ";
        }
      }
    }
  }
  return testing::AssertionSuccess();
}

template <typename T>
auto eval_recall(const std::vector<T>& expected_idx,
                 const std::vector<T>& actual_idx,
                 size_t rows,
                 size_t cols,
                 double eps,
                 double min_recall,
                 bool test_unique = true) -> testing::AssertionResult
{
  auto [actual_recall, match_count, total_count] =
    calc_recall(expected_idx, actual_idx, rows, cols);
  double error_margin = (actual_recall - min_recall) / std::max(1.0 - min_recall, eps);
  RAFT_LOG_INFO("Recall = %f (%zu/%zu), the error is %2.1f%% %s the threshold (eps = %f).",
                actual_recall,
                match_count,
                total_count,
                std::abs(error_margin * 100.0),
                error_margin < 0 ? "above" : "below",
                eps);
  if (actual_recall < min_recall - eps) {
    return testing::AssertionFailure()
           << "actual recall (" << actual_recall << ") is lower than the minimum expected recall ("
           << min_recall << "); eps = " << eps << ". ";
  }
  if (test_unique)
    return check_unique_indices(actual_idx, rows, cols);
  else
    return testing::AssertionSuccess();
}

/** Overload of calc_recall to account for distances
 */
template <typename T, typename DistT>
auto calc_recall(const std::vector<T>& expected_idx,
                 const std::vector<T>& actual_idx,
                 const std::vector<DistT>& expected_dist,
                 const std::vector<DistT>& actual_dist,
                 size_t rows,
                 size_t cols,
                 double eps)
{
  size_t match_count       = 0;
  size_t index_match_count = 0;
  size_t total_count       = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  for (size_t i = 0; i < rows; ++i) {
    for (size_t k = 0; k < cols; ++k) {
      size_t idx_k  = i * cols + k;  // row major assumption!
      auto act_idx  = actual_idx[idx_k];
      auto act_dist = actual_dist[idx_k];
      for (size_t j = 0; j < cols; ++j) {
        size_t idx    = i * cols + j;  // row major assumption!
        auto exp_idx  = expected_idx[idx];
        auto exp_dist = expected_dist[idx];
        idx_dist_pair exp_kvp(exp_idx, exp_dist, cuvs::CompareApprox<DistT>(eps));
        idx_dist_pair act_kvp(act_idx, act_dist, cuvs::CompareApprox<DistT>(eps));
        if (exp_kvp == act_kvp) {
          match_count++;
          break;
        }
      }
    }
  }

  // Index based recall
  for (size_t i = 0; i < rows; ++i) {
    for (size_t k = 0; k < cols; ++k) {
      size_t idx_k = i * cols + k;  // row major assumption!
      auto act_idx = actual_idx[idx_k];
      for (size_t j = 0; j < cols; ++j) {
        size_t idx   = i * cols + j;  // row major assumption!
        auto exp_idx = expected_idx[idx];

        if (act_idx == exp_idx) {
          index_match_count++;
          break;
        }
      }
    }
  }

  return std::make_tuple(static_cast<double>(match_count) / static_cast<double>(total_count),
                         static_cast<double>(index_match_count) / static_cast<double>(total_count),
                         match_count,
                         total_count);
}

/** same as eval_recall, but in case indices do not match,
 * then check distances as well, and accept match if actual dist is equal to expected_dist */
template <typename T, typename DistT>
auto eval_neighbours(const std::vector<T>& expected_idx,
                     const std::vector<T>& actual_idx,
                     const std::vector<DistT>& expected_dist,
                     const std::vector<DistT>& actual_dist,
                     size_t rows,
                     size_t cols,
                     double eps,
                     double min_recall,
                     bool test_unique      = true,
                     size_t max_duplicates = 0) -> testing::AssertionResult
{
  auto [actual_recall, index_based_actual_recall, match_count, total_count] =
    calc_recall(expected_idx, actual_idx, expected_dist, actual_dist, rows, cols, eps);
  double error_margin = (actual_recall - min_recall) / std::max(1.0 - min_recall, eps);

  RAFT_LOG_INFO("Recall = %f (%zu/%zu), the error is %2.1f%% %s the threshold (eps = %f).",
                actual_recall,
                match_count,
                total_count,
                std::abs(error_margin * 100.0),
                error_margin < 0 ? "above" : "below",
                eps);

  if (actual_recall < min_recall - eps) {
    return testing::AssertionFailure()
           << "actual recall (" << actual_recall << ") is lower than the minimum expected recall ("
           << min_recall << "); eps = " << eps << ". ";
  }
  if (test_unique)
    return check_unique_indices(actual_idx, rows, cols, max_duplicates);
  else
    return testing::AssertionSuccess();
}

template <typename T, typename DistT, typename IdxT>
auto eval_distances(raft::resources const& handle,
                    const T* x,              // dataset, n_rows * n_cols
                    const T* queries,        // n_queries * n_cols
                    const IdxT* neighbors,   // n_queries * k
                    const DistT* distances,  // n_queries *k
                    size_t n_rows,
                    size_t n_cols,
                    size_t n_queries,
                    uint32_t k,
                    cuvs::distance::DistanceType metric,
                    double eps) -> testing::AssertionResult
{
  // for each vector, we calculate the actual distance to the k neighbors
  auto stream     = raft::resource::get_cuda_stream(handle);
  size_t n_dists  = n_queries * k;
  auto naive_dist = raft::make_device_vector<DistT, size_t>(handle, n_dists);
  if (n_dists > 0) {
    constexpr int block_size = 256;
    naive_neighbor_distance_kernel<DistT, T, IdxT>
      <<<raft::ceildiv<size_t>(n_dists, block_size), block_size, 0, stream.get()>>>(
        naive_dist.data_handle(), x, queries, neighbors, n_queries, k, n_cols, metric);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  std::vector<DistT> dist_h(n_dists);
  std::vector<DistT> naive_dist_h(n_dists);
  raft::update_host(dist_h.data(), distances, n_dists, stream);
  raft::update_host(naive_dist_h.data(), naive_dist.data_handle(), n_dists, stream);
  raft::resource::sync_stream(handle, stream);

  CompareApprox<float> eq_compare(eps);
  for (size_t i = 0; i < n_queries; i++) {
    for (size_t j = 0; j < k; j++) {
      if (!eq_compare(dist_h[i * k + j], naive_dist_h[i * k + j])) {
        std::cout << n_rows << "x" << n_cols << ", " << k << std::endl;
        std::cout << "query " << i << std::endl;
        raft::print_vector(" indices", neighbors + i * k, k, std::cout);
        raft::print_vector("n dist", distances + i * k, k, std::cout);
        raft::print_vector("c dist", naive_dist.data_handle() + i * k, k, std::cout);

        return testing::AssertionFailure() << "actual=" << naive_dist_h[i * k + j]
                                           << " != expected=" << dist_h[i * k + j] << " @" << j;
      }
    }
  }
  return testing::AssertionSuccess();
}

/**
 * A helper class to create a temporary file for a cuVS index object in the system's temp directory.
 * The file will be automatically deleted when the object is destroyed.
 */
struct tmp_index_file {
  // Ideally, we should use std::tmpfile() or another system-provided API to create a temporary
  // file. However, our API requires a file name, so we cannot use the file descriptors. There's no
  // recommended way to generate a robust unique temp filenames, so we use a combination of a
  // counter, process id, and random number.
  std::string filename = (std::filesystem::temp_directory_path() /
                          ("cuvs_" + std::to_string(getpid()) + "_" + std::to_string(counter++) +
                           "_" + std::to_string(std::rand())))
                           .string();
  ~tmp_index_file()
  {
    if (std::filesystem::exists(filename)) { std::filesystem::remove(filename); }
  }

 private:
  static inline std::atomic<uint64_t> counter = 0;
};

}  // namespace cuvs::neighbors
