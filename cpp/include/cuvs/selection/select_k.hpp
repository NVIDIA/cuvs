/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/cuda_fp16.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/select_k_types.hpp>

#include <cuvs/core/export.hpp>
#include <optional>

namespace CUVS_EXPORT cuvs {
namespace selection {
using SelectAlgo = raft::matrix::SelectAlgo;

/**
 * @defgroup select_k Batched-select k smallest or largest key/values
 * @{
 */

/**
 * Select k smallest or largest key/values from each row in the input data.
 *
 * If you think of the input data `in_val` as a row-major matrix with `len` columns and
 * `batch_size` rows, then this function selects `k` smallest/largest values in each row and fills
 * in the row-major matrix `out_val` of size (batch_size, k).
 *
 * Example usage
 * @code{.cpp}
 *   using namespace raft;
 *   // get a 2D row-major array of values to search through
 *   auto in_values = {... input device_matrix_view<const float, int64_t, row_major> ...}
 *   // prepare output arrays
 *   auto out_extents = make_extents<int64_t>(in_values.extent(0), k);
 *   auto out_values  = make_device_mdarray<float>(handle, out_extents);
 *   auto out_indices = make_device_mdarray<int64_t>(handle, out_extents);
 *   // search `k` smallest values in each row
 *   cuvs::selection::select_k(
 *     handle, in_values, std::nullopt, out_values.view(), out_indices.view(), true);
 * @endcode
 *
 * @param[in] handle container of reusable resources
 * @param[in] in_val
 *   inputs values [batch_size, len];
 *   these are compared and selected.
 * @param[in] in_idx
 *   optional input payload [batch_size, len];
 *   typically, these are indices of the corresponding `in_val`.
 *   If `in_idx` is `std::nullopt`, a contiguous array `0...len-1` is implied.
 * @param[out] out_val
 *   output values [batch_size, k];
 *   the k smallest/largest values from each row of the `in_val`.
 * @param[out] out_idx
 *   output payload (e.g. indices) [batch_size, k];
 *   the payload selected together with `out_val`.
 * @param[in] select_min
 *   whether to select k smallest (true) or largest (false) keys.
 * @param[in] sorted
 *   whether to make sure selected pairs are sorted by value
 * @param[in] algo
 *   the selection algorithm to use
 * @param[in] len_i
 *  optional array of size (batch_size) providing lengths for each individual row
 */
void select_k(
  raft::resources const& handle,
  raft::device_matrix_view<const float, int64_t, raft::row_major> in_val,
  std::optional<raft::device_matrix_view<const int64_t, int64_t, raft::row_major>> in_idx,
  raft::device_matrix_view<float, int64_t, raft::row_major> out_val,
  raft::device_matrix_view<int64_t, int64_t, raft::row_major> out_idx,
  bool select_min,
  bool sorted                                                           = false,
  SelectAlgo algo                                                       = SelectAlgo::kAuto,
  std::optional<raft::device_vector_view<const int64_t, int64_t>> len_i = std::nullopt);

void select_k(raft::resources const& handle,
              raft::device_matrix_view<const float, int64_t, raft::row_major> in_val,
              std::optional<raft::device_matrix_view<const int, int64_t, raft::row_major>> in_idx,
              raft::device_matrix_view<float, int64_t, raft::row_major> out_val,
              raft::device_matrix_view<int, int64_t, raft::row_major> out_idx,
              bool select_min,
              bool sorted                                                       = false,
              SelectAlgo algo                                                   = SelectAlgo::kAuto,
              std::optional<raft::device_vector_view<const int, int64_t>> len_i = std::nullopt);

/**
 * Select k smallest or largest key/values from each row in the input data.
 *
 * If you think of the input data `in_val` as a row-major matrix with `len` columns and
 * `batch_size` rows, then this function selects `k` smallest/largest values in each row and fills
 * in the row-major matrix `out_val` of size (batch_size, k).
 *
 * Example usage
 * @code{.cpp}
 *   using namespace raft;
 *   // get a 2D row-major array of values to search through
 *   auto in_values = {... input device_matrix_view<const float, int64_t, row_major> ...}
 *   // prepare output arrays
 *   auto out_extents = make_extents<int64_t>(in_values.extent(0), k);
 *   auto out_values  = make_device_mdarray<float>(handle, out_extents);
 *   auto out_indices = make_device_mdarray<uint32_t>(handle, out_extents);
 *   // search `k` smallest values in each row
 *   cuvs::selection::select_k(
 *     handle, in_values, std::nullopt, out_values.view(), out_indices.view(), true);
 * @endcode
 *
 * @param[in] handle container of reusable resources
 * @param[in] in_val
 *   inputs values [batch_size, len];
 *   these are compared and selected.
 * @param[in] in_idx
 *   optional input payload [batch_size, len];
 *   typically, these are indices of the corresponding `in_val`.
 *   If `in_idx` is `std::nullopt`, a contiguous array `0...len-1` is implied.
 * @param[out] out_val
 *   output values [batch_size, k];
 *   the k smallest/largest values from each row of the `in_val`.
 * @param[out] out_idx
 *   output payload (e.g. indices) [batch_size, k];
 *   the payload selected together with `out_val`.
 * @param[in] select_min
 *   whether to select k smallest (true) or largest (false) keys.
 * @param[in] sorted
 *   whether to make sure selected pairs are sorted by value
 * @param[in] algo
 *   the selection algorithm to use
 * @param[in] len_i
 *  optional array of size (batch_size) providing lengths for each individual row
 */
void select_k(
  raft::resources const& handle,
  raft::device_matrix_view<const float, int64_t, raft::row_major> in_val,
  std::optional<raft::device_matrix_view<const uint32_t, int64_t, raft::row_major>> in_idx,
  raft::device_matrix_view<float, int64_t, raft::row_major> out_val,
  raft::device_matrix_view<uint32_t, int64_t, raft::row_major> out_idx,
  bool select_min,
  bool sorted                                                            = false,
  SelectAlgo algo                                                        = SelectAlgo::kAuto,
  std::optional<raft::device_vector_view<const uint32_t, int64_t>> len_i = std::nullopt);

/**
 * Select k smallest or largest key/values from each row in the input data.
 *
 * If you think of the input data `in_val` as a row-major matrix with `len` columns and
 * `batch_size` rows, then this function selects `k` smallest/largest values in each row and fills
 * in the row-major matrix `out_val` of size (batch_size, k).
 *
 * Example usage
 * @code{.cpp}
 *   using namespace raft;
 *   // get a 2D row-major array of values to search through
 *   auto in_values = {... input device_matrix_view<const half, int64_t, row_major> ...}
 *   // prepare output arrays
 *   auto out_extents = make_extents<int64_t>(in_values.extent(0), k);
 *   auto out_values  = make_device_mdarray<half>(handle, out_extents);
 *   auto out_indices = make_device_mdarray<uint32_t>(handle, out_extents);
 *   // search `k` smallest values in each row
 *   cuvs::selection::select_k(
 *     handle, in_values, std::nullopt, out_values.view(), out_indices.view(), true);
 * @endcode
 *
 * @param[in] handle container of reusable resources
 * @param[in] in_val
 *   inputs values [batch_size, len];
 *   these are compared and selected.
 * @param[in] in_idx
 *   optional input payload [batch_size, len];
 *   typically, these are indices of the corresponding `in_val`.
 *   If `in_idx` is `std::nullopt`, a contiguous array `0...len-1` is implied.
 * @param[out] out_val
 *   output values [batch_size, k];
 *   the k smallest/largest values from each row of the `in_val`.
 * @param[out] out_idx
 *   output payload (e.g. indices) [batch_size, k];
 *   the payload selected together with `out_val`.
 * @param[in] select_min
 *   whether to select k smallest (true) or largest (false) keys.
 * @param[in] sorted
 *   whether to make sure selected pairs are sorted by value
 * @param[in] algo
 *   the selection algorithm to use
 * @param[in] len_i
 *  optional array of size (batch_size) providing lengths for each individual row
 */
void select_k(
  raft::resources const& handle,
  raft::device_matrix_view<const half, int64_t, raft::row_major> in_val,
  std::optional<raft::device_matrix_view<const uint32_t, int64_t, raft::row_major>> in_idx,
  raft::device_matrix_view<half, int64_t, raft::row_major> out_val,
  raft::device_matrix_view<uint32_t, int64_t, raft::row_major> out_idx,
  bool select_min,
  bool sorted                                                            = false,
  SelectAlgo algo                                                        = SelectAlgo::kAuto,
  std::optional<raft::device_vector_view<const uint32_t, int64_t>> len_i = std::nullopt);
/** @} */  // end of group select_k

}  // namespace selection
}  // namespace CUVS_EXPORT cuvs
