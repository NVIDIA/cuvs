/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../core/nvtx.hpp"
#include "../../core/omp_wrapper.hpp"
#include "refine_common.hpp"

#include <raft/core/host_mdspan.hpp>
#include <raft/util/integer_utils.hpp>

#include <algorithm>

#if defined(__arm__) || defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace cuvs::neighbors {

namespace detail {

// -----------------------------------------------------------------------------
//  Generic implementation
// -----------------------------------------------------------------------------

template <typename DC, typename DistanceT, typename DataT>
DistanceT euclidean_distance_squared_generic(DataT const* a, DataT const* b, size_t n)
{
  size_t constexpr max_vreg_len = 512 / (8 * sizeof(DistanceT));

  // max_vreg_len is a power of two
  size_t n_rounded                 = n - (n % max_vreg_len);
  DistanceT distance[max_vreg_len] = {0};

  for (size_t i = 0; i < n_rounded; i += max_vreg_len) {
    for (size_t j = 0; j < max_vreg_len; ++j) {
      distance[j] += DC::template eval<DistanceT>(a[i + j], b[i + j]);
    }
  }

  for (size_t i = n_rounded; i < n; ++i) {
    distance[i - n_rounded] += DC::template eval<DistanceT>(a[i], b[i]);
  }

  for (size_t i = 1; i < max_vreg_len; ++i) {
    distance[0] += distance[i];
  }

  return distance[0];
}

// -----------------------------------------------------------------------------
//  NEON implementation
// -----------------------------------------------------------------------------

struct distance_comp_l2;
struct distance_comp_inner;
struct distance_comp_cosine;

// fallback
template <typename DC, typename DistanceT, typename DataT>
DistanceT euclidean_distance_squared(DataT const* a, DataT const* b, size_t n)
{
  return euclidean_distance_squared_generic<DC, DistanceT, DataT>(a, b, n);
}

#if defined(__arm__) || defined(__aarch64__)

template <>
inline float euclidean_distance_squared<distance_comp_l2, float, float>(float const* a,
                                                                        float const* b,
                                                                        size_t n)
{
  size_t n_rounded = n - (n % 4);

  float32x4_t vreg_dsum = vdupq_n_f32(0.f);
  for (size_t i = 0; i < n_rounded; i += 4) {
    float32x4_t vreg_a = vld1q_f32(&a[i]);
    float32x4_t vreg_b = vld1q_f32(&b[i]);
    float32x4_t vreg_d = vsubq_f32(vreg_a, vreg_b);
    vreg_dsum          = vfmaq_f32(vreg_dsum, vreg_d, vreg_d);
  }

  float dsum = vaddvq_f32(vreg_dsum);
  for (size_t i = n_rounded; i < n; ++i) {
    float d = a[i] - b[i];
    dsum += d * d;
  }

  return dsum;
}

template <>
inline float euclidean_distance_squared<distance_comp_l2, float, ::std::int8_t>(
  ::std::int8_t const* a, ::std::int8_t const* b, size_t n)
{
  size_t n_rounded = n - (n % 16);
  float dsum       = 0.f;

  if (n_rounded > 0) {
    float32x4_t vreg_dsum_fp32_0 = vdupq_n_f32(0.f);
    float32x4_t vreg_dsum_fp32_1 = vreg_dsum_fp32_0;
    float32x4_t vreg_dsum_fp32_2 = vreg_dsum_fp32_0;
    float32x4_t vreg_dsum_fp32_3 = vreg_dsum_fp32_0;

    for (size_t i = 0; i < n_rounded; i += 16) {
      int8x16_t vreg_a       = vld1q_s8(&a[i]);
      int16x8_t vreg_a_s16_0 = vmovl_s8(vget_low_s8(vreg_a));
      int16x8_t vreg_a_s16_1 = vmovl_s8(vget_high_s8(vreg_a));

      int8x16_t vreg_b       = vld1q_s8(&b[i]);
      int16x8_t vreg_b_s16_0 = vmovl_s8(vget_low_s8(vreg_b));
      int16x8_t vreg_b_s16_1 = vmovl_s8(vget_high_s8(vreg_b));

      int16x8_t vreg_d_s16_0 = vsubq_s16(vreg_a_s16_0, vreg_b_s16_0);
      int16x8_t vreg_d_s16_1 = vsubq_s16(vreg_a_s16_1, vreg_b_s16_1);

      float32x4_t vreg_d_fp32_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vreg_d_s16_0)));
      float32x4_t vreg_d_fp32_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vreg_d_s16_0)));
      float32x4_t vreg_d_fp32_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vreg_d_s16_1)));
      float32x4_t vreg_d_fp32_3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vreg_d_s16_1)));

      vreg_dsum_fp32_0 = vfmaq_f32(vreg_dsum_fp32_0, vreg_d_fp32_0, vreg_d_fp32_0);
      vreg_dsum_fp32_1 = vfmaq_f32(vreg_dsum_fp32_1, vreg_d_fp32_1, vreg_d_fp32_1);
      vreg_dsum_fp32_2 = vfmaq_f32(vreg_dsum_fp32_2, vreg_d_fp32_2, vreg_d_fp32_2);
      vreg_dsum_fp32_3 = vfmaq_f32(vreg_dsum_fp32_3, vreg_d_fp32_3, vreg_d_fp32_3);
    }

    vreg_dsum_fp32_0 = vaddq_f32(vreg_dsum_fp32_0, vreg_dsum_fp32_1);
    vreg_dsum_fp32_2 = vaddq_f32(vreg_dsum_fp32_2, vreg_dsum_fp32_3);
    vreg_dsum_fp32_0 = vaddq_f32(vreg_dsum_fp32_0, vreg_dsum_fp32_2);

    dsum = vaddvq_f32(vreg_dsum_fp32_0);  // faddp
  }

  for (size_t i = n_rounded; i < n; ++i) {
    float d = a[i] - b[i];
    dsum += d * d;  // [nvc++] faddp, [clang] fadda, [gcc] vecsum+fadda
  }

  return dsum;
}

template <>
inline float euclidean_distance_squared<distance_comp_l2, float, ::std::uint8_t>(
  ::std::uint8_t const* a, ::std::uint8_t const* b, size_t n)
{
  size_t n_rounded = n - (n % 16);
  float dsum       = 0.f;

  if (n_rounded > 0) {
    float32x4_t vreg_dsum_fp32_0 = vdupq_n_f32(0.f);
    float32x4_t vreg_dsum_fp32_1 = vreg_dsum_fp32_0;
    float32x4_t vreg_dsum_fp32_2 = vreg_dsum_fp32_0;
    float32x4_t vreg_dsum_fp32_3 = vreg_dsum_fp32_0;

    for (size_t i = 0; i < n_rounded; i += 16) {
      uint8x16_t vreg_a         = vld1q_u8(&a[i]);
      uint16x8_t vreg_a_u16_0   = vmovl_u8(vget_low_u8(vreg_a));
      uint16x8_t vreg_a_u16_1   = vmovl_u8(vget_high_u8(vreg_a));
      float32x4_t vreg_a_fp32_0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vreg_a_u16_0)));
      float32x4_t vreg_a_fp32_1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vreg_a_u16_0)));
      float32x4_t vreg_a_fp32_2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vreg_a_u16_1)));
      float32x4_t vreg_a_fp32_3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vreg_a_u16_1)));

      uint8x16_t vreg_b         = vld1q_u8(&b[i]);
      uint16x8_t vreg_b_u16_0   = vmovl_u8(vget_low_u8(vreg_b));
      uint16x8_t vreg_b_u16_1   = vmovl_u8(vget_high_u8(vreg_b));
      float32x4_t vreg_b_fp32_0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vreg_b_u16_0)));
      float32x4_t vreg_b_fp32_1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vreg_b_u16_0)));
      float32x4_t vreg_b_fp32_2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vreg_b_u16_1)));
      float32x4_t vreg_b_fp32_3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vreg_b_u16_1)));

      float32x4_t vreg_d_fp32_0 = vsubq_f32(vreg_a_fp32_0, vreg_b_fp32_0);
      float32x4_t vreg_d_fp32_1 = vsubq_f32(vreg_a_fp32_1, vreg_b_fp32_1);
      float32x4_t vreg_d_fp32_2 = vsubq_f32(vreg_a_fp32_2, vreg_b_fp32_2);
      float32x4_t vreg_d_fp32_3 = vsubq_f32(vreg_a_fp32_3, vreg_b_fp32_3);

      vreg_dsum_fp32_0 = vfmaq_f32(vreg_dsum_fp32_0, vreg_d_fp32_0, vreg_d_fp32_0);
      vreg_dsum_fp32_1 = vfmaq_f32(vreg_dsum_fp32_1, vreg_d_fp32_1, vreg_d_fp32_1);
      vreg_dsum_fp32_2 = vfmaq_f32(vreg_dsum_fp32_2, vreg_d_fp32_2, vreg_d_fp32_2);
      vreg_dsum_fp32_3 = vfmaq_f32(vreg_dsum_fp32_3, vreg_d_fp32_3, vreg_d_fp32_3);
    }

    vreg_dsum_fp32_0 = vaddq_f32(vreg_dsum_fp32_0, vreg_dsum_fp32_1);
    vreg_dsum_fp32_2 = vaddq_f32(vreg_dsum_fp32_2, vreg_dsum_fp32_3);
    vreg_dsum_fp32_0 = vaddq_f32(vreg_dsum_fp32_0, vreg_dsum_fp32_2);

    dsum = vaddvq_f32(vreg_dsum_fp32_0);  // faddp
  }

  for (size_t i = n_rounded; i < n; ++i) {
    float d = a[i] - b[i];
    dsum += d * d;  // [nvc++] faddp, [clang] fadda, [gcc] vecsum+fadda
  }

  return dsum;
}

template <>
inline float euclidean_distance_squared<distance_comp_inner, float, float>(float const* a,
                                                                           float const* b,
                                                                           size_t n)
{
  size_t n_rounded = n - (n % 4);

  float32x4_t vreg_dsum = vdupq_n_f32(0.f);
  for (size_t i = 0; i < n_rounded; i += 4) {
    float32x4_t vreg_a = vld1q_f32(&a[i]);
    float32x4_t vreg_b = vld1q_f32(&b[i]);
    vreg_a             = vnegq_f32(vreg_a);
    vreg_dsum          = vfmaq_f32(vreg_dsum, vreg_a, vreg_b);
  }

  float dsum = vaddvq_f32(vreg_dsum);
  for (size_t i = n_rounded; i < n; ++i) {
    dsum += -a[i] * b[i];
  }

  return dsum;
}

template <>
inline float euclidean_distance_squared<distance_comp_inner, float, ::std::int8_t>(
  ::std::int8_t const* a, ::std::int8_t const* b, size_t n)
{
  size_t n_rounded = n - (n % 16);
  float dsum       = 0.f;

  if (n_rounded > 0) {
    float32x4_t vreg_dsum_fp32_0 = vdupq_n_f32(0.f);
    float32x4_t vreg_dsum_fp32_1 = vreg_dsum_fp32_0;
    float32x4_t vreg_dsum_fp32_2 = vreg_dsum_fp32_0;
    float32x4_t vreg_dsum_fp32_3 = vreg_dsum_fp32_0;

    for (size_t i = 0; i < n_rounded; i += 16) {
      int8x16_t vreg_a       = vld1q_s8(&a[i]);
      int16x8_t vreg_a_s16_0 = vmovl_s8(vget_low_s8(vreg_a));
      int16x8_t vreg_a_s16_1 = vmovl_s8(vget_high_s8(vreg_a));

      int8x16_t vreg_b       = vld1q_s8(&b[i]);
      int16x8_t vreg_b_s16_0 = vmovl_s8(vget_low_s8(vreg_b));
      int16x8_t vreg_b_s16_1 = vmovl_s8(vget_high_s8(vreg_b));

      vreg_a_s16_0 = vmulq_s16(vreg_a_s16_0, vreg_b_s16_0);
      vreg_a_s16_1 = vmulq_s16(vreg_a_s16_1, vreg_b_s16_1);

      float32x4_t vreg_res_fp32_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vreg_a_s16_0)));
      float32x4_t vreg_res_fp32_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vreg_a_s16_0)));
      float32x4_t vreg_res_fp32_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vreg_a_s16_1)));
      float32x4_t vreg_res_fp32_3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vreg_a_s16_1)));

      vreg_dsum_fp32_0 = vsubq_f32(vreg_dsum_fp32_0, vreg_res_fp32_0);
      vreg_dsum_fp32_1 = vsubq_f32(vreg_dsum_fp32_1, vreg_res_fp32_1);
      vreg_dsum_fp32_2 = vsubq_f32(vreg_dsum_fp32_2, vreg_res_fp32_2);
      vreg_dsum_fp32_3 = vsubq_f32(vreg_dsum_fp32_3, vreg_res_fp32_3);
    }

    vreg_dsum_fp32_0 = vaddq_f32(vreg_dsum_fp32_0, vreg_dsum_fp32_1);
    vreg_dsum_fp32_2 = vaddq_f32(vreg_dsum_fp32_2, vreg_dsum_fp32_3);
    vreg_dsum_fp32_0 = vaddq_f32(vreg_dsum_fp32_0, vreg_dsum_fp32_2);

    dsum = vaddvq_f32(vreg_dsum_fp32_0);  // faddp
  }

  for (size_t i = n_rounded; i < n; ++i) {
    dsum += -a[i] * b[i];
  }

  return dsum;
}

template <>
inline float euclidean_distance_squared<distance_comp_inner, float, ::std::uint8_t>(
  ::std::uint8_t const* a, ::std::uint8_t const* b, size_t n)
{
  size_t n_rounded = n - (n % 16);
  float dsum       = 0.f;

  if (n_rounded > 0) {
    float32x4_t vreg_dsum_fp32_0 = vdupq_n_f32(0.f);
    float32x4_t vreg_dsum_fp32_1 = vreg_dsum_fp32_0;
    float32x4_t vreg_dsum_fp32_2 = vreg_dsum_fp32_0;
    float32x4_t vreg_dsum_fp32_3 = vreg_dsum_fp32_0;

    for (size_t i = 0; i < n_rounded; i += 16) {
      uint8x16_t vreg_a       = vld1q_u8(&a[i]);
      uint16x8_t vreg_a_u16_0 = vmovl_u8(vget_low_u8(vreg_a));
      uint16x8_t vreg_a_u16_1 = vmovl_u8(vget_high_u8(vreg_a));

      uint8x16_t vreg_b       = vld1q_u8(&b[i]);
      uint16x8_t vreg_b_u16_0 = vmovl_u8(vget_low_u8(vreg_b));
      uint16x8_t vreg_b_u16_1 = vmovl_u8(vget_high_u8(vreg_b));

      vreg_a_u16_0 = vmulq_u16(vreg_a_u16_0, vreg_b_u16_0);
      vreg_a_u16_1 = vmulq_u16(vreg_a_u16_1, vreg_b_u16_1);

      float32x4_t vreg_res_fp32_0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vreg_a_u16_0)));
      float32x4_t vreg_res_fp32_1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vreg_a_u16_0)));
      float32x4_t vreg_res_fp32_2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vreg_a_u16_1)));
      float32x4_t vreg_res_fp32_3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vreg_a_u16_1)));

      vreg_dsum_fp32_0 = vsubq_f32(vreg_dsum_fp32_0, vreg_res_fp32_0);
      vreg_dsum_fp32_1 = vsubq_f32(vreg_dsum_fp32_1, vreg_res_fp32_1);
      vreg_dsum_fp32_2 = vsubq_f32(vreg_dsum_fp32_2, vreg_res_fp32_2);
      vreg_dsum_fp32_3 = vsubq_f32(vreg_dsum_fp32_3, vreg_res_fp32_3);
    }

    vreg_dsum_fp32_0 = vaddq_f32(vreg_dsum_fp32_0, vreg_dsum_fp32_1);
    vreg_dsum_fp32_2 = vaddq_f32(vreg_dsum_fp32_2, vreg_dsum_fp32_3);
    vreg_dsum_fp32_0 = vaddq_f32(vreg_dsum_fp32_0, vreg_dsum_fp32_2);

    dsum = vaddvq_f32(vreg_dsum_fp32_0);  // faddp
  }

  for (size_t i = n_rounded; i < n; ++i) {
    dsum += -a[i] * b[i];
  }

  return dsum;
}

#endif  // defined(__arm__) || defined(__aarch64__)

// -----------------------------------------------------------------------------
//  Refine kernel
// -----------------------------------------------------------------------------

// Cosine distance: 1 - (a·b)/(||a||·||b||)
template <typename DistanceT, typename DataT>
inline DistanceT cosine_distance(DataT const* a, DataT const* b, size_t n)
{
  using AccT = double;
  AccT dot   = AccT{0};
  AccT na2   = AccT{0};
  AccT nb2   = AccT{0};
  for (size_t i = 0; i < n; ++i) {
    AccT va = static_cast<AccT>(a[i]);
    AccT vb = static_cast<AccT>(b[i]);
    dot += va * vb;
    na2 += va * va;
    nb2 += vb * vb;
  }
  AccT denom = std::sqrt(na2) * std::sqrt(nb2);
  AccT dist  = denom > AccT{0} ? AccT{1} - (dot / denom) : AccT{1};
  return static_cast<DistanceT>(dist);
}

template <typename DC, typename IdxT, typename DataT, typename DistanceT, typename ExtentsT>
[[gnu::optimize(3), gnu::optimize("tree-vectorize")]] void refine_host_impl(
  raft::host_matrix_view<const DataT, ExtentsT, raft::row_major> dataset,
  raft::host_matrix_view<const DataT, ExtentsT, raft::row_major> queries,
  raft::host_matrix_view<const IdxT, ExtentsT, raft::row_major> neighbor_candidates,
  raft::host_matrix_view<IdxT, ExtentsT, raft::row_major> indices,
  raft::host_matrix_view<DistanceT, ExtentsT, raft::row_major> distances)
{
  size_t n_queries = queries.extent(0);
  size_t n_rows    = dataset.extent(0);
  size_t dim       = dataset.extent(1);
  size_t orig_k    = neighbor_candidates.extent(1);
  size_t refined_k = indices.extent(1);

  cuvs::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "neighbors::refine_host(%zu, %zu -> %zu)", n_queries, orig_k, refined_k);

  auto suggested_n_threads =
    std::max(1, std::min(cuvs::core::omp::get_num_procs(), cuvs::core::omp::get_max_threads()));

  // If the number of queries is small, separate the distance calculation and
  // the top-k calculation into separate loops, and apply finer-grained thread
  // parallelism to the distance calculation loop.
  if (n_queries < size_t(suggested_n_threads)) {
    std::vector<std::vector<std::tuple<DistanceT, IdxT>>> refined_pairs(
      n_queries, std::vector<std::tuple<DistanceT, IdxT>>(orig_k));

    // For efficiency, each thread should read a certain amount of array
    // elements. The number of threads for distance computation is determined
    // taking this into account.
    auto n_elements    = std::max(size_t(512), dim);
    auto max_n_threads = raft::div_rounding_up_safe<size_t>(n_queries * orig_k * dim, n_elements);
    [[maybe_unused]] auto suggested_n_threads_for_distance =
      std::min(size_t(suggested_n_threads), max_n_threads);

    // The max number of threads for topk computation is the number of queries.
    [[maybe_unused]] auto suggested_n_threads_for_topk =
      std::min(size_t(suggested_n_threads), n_queries);

    // Compute the refined distance using original dataset vectors
#pragma omp parallel for collapse(2) num_threads(suggested_n_threads_for_distance)
    for (size_t i = 0; i < n_queries; i++) {
      for (size_t j = 0; j < orig_k; j++) {
        const DataT* query = queries.data_handle() + dim * i;
        IdxT id            = neighbor_candidates(i, j);
        DistanceT distance = 0.0;
        if (static_cast<size_t>(id) >= n_rows) {
          distance = std::numeric_limits<DistanceT>::max();
        } else {
          const DataT* row = dataset.data_handle() + dim * id;
          for (size_t k = 0; k < dim; k++) {
            distance += DC::template eval<DistanceT>(query[k], row[k]);
          }
        }
        refined_pairs[i][j] = std::make_tuple(distance, id);
      }
    }

    // Sort the query neighbors by their refined distances
#pragma omp parallel for num_threads(suggested_n_threads_for_topk)
    for (size_t i = 0; i < n_queries; i++) {
      std::sort(refined_pairs[i].begin(), refined_pairs[i].end());
      // Store first refined_k neighbors
      for (size_t j = 0; j < refined_k; j++) {
        indices(i, j) = std::get<1>(refined_pairs[i][j]);
        if (distances.data_handle() != nullptr) {
          distances(i, j) = DC::template postprocess<DistanceT>(std::get<0>(refined_pairs[i][j]));
        }
      }
    }
    return;
  }

  if (size_t(suggested_n_threads) > n_queries) { suggested_n_threads = n_queries; }

  {
    std::vector<std::vector<std::tuple<DistanceT, IdxT>>> refined_pairs(
      suggested_n_threads, std::vector<std::tuple<DistanceT, IdxT>>(orig_k));
#pragma omp parallel num_threads(suggested_n_threads)
    {
      auto tid = cuvs::core::omp::get_thread_num();
      for (size_t i = tid; i < n_queries; i += cuvs::core::omp::get_num_threads()) {
        // Compute the refined distance using original dataset vectors
        const DataT* query = queries.data_handle() + dim * i;
        for (size_t j = 0; j < orig_k; j++) {
          IdxT id            = neighbor_candidates(i, j);
          DistanceT distance = 0.0;
          if (static_cast<size_t>(id) >= n_rows) {
            distance = std::numeric_limits<DistanceT>::max();
          } else {
            const DataT* row = dataset.data_handle() + dim * id;
            if constexpr (std::is_same_v<DC, distance_comp_cosine>) {
              distance = cosine_distance<DistanceT, DataT>(query, row, dim);
            } else {
              distance = euclidean_distance_squared<DC, DistanceT, DataT>(query, row, dim);
            }
          }
          refined_pairs[tid][j] = std::make_tuple(distance, id);
        }
        // Sort the query neighbors by their refined distances
        std::sort(refined_pairs[tid].begin(), refined_pairs[tid].end());
        // Store first refined_k neighbors
        for (size_t j = 0; j < refined_k; j++) {
          indices(i, j) = std::get<1>(refined_pairs[tid][j]);
          if (distances.data_handle() != nullptr) {
            distances(i, j) =
              DC::template postprocess<DistanceT>(std::get<0>(refined_pairs[tid][j]));
          }
        }
      }
    }
  }
}

struct distance_comp_l2 {
  template <typename DistanceT>
  static inline auto eval(const DistanceT& a, const DistanceT& b) -> DistanceT
  {
    auto d = a - b;
    return d * d;
  }
  template <typename DistanceT>
  static inline auto postprocess(const DistanceT& a) -> DistanceT
  {
    return a;
  }
};

struct distance_comp_inner {
  template <typename DistanceT>
  static inline auto eval(const DistanceT& a, const DistanceT& b) -> DistanceT
  {
    return -a * b;
  }
  template <typename DistanceT>
  static inline auto postprocess(const DistanceT& a) -> DistanceT
  {
    return -a;
  }
};

struct distance_comp_cosine {
  template <typename DistanceT>
  static inline auto eval(const DistanceT&, const DistanceT&) -> DistanceT
  {
    return DistanceT{0};
  }
  template <typename DistanceT>
  static inline auto postprocess(const DistanceT& a) -> DistanceT
  {
    return a;
  }
};

/**
 * Naive CPU implementation of refine operation
 *
 * All pointers are expected to be accessible on the host.
 */
template <typename IdxT, typename DataT, typename DistanceT, typename ExtentsT>
[[gnu::optimize(3), gnu::optimize("tree-vectorize")]] void refine_host(
  raft::host_matrix_view<const DataT, ExtentsT, raft::row_major> dataset,
  raft::host_matrix_view<const DataT, ExtentsT, raft::row_major> queries,
  raft::host_matrix_view<const IdxT, ExtentsT, raft::row_major> neighbor_candidates,
  raft::host_matrix_view<IdxT, ExtentsT, raft::row_major> indices,
  raft::host_matrix_view<DistanceT, ExtentsT, raft::row_major> distances,
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded)
{
  refine_check_input(dataset.extents(),
                     queries.extents(),
                     neighbor_candidates.extents(),
                     indices.extents(),
                     distances.extents(),
                     metric);

  switch (metric) {
    case cuvs::distance::DistanceType::L2Expanded:
      return refine_host_impl<distance_comp_l2>(
        dataset, queries, neighbor_candidates, indices, distances);
    case cuvs::distance::DistanceType::InnerProduct:
      return refine_host_impl<distance_comp_inner>(
        dataset, queries, neighbor_candidates, indices, distances);
    case cuvs::distance::DistanceType::CosineExpanded:
      return refine_host_impl<distance_comp_cosine>(
        dataset, queries, neighbor_candidates, indices, distances);
    default: throw raft::logic_error("Unsupported metric");
  }
}

}  // namespace detail

template <typename idx_t, typename data_t, typename distance_t, typename matrix_idx>
void refine_impl(
  raft::resources const& handle,
  raft::host_matrix_view<const data_t, matrix_idx, raft::row_major> dataset,
  raft::host_matrix_view<const data_t, matrix_idx, raft::row_major> queries,
  raft::host_matrix_view<const idx_t, matrix_idx, raft::row_major> neighbor_candidates,
  raft::host_matrix_view<idx_t, matrix_idx, raft::row_major> indices,
  raft::host_matrix_view<distance_t, matrix_idx, raft::row_major> distances,
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded)
{
  detail::refine_host(dataset, queries, neighbor_candidates, indices, distances, metric);
}

}  // namespace cuvs::neighbors
