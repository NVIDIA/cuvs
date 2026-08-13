/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "extern_device_functions.cuh"

#include <raft/core/operators.hpp>

namespace cuvs::neighbors::cagra::detail {

template <typename QUERY_T, typename DISTANCE_T>
__device__ DISTANCE_T dist_op_l2_impl(QUERY_T a, QUERY_T b)
{
  DISTANCE_T diff = a - b;
  return diff * diff;
}

template <typename QUERY_T, typename DISTANCE_T>
__device__ DISTANCE_T dist_op_inner_product_impl(QUERY_T a, QUERY_T b)
{
  return -static_cast<DISTANCE_T>(a) * static_cast<DISTANCE_T>(b);
}

template <typename QUERY_T, typename DISTANCE_T>
__device__ DISTANCE_T dist_op_hamming_impl(QUERY_T a, QUERY_T b)
{
  const auto v = (a ^ b) & 0xffu;
  return __popc(v);
}

template <typename QUERY_T, typename DISTANCE_T>
__device__ DISTANCE_T dist_op_l1_impl(QUERY_T a, QUERY_T b)
{
  DISTANCE_T diff = a - b;
  return raft::abs(diff);
}

}  // namespace cuvs::neighbors::cagra::detail
