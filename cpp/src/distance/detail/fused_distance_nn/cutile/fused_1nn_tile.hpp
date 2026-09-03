/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <cuda_runtime.h>

#include <cuvs/detail/jit_lto/tileir_compat.hpp>
#include <cuvs/distance/distance.hpp>

#ifndef CUVS_CUTILE_ENABLED
#define CUVS_CUTILE_ENABLED 0
#endif

namespace cuvs {
namespace distance {
namespace detail {

template <typename DataT>
inline constexpr bool is_fused_1nn_cutile_data_v =
  std::is_same_v<DataT, float> || std::is_same_v<DataT, half>;

// Tensor-core products accumulate in FP32; FP16 norms must remain FP32 through the epilogue.
template <typename DataT>
using fused_1nn_cutile_norm_t = std::conditional_t<std::is_same_v<DataT, half>, float, DataT>;

template <typename DataT>
inline constexpr int64_t fused_1nn_cutile_max_batch_m = [] {
  constexpr int64_t max_i32         = std::numeric_limits<int>::max();
  constexpr int64_t batch_alignment = 16 / sizeof(DataT);
  return max_i32 - max_i32 % batch_alignment;
}();

template <typename DataT, typename IdxT>
constexpr size_t fused_1nn_cutile_index_workspace_rows(IdxT m)
{
  const auto rows = static_cast<int64_t>(m);
  if (rows <= 0) { return 0; }
  return static_cast<size_t>(
    rows < fused_1nn_cutile_max_batch_m<DataT> ? rows : fused_1nn_cutile_max_batch_m<DataT>);
}

#if CUVS_CUTILE_ENABLED
/**
 * Return whether the input problem has a compatible cuTile launcher.
 *
 * This output-independent probe lets callers select native result storage before allocating it.
 */
template <typename DataT, typename IdxT>
  requires is_fused_1nn_cutile_data_v<DataT>
bool can_launch_fused_1nn_tile(
  const DataT* x, const DataT* y, IdxT m, IdxT n, IdxT k, cuvs::distance::DistanceType metric);

/**
 * Return whether the supplied problem can use cuTile without fallback scratch.
 *
 * The result includes runtime/device support, exported ABI constraints, and launcher construction.
 * A successful probe populates the shared launcher cache used by try_fused_1nn_tile.
 * An int64 output index still requires an int32 workspace sized to the largest launch chunk.
 */
template <typename DataT, typename IdxT>
  requires is_fused_1nn_cutile_data_v<DataT>
bool can_launch_fused_1nn_tile(IdxT* nearest_idx,
                               DataT* nearest_dist,
                               const DataT* x,
                               const DataT* y,
                               IdxT m,
                               IdxT n,
                               IdxT k,
                               cuvs::distance::DistanceType metric);

/**
 * Return whether the supplied problem and existing norm buffers can use cuTile.
 *
 * The overload without norm pointers is a preflight probe for callers that allocate aligned norm
 * buffers only after the remaining launch requirements have been validated.
 */
template <typename DataT, typename IdxT>
  requires is_fused_1nn_cutile_data_v<DataT>
bool can_launch_fused_1nn_tile(IdxT* nearest_idx,
                               DataT* nearest_dist,
                               const DataT* x,
                               const DataT* y,
                               const fused_1nn_cutile_norm_t<DataT>* xn,
                               const fused_1nn_cutile_norm_t<DataT>* yn,
                               IdxT m,
                               IdxT n,
                               IdxT k,
                               cuvs::distance::DistanceType metric);

template <typename DataT, typename IdxT>
  requires is_fused_1nn_cutile_data_v<DataT>
bool try_fused_1nn_tile(IdxT* nearest_idx,
                        DataT* nearest_dist,
                        const DataT* x,
                        const DataT* y,
                        const fused_1nn_cutile_norm_t<DataT>* xn,
                        const fused_1nn_cutile_norm_t<DataT>* yn,
                        IdxT m,
                        IdxT n,
                        IdxT k,
                        cuvs::distance::DistanceType metric,
                        bool is_sqrt,
                        void* index_workspace,
                        cudaStream_t stream);
#else
template <typename DataT, typename IdxT>
bool can_launch_fused_1nn_tile(
  const DataT*, const DataT*, IdxT, IdxT, IdxT, cuvs::distance::DistanceType)
{
  return false;
}

template <typename DataT, typename IdxT>
bool can_launch_fused_1nn_tile(
  IdxT*, DataT*, const DataT*, const DataT*, IdxT, IdxT, IdxT, cuvs::distance::DistanceType)
{
  return false;
}

template <typename DataT, typename IdxT>
bool can_launch_fused_1nn_tile(IdxT*,
                               DataT*,
                               const DataT*,
                               const DataT*,
                               const fused_1nn_cutile_norm_t<DataT>*,
                               const fused_1nn_cutile_norm_t<DataT>*,
                               IdxT,
                               IdxT,
                               IdxT,
                               cuvs::distance::DistanceType)
{
  return false;
}

template <typename DataT, typename IdxT>
bool try_fused_1nn_tile(IdxT*,
                        DataT*,
                        const DataT*,
                        const DataT*,
                        const fused_1nn_cutile_norm_t<DataT>*,
                        const fused_1nn_cutile_norm_t<DataT>*,
                        IdxT,
                        IdxT,
                        IdxT,
                        cuvs::distance::DistanceType,
                        bool,
                        void*,
                        cudaStream_t)
{
  return false;
}
#endif

}  // namespace detail
}  // namespace distance
}  // namespace cuvs
