/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

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

#if CUVS_CUTILE_ENABLED
template <typename DataT, typename IdxT>
  requires is_fused_1nn_cutile_data_v<DataT>
bool try_fused_1nn_tile(IdxT* nearest_idx,
                        DataT* nearest_dist,
                        const DataT* x,
                        const DataT* y,
                        const DataT* xn,
                        const DataT* yn,
                        IdxT m,
                        IdxT n,
                        IdxT k,
                        cuvs::distance::DistanceType metric,
                        bool is_sqrt,
                        void* index_workspace,
                        cudaStream_t stream);
#else
template <typename DataT, typename IdxT>
bool try_fused_1nn_tile(IdxT*,
                        DataT*,
                        const DataT*,
                        const DataT*,
                        const DataT*,
                        const DataT*,
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
