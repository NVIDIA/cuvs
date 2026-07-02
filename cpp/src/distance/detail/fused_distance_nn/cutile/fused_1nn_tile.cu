/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "fused_1nn_tile.hpp"

#include "fused_1nn_planner.hpp"

#include <cuvs/core/export.hpp>
#include <cuvs/detail/jit_lto/fused_distance_nn/fused_1nn_fragments.hpp>
#include <raft/util/cuda_utils.cuh>

#include <type_traits>

namespace cuvs {
namespace distance {
namespace detail {

namespace {

template <cuvs::distance::DistanceType Metric, typename DataT, typename IdxT>
bool launch_fused_1nn_tile(IdxT* nearest_idx,
                           DataT* nearest_dist,
                           const DataT* x,
                           const DataT* y,
                           const DataT* xn,
                           const DataT* yn,
                           IdxT m,
                           IdxT n,
                           IdxT k,
                           bool is_sqrt,
                           cudaStream_t stream)
{
  if constexpr (!std::is_same_v<DataT, float> && !std::is_same_v<DataT, half>) { return false; }

  if (nearest_dist == nullptr) { return false; }

  Fused1nnTilePlanner<DataT, Metric, IdxT> planner;
  planner.add_entrypoint();
  planner.add_tileir_fallback();
  const CutileTileConfig tile_cfg = planner.tile_config();
  auto launcher                   = planner.try_get_launcher();
  if (!launcher) { return false; }

  const bool apply_sqrt = fused_1nn_apply_sqrt_at_pack<Metric>(is_sqrt);

  IdxT shape_x[2]  = {m, k};
  IdxT stride_x[2] = {k, IdxT{1}};
  IdxT shape_y[2]  = {n, k};
  IdxT stride_y[2] = {k, IdxT{1}};
  IdxT shape_xn    = m;
  IdxT stride_xn   = IdxT{1};
  IdxT shape_yn    = n;
  IdxT stride_yn   = IdxT{1};
  IdxT shape_idx   = m;
  IdxT stride_idx  = IdxT{1};
  IdxT shape_dist  = m;
  IdxT stride_dist = IdxT{1};

  IdxT M = m;
  IdxT N = n;
  IdxT K = k;

  void* x_ptr  = const_cast<DataT*>(x);
  void* y_ptr  = const_cast<DataT*>(y);
  void* xn_ptr = const_cast<DataT*>(xn);
  void* yn_ptr = const_cast<DataT*>(yn);
  // OutIdx must be a valid device pointer for the launch ABI; when store_idx is 0 the kernel
  // does not write it (dist-only callers pass nearest_dist as a stand-in).
  const IdxT store_idx = nearest_idx != nullptr ? IdxT{1} : IdxT{0};
  void* idx_ptr =
    nearest_idx != nullptr ? static_cast<void*>(nearest_idx) : static_cast<void*>(nearest_dist);
  void* dist_ptr = nearest_dist;

  const int tile_m = tile_cfg.tile_m;
  dim3 grid((m + tile_m - 1) / tile_m, 1, 1);
  dim3 block(1, 1, 1);

  using fused_1nn_cutile_kernel_t = void(void*,
                                         IdxT,
                                         IdxT,
                                         IdxT,
                                         IdxT,
                                         void*,
                                         IdxT,
                                         IdxT,
                                         IdxT,
                                         IdxT,
                                         void*,
                                         IdxT,
                                         IdxT,
                                         void*,
                                         IdxT,
                                         IdxT,
                                         void*,
                                         IdxT,
                                         IdxT,
                                         void*,
                                         IdxT,
                                         IdxT,
                                         IdxT,
                                         IdxT,
                                         IdxT,
                                         IdxT,
                                         IdxT);
  launcher->template dispatch<fused_1nn_cutile_kernel_t>(stream,
                                                         grid,
                                                         block,
                                                         0,
                                                         x_ptr,
                                                         shape_x[0],
                                                         shape_x[1],
                                                         stride_x[0],
                                                         stride_x[1],
                                                         y_ptr,
                                                         shape_y[0],
                                                         shape_y[1],
                                                         stride_y[0],
                                                         stride_y[1],
                                                         xn_ptr,
                                                         shape_xn,
                                                         stride_xn,
                                                         yn_ptr,
                                                         shape_yn,
                                                         stride_yn,
                                                         idx_ptr,
                                                         shape_idx,
                                                         stride_idx,
                                                         dist_ptr,
                                                         shape_dist,
                                                         stride_dist,
                                                         M,
                                                         N,
                                                         K,
                                                         static_cast<IdxT>(apply_sqrt),
                                                         store_idx);
  RAFT_CUDA_TRY(cudaGetLastError());
  return true;
}

template <typename DataT, typename IdxT>
bool try_fused_1nn_tile_dispatch(IdxT* nearest_idx,
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
                                 cudaStream_t stream)
{
  switch (metric) {
    case cuvs::distance::DistanceType::InnerProduct:
      return launch_fused_1nn_tile<cuvs::distance::DistanceType::InnerProduct, DataT, IdxT>(
        nearest_idx, nearest_dist, x, y, xn, yn, m, n, k, is_sqrt, stream);
    case cuvs::distance::DistanceType::L2Expanded:
      return launch_fused_1nn_tile<cuvs::distance::DistanceType::L2Expanded, DataT, IdxT>(
        nearest_idx, nearest_dist, x, y, xn, yn, m, n, k, is_sqrt, stream);
    case cuvs::distance::DistanceType::L2SqrtExpanded:
      return launch_fused_1nn_tile<cuvs::distance::DistanceType::L2SqrtExpanded, DataT, IdxT>(
        nearest_idx, nearest_dist, x, y, xn, yn, m, n, k, is_sqrt, stream);
    case cuvs::distance::DistanceType::CosineExpanded:
      return launch_fused_1nn_tile<cuvs::distance::DistanceType::CosineExpanded, DataT, IdxT>(
        nearest_idx, nearest_dist, x, y, xn, yn, m, n, k, is_sqrt, stream);
    default: return false;
  }
}

}  // namespace

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
                        cudaStream_t stream)
{
  if (!cuvs::detail::jit_lto::cutile_launch_available_on_current_device()) { return false; }
  return try_fused_1nn_tile_dispatch<DataT, IdxT>(
    nearest_idx, nearest_dist, x, y, xn, yn, m, n, k, metric, is_sqrt, stream);
}

#define CUVS_INST_TRY_FUSED_1NN_TILE(DataT, IdxT)                                         \
  template CUVS_EXPORT bool try_fused_1nn_tile<DataT, IdxT>(IdxT*,                        \
                                                            DataT*,                       \
                                                            const DataT*,                 \
                                                            const DataT*,                 \
                                                            const DataT*,                 \
                                                            const DataT*,                 \
                                                            IdxT,                         \
                                                            IdxT,                         \
                                                            IdxT,                         \
                                                            cuvs::distance::DistanceType, \
                                                            bool,                         \
                                                            cudaStream_t)

CUVS_INST_TRY_FUSED_1NN_TILE(float, int);
CUVS_INST_TRY_FUSED_1NN_TILE(float, int64_t);
CUVS_INST_TRY_FUSED_1NN_TILE(half, int);
CUVS_INST_TRY_FUSED_1NN_TILE(half, int64_t);

#undef CUVS_INST_TRY_FUSED_1NN_TILE

}  // namespace detail
}  // namespace distance
}  // namespace cuvs
