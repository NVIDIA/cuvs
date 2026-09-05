/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "fused_1nn_tile.hpp"

#include "fused_1nn_planner.hpp"

#include <cuvs/core/export.hpp>
#include <cuvs/detail/jit_lto/cutile_module.hpp>
#include <cuvs/detail/jit_lto/fused_distance_nn/fused_1nn_fragments.hpp>
#include <raft/core/operators.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/util/cuda_utils.cuh>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace cuvs {
namespace distance {
namespace detail {

namespace {

bool is_16_byte_aligned(const void* ptr)
{
  return ptr == nullptr || reinterpret_cast<std::uintptr_t>(ptr) % 16 == 0;
}

bool byte_ranges_overlap(const void* lhs, size_t lhs_bytes, const void* rhs, size_t rhs_bytes)
{
  if (lhs == nullptr || rhs == nullptr || lhs_bytes == 0 || rhs_bytes == 0) { return false; }
  const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs);
  const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs);
  if (lhs_bytes > std::numeric_limits<std::uintptr_t>::max() - lhs_begin ||
      rhs_bytes > std::numeric_limits<std::uintptr_t>::max() - rhs_begin) {
    return true;
  }
  return lhs_begin < rhs_begin + rhs_bytes && rhs_begin < lhs_begin + lhs_bytes;
}

template <typename IdxT>
size_t checked_tensor_bytes(IdxT rows, IdxT cols, size_t element_size)
{
  const auto rows_u       = static_cast<uint64_t>(rows);
  const auto cols_u       = static_cast<uint64_t>(cols);
  constexpr auto max_size = std::numeric_limits<size_t>::max();
  if (cols_u != 0 && rows_u > max_size / cols_u) { return max_size; }
  const auto elements = rows_u * cols_u;
  if (element_size != 0 && elements > max_size / element_size) { return max_size; }
  return static_cast<size_t>(elements) * element_size;
}

template <typename DataT, typename AbiTag>
bool has_fused_1nn_tile_launcher()
{
  Fused1nnTilePlanner<DataT, AbiTag> planner;
  planner.add_entrypoint();
  planner.add_tileir_fallback();
  return planner.try_get_launcher() != nullptr;
}

template <typename DataT, typename IdxT, typename AbiTag>
void launch_fused_1nn_tile_impl(IdxT* nearest_idx,
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
                                cudaStream_t stream)
{
  Fused1nnTilePlanner<DataT, AbiTag> planner;
  planner.add_entrypoint();
  planner.add_tileir_fallback();
  auto launcher = planner.try_get_launcher();
  RAFT_EXPECTS(launcher != nullptr, "Requested cuTile fused 1-NN launcher is unavailable");
  const cuvs::detail::jit_lto::CutileTileConfig tile_cfg = planner.tile_config();

  int metric_code;
  bool apply_sqrt = false;
  switch (metric) {
    case cuvs::distance::DistanceType::InnerProduct:
      metric_code = static_cast<int>(cuvs::distance::DistanceType::InnerProduct);
      break;
    case cuvs::distance::DistanceType::L2Expanded:
    case cuvs::distance::DistanceType::L2SqrtExpanded:
      metric_code = static_cast<int>(cuvs::distance::DistanceType::L2Expanded);
      apply_sqrt  = is_sqrt;
      break;
    case cuvs::distance::DistanceType::CosineExpanded:
      metric_code = static_cast<int>(cuvs::distance::DistanceType::CosineExpanded);
      break;
    default: RAFT_FAIL("Unsupported cuTile fused 1-NN metric");
  }

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

  void* x_ptr          = const_cast<DataT*>(x);
  void* y_ptr          = const_cast<DataT*>(y);
  void* xn_ptr         = const_cast<fused_1nn_cutile_norm_t<DataT>*>(xn);
  void* yn_ptr         = const_cast<fused_1nn_cutile_norm_t<DataT>*>(yn);
  const IdxT store_idx = nearest_idx != nullptr ? IdxT{1} : IdxT{0};
  void* idx_ptr        = nearest_idx;
  void* dist_ptr       = nearest_dist;

  const int tile_m = tile_cfg.tile_m;
  dim3 grid((static_cast<uint64_t>(m) + tile_m - 1) / tile_m, 1, 1);
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
                                         IdxT,
                                         int);
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
                                                         store_idx,
                                                         metric_code);
  RAFT_CUDA_TRY(cudaGetLastError());
}

template <typename DataT, typename IdxT>
void validate_fused_1nn_tile_launch(IdxT* nearest_idx,
                                    DataT* nearest_dist,
                                    const DataT* x,
                                    const DataT* y,
                                    const fused_1nn_cutile_norm_t<DataT>* xn,
                                    const fused_1nn_cutile_norm_t<DataT>* yn,
                                    IdxT m,
                                    IdxT n,
                                    IdxT k,
                                    cuvs::distance::DistanceType metric,
                                    void* index_workspace)
{
  RAFT_EXPECTS(is_fused_1nn_tile_available(x, y, m, n, k, metric),
               "Requested cuTile fused 1-NN backend is unavailable for this input/device");
  RAFT_EXPECTS(nearest_dist != nullptr && is_16_byte_aligned(nearest_dist),
               "cuTile fused 1-NN requires a 16-byte-aligned distance output");
  if constexpr (std::is_same_v<IdxT, int>) {
    RAFT_EXPECTS(is_16_byte_aligned(nearest_idx),
                 "cuTile fused 1-NN requires a 16-byte-aligned int32 index output");
  }
  RAFT_EXPECTS(
    metric == cuvs::distance::DistanceType::InnerProduct || (xn != nullptr && yn != nullptr),
    "cuTile fused 1-NN requires norm buffers for this metric");
  RAFT_EXPECTS(is_16_byte_aligned(xn) && is_16_byte_aligned(yn),
               "cuTile fused 1-NN requires 16-byte-aligned norm buffers");

  const auto x_bytes    = checked_tensor_bytes(m, k, sizeof(DataT));
  const auto y_bytes    = checked_tensor_bytes(n, k, sizeof(DataT));
  const auto dist_bytes = checked_tensor_bytes(m, IdxT{1}, sizeof(DataT));
  const auto idx_bytes  = checked_tensor_bytes(m, IdxT{1}, sizeof(IdxT));
  const auto xn_bytes   = checked_tensor_bytes(m, IdxT{1}, sizeof(*xn));
  const auto yn_bytes   = checked_tensor_bytes(n, IdxT{1}, sizeof(*yn));
  RAFT_EXPECTS(!byte_ranges_overlap(nearest_dist, dist_bytes, x, x_bytes) &&
                 !byte_ranges_overlap(nearest_dist, dist_bytes, y, y_bytes) &&
                 !byte_ranges_overlap(nearest_idx, idx_bytes, x, x_bytes) &&
                 !byte_ranges_overlap(nearest_idx, idx_bytes, y, y_bytes) &&
                 !byte_ranges_overlap(nearest_idx, idx_bytes, nearest_dist, dist_bytes) &&
                 !byte_ranges_overlap(nearest_dist, dist_bytes, xn, xn_bytes) &&
                 !byte_ranges_overlap(nearest_dist, dist_bytes, yn, yn_bytes) &&
                 !byte_ranges_overlap(nearest_idx, idx_bytes, xn, xn_bytes) &&
                 !byte_ranges_overlap(nearest_idx, idx_bytes, yn, yn_bytes),
               "cuTile fused 1-NN input, norm, and output buffers must not overlap");

  if constexpr (std::is_same_v<IdxT, int64_t>) {
    RAFT_EXPECTS(nearest_idx == nullptr || index_workspace != nullptr,
                 "cuTile fused 1-NN requires int32 workspace for int64 index output");
    RAFT_EXPECTS(is_16_byte_aligned(index_workspace),
                 "cuTile fused 1-NN requires 16-byte-aligned index workspace");
    const auto workspace_rows  = static_cast<IdxT>(fused_1nn_cutile_index_workspace_rows<DataT>(m));
    const auto workspace_bytes = checked_tensor_bytes(workspace_rows, IdxT{1}, sizeof(int));
    RAFT_EXPECTS(
      !byte_ranges_overlap(index_workspace, workspace_bytes, x, x_bytes) &&
        !byte_ranges_overlap(index_workspace, workspace_bytes, y, y_bytes) &&
        !byte_ranges_overlap(index_workspace, workspace_bytes, xn, xn_bytes) &&
        !byte_ranges_overlap(index_workspace, workspace_bytes, yn, yn_bytes) &&
        !byte_ranges_overlap(index_workspace, workspace_bytes, nearest_dist, dist_bytes) &&
        !byte_ranges_overlap(index_workspace, workspace_bytes, nearest_idx, idx_bytes),
      "cuTile fused 1-NN index workspace must not overlap input or output buffers");
  }
}

}  // namespace

template <typename DataT, typename IdxT>
  requires is_fused_1nn_cutile_data_v<DataT>
bool is_fused_1nn_tile_available(
  const DataT* x, const DataT* y, IdxT m, IdxT n, IdxT k, cuvs::distance::DistanceType metric)
{
  if (!cuvs::detail::jit_lto::cutile_launch_available_on_current_device()) { return false; }
  static_assert(std::is_same_v<IdxT, int> || std::is_same_v<IdxT, int64_t>);

  if (x == nullptr || y == nullptr || m <= 0 || n <= 0 || k <= 0) { return false; }
  if (metric != cuvs::distance::DistanceType::InnerProduct &&
      metric != cuvs::distance::DistanceType::L2Expanded &&
      metric != cuvs::distance::DistanceType::L2SqrtExpanded &&
      metric != cuvs::distance::DistanceType::CosineExpanded) {
    return false;
  }

  if (!is_16_byte_aligned(x) || !is_16_byte_aligned(y)) { return false; }
  if constexpr (std::is_same_v<IdxT, int64_t>) {
    constexpr int64_t max_i32 = std::numeric_limits<int>::max();
    if (n > max_i32 || k > max_i32) { return false; }
  }

  constexpr int strict_pitch_elements = 16 / sizeof(DataT);
  return k % strict_pitch_elements == 0 ? has_fused_1nn_tile_launcher<DataT, cutile_abi_strict>()
                                        : has_fused_1nn_tile_launcher<DataT, cutile_abi_relaxed>();
}

template <typename DataT, typename IdxT>
  requires is_fused_1nn_cutile_data_v<DataT>
void launch_fused_1nn_tile(IdxT* nearest_idx,
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
                           cudaStream_t stream)
{
  validate_fused_1nn_tile_launch(
    nearest_idx, nearest_dist, x, y, xn, yn, m, n, k, metric, index_workspace);

  constexpr int strict_pitch_elements = 16 / sizeof(DataT);
  const bool use_strict_abi           = k % strict_pitch_elements == 0;

  if constexpr (std::is_same_v<IdxT, int>) {
    if (use_strict_abi) {
      launch_fused_1nn_tile_impl<DataT, int, cutile_abi_strict>(
        nearest_idx, nearest_dist, x, y, xn, yn, m, n, k, metric, is_sqrt, stream);
    } else {
      launch_fused_1nn_tile_impl<DataT, int, cutile_abi_relaxed>(
        nearest_idx, nearest_dist, x, y, xn, yn, m, n, k, metric, is_sqrt, stream);
    }
  } else {
    constexpr int64_t max_batch_m = fused_1nn_cutile_max_batch_m<DataT>;
    auto* tmp_idx                 = static_cast<int*>(index_workspace);
    for (int64_t offset = 0; offset < m;) {
      const int64_t batch_m64 = std::min<int64_t>(max_batch_m, m - offset);
      const int batch_m       = static_cast<int>(batch_m64);
      const auto* batch_x     = x + static_cast<size_t>(offset) * static_cast<size_t>(k);
      const auto* batch_xn    = xn == nullptr ? nullptr : xn + offset;
      auto* batch_dist        = nearest_dist + offset;

      if (use_strict_abi) {
        launch_fused_1nn_tile_impl<DataT, int, cutile_abi_strict>(tmp_idx,
                                                                  batch_dist,
                                                                  batch_x,
                                                                  y,
                                                                  batch_xn,
                                                                  yn,
                                                                  batch_m,
                                                                  static_cast<int>(n),
                                                                  static_cast<int>(k),
                                                                  metric,
                                                                  is_sqrt,
                                                                  stream);
      } else {
        launch_fused_1nn_tile_impl<DataT, int, cutile_abi_relaxed>(tmp_idx,
                                                                   batch_dist,
                                                                   batch_x,
                                                                   y,
                                                                   batch_xn,
                                                                   yn,
                                                                   batch_m,
                                                                   static_cast<int>(n),
                                                                   static_cast<int>(k),
                                                                   metric,
                                                                   is_sqrt,
                                                                   stream);
      }

      if (nearest_idx != nullptr) {
        raft::linalg::unaryOp(
          nearest_idx + offset, tmp_idx, batch_m, raft::cast_op<int64_t>{}, stream);
      }
      offset += batch_m64;
    }
  }
}

#define CUVS_INST_IS_FUSED_1NN_TILE_AVAILABLE(DataT, IdxT)            \
  template CUVS_EXPORT bool is_fused_1nn_tile_available<DataT, IdxT>( \
    const DataT*, const DataT*, IdxT, IdxT, IdxT, cuvs::distance::DistanceType)

CUVS_INST_IS_FUSED_1NN_TILE_AVAILABLE(float, int);
CUVS_INST_IS_FUSED_1NN_TILE_AVAILABLE(float, int64_t);
CUVS_INST_IS_FUSED_1NN_TILE_AVAILABLE(half, int);
CUVS_INST_IS_FUSED_1NN_TILE_AVAILABLE(half, int64_t);

#undef CUVS_INST_IS_FUSED_1NN_TILE_AVAILABLE

#define CUVS_INST_LAUNCH_FUSED_1NN_TILE(DataT, IdxT)            \
  template CUVS_EXPORT void launch_fused_1nn_tile<DataT, IdxT>( \
    IdxT*,                                                      \
    DataT*,                                                     \
    const DataT*,                                               \
    const DataT*,                                               \
    const fused_1nn_cutile_norm_t<DataT>*,                      \
    const fused_1nn_cutile_norm_t<DataT>*,                      \
    IdxT,                                                       \
    IdxT,                                                       \
    IdxT,                                                       \
    cuvs::distance::DistanceType,                               \
    bool,                                                       \
    void*,                                                      \
    cudaStream_t)

CUVS_INST_LAUNCH_FUSED_1NN_TILE(float, int);
CUVS_INST_LAUNCH_FUSED_1NN_TILE(float, int64_t);
CUVS_INST_LAUNCH_FUSED_1NN_TILE(half, int);
CUVS_INST_LAUNCH_FUSED_1NN_TILE(half, int64_t);

#undef CUVS_INST_LAUNCH_FUSED_1NN_TILE

}  // namespace detail
}  // namespace distance
}  // namespace cuvs
