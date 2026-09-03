/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __FUSED_DISTANCE_NN_H
#define __FUSED_DISTANCE_NN_H

#pragma once

#include "detail/fused_distance_nn.cuh"
#include "fused_distance_nn_helpers.cuh"
#include "top_1_nn.cuh"
#include "unfused_distance_nn.cuh"
#include <raft/core/resources.hpp>
#include <raft/linalg/contractions.cuh>
#include <raft/linalg/map.cuh>
#include <raft/util/cuda_utils.cuh>

#include <cub/util_type.cuh>

#include <stdint.h>

#include <algorithm>
#include <limits>
#include <type_traits>

namespace cuvs {
namespace distance {

/**
 * \ingroup fused_l2_nn
 * @{
 */
/**
 * @brief Fused L2 distance and 1-nearest-neighbor computation in a single call.
 *
 * The benefits of such a call are 2-fold: 1) eliminate the need for an
 * intermediate buffer to store the output of gemm 2) reduce the memory read
 * traffic on this intermediate buffer, otherwise needed during the reduction
 * phase for 1-NN.
 *
 * @tparam DataT      data type
 * @tparam OutT       output type to either store 1-NN indices and their minimum
 *                    distances or store only the min distances. Accordingly, one
 *                    has to pass an appropriate `ReduceOpT`
 * @tparam IdxT       indexing arithmetic type
 * @tparam ReduceOpT  A struct to perform the final needed reduction operation
 *                    and also to initialize the output array elements with the
 *                    appropriate initial value needed for reduction.
 * @tparam KVPReduceOpT A struct providing functions for key-value pair comparison.
 *
 * @param[out] min           will contain the reduced output (Length = `m`)
 *                           (on device)
 * @param[in]  x             first matrix. Row major. Dim = `m x k`.
 *                           (on device).
 * @param[in]  y             second matrix. Row major. Dim = `n x k`.
 *                           (on device).
 * @param[in]  xn            L2 squared norm of `x`. Length = `m`. (on device).
 * @param[in]  yn            L2 squared norm of `y`. Length = `n`. (on device)
 * @param[in]  m             gemm m
 * @param[in]  n             gemm n
 * @param[in]  k             gemm k
 * @param[in]  workspace     temp workspace. Size = sizeof(int)*m. (on device)
 * @param[in]  redOp         reduction operator in the epilogue
 * @param[in]  pairRedOp     reduction operation on key value pairs
 * @param[in]  sqrt          Whether the output `minDist` should contain L2-sqrt
 * @param[in]  initOutBuffer whether to initialize the output buffer before the
 *                           main kernel launch
 * @param[in]  isRowMajor    whether the input/output is row or column major.
 * @param[in]  metric        Distance metric to be used (supports L2, cosine)
 * @param[in]  metric_arg    power argument for distances like Minkowski (not supported for now)
 * @param[in]  stream        cuda stream
 */
template <typename DataT, typename OutT, typename IdxT, typename ReduceOpT, typename KVPReduceOpT>
void fusedDistanceNN(OutT* min,
                     const DataT* x,
                     const DataT* y,
                     const DataT* xn,
                     const DataT* yn,
                     IdxT m,
                     IdxT n,
                     IdxT k,
                     void* workspace,
                     ReduceOpT redOp,
                     KVPReduceOpT pairRedOp,
                     bool sqrt,
                     bool initOutBuffer,
                     bool isRowMajor,
                     cuvs::distance::DistanceType metric,
                     float metric_arg,
                     cudaStream_t stream)
{
  ASSERT(isRowMajor, "fusedDistanceNN only supports row major inputs");
  // When k is smaller than 32, the Policy4x4 results in redundant calculations
  // as it uses tiles that have k=32. Therefore, use a "skinny" policy instead
  // that uses tiles with a smaller value of k.
  bool is_skinny = k < 32;

  size_t bytes = sizeof(DataT) * k;
  auto px      = reinterpret_cast<uintptr_t>(x);
  auto py      = reinterpret_cast<uintptr_t>(y);
  if (16 % sizeof(DataT) == 0 && bytes % 16 == 0 && px % 16 == 0 && py % 16 == 0) {
    if (is_skinny) {
      detail::fusedDistanceNNImpl<
        DataT,
        OutT,
        IdxT,
        typename raft::linalg::Policy4x4Skinny<DataT, 16 / sizeof(DataT)>::Policy,
        ReduceOpT>(min,
                   x,
                   y,
                   xn,
                   yn,
                   m,
                   n,
                   k,
                   (int*)workspace,
                   redOp,
                   pairRedOp,
                   sqrt,
                   initOutBuffer,
                   isRowMajor,
                   metric,
                   metric_arg,
                   stream);
    } else {
      detail::fusedDistanceNNImpl<
        DataT,
        OutT,
        IdxT,
        typename raft::linalg::Policy4x4<DataT, 16 / sizeof(DataT)>::Policy,
        ReduceOpT>(min,
                   x,
                   y,
                   xn,
                   yn,
                   m,
                   n,
                   k,
                   (int*)workspace,
                   redOp,
                   pairRedOp,
                   sqrt,
                   initOutBuffer,
                   isRowMajor,
                   metric,
                   metric_arg,
                   stream);
    }
  } else if (8 % sizeof(DataT) == 0 && bytes % 8 == 0 && px % 8 == 0 && py % 8 == 0) {
    if (is_skinny) {
      detail::fusedDistanceNNImpl<
        DataT,
        OutT,
        IdxT,
        typename raft::linalg::Policy4x4Skinny<DataT, 8 / sizeof(DataT)>::Policy,
        ReduceOpT>(min,
                   x,
                   y,
                   xn,
                   yn,
                   m,
                   n,
                   k,
                   (int*)workspace,
                   redOp,
                   pairRedOp,
                   sqrt,
                   initOutBuffer,
                   isRowMajor,
                   metric,
                   metric_arg,
                   stream);
    } else {
      detail::fusedDistanceNNImpl<
        DataT,
        OutT,
        IdxT,
        typename raft::linalg::Policy4x4<DataT, 8 / sizeof(DataT)>::Policy,
        ReduceOpT>(min,
                   x,
                   y,
                   xn,
                   yn,
                   m,
                   n,
                   k,
                   (int*)workspace,
                   redOp,
                   pairRedOp,
                   sqrt,
                   initOutBuffer,
                   isRowMajor,
                   metric,
                   metric_arg,
                   stream);
    }
  } else {
    if (is_skinny) {
      detail::fusedDistanceNNImpl<DataT,
                                  OutT,
                                  IdxT,
                                  typename raft::linalg::Policy4x4Skinny<DataT, 1>::Policy,
                                  ReduceOpT>(min,
                                             x,
                                             y,
                                             xn,
                                             yn,
                                             m,
                                             n,
                                             k,
                                             (int*)workspace,
                                             redOp,
                                             pairRedOp,
                                             sqrt,
                                             initOutBuffer,
                                             isRowMajor,
                                             metric,
                                             metric_arg,
                                             stream);
    } else {
      detail::fusedDistanceNNImpl<DataT,
                                  OutT,
                                  IdxT,
                                  typename raft::linalg::Policy4x4<DataT, 1>::Policy,
                                  ReduceOpT>(min,
                                             x,
                                             y,
                                             xn,
                                             yn,
                                             m,
                                             n,
                                             k,
                                             (int*)workspace,
                                             redOp,
                                             pairRedOp,
                                             sqrt,
                                             initOutBuffer,
                                             isRowMajor,
                                             metric,
                                             metric_arg,
                                             stream);
    }
  }
}

/**
 * @brief Wrapper around fusedDistanceNN with minimum reduction operators.
 *
 * fusedDistanceNN cannot be compiled in the distance library due to the lambda
 * operators, so this wrapper covers the most common case (minimum).
 *
 * @tparam DataT     data type
 * @tparam OutT      output type to either store 1-NN indices and their minimum
 *                   distances (e.g. raft::KeyValuePair<int, float>) or store only the min
 * distances.
 * @tparam IdxT      indexing arithmetic type
 * @param[out] min           will contain the reduced output (Length = `m`)
 *                           (on device)
 * @param[in]  x             first matrix. Row major. Dim = `m x k`.
 *                           (on device).
 * @param[in]  y             second matrix. Row major. Dim = `n x k`.
 *                           (on device).
 * @param[in]  xn            L2 squared norm of `x`. Length = `m`. (on device).
 * @param[in]  yn            L2 squared norm of `y`. Length = `n`. (on device)
 * @param[in]  m             gemm m
 * @param[in]  n             gemm n
 * @param[in]  k             gemm k
 * @param[in]  workspace     temp workspace. Size = sizeof(int)*m. (on device)
 * @param[in]  sqrt          Whether the output `minDist` should contain L2-sqrt
 * @param[in]  initOutBuffer whether to initialize the output buffer before the
 *                           main kernel launch
 * @param[in]  isRowMajor    whether the input/output is row or column major.
 * @param[in]  metric        Distance metric to be used (supports L2, cosine)
 * @param[in]  metric_arg    power argument for distances like Minkowski (not supported for now)
 * @param[in]  stream        cuda stream
 */
template <typename DataT, typename OutT, typename IdxT>
void fusedDistanceNNMinReduce(OutT* min,
                              const DataT* x,
                              const DataT* y,
                              const DataT* xn,
                              const DataT* yn,
                              IdxT m,
                              IdxT n,
                              IdxT k,
                              void* workspace,
                              bool sqrt,
                              bool initOutBuffer,
                              bool isRowMajor,
                              cuvs::distance::DistanceType metric,
                              float metric_arg,
                              cudaStream_t stream)
{
  if constexpr (std::is_same_v<OutT, raft::KeyValuePair<IdxT, DataT>>) {
    detail::Top1nnTuning tuning{};
    top_1_nn<DataT, IdxT>(nullptr,
                          nullptr,
                          x,
                          y,
                          xn,
                          yn,
                          m,
                          n,
                          k,
                          tuning,
                          workspace,
                          0,
                          sqrt,
                          initOutBuffer,
                          isRowMajor,
                          metric,
                          metric_arg,
                          detail::Fused1nnBackend::Cutlass,
                          min,
                          stream);
    return;
  } else if constexpr (std::is_same_v<OutT, DataT>) {
    detail::Top1nnTuning tuning{};
    top_1_nn<DataT, IdxT>(nullptr,
                          min,
                          x,
                          y,
                          xn,
                          yn,
                          m,
                          n,
                          k,
                          tuning,
                          workspace,
                          0,
                          sqrt,
                          initOutBuffer,
                          isRowMajor,
                          metric,
                          metric_arg,
                          detail::Fused1nnBackend::Cutlass,
                          nullptr,
                          stream);
    return;
  }

  MinAndDistanceReduceOp<IdxT, DataT> redOp;
  KVPMinReduce<IdxT, DataT> pairRedOp;

  fusedDistanceNN<DataT, OutT, IdxT>(min,
                                     x,
                                     y,
                                     xn,
                                     yn,
                                     m,
                                     n,
                                     k,
                                     workspace,
                                     redOp,
                                     pairRedOp,
                                     sqrt,
                                     initOutBuffer,
                                     isRowMajor,
                                     metric,
                                     metric_arg,
                                     stream);
}

template <typename DataT, typename IdxT, typename NormT>
void top_1_nn(raft::resources const& handle,
              IdxT* nearest_idx,
              DataT* nearest_dist,
              const DataT* x,
              const DataT* y,
              const NormT* xn,
              const NormT* yn,
              IdxT m,
              IdxT n,
              IdxT k,
              const detail::Top1nnTuning& tuning,
              void* workspace,
              std::size_t workspace_bytes,
              bool sqrt,
              bool init_out_buffer,
              bool is_row_major,
              cuvs::distance::DistanceType metric,
              float metric_arg,
              detail::Fused1nnBackend backend,
              raft::KeyValuePair<IdxT, DataT>* cutlass_kvp_output,
              cudaStream_t stream)
{
  RAFT_EXPECTS(is_row_major, "fusedDistanceNN only supports row-major inputs");
  RAFT_EXPECTS(backend != detail::Fused1nnBackend::Auto,
               "top_1_nn requires AUTO to be resolved before dispatch");
  if (backend == detail::Fused1nnBackend::Cutile) {
    RAFT_EXPECTS(cutlass_kvp_output == nullptr,
                 "cuTile top_1_nn requires its native separate output arrays");
    if constexpr (detail::is_fused_1nn_cutile_data_v<DataT> &&
                  std::is_same_v<NormT, detail::fused_1nn_cutile_norm_t<DataT>>) {
      if constexpr (std::is_same_v<IdxT, int64_t>) {
        const auto required_workspace_bytes =
          sizeof(int) * detail::fused_1nn_cutile_index_workspace_rows<DataT>(m);
        RAFT_EXPECTS(workspace != nullptr && workspace_bytes >= required_workspace_bytes,
                     "cuTile top_1_nn workspace is too small for int64 output batching");
      }
      const bool launched = detail::try_fused_1nn_tile<DataT, IdxT>(
        nearest_idx, nearest_dist, x, y, xn, yn, m, n, k, metric, sqrt, workspace, stream);
      RAFT_EXPECTS(launched,
                   "Requested cuTile fused 1-NN backend is unavailable for this input/device");
      return;
    } else {
      RAFT_FAIL("Requested cuTile fused 1-NN backend does not support these data/norm types");
    }
  }
  RAFT_EXPECTS(detail::can_launch_fused_1nn_backend(backend, x, y, m, n, k, metric),
               "Requested fused 1-NN backend is unavailable for this input");
  RAFT_EXPECTS(metric != cuvs::distance::DistanceType::InnerProduct,
               "Only cuTile top_1_nn supports InnerProduct (as a maximum reduction)");
  RAFT_EXPECTS(nearest_idx == nullptr && nearest_dist == nullptr,
               "CUTLASS and unfused top_1_nn require their native KVP output buffer");
  constexpr bool matching_norm_type = std::is_same_v<NormT, DataT>;
  RAFT_EXPECTS(matching_norm_type, "CUTLASS and unfused top_1_nn require matching norm types");
  if constexpr (matching_norm_type) {
    if (backend == detail::Fused1nnBackend::Unfused) {
      RAFT_EXPECTS(cutlass_kvp_output != nullptr,
                   "Unfused top_1_nn requires its native KVP output buffer");
      RAFT_EXPECTS(tuning.unfused.row_tile > 0 && tuning.unfused.candidate_tile > 0,
                   "Unfused top_1_nn tile dimensions must be positive");

      const auto max_row_tile       = static_cast<std::size_t>(m);
      const auto max_candidate_tile = static_cast<std::size_t>(n);
      const auto row_tile = static_cast<IdxT>(std::min(tuning.unfused.row_tile, max_row_tile));
      const auto candidate_tile =
        static_cast<IdxT>(std::min(tuning.unfused.candidate_tile, max_candidate_tile));
      const auto distance_workspace_bytes = static_cast<std::size_t>(row_tile) *
                                            static_cast<std::size_t>(candidate_tile) *
                                            sizeof(DataT);
      const auto candidate_min_offset =
        raft::alignTo(distance_workspace_bytes, alignof(raft::KeyValuePair<IdxT, DataT>));
      const auto candidate_min_bytes =
        candidate_tile < n
          ? static_cast<std::size_t>(row_tile) * sizeof(raft::KeyValuePair<IdxT, DataT>)
          : std::size_t{0};
      const auto required_workspace_bytes = candidate_min_offset + candidate_min_bytes;
      RAFT_EXPECTS(workspace != nullptr && workspace_bytes >= required_workspace_bytes,
                   "Unfused top_1_nn workspace is smaller than its configured tile");

      using KeyValueT = raft::KeyValuePair<IdxT, DataT>;
      auto* candidate_min =
        candidate_tile < n
          ? reinterpret_cast<KeyValueT*>(static_cast<std::byte*>(workspace) + candidate_min_offset)
          : nullptr;
      for (IdxT row_offset = 0; row_offset < m; row_offset += row_tile) {
        const auto rows = std::min(row_tile, static_cast<IdxT>(m - row_offset));
        auto output =
          raft::make_device_vector_view<KeyValueT, IdxT>(cutlass_kvp_output + row_offset, rows);
        for (IdxT candidate_offset = 0; candidate_offset < n; candidate_offset += candidate_tile) {
          const auto candidates = std::min(candidate_tile, static_cast<IdxT>(n - candidate_offset));
          auto* tile_output     = candidate_offset == 0 ? output.data_handle() : candidate_min;
          unfusedDistanceNNMinReduce<DataT, DataT, KeyValueT, IdxT>(
            handle,
            tile_output,
            x + row_offset * k,
            y + candidate_offset * k,
            xn + row_offset,
            yn + candidate_offset,
            rows,
            candidates,
            k,
            workspace,
            sqrt,
            candidate_offset != 0 || init_out_buffer,
            is_row_major,
            metric,
            metric_arg,
            stream);
          if (candidate_offset != 0) {
            auto candidate_output =
              raft::make_device_vector_view<const KeyValueT, IdxT>(candidate_min, rows);
            raft::linalg::map(
              handle,
              output,
              [candidate_offset] __device__(KeyValueT current, KeyValueT candidate) {
                candidate.key += candidate_offset;
                return candidate.value < current.value ? candidate : current;
              },
              raft::make_const_mdspan(output),
              candidate_output);
          }
        }
      }
      return;
    }
    RAFT_EXPECTS(backend == detail::Fused1nnBackend::Cutlass, "Unknown fused 1-NN backend");
    RAFT_EXPECTS(cutlass_kvp_output != nullptr,
                 "CUTLASS fused 1-NN requires its native KVP output buffer");
    RAFT_EXPECTS(
      workspace != nullptr && workspace_bytes >= sizeof(int) * static_cast<std::size_t>(m),
      "CUTLASS top_1_nn workspace is too small");
    MinAndDistanceReduceOp<IdxT, DataT> red_op;
    KVPMinReduce<IdxT, DataT> pair_red_op;
    fusedDistanceNN<DataT, raft::KeyValuePair<IdxT, DataT>, IdxT>(cutlass_kvp_output,
                                                                  x,
                                                                  y,
                                                                  xn,
                                                                  yn,
                                                                  m,
                                                                  n,
                                                                  k,
                                                                  workspace,
                                                                  red_op,
                                                                  pair_red_op,
                                                                  sqrt,
                                                                  init_out_buffer,
                                                                  is_row_major,
                                                                  metric,
                                                                  metric_arg,
                                                                  stream);
  }
}

template <typename DataT, typename IdxT, typename NormT>
void top_1_nn(IdxT* nearest_idx,
              DataT* nearest_dist,
              const DataT* x,
              const DataT* y,
              const NormT* xn,
              const NormT* yn,
              IdxT m,
              IdxT n,
              IdxT k,
              const detail::Top1nnTuning&,
              void* workspace,
              std::size_t,
              bool sqrt,
              bool init_out_buffer,
              bool is_row_major,
              cuvs::distance::DistanceType metric,
              float metric_arg,
              detail::Fused1nnBackend backend,
              raft::KeyValuePair<IdxT, DataT>* cutlass_kvp_output,
              cudaStream_t stream)
{
  RAFT_EXPECTS(backend == detail::Fused1nnBackend::Cutlass,
               "The no-handle top_1_nn compatibility path supports CUTLASS only");
  RAFT_EXPECTS(is_row_major, "fusedDistanceNN only supports row-major inputs");
  RAFT_EXPECTS(detail::can_launch_fused_1nn_backend(backend, x, y, m, n, k, metric),
               "Requested CUTLASS fused 1-NN backend is unavailable for this input");
  constexpr bool matching_norm_type = std::is_same_v<NormT, DataT>;
  RAFT_EXPECTS(matching_norm_type, "CUTLASS top_1_nn requires matching norm types");

  if constexpr (std::is_same_v<NormT, DataT>) {
    MinAndDistanceReduceOp<IdxT, DataT> red_op;
    KVPMinReduce<IdxT, DataT> pair_red_op;
    if (cutlass_kvp_output != nullptr) {
      fusedDistanceNN<DataT, raft::KeyValuePair<IdxT, DataT>, IdxT>(cutlass_kvp_output,
                                                                    x,
                                                                    y,
                                                                    xn,
                                                                    yn,
                                                                    m,
                                                                    n,
                                                                    k,
                                                                    workspace,
                                                                    red_op,
                                                                    pair_red_op,
                                                                    sqrt,
                                                                    init_out_buffer,
                                                                    is_row_major,
                                                                    metric,
                                                                    metric_arg,
                                                                    stream);
      return;
    }
    RAFT_EXPECTS(nearest_idx == nullptr && nearest_dist != nullptr,
                 "CUTLASS scalar top_1_nn requires a distance output and no index output");
    fusedDistanceNN<DataT, DataT, IdxT>(nearest_dist,
                                        x,
                                        y,
                                        xn,
                                        yn,
                                        m,
                                        n,
                                        k,
                                        workspace,
                                        red_op,
                                        pair_red_op,
                                        sqrt,
                                        init_out_buffer,
                                        is_row_major,
                                        metric,
                                        metric_arg,
                                        stream);
  }
}

/** @} */

}  // namespace distance
}  // namespace cuvs

#endif
