/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "detail/fused_distance_nn.cuh"

#include <cuvs/core/export.hpp>

#include <raft/core/kvp.hpp>
#include <raft/core/resources.hpp>

#include <cuda_runtime.h>

namespace cuvs::distance {

/** Dispatch 1-NN to a selected backend using backend-native output storage. */
template <typename DataT, typename IdxT, typename NormT = DataT>
CUVS_EXPORT void top_1_nn(raft::resources const& handle,
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
                          DistanceType metric,
                          float metric_arg,
                          detail::Fused1nnBackend backend,
                          raft::KeyValuePair<IdxT, DataT>* cutlass_kvp_output,
                          cudaStream_t stream);

/** CUTLASS-only overload used by the no-handle legacy compatibility wrapper. */
template <typename DataT, typename IdxT, typename NormT = DataT>
CUVS_EXPORT void top_1_nn(IdxT* nearest_idx,
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
                          DistanceType metric,
                          float metric_arg,
                          detail::Fused1nnBackend backend,
                          raft::KeyValuePair<IdxT, DataT>* cutlass_kvp_output,
                          cudaStream_t stream);

#define CUVS_EXTERN_TOP_1_NN(DataT, IdxT, NormT)                                      \
  extern template void top_1_nn<DataT, IdxT, NormT>(raft::resources const&,           \
                                                    IdxT*,                            \
                                                    DataT*,                           \
                                                    const DataT*,                     \
                                                    const DataT*,                     \
                                                    const NormT*,                     \
                                                    const NormT*,                     \
                                                    IdxT,                             \
                                                    IdxT,                             \
                                                    IdxT,                             \
                                                    const detail::Top1nnTuning&,      \
                                                    void*,                            \
                                                    std::size_t,                      \
                                                    bool,                             \
                                                    bool,                             \
                                                    bool,                             \
                                                    DistanceType,                     \
                                                    float,                            \
                                                    detail::Fused1nnBackend,          \
                                                    raft::KeyValuePair<IdxT, DataT>*, \
                                                    cudaStream_t);                    \
  extern template void top_1_nn<DataT, IdxT, NormT>(IdxT*,                            \
                                                    DataT*,                           \
                                                    const DataT*,                     \
                                                    const DataT*,                     \
                                                    const NormT*,                     \
                                                    const NormT*,                     \
                                                    IdxT,                             \
                                                    IdxT,                             \
                                                    IdxT,                             \
                                                    const detail::Top1nnTuning&,      \
                                                    void*,                            \
                                                    std::size_t,                      \
                                                    bool,                             \
                                                    bool,                             \
                                                    bool,                             \
                                                    DistanceType,                     \
                                                    float,                            \
                                                    detail::Fused1nnBackend,          \
                                                    raft::KeyValuePair<IdxT, DataT>*, \
                                                    cudaStream_t)

CUVS_EXTERN_TOP_1_NN(float, int, float);
CUVS_EXTERN_TOP_1_NN(float, int64_t, float);
CUVS_EXTERN_TOP_1_NN(double, int, double);
CUVS_EXTERN_TOP_1_NN(double, int64_t, double);
CUVS_EXTERN_TOP_1_NN(half, int, float);
CUVS_EXTERN_TOP_1_NN(half, int64_t, float);

#undef CUVS_EXTERN_TOP_1_NN

}  // namespace cuvs::distance
