/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "fused_distance_nn.cuh"

namespace cuvs::distance {

#define CUVS_INSTANTIATE_TOP_1_NN_WORKSPACE_SIZE(DataT, IdxT)            \
  template CUVS_EXPORT std::size_t top_1_nn_workspace_size<DataT, IdxT>( \
    IdxT, IdxT, const detail::Top1nnTuning&, detail::Top1nnBackend)

CUVS_INSTANTIATE_TOP_1_NN_WORKSPACE_SIZE(float, int);
CUVS_INSTANTIATE_TOP_1_NN_WORKSPACE_SIZE(float, int64_t);
CUVS_INSTANTIATE_TOP_1_NN_WORKSPACE_SIZE(double, int);
CUVS_INSTANTIATE_TOP_1_NN_WORKSPACE_SIZE(double, int64_t);
CUVS_INSTANTIATE_TOP_1_NN_WORKSPACE_SIZE(half, int);
CUVS_INSTANTIATE_TOP_1_NN_WORKSPACE_SIZE(half, int64_t);

#undef CUVS_INSTANTIATE_TOP_1_NN_WORKSPACE_SIZE

#define CUVS_INSTANTIATE_TOP_1_NN(DataT, IdxT, NormT, OutputKind)                            \
  template CUVS_EXPORT void                                                                  \
  top_1_nn<DataT, IdxT, typename detail::Top1nnOutputTypes<DataT, IdxT>::OutputKind, NormT>( \
    raft::resources const&,                                                                  \
    typename detail::Top1nnOutputTypes<DataT, IdxT>::OutputKind,                             \
    const DataT*,                                                                            \
    const DataT*,                                                                            \
    const NormT*,                                                                            \
    const NormT*,                                                                            \
    IdxT,                                                                                    \
    IdxT,                                                                                    \
    IdxT,                                                                                    \
    const detail::Top1nnTuning&,                                                             \
    void*,                                                                                   \
    std::size_t,                                                                             \
    bool,                                                                                    \
    bool,                                                                                    \
    bool,                                                                                    \
    DistanceType,                                                                            \
    float,                                                                                   \
    detail::Top1nnBackend,                                                                   \
    cudaStream_t)

CUVS_INSTANTIATE_TOP_1_NN(float, int, float, kvp);
CUVS_INSTANTIATE_TOP_1_NN(float, int, float, scalar);
CUVS_INSTANTIATE_TOP_1_NN(float, int, float, separate);
CUVS_INSTANTIATE_TOP_1_NN(float, int64_t, float, kvp);
CUVS_INSTANTIATE_TOP_1_NN(float, int64_t, float, scalar);
CUVS_INSTANTIATE_TOP_1_NN(float, int64_t, float, separate);
CUVS_INSTANTIATE_TOP_1_NN(double, int, double, kvp);
CUVS_INSTANTIATE_TOP_1_NN(double, int, double, scalar);
CUVS_INSTANTIATE_TOP_1_NN(double, int64_t, double, kvp);
CUVS_INSTANTIATE_TOP_1_NN(double, int64_t, double, scalar);
CUVS_INSTANTIATE_TOP_1_NN(half, int, float, separate);
CUVS_INSTANTIATE_TOP_1_NN(half, int64_t, float, separate);

#undef CUVS_INSTANTIATE_TOP_1_NN

}  // namespace cuvs::distance
