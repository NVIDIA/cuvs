/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "fused_distance_nn.cuh"

namespace cuvs::distance {

#define CUVS_INSTANTIATE_TOP_1_NN(DataT, IdxT, NormT)                                      \
  template CUVS_EXPORT void top_1_nn<DataT, IdxT, NormT>(raft::resources const&,           \
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
  template CUVS_EXPORT void top_1_nn<DataT, IdxT, NormT>(IdxT*,                            \
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

CUVS_INSTANTIATE_TOP_1_NN(float, int, float);
CUVS_INSTANTIATE_TOP_1_NN(float, int64_t, float);
CUVS_INSTANTIATE_TOP_1_NN(double, int, double);
CUVS_INSTANTIATE_TOP_1_NN(double, int64_t, double);
CUVS_INSTANTIATE_TOP_1_NN(half, int, float);
CUVS_INSTANTIATE_TOP_1_NN(half, int64_t, float);

#undef CUVS_INSTANTIATE_TOP_1_NN

}  // namespace cuvs::distance
