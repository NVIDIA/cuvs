/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gmm.cuh"

#include <cuvs/cluster/gmm.hpp>
#include <cuvs/core/export.hpp>

namespace cuvs::cluster::gmm {

template CUVS_EXPORT void fit<float>(raft::resources const&,
                                     const params&,
                                     raft::device_matrix_view<const float, int64_t>,
                                     raft::device_vector_view<float, int64_t>,
                                     raft::device_matrix_view<float, int64_t>,
                                     raft::device_vector_view<float, int64_t>,
                                     raft::device_vector_view<float, int64_t>,
                                     raft::device_vector_view<float, int64_t>,
                                     raft::device_vector_view<int, int64_t>,
                                     raft::host_scalar_view<float>,
                                     raft::host_scalar_view<int>,
                                     raft::host_scalar_view<bool>,
                                     bool);

template CUVS_EXPORT void predict<float>(raft::resources const&,
                                         const params&,
                                         raft::device_matrix_view<const float, int64_t>,
                                         raft::device_vector_view<const float, int64_t>,
                                         raft::device_matrix_view<const float, int64_t>,
                                         raft::device_vector_view<const float, int64_t>,
                                         raft::device_vector_view<int, int64_t>);

template CUVS_EXPORT void predict_proba<float>(raft::resources const&,
                                               const params&,
                                               raft::device_matrix_view<const float, int64_t>,
                                               raft::device_vector_view<const float, int64_t>,
                                               raft::device_matrix_view<const float, int64_t>,
                                               raft::device_vector_view<const float, int64_t>,
                                               raft::device_matrix_view<float, int64_t>);

template CUVS_EXPORT void score_samples<float>(raft::resources const&,
                                               const params&,
                                               raft::device_matrix_view<const float, int64_t>,
                                               raft::device_vector_view<const float, int64_t>,
                                               raft::device_matrix_view<const float, int64_t>,
                                               raft::device_vector_view<const float, int64_t>,
                                               raft::device_vector_view<float, int64_t>);

}  // namespace cuvs::cluster::gmm
