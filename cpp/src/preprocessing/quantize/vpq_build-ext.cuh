/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuvs/neighbors/common.hpp>
#include <cuvs/preprocessing/quantize/vpq_dataset.hpp>
#include <raft/core/resources.hpp>

namespace cuvs::preprocessing::quantize::pq {

#define CUVS_INST_VPQ_BUILD(T)                                                 \
  vpq_dataset<half, int64_t> vpq_build(                                        \
    const raft::resources& res,                                                \
    const cuvs::neighbors::vpq_params& params,                                 \
    const raft::host_matrix_view<const T, int64_t, raft::row_major>& dataset); \
  vpq_dataset<half, int64_t> vpq_build(                                        \
    const raft::resources& res,                                                \
    const cuvs::neighbors::vpq_params& params,                                 \
    const raft::device_matrix_view<const T, int64_t, raft::row_major>& dataset);

CUVS_INST_VPQ_BUILD(float);
CUVS_INST_VPQ_BUILD(half);
CUVS_INST_VPQ_BUILD(int8_t);
CUVS_INST_VPQ_BUILD(uint8_t);

#undef CUVS_INST_VPQ_BUILD
}  // namespace cuvs::preprocessing::quantize::pq
