/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

namespace cuvs::detail {
using raft::row_major;
template <typename math_t, typename idx_t, typename matrix_idx_t>
void argmin(raft::resources const& handle,
            raft::device_matrix_view<const math_t, matrix_idx_t, row_major> in,
            raft::device_vector_view<idx_t, matrix_idx_t> out);
}  // namespace cuvs::detail
