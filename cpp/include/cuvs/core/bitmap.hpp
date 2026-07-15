/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/export.hpp>
#include <raft/core/bitmap.hpp>

namespace CUVS_EXPORT cuvs {
namespace core {
/* To use bitmap functions containing CUDA code, include <raft/core/bitmap.cuh> */

template <typename bitmap_t, typename index_t>
using bitmap_view = raft::core::bitmap_view<bitmap_t, index_t>;

}  // namespace core
}  // namespace CUVS_EXPORT cuvs
