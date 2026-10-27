/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>
#include <cuvs/core/export.hpp>
#include <raft/core/resources.hpp>
#include <rmm/device_uvector.hpp>

namespace cuvs::detail {
template <typename value_t>
CUVS_EXPORT int getUniquelabels(raft::resources const& handle,
                                rmm::device_uvector<value_t>& unique,
                                value_t* y,
                                size_t n);

template <typename value_t>
CUVS_EXPORT int getUniquelabels(rmm::device_uvector<value_t>& unique,
                                value_t* y,
                                size_t n,
                                cudaStream_t stream);

template <typename Type, typename Lambda>
CUVS_EXPORT void make_monotonic(
  Type* out, Type* in, size_t N, cudaStream_t stream, Lambda filter_op, bool zero_based = false);

template <typename Type>
CUVS_EXPORT void make_monotonic(
  Type* out, Type* in, size_t N, cudaStream_t stream, bool zero_based = false);
}  // namespace cuvs::detail
