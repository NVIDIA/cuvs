/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cstddef>
#include <cuda/std/utility>
#include <cuda_runtime_api.h>
#include <cuvs/core/export.hpp>
#include <iterator>

namespace cuvs::detail {
template <typename Policy, typename Map, typename Input, typename Output>
CUVS_EXPORT Output
thrust_gather(Policy const& policy, Map first, Map last, Input input, Output output);

template <typename Policy, typename Input, typename Output>
CUVS_EXPORT Output thrust_copy(Policy const& policy, Input first, Input last, Output output);
}  // namespace cuvs::detail
