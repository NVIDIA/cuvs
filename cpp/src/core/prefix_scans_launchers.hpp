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
template <typename InputIt, typename OutputIt, typename NumItems>
cudaError_t cub_exclusive_sum(void* storage,
                              size_t& bytes,
                              InputIt input,
                              OutputIt output,
                              NumItems count,
                              cudaStream_t stream = 0);
}  // namespace cuvs::detail

namespace cuvs::detail {
template <typename Policy, typename Input, typename Output>
CUVS_EXPORT Output
thrust_exclusive_scan(Policy const& policy, Input first, Input last, Output output);

template <typename Policy, typename Input, typename Output, typename Init>
CUVS_EXPORT Output
thrust_exclusive_scan(Policy const& policy, Input first, Input last, Output output, Init init);

template <typename Policy, typename Input, typename Output, typename Init, typename Op>
CUVS_EXPORT Output thrust_exclusive_scan(
  Policy const& policy, Input first, Input last, Output output, Init init, Op op);

template <typename Policy, typename Input, typename Output>
Output thrust_inclusive_scan(Policy const& policy, Input first, Input last, Output output);

template <typename Policy, typename Input, typename Output, typename Op>
Output thrust_inclusive_scan(Policy const& policy, Input first, Input last, Output output, Op op);
}  // namespace cuvs::detail
