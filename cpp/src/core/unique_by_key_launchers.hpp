/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cstddef>
#include <cuda/std/utility>
#include <cuda_runtime_api.h>
#include <iterator>

namespace cuvs::detail {
template <typename Policy, typename Key, typename Value>
cuda::std::pair<Key, Value> thrust_unique_by_key(Policy const& policy,
                                                 Key first,
                                                 Key last,
                                                 Value values);

template <typename Policy, typename Key, typename Value, typename Predicate>
cuda::std::pair<Key, Value> thrust_unique_by_key(
  Policy const& policy, Key first, Key last, Value values, Predicate predicate);
}  // namespace cuvs::detail
