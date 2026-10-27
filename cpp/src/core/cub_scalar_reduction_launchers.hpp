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
template <typename InputIt, typename OutputIt, typename NumItems>
cudaError_t cub_reduce_sum(void* storage,
                           size_t& bytes,
                           InputIt input,
                           OutputIt output,
                           NumItems count,
                           cudaStream_t stream);
}

namespace cuvs::detail {
template <typename Policy, typename Iterator>
typename std::iterator_traits<Iterator>::value_type thrust_reduce(Policy const& policy,
                                                                  Iterator first,
                                                                  Iterator last);

template <typename Policy, typename Iterator, typename Value>
Value thrust_reduce(Policy const& policy, Iterator first, Iterator last, Value init);

template <typename Policy, typename Iterator, typename Value, typename Op>
Value thrust_reduce(Policy const& policy, Iterator first, Iterator last, Value init, Op op);
}  // namespace cuvs::detail
