/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cub_select_launchers.hpp"
#include <cstddef>
#include <cstdint>
#include <cuda/std/utility>
#include <cuda_runtime_api.h>
#include <iterator>
#include <raft/core/kvp.hpp>

namespace cuvs::detail {
template <typename InputIt,
          typename OutputIt,
          typename CountIt,
          typename NumItems,
          typename Predicate>
cudaError_t cub_select_if(void* storage,
                          size_t& bytes,
                          InputIt input,
                          OutputIt output,
                          CountIt selected,
                          NumItems count,
                          Predicate predicate,
                          cudaStream_t stream);
}

namespace cuvs::detail {
template <typename Policy, typename Iterator>
Iterator thrust_unique(Policy const& policy, Iterator first, Iterator last);

template <typename Policy, typename Iterator, typename Predicate>
Iterator thrust_unique(Policy const& policy, Iterator first, Iterator last, Predicate predicate);

template <typename Policy, typename Input, typename Output, typename Predicate>
Output thrust_copy_if(
  Policy const& policy, Input first, Input last, Output output, Predicate predicate);
}  // namespace cuvs::detail
