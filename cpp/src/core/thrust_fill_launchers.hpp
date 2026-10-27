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
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resources.hpp>

namespace cuvs::detail {
template <typename Policy, typename Iterator, typename Value>
void thrust_fill(Policy const& policy, Iterator first, Iterator last, Value const& value);

template <typename Policy, typename Iterator, typename Count, typename Value>
Iterator thrust_fill_n(Policy const& policy, Iterator first, Count count, Value const& value);

template <typename Policy, typename Iterator>
void thrust_sequence(Policy const& policy, Iterator first, Iterator last);

template <typename Policy, typename Iterator, typename Init>
void thrust_sequence(Policy const& policy, Iterator first, Iterator last, Init init);

template <typename Policy, typename Iterator, typename Init, typename Step>
void thrust_sequence(Policy const& policy, Iterator first, Iterator last, Init init, Step step);
}  // namespace cuvs::detail

namespace cuvs::detail {
using raft::row_major;
template <typename math_t, typename extents, typename layout>
CUVS_EXPORT void matrix_fill(raft::resources const& handle,
                             raft::device_mdspan<const math_t, extents, layout> in,
                             raft::device_mdspan<math_t, extents, layout> out,
                             raft::host_scalar_view<math_t> scalar);
template <typename math_t, typename extents, typename layout>
CUVS_EXPORT void matrix_fill(raft::resources const& handle,
                             raft::device_mdspan<math_t, extents, layout> inout,
                             math_t scalar);
}  // namespace cuvs::detail
