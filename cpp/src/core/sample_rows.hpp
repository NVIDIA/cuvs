/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/rng_state.hpp>

#include <type_traits>

namespace cuvs::detail {

template <typename T, typename Index>
void sample_rows(raft::resources const&,
                 raft::random::RngState,
                 const T*,
                 Index,
                 Index,
                 raft::device_matrix_view<T, Index>);

// Preserve the RAFT adapter's layout checks, stride, allocation and by-value RNG state.
template <typename T, typename Index, typename Accessor, typename Layout>
void sample_rows(raft::resources const& res,
                 raft::random::RngState state,
                 raft::mdspan<const T, raft::matrix_extent<Index>, Layout, Accessor> dataset,
                 raft::device_matrix_view<T, Index> output);

template <typename T, typename Index, typename Accessor, typename Layout>
raft::device_matrix<T, Index> sample_rows(
  raft::resources const& res,
  raft::random::RngState state,
  raft::mdspan<const T, raft::matrix_extent<Index>, Layout, Accessor> dataset,
  Index n_samples);

}  // namespace cuvs::detail
