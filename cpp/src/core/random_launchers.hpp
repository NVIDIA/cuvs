/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/rng_state.hpp>

// Private cuVS launchers keep cuRAND implementation headers in the CUDA owner.
// RAFT's public interfaces remain header-only.
namespace cuvs::detail::random {

template <typename T, typename Index>
void uniform(raft::resources const&, raft::random::RngState&, T*, Index, T, T);

template <typename T, typename Index>
void normal(raft::resources const&, raft::random::RngState&, T*, Index, T, T);

template <typename T, typename Weight, typename Index>
void discrete(raft::resources const&,
              raft::random::RngState&,
              raft::device_vector_view<T, Index>,
              raft::device_vector_view<const Weight, Index>);

template <typename T, typename Weight, typename Index>
void sampleWithoutReplacement(raft::resources const&,
                              raft::random::RngState&,
                              T*,
                              Index*,
                              const T*,
                              const Weight*,
                              Index,
                              Index);

template <typename Index, typename MatrixIndex = Index>
auto excess_subsample(raft::resources const&, raft::random::RngState&, Index, Index)
  -> raft::device_vector<Index, Index>;

template <typename Index>
void permute_indices(Index* indices, Index columns, Index rows, cudaStream_t stream);

template <typename T, typename Index>
void permute(Index* indices, Index columns, Index rows, cudaStream_t stream);

}  // namespace cuvs::detail::random
