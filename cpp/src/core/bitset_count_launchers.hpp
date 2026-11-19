/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <raft/core/bitmap.hpp>
#include <raft/core/bitset.hpp>
#include <raft/core/resources.hpp>
namespace cuvs::detail {
template <typename Bits, typename Index>
Index count_bits(raft::resources const& handle, raft::core::bitset_view<Bits, Index> view);
template <typename Bits, typename Index>
Index count_bits(raft::resources const& handle, raft::core::bitmap_view<Bits, Index> view);
}  // namespace cuvs::detail
