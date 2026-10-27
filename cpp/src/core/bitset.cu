/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuvs/core/bitset.hpp>
#include <raft/core/bitset.cuh>
#include <raft/linalg/map.cuh>

extern template void
raft::linalg::detail::map<false, uint32_t, unsigned int, raft::const_op<uint32_t>>(
  cuda::stream_ref, uint32_t*, unsigned int, raft::const_op<uint32_t>);

extern template void
raft::linalg::detail::map<false, unsigned int, long, raft::const_op<unsigned int>>(
  cuda::stream_ref, unsigned int*, long, raft::const_op<unsigned int>);

template struct raft::core::bitset<uint8_t, uint32_t>;
template struct raft::core::bitset<uint16_t, uint32_t>;
template struct raft::core::bitset<uint32_t, uint32_t>;
template struct raft::core::bitset<uint32_t, int64_t>;
template struct raft::core::bitset<uint64_t, int64_t>;
