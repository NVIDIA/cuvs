/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuvs/core/bitset.hpp>
#include <raft/core/bitset.cuh>
#include <raft/util/popc.cuh>

extern template void raft::popc<uint8_t, uint32_t>(
  raft::resources const&,
  raft::device_vector_view<const uint8_t, uint32_t>,
  raft::host_scalar_view<const uint32_t, uint32_t>,
  raft::device_scalar_view<uint32_t>);

extern template void raft::popc<uint16_t, uint32_t>(
  raft::resources const&,
  raft::device_vector_view<const uint16_t, uint32_t>,
  raft::host_scalar_view<const uint32_t, uint32_t>,
  raft::device_scalar_view<uint32_t>);

extern template void raft::popc<uint32_t, uint32_t>(
  raft::resources const&,
  raft::device_vector_view<const uint32_t, uint32_t>,
  raft::host_scalar_view<const uint32_t, uint32_t>,
  raft::device_scalar_view<uint32_t>);

extern template void raft::popc<uint32_t, int64_t>(
  raft::resources const&,
  raft::device_vector_view<const uint32_t, int64_t>,
  raft::host_scalar_view<const int64_t, int64_t>,
  raft::device_scalar_view<int64_t>);

extern template void raft::popc<uint64_t, int64_t>(
  raft::resources const&,
  raft::device_vector_view<const uint64_t, int64_t>,
  raft::host_scalar_view<const int64_t, int64_t>,
  raft::device_scalar_view<int64_t>);

template struct raft::core::bitset<uint8_t, uint32_t>;
template struct raft::core::bitset<uint16_t, uint32_t>;
template struct raft::core::bitset<uint32_t, uint32_t>;
template struct raft::core::bitset<uint32_t, int64_t>;
template struct raft::core::bitset<uint64_t, int64_t>;
