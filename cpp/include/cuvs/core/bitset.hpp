/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/export.hpp>
#include <raft/core/bitset.hpp>

extern template struct raft::core::bitset<uint8_t, uint32_t>;
extern template struct raft::core::bitset<uint16_t, uint32_t>;
extern template struct raft::core::bitset<uint32_t, uint32_t>;
extern template struct raft::core::bitset<uint32_t, int64_t>;
extern template struct raft::core::bitset<uint64_t, int64_t>;

namespace CUVS_EXPORT cuvs {
namespace core {
/* To use bitset functions containing CUDA code, include <raft/core/bitset.cuh> */

template <typename bitset_t, typename index_t>
using bitset_view = raft::core::bitset_view<bitset_t, index_t>;

template <typename bitset_t, typename index_t>
using bitset = raft::core::bitset<bitset_t, index_t>;

}  // namespace core
}  // namespace CUVS_EXPORT cuvs
