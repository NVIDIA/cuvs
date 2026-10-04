/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <neighbors/detail/flowann/flowann.hpp>

namespace cuvs::neighbors::cagra::detail {
// Transitional internal bridge for legacy importers. Moves the owner without copying its buffers.
#define CUVS_DECLARE_ADOPT_TIERED(T)                                                    \
  CUVS_EXPORT auto adopt_tiered(raft::resources const&,                                 \
                                experimental::flowann::device_padded_index_bundle<T>&&) \
    -> device_padded_index<T>;                                                          \
  CUVS_EXPORT auto adopt_tiered(raft::resources const&,                                 \
                                experimental::flowann::vpq_f16_index_bundle<T>&&)       \
    -> device_pq_index<T>;
CUVS_DECLARE_ADOPT_TIERED(float)
CUVS_DECLARE_ADOPT_TIERED(half)
CUVS_DECLARE_ADOPT_TIERED(std::int8_t)
CUVS_DECLARE_ADOPT_TIERED(std::uint8_t)
#undef CUVS_DECLARE_ADOPT_TIERED
CUVS_EXPORT auto tiered_statistics(device_pq_index<float> const&)
  -> experimental::flowann::queue_statistics;
}  // namespace cuvs::neighbors::cagra::detail
