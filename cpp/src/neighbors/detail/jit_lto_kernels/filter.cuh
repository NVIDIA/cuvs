/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace cuvs::neighbors::detail {

template <typename BitsetT, typename IndexT>
__device__ bool sample_filter_none(const IndexT* const* const inds_ptr,
                                   const uint32_t query_ix,
                                   const uint32_t cluster_ix,
                                   const uint32_t sample_ix,
                                   uint32_t* bitset_ptr,
                                   IndexT bitset_len,
                                   IndexT original_nbits);

template <typename BitsetT, typename IndexT>
__device__ bool sample_filter_bitset(const IndexT* const* const inds_ptr,
                                     const uint32_t query_ix,
                                     const uint32_t cluster_ix,
                                     const uint32_t sample_ix,
                                     uint32_t* bitset_ptr,
                                     IndexT bitset_len,
                                     IndexT original_nbits);

}  // namespace cuvs::neighbors::detail
