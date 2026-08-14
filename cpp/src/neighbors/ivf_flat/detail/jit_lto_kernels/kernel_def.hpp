/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace cuvs::neighbors::ivf_flat::detail {

template <typename T, typename IdxT>
using interleaved_scan_func_t = void(const uint32_t query_smem_elems,
                                     const T* query,
                                     const uint32_t* coarse_index,
                                     const T* const* list_data_ptrs,
                                     const uint32_t* list_sizes,
                                     const uint32_t queries_offset,
                                     const uint32_t n_probes,
                                     const uint32_t k,
                                     const uint32_t max_samples,
                                     const uint32_t* chunk_indices,
                                     const uint32_t dim,
                                     IdxT* const* const inds_ptrs,
                                     uint32_t* bitset_ptr,
                                     IdxT bitset_len,
                                     IdxT original_nbits,
                                     uint32_t* neighbors,
                                     float* distances);

}  // namespace cuvs::neighbors::ivf_flat::detail
