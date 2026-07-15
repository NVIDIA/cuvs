/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace cuvs::neighbors::ivf_flat::detail {

template <typename T, typename AccT>
__device__ float load_and_compute_dist(AccT& dist,
                                       AccT& norm_query,
                                       AccT& norm_dataset,
                                       uint32_t shm_assisted_dim,
                                       const T*& data,
                                       const T* query,
                                       T* query_shared,
                                       const uint32_t dim,
                                       const uint32_t query_smem_elems);

template <typename AccT>
__device__ void compute_dist(AccT& acc, AccT x, AccT y);

template <typename T>
__device__ T post_process(T val);

template <typename IndexT>
__device__ bool sample_filter(const IndexT* const* const inds_ptrs,
                              const uint32_t query_ix,
                              const uint32_t cluster_ix,
                              const uint32_t sample_ix,
                              uint32_t* bitset_ptr,
                              IndexT bitset_len,
                              IndexT original_nbits);

}  // namespace cuvs::neighbors::ivf_flat::detail
