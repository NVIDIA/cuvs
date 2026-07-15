/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/neighbors/common.hpp>

// Include explicit instantiations before namespace (launcher includes JIT LTO headers with
// namespace definitions)
#include "search_single_cta_kernel_explicit_inst.cuh"

namespace cuvs::neighbors::cagra::detail::single_cta_search {

#define instantiate_kernel_selection(DataT, IndexT, DistanceT, SampleFilterT)      \
  template void select_and_run<DataT, uint32_t, DistanceT, IndexT, SampleFilterT>( \
    const dataset_descriptor_host<DataT, uint32_t, DistanceT>& dataset_desc,       \
    raft::device_matrix_view<const uint32_t, int64_t, raft::row_major> graph,      \
    std::optional<raft::device_vector_view<const IndexT, int64_t>> source_indices, \
    uintptr_t topk_indices_ptr,                                                    \
    DistanceT* topk_distances_ptr,                                                 \
    const DataT* queries_ptr,                                                      \
    uint32_t num_queries,                                                          \
    const uint32_t* dev_seed_ptr,                                                  \
    uint32_t* num_executed_iterations,                                             \
    const search_params& ps,                                                       \
    uint32_t topk,                                                                 \
    uint32_t num_itopk_candidates,                                                 \
    uint32_t block_size,                                                           \
    uint32_t smem_size,                                                            \
    int64_t hash_bitlen,                                                           \
    uint32_t* hashmap_ptr,                                                         \
    size_t small_hash_bitlen,                                                      \
    size_t small_hash_reset_interval,                                              \
    uint32_t num_seeds,                                                            \
    SampleFilterT sample_filter,                                                   \
    cudaStream_t stream);

}  // namespace cuvs::neighbors::cagra::detail::single_cta_search
