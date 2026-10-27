/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cuda/std/utility>
#include <cuda_runtime_api.h>
#include <cuvs/core/export.hpp>

#include <cstdint>
#include <raft/core/kvp.hpp>
#include <raft/core/resources.hpp>
namespace cuvs::detail {
template <typename Input, typename Output, typename Count>
cudaError_t cub_argmin(void* storage,
                       size_t& bytes,
                       Input input,
                       Output output,
                       Count count,
                       cudaStream_t stream = nullptr);

template <typename Input, typename Output, typename Count>
cudaError_t cub_max(void* storage,
                    size_t& bytes,
                    Input input,
                    Output output,
                    Count count,
                    cudaStream_t stream = nullptr);

template <typename T, typename Count>
cudaError_t cub_merge_sort_keys(
  void* storage, size_t& bytes, T* keys, Count count, cudaStream_t stream);

template <typename Policy, typename Input, typename Map, typename Output>
CUVS_EXPORT void thrust_scatter(
  Policy const& policy, Input first, Input last, Map map, Output output);

template <typename Policy, typename Iterator>
Iterator thrust_max_element(Policy const& policy, Iterator first, Iterator last);

template <typename Policy, typename Iterator, typename Op>
void thrust_for_each(Policy const& policy, Iterator first, Iterator last, Op op);

template <typename Policy, typename Input, typename Output, typename Op>
Output thrust_transform(Policy const& policy, Input first, Input last, Output output, Op op);

template <typename Policy, typename Key, typename Value>
void thrust_stable_sort_by_key(Policy const& policy, Key first, Key last, Value values);

template <typename Policy, typename Key, typename Value, typename OutputKey, typename OutputValue>
cuda::std::pair<OutputKey, OutputValue> thrust_reduce_by_key(Policy const& policy,
                                                             Key first,
                                                             Key last,
                                                             Value values,
                                                             OutputKey keys_out,
                                                             OutputValue values_out);

}  // namespace cuvs::detail

namespace cuvs::detail {
template <typename Index, typename Distance>
void mask_distance_tile(raft::resources const& handle,
                        Distance* distances_ptr,
                        const uint32_t* filter_bits,
                        size_t i,
                        size_t j,
                        size_t current_query_size,
                        size_t current_centroid_size,
                        size_t n_cols,
                        Distance masked_distance);

template <typename Index, typename Distance>
CUVS_EXPORT void copy_distance_tile(raft::resources const& handle,
                                    const Distance* in_distances,
                                    const Index* in_indices,
                                    Distance* out_distances,
                                    Index* out_indices,
                                    size_t current_query_size,
                                    size_t current_k,
                                    size_t temp_out_cols,
                                    size_t j,
                                    size_t k,
                                    size_t tile_cols);

template <typename Index, typename Distance>
void zero_self_diagonal(cudaStream_t stream, Distance* out_dists, Index rows, Index b_nrows);

template <typename T, typename Index>
void mark_sampled_centroids(raft::resources const& handle,
                            raft::KeyValuePair<ptrdiff_t, T>* values,
                            Index count,
                            uint8_t* flags);

CUVS_HIDDEN bool has_non_finite(raft::resources const& handle, const float* values, int64_t count);
}  // namespace cuvs::detail
