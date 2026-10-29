/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cub_ptx_query.hpp"

#include "cub_runtime_launchers.hpp"
#include <cuda/iterator>
#include <raft/core/resource/thrust_policy.hpp>
#include <thrust/for_each.h>

namespace cuvs::detail {

template <typename Index, typename Distance>
void copy_distance_tile(raft::resources const& handle,
                        const Distance* in_distances,
                        const Index* in_indices,
                        Distance* out_distances,
                        Index* out_indices,
                        size_t current_query_size,
                        size_t current_k,
                        size_t temp_out_cols,
                        size_t j,
                        size_t k,
                        size_t tile_cols)
{
  auto count = cuda::make_counting_iterator<Index>(0);
  thrust::for_each(raft::resource::get_thrust_policy(handle),
                   count,
                   count + current_query_size * current_k,
                   [=] __device__(Index i) {
                     Index row = i / current_k, col = i % current_k;
                     Index out_index          = row * temp_out_cols + j * k / tile_cols + col;
                     out_distances[out_index] = in_distances[i];
                     out_indices[out_index]   = in_indices[i] + j;
                   });
}

}  // namespace cuvs::detail

template CUVS_EXPORT void cuvs::detail::copy_distance_tile(raft::resources const&,
                                                           float const*,
                                                           long const*,
                                                           float*,
                                                           long*,
                                                           unsigned long,
                                                           unsigned long,
                                                           unsigned long,
                                                           unsigned long,
                                                           unsigned long,
                                                           unsigned long);
