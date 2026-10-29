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
void mask_distance_tile(raft::resources const& handle,
                        Distance* distances_ptr,
                        const uint32_t* filter_bits,
                        size_t i,
                        size_t j,
                        size_t current_query_size,
                        size_t current_centroid_size,
                        size_t n_cols,
                        Distance masked_distance)
{
  auto count = cuda::make_counting_iterator<Index>(0);
  thrust::for_each(raft::resource::get_thrust_policy(handle),
                   count,
                   count + current_query_size * current_centroid_size,
                   [=] __device__(Index idx) {
                     Index row        = i + (idx / current_centroid_size);
                     Index col        = j + (idx % current_centroid_size);
                     Index g_idx      = row * n_cols + col;
                     Index item_idx   = (g_idx) >> 5;
                     uint32_t bit_idx = (g_idx) & 31;
                     uint32_t filter  = filter_bits[item_idx];
                     if ((filter & (uint32_t(1) << bit_idx)) == 0) {
                       distances_ptr[idx] = masked_distance;
                     }
                   });
}

}  // namespace cuvs::detail

template void cuvs::detail::mask_distance_tile<long>(raft::resources const&,
                                                     float*,
                                                     unsigned int const*,
                                                     unsigned long,
                                                     unsigned long,
                                                     unsigned long,
                                                     unsigned long,
                                                     unsigned long,
                                                     float);
