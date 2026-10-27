/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <raft/core/resources.hpp>
namespace cuvs::neighbors::ivf_rabitq::detail {
void launch_random_scaling_factors(float* output,
                                   int rows,
                                   int cols,
                                   int ex_bits,
                                   unsigned long long seed,
                                   int block_size,
                                   size_t shared_mem_size,
                                   cudaStream_t stream);
void exrabitq_codes_and_factors_fused_ori(const int* d_bin_XP,
                                          const float* d_XP_norm,
                                          float* d_XP,
                                          uint8_t* d_long_code,
                                          float* d_ex_factor,
                                          const float* d_centroid,
                                          size_t num_points,
                                          size_t D,
                                          size_t EX_BITS,
                                          raft::resources const& handle);
}  // namespace cuvs::neighbors::ivf_rabitq::detail
