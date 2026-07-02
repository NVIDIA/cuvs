/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace cuvs::neighbors::ivf_sq::detail {

static constexpr int kSqScanThreads = 128;

// Function-pointer signature for the JIT-LTO scan entrypoint.
// Must exactly match the extern "C" __global__ ivf_sq_scan(...) signature
// produced by scan_kernel.cu.in.
template <typename IdxT>
using ivf_sq_scan_func_t = void(const uint8_t* const* data_ptrs,
                                const uint32_t* list_sizes,
                                const uint32_t* coarse_indices,
                                const float* queries_float,
                                const float* centers,
                                const float* sq_vmin,
                                const float* sq_delta,
                                const float* query_norms,
                                uint32_t n_probes,
                                uint32_t dim,
                                uint32_t k,
                                uint32_t max_samples,
                                const uint32_t* chunk_indices,
                                float* out_distances,
                                uint32_t* out_indices,
                                IdxT* const* inds_ptrs,
                                uint32_t* bitset_ptr,
                                IdxT bitset_len,
                                IdxT original_nbits);

}  // namespace cuvs::neighbors::ivf_sq::detail
