/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace cuvs::neighbors::ivf_sq::detail {

// Forward declaration of the sample filter device function.
// The concrete implementation is provided by a JIT-LTO filter-adapter fragment
// (see filter_kernel.cu.in) that delegates to the shared
// cuvs::neighbors::detail::sample_filter_<name><uint32_t, int64_t> fragment.
template <typename IndexT>
__device__ bool sample_filter(const IndexT* const* const inds_ptrs,
                              const uint32_t query_ix,
                              const uint32_t cluster_ix,
                              const uint32_t sample_ix,
                              uint32_t* bitset_ptr,
                              IndexT bitset_len,
                              IndexT original_nbits);

// Forward declarations of the metric-specific scan device functions. The
// concrete implementations are provided by JIT-LTO fragments generated from
// setup_invariant_smem_kernel.cu.in, setup_per_probe_smem_kernel.cu.in,
// accumulate_distance_kernel.cu.in and finalize_distance_kernel.cu.in. After
// nvJitLink LTO inlines these into ivf_sq_scan_impl, the codegen matches the
// pre-refactor `if constexpr (kIsL2 / kIsCosine)` form.

// Phase 1: load the metric-invariant smem array. Called once per query.
__device__ void setup_invariant_smem(uint32_t dim,
                                     const float* __restrict__ query,
                                     const float* __restrict__ sq_vmin,
                                     float* __restrict__ s_aux,
                                     float* __restrict__ s_query_term);

// Phase 2: load the per-probe smem array. Called once per (query, probe).
__device__ void setup_per_probe_smem(uint32_t dim,
                                     const float* __restrict__ centroid,
                                     const float* __restrict__ sq_vmin,
                                     float* __restrict__ s_aux,
                                     float* __restrict__ s_query_term);

// Per-element distance accumulator. Called inside the unrolled inner loop.
__device__ void accumulate_distance(
  float qt, float aux, float scale, uint8_t code, float& dist, float& v_norm_sq);

// Per-row distance finalize. Called once per scanned row.
__device__ float finalize_distance(float dist, float v_norm_sq, float query_norm);

}  // namespace cuvs::neighbors::ivf_sq::detail
