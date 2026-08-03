/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Imported from the standalone RaBitQ GPU quantizer. Behaviour unchanged;
 * only include paths, error macros and the enclosing namespace differ.
 */

//
// Standalone GPU RaBitQ scalar quantization on pre-computed residuals.
// Trimmed to just the fused-kernel free functions used by the benchmark.
//

#ifndef RABITQ_GPU_QUANTIZER_STANDALONE_CUH
#define RABITQ_GPU_QUANTIZER_STANDALONE_CUH

#include <cuda_runtime.h>

#include <cstdint>
#include <cstddef>

namespace cuvs::preprocessing::quantize::rabitq::detail {

/// Quantize pre-computed, rotated residuals into RaBitQ scalar codes.
///
/// stream      : the caller owns the stream. Every one of these entry points
///               only enqueues kernel launches on it and is therefore
///               *asynchronous*: the outputs are not readable until the caller
///               synchronizes the stream (or orders subsequent work on it).
///               All input and output buffers must stay alive and unmodified
///               until that work completes.
/// d_residuals : must be N × padded_dim on device (rotated, zero-centroid).
///
/// Writes N × padded_dim codes, plus a per-vector (delta, vl) pair.
///
/// delta_mode: 0 = RECONSTRUCTION, 1 = UNBIASED, 2 = PLAIN.
void quantize_fused_on_residuals(
    cudaStream_t stream,
    const float* d_residuals, size_t N, size_t padded_dim,
    size_t ex_bits, float const_scaling_factor, bool use_fast,
    uint16_t* d_total_code, float* d_delta, float* d_vl, int delta_mode = 0,
    int coarse_samples = 64, int fine_samples = 64);

void quantize_fused_on_residuals(
    cudaStream_t stream,
    const float* d_residuals, size_t N, size_t padded_dim,
    size_t ex_bits, float const_scaling_factor, bool use_fast,
    uint8_t* d_total_code, float* d_delta, float* d_vl, int delta_mode = 0,
    int coarse_samples = 64, int fine_samples = 64);

/// Quantize pre-computed, rotated residuals and produce the full
/// (f_add, f_rescale, f_error) factor triplet used for approximate-distance
/// estimation during search.
///
/// stream      : the caller owns the stream; the call is asynchronous on it
///               (see quantize_fused_on_residuals above).
/// d_residuals : N × padded_dim on device (rotated, centroid already subtracted).
/// d_centroid  : padded_dim floats on device (the rotated centroid) — pass a
///               zero-filled buffer when there is no centroid.
/// d_factors   : N × 3 floats on device — stride 3, layout
///               [f_add, f_rescale, f_error] per vector.
void quantize_full_on_residuals(
    cudaStream_t stream,
    const float* d_residuals, const float* d_centroid,
    size_t N, size_t padded_dim,
    size_t ex_bits, float const_scaling_factor, bool use_fast,
    uint16_t* d_total_code, float* d_factors);

void quantize_full_on_residuals(
    cudaStream_t stream,
    const float* d_residuals, const float* d_centroid,
    size_t N, size_t padded_dim,
    size_t ex_bits, float const_scaling_factor, bool use_fast,
    uint8_t* d_total_code, float* d_factors);

}  // namespace cuvs::preprocessing::quantize::rabitq::detail

#endif // RABITQ_GPU_QUANTIZER_STANDALONE_CUH
