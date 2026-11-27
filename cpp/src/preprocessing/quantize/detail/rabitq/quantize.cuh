/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Packed RaBitQ quantization and per-vector factors on rotated residuals.

#ifndef RABITQ_GPU_QUANTIZER_STANDALONE_CUH
#define RABITQ_GPU_QUANTIZER_STANDALONE_CUH

#include <raft/core/device_mdspan.hpp>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace cuvs::preprocessing::quantize::rabitq::detail {

/** Output pointers for a row batch; sign_stride is the full output row count. */
struct packed_codes_output {
  uint32_t* signs;
  uint8_t* extra_bits;
  int64_t sign_stride;
};

/// Quantize pre-computed, rotated residuals into packed RaBitQ codes.
///
/// stream      : the caller owns the stream. Every one of these entry points
///               only enqueues kernel launches on it and is therefore
///               *asynchronous*: the outputs are not readable until the caller
///               synchronizes the stream (or orders subsequent work on it).
///               All input and output buffers must stay alive and unmodified
///               until that work completes.
/// residuals   : must be N × padded_dim on device (rotated, zero-centroid).
///
/// Writes packed signs and extra bits, plus a per-vector (delta, vl) pair.
///
/// delta_mode: 0 = RECONSTRUCTION, 1 = UNBIASED, 2 = PLAIN.
void quantize_fused_on_residuals(
  cudaStream_t stream,
  raft::device_matrix_view<const float, int64_t, raft::row_major> residuals,
  size_t ex_bits,
  float const_scaling_factor,
  bool use_fast,
  packed_codes_output codes,
  raft::device_vector_view<float, int64_t> delta,
  raft::device_vector_view<float, int64_t> vl,
  int delta_mode     = 0,
  int coarse_samples = 64,
  int fine_samples   = 64);

/// Quantize pre-computed, rotated residuals and produce the full
/// (f_add, f_rescale, f_error) factor triplet used for approximate-distance
/// estimation during search.
///
/// stream      : the caller owns the stream; the call is asynchronous on it
///               (see quantize_fused_on_residuals above).
/// residuals   : N × padded_dim on device (rotated, centroid already subtracted).
/// centroid    : padded_dim floats on device (the rotated centroid) — pass a
///               zero-filled buffer when there is no centroid.
/// factors     : N × 3 floats on device — layout
///               [f_add, f_rescale, f_error] per vector.
void quantize_full_on_residuals(
  cudaStream_t stream,
  raft::device_matrix_view<const float, int64_t, raft::row_major> residuals,
  raft::device_vector_view<const float, int64_t> centroid,
  size_t ex_bits,
  float const_scaling_factor,
  bool use_fast,
  packed_codes_output codes,
  raft::device_matrix_view<float, int64_t, raft::row_major> factors,
  int coarse_samples,
  int fine_samples);

}  // namespace cuvs::preprocessing::quantize::rabitq::detail

#endif  // RABITQ_GPU_QUANTIZER_STANDALONE_CUH
