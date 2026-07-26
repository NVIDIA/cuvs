/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Imported from the standalone RaBitQ GPU quantizer. Behaviour unchanged;
 * only include paths, error macros and the enclosing namespace differ.
 */

//
// One-call RaBitQ quantization pipeline.
//
// This layer only orchestrates the two underlying components — the rotator
// (rotator.cuh) and the quantizer (quantize.cuh) — plus two trivial glue
// kernels (row padding, centroid subtraction). It adds no quantization logic
// of its own; use the components directly for custom pipelines (e.g.
// externally computed residuals).
//
// Pipeline: pad(dim -> rotator.padded_dim) -> rotate -> [subtract rotated
// centroid] -> fused quantization on the rotated residuals.
//
// Design note: the centroid is rotated with a separate 1-row rotate() call
// rather than appended to the data batch (as cuVS does). Appending is only
// free when a gather/copy pass over the data exists anyway; for contiguous
// input it would cost a full extra N x D copy, while a 1-row rotate is a
// single tiny launch.
//
// The rotator state is NOT owned here: callers pass a non-owning rotator_ref
// that borrows the buffers held by the public `rabitq::rotator` struct.
//

#ifndef RABITQ_GPU_PIPELINE_CUH
#define RABITQ_GPU_PIPELINE_CUH

#include <cstddef>
#include <cstdint>

#include <raft/core/resources.hpp>

#include "rotator.cuh"

namespace cuvs::preprocessing::quantize::rabitq::detail {

/// Quantize raw (unrotated, unpadded) vectors end to end, emitting per-vector
/// scalar factors (delta, vl).
///
/// d_data              N x dim on device.
/// d_centroid          optional (nullable) dim floats on device, UNrotated.
///                     When non-null, residuals = rotate(pad(data)) -
///                     rotate(pad(centroid)) and d_rotated_centroid must be
///                     non-null (receives the rotated centroid, D floats).
///                     When null, residuals = rotate(pad(data)) and
///                     d_rotated_centroid may be null.
/// d_residuals         N x D output (D = rotator.padded_dim): the rotated
///                     residuals that the emitted codes refer to.
/// Remaining parameters mirror quantize_fused_on_residuals.
void quantize_data(raft::resources const& res,
                   const float* d_data,
                   size_t N,
                   size_t dim,
                   rotator_ref const& rotator,
                   const float* d_centroid,
                   float* d_rotated_centroid,
                   size_t ex_bits,
                   float const_scaling_factor,
                   bool use_fast,
                   uint16_t* d_total_code,
                   float* d_delta,
                   float* d_vl,
                   float* d_residuals,
                   int delta_mode      = 0,
                   int coarse_samples  = 64,
                   int fine_samples    = 64);

void quantize_data(raft::resources const& res,
                   const float* d_data,
                   size_t N,
                   size_t dim,
                   rotator_ref const& rotator,
                   const float* d_centroid,
                   float* d_rotated_centroid,
                   size_t ex_bits,
                   float const_scaling_factor,
                   bool use_fast,
                   uint8_t* d_total_code,
                   float* d_delta,
                   float* d_vl,
                   float* d_residuals,
                   int delta_mode      = 0,
                   int coarse_samples  = 64,
                   int fine_samples    = 64);

/// Full-factor variant: emits the (f_add, f_rescale, f_error) triplet used
/// for approximate-distance estimation. d_rotated_centroid (D floats) is
/// always required here — it participates in the factor computation — and is
/// zero-filled when d_centroid is null.
void quantize_data_full(raft::resources const& res,
                        const float* d_data,
                        size_t N,
                        size_t dim,
                        rotator_ref const& rotator,
                        const float* d_centroid,
                        float* d_rotated_centroid,
                        size_t ex_bits,
                        float const_scaling_factor,
                        bool use_fast,
                        uint16_t* d_total_code,
                        float* d_factors,
                        float* d_residuals);

void quantize_data_full(raft::resources const& res,
                        const float* d_data,
                        size_t N,
                        size_t dim,
                        rotator_ref const& rotator,
                        const float* d_centroid,
                        float* d_rotated_centroid,
                        size_t ex_bits,
                        float const_scaling_factor,
                        bool use_fast,
                        uint8_t* d_total_code,
                        float* d_factors,
                        float* d_residuals);

}  // namespace cuvs::preprocessing::quantize::rabitq::detail

#endif  // RABITQ_GPU_PIPELINE_CUH
