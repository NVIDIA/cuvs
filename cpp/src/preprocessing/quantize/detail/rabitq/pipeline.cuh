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
// The rotator state is NOT owned here: callers pass a non-owning rotator_view
// that borrows the buffers held by the public `rabitq::rotator` struct.
//

#ifndef RABITQ_GPU_PIPELINE_CUH
#define RABITQ_GPU_PIPELINE_CUH

#include <cstddef>
#include <cstdint>
#include <optional>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

#include "rotator.cuh"

namespace cuvs::preprocessing::quantize::rabitq::detail {

/// Quantize raw (unrotated, unpadded) vectors end to end, emitting per-vector
/// scalar factors (delta, vl).
///
/// data                N x dim on device.
/// centroid            optional dim floats on device, UNrotated.
///                     When given, residuals = rotate(pad(data)) -
///                     rotate(pad(centroid)) and rotated_centroid receives the
///                     rotated centroid (D floats).
///                     When absent, residuals = rotate(pad(data)) and
///                     rotated_centroid is not used (it may be empty).
/// residuals           N x D output (D = rotator.padded_dim): the rotated
///                     residuals that the emitted codes refer to.
/// Remaining parameters mirror quantize_fused_on_residuals.
void quantize_data(raft::resources const& res,
                   raft::device_matrix_view<const float, int64_t, raft::row_major> data,
                   rotator_view<const float> rotator,
                   std::optional<raft::device_vector_view<const float, int64_t>> centroid,
                   raft::device_vector_view<float, int64_t> rotated_centroid,
                   size_t ex_bits,
                   float const_scaling_factor,
                   bool use_fast,
                   raft::device_matrix_view<uint16_t, int64_t, raft::row_major> codes,
                   raft::device_vector_view<float, int64_t> delta,
                   raft::device_vector_view<float, int64_t> vl,
                   raft::device_matrix_view<float, int64_t, raft::row_major> residuals,
                   int delta_mode     = 0,
                   int coarse_samples = 64,
                   int fine_samples   = 64);

void quantize_data(raft::resources const& res,
                   raft::device_matrix_view<const float, int64_t, raft::row_major> data,
                   rotator_view<const float> rotator,
                   std::optional<raft::device_vector_view<const float, int64_t>> centroid,
                   raft::device_vector_view<float, int64_t> rotated_centroid,
                   size_t ex_bits,
                   float const_scaling_factor,
                   bool use_fast,
                   raft::device_matrix_view<uint8_t, int64_t, raft::row_major> codes,
                   raft::device_vector_view<float, int64_t> delta,
                   raft::device_vector_view<float, int64_t> vl,
                   raft::device_matrix_view<float, int64_t, raft::row_major> residuals,
                   int delta_mode     = 0,
                   int coarse_samples = 64,
                   int fine_samples   = 64);

/// Full-factor variant: emits the (f_add, f_rescale, f_error) triplet used
/// for approximate-distance estimation. rotated_centroid (D floats) is
/// always required here — it participates in the factor computation — and is
/// zero-filled when there is no centroid.
void quantize_data_full(raft::resources const& res,
                        raft::device_matrix_view<const float, int64_t, raft::row_major> data,
                        rotator_view<const float> rotator,
                        std::optional<raft::device_vector_view<const float, int64_t>> centroid,
                        raft::device_vector_view<float, int64_t> rotated_centroid,
                        size_t ex_bits,
                        float const_scaling_factor,
                        bool use_fast,
                        raft::device_matrix_view<uint16_t, int64_t, raft::row_major> codes,
                        raft::device_matrix_view<float, int64_t, raft::row_major> factors,
                        raft::device_matrix_view<float, int64_t, raft::row_major> residuals,
                        int coarse_samples,
                        int fine_samples);

void quantize_data_full(raft::resources const& res,
                        raft::device_matrix_view<const float, int64_t, raft::row_major> data,
                        rotator_view<const float> rotator,
                        std::optional<raft::device_vector_view<const float, int64_t>> centroid,
                        raft::device_vector_view<float, int64_t> rotated_centroid,
                        size_t ex_bits,
                        float const_scaling_factor,
                        bool use_fast,
                        raft::device_matrix_view<uint8_t, int64_t, raft::row_major> codes,
                        raft::device_matrix_view<float, int64_t, raft::row_major> factors,
                        raft::device_matrix_view<float, int64_t, raft::row_major> residuals,
                        int coarse_samples,
                        int fine_samples);

}  // namespace cuvs::preprocessing::quantize::rabitq::detail

#endif  // RABITQ_GPU_PIPELINE_CUH
