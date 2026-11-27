/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// One-call RaBitQ quantization pipeline.
//
// This layer only orchestrates the two underlying components — the rotator
// (rotator.cuh, including padding) and the quantizer (quantize.cuh) — plus
// centroid subtraction. It adds no quantization logic of its own; use the
// components directly for custom pipelines (e.g. externally computed residuals).
//
// Pipeline: rotate (including implicit zero-padding) -> [subtract rotated
// centroid] -> fused quantization on the rotated residuals.
//
// Design note: the centroid is rotated with a separate 1-row rotate() call
// rather than appended to the data batch (as IVF-RaBitQ does). Appending is only
// free when a gather/copy pass over the data exists anyway; for contiguous
// input it would cost a full extra n_rows x padded_dim copy, while rotating the
// centroid needs only one output row.
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
/// data                n_rows x dim on device.
/// centroid            optional dim floats on device, UNrotated.
///                     When given, residuals = rotate(pad(data)) -
///                     rotate(pad(centroid)) and rotated_centroid receives the
///                     rotated centroid (D floats).
///                     When absent, residuals = rotate(pad(data)) and
///                     rotated_centroid is not used (it may be empty).
/// residuals           n_rows x D output (D = rotator.padded_dim): the rotated
///                     residuals that the emitted codes refer to.
/// Remaining parameters mirror quantize_fused_on_residuals.
template <typename CodeT>
void quantize_data(raft::resources const& res,
                   raft::device_matrix_view<const float, int64_t, raft::row_major> data,
                   rotator_view<const float> rotator,
                   std::optional<raft::device_vector_view<const float, int64_t>> centroid,
                   raft::device_vector_view<float, int64_t> rotated_centroid,
                   size_t ex_bits,
                   float const_scaling_factor,
                   bool use_fast,
                   raft::device_matrix_view<CodeT, int64_t, raft::row_major> codes,
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
template <typename CodeT>
void quantize_data_full(raft::resources const& res,
                        raft::device_matrix_view<const float, int64_t, raft::row_major> data,
                        rotator_view<const float> rotator,
                        std::optional<raft::device_vector_view<const float, int64_t>> centroid,
                        raft::device_vector_view<float, int64_t> rotated_centroid,
                        size_t ex_bits,
                        float const_scaling_factor,
                        bool use_fast,
                        raft::device_matrix_view<CodeT, int64_t, raft::row_major> codes,
                        raft::device_matrix_view<float, int64_t, raft::row_major> factors,
                        raft::device_matrix_view<float, int64_t, raft::row_major> residuals,
                        int coarse_samples,
                        int fine_samples);

}  // namespace cuvs::preprocessing::quantize::rabitq::detail

#endif  // RABITQ_GPU_PIPELINE_CUH
