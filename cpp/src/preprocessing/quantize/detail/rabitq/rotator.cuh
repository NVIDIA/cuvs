/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// GPU rotator supporting two implementations (D is the padded width; dim is
// the logical input width):
//   - matmul  : D x D random orthogonal matrix via RAFT GEMM, O(n_rows*D*dim)
//   - fht_kac : Fast Hadamard Transform + Kac's walk, O(n_rows*D*logD)
//
// This layer owns NO state. The rotation matrix / flip bits live in device
// buffers owned by the caller (the public `rabitq::rotator` struct holds them
// in raft containers); the functions below only
//   (a) report the buffer sizes the caller must allocate,
//   (b) generate the state into caller-provided buffers, and
//   (c) launch the rotation kernels against that state.
//
// The derived fht_kac quantities (trunc_dim, fac, log_dim) are recomputed from
// the padded dimension on every launch, so nothing has to be cached.
//

#ifndef RABITQ_GPU_ROTATOR_GPU_CUH
#define RABITQ_GPU_ROTATOR_GPU_CUH

#include <cstdint>

#include <cuda_runtime.h>
#include <cuvs/preprocessing/quantize/rabitq.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

namespace cuvs::preprocessing::quantize::rabitq::detail {

// ---------------------------------------------------------------------------
// Sizes. The public layer allocates its raft containers from these.
// ---------------------------------------------------------------------------

/// Padded working dimension: `dim` rounded up to a multiple of 32. Both
/// rotator kinds implicitly zero-extend shorter input rows to this width.
[[nodiscard]] int64_t padded_dim(int64_t dim);

/// Bytes of flip-bit state the fht_kac rotator consumes at padded dimension
/// `D`: four sign passes of D bits each == 4 * D / 8 bytes.
[[nodiscard]] int64_t flip_bytes(int64_t D);

// ---------------------------------------------------------------------------
// State generation. Writes into device buffers provided by the caller.
// ---------------------------------------------------------------------------

/// Fill `P_out` (D x D, row-major) with a random orthogonal matrix,
/// drawn on the device with the preprocessing helper IVF-PQ uses too.
void init_state_matmul(raft::resources const& res,
                       uint64_t seed,
                       raft::device_matrix_view<float, int64_t, raft::row_major> P_out);

/// Fill `flip_out` (`flip_bytes(D)` bytes) with the Kac's-walk sign bits.
/// The draw happens on the host (std::mt19937, identical to the standalone
/// code), so this stages through host memory and synchronizes `stream`.
void init_state_fht_kac(cudaStream_t stream,
                        uint64_t seed,
                        raft::device_vector_view<uint8_t, int64_t> flip_out);

// ---------------------------------------------------------------------------
// Launchers. `in` is n_rows x dim (dim <= D); `out` is n_rows x D, both row-major.
// In-place aliasing (`in == out`) requires dim == D and fht_kac.
// ---------------------------------------------------------------------------

/// out_i = P * pad(in_i), computed with the cuBLAS handle carried by `res` (which is
/// bound to `res`'s stream). `P` is D x D, row-major.
void rotate_matmul(raft::resources const& res,
                   raft::device_matrix_view<const float, int64_t, raft::row_major> P,
                   raft::device_matrix_view<const float, int64_t, raft::row_major> in,
                   raft::device_matrix_view<float, int64_t, raft::row_major> out);

/// Fused FHT + Kac's-walk rotation. `flip` holds `flip_bytes(D)` bytes.
///
/// `D` is the padded row width, and the truncated-FHT segment size is derived
/// from it (trunc_dim = 2^floor_log2(D)) — the padded width is what the flip bits
/// and the kernels operate on. The transform depends only on `D`; the input
/// width determines the row pitch and which coordinates to load before
/// zero-extending to `D`.
void rotate_fht_kac(cudaStream_t stream,
                    raft::device_vector_view<const uint8_t, int64_t> flip,
                    raft::device_matrix_view<const float, int64_t, raft::row_major> in,
                    raft::device_matrix_view<float, int64_t, raft::row_major> out);

// ---------------------------------------------------------------------------
// Kind-agnostic entry points for internal callers (the pipeline). They only
// borrow the caller's state through the public non-owning `rotator_view`.
// ---------------------------------------------------------------------------

/// Only fht_kac can rotate a buffer onto itself.
[[nodiscard]] inline bool supports_inplace_rotate(rotator_view<const float> rot)
{
  return rot.kind == rotator_kind::fht_kac;
}

/// Zero-pad input rows to `rot.padded_dim` if needed, then dispatch according to `rot.kind`.
/// Input width must be in [1, rot.padded_dim]; output width must be rot.padded_dim.
/// Input and output must be disjoint unless already padded and using fht_kac.
/// Runs on `raft::resource::get_cuda_stream(res)` in both cases. Padding is
/// implicit in GEMM's reduction dimension or fused into the FHT loads.
void rotate(raft::resources const& res,
            rotator_view<const float> rot,
            raft::device_matrix_view<const float, int64_t, raft::row_major> in,
            raft::device_matrix_view<float, int64_t, raft::row_major> out);

}  // namespace cuvs::preprocessing::quantize::rabitq::detail

#endif  // RABITQ_GPU_ROTATOR_GPU_CUH
