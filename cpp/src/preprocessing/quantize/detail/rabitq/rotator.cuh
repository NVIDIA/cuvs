/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Imported from the standalone RaBitQ GPU quantizer. Behaviour unchanged;
 * only include paths, error macros and the enclosing namespace differ.
 */

//
// GPU rotator supporting two implementations:
//   - matmul  : full D x D random orthogonal matrix via cuBLAS sgemm, O(N*D^2)
//   - fht_kac : Fast Hadamard Transform + Kac's walk, O(N*D*logD)
//
// This layer owns NO state. The rotation matrix / flip bits live in device
// buffers owned by the caller (the public `rabitq::rotator` struct holds them
// in raft containers); the functions below only
//   (a) report the buffer sizes the caller must allocate,
//   (b) generate the state into caller-provided buffers, and
//   (c) launch the rotation kernels against that state.
//
// The derived fht_kac quantities (trunc_dim, fac, log_n) are recomputed from
// the padded dimension on every launch, so nothing has to be cached.
//

#ifndef RABITQ_GPU_ROTATOR_GPU_CUH
#define RABITQ_GPU_ROTATOR_GPU_CUH

#include <cstdint>

#include <cuda_runtime.h>
#include <raft/core/resources.hpp>

namespace cuvs::preprocessing::quantize::rabitq::detail {

// ---------------------------------------------------------------------------
// Sizes. The public layer allocates its raft containers from these.
// ---------------------------------------------------------------------------

/// Padded working dimension: `dim` rounded up to a multiple of 32. Both
/// rotator kinds operate on padded rows of this width.
[[nodiscard]] int64_t padded_dim(int64_t dim);

/// Bytes of flip-bit state the fht_kac rotator consumes at padded dimension
/// `D`: four sign passes of D bits each == 4 * D / 8 bytes.
[[nodiscard]] int64_t flip_bytes(int64_t D);

// ---------------------------------------------------------------------------
// State generation. Writes into device buffers provided by the caller. The
// draw happens on the host (std::mt19937, identical to the standalone code),
// so both calls stage through pinned-free host memory and therefore
// synchronize `stream` before returning.
// ---------------------------------------------------------------------------

/// Fill `d_P_out` (D * D floats, row-major) with a random orthogonal matrix.
void init_state_matmul(cudaStream_t stream, uint64_t seed, int64_t D, float* d_P_out);

/// Fill `d_flip_out` (`flip_bytes(D)` bytes) with the Kac's-walk sign bits.
void init_state_fht_kac(cudaStream_t stream, uint64_t seed, int64_t D, uint8_t* d_flip_out);

// ---------------------------------------------------------------------------
// Launchers. `in` and `out` are N x D row-major float matrices.
// In-place aliasing (`in == out`) is permitted for fht_kac only.
// ---------------------------------------------------------------------------

/// out = P * in, computed with the cuBLAS handle carried by `res` (which is
/// bound to `res`'s stream). `d_P` is D * D floats, row-major.
void rotate_matmul(raft::resources const& res,
                   const float* d_P,
                   const float* in,
                   float* out,
                   int64_t N,
                   int64_t D);

/// Fused FHT + Kac's-walk rotation. `d_flip` is `flip_bytes(D)` bytes.
///
/// `D` is the padded row width, and the truncated-FHT segment size is derived
/// from it (trunc_dim = 2^floor_log2(D)) — the padded width is what the flip bits
/// and the kernels operate on. The transform therefore depends only on `D`; the
/// original unpadded dimension is not needed here.
void rotate_fht_kac(cudaStream_t stream,
                    const uint8_t* d_flip,
                    const float* in,
                    float* out,
                    int64_t N,
                    int64_t D);

// ---------------------------------------------------------------------------
// Non-owning shim over the two kinds, for internal callers (the pipeline)
// that are kind-agnostic. It only *borrows* the caller's state; the kind tag
// stays a plain bool here so that the detail layer does not have to duplicate
// the public `rabitq::rotator_kind` enum.
// ---------------------------------------------------------------------------
struct rotator_ref {
  /// Padded row width; must equal padded_dim(dim) of the owning rotator. Both
  /// kinds are fully determined by this width plus the state pointers below.
  int64_t padded_dim = 0;
  /// true -> fht_kac (uses `flip_bits`), false -> matmul (uses `rotation_matrix`).
  bool use_fht_kac = true;
  const float* rotation_matrix = nullptr;
  const uint8_t* flip_bits     = nullptr;
};

/// Only fht_kac can rotate a buffer onto itself.
[[nodiscard]] inline bool supports_inplace_rotate(rotator_ref const& rot)
{
  return rot.use_fht_kac;
}

/// Dispatch to rotate_fht_kac / rotate_matmul according to `rot.use_fht_kac`.
/// Runs on `raft::resource::get_cuda_stream(res)` in both cases.
void rotate(
  raft::resources const& res, rotator_ref const& rot, const float* in, float* out, int64_t N);

}  // namespace cuvs::preprocessing::quantize::rabitq::detail

#endif  // RABITQ_GPU_ROTATOR_GPU_CUH
