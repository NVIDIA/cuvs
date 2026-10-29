/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Imported from the standalone RaBitQ GPU quantizer. Behaviour unchanged
 * except for the matmul rotator: its matrix comes from the shared
 * preprocessing helper and is applied as y = P x.
 */

//
// Rotator state generation and launchers — matmul and fht_kac.
// The random orthogonal matrix comes from the preprocessing helper that
// IVF-PQ uses too.
//

#include "../../../detail/rotation_matrix.cuh"
#include "fht.cuh"
#include "rotator.cuh"

#include <raft/core/cublas_macros.hpp>
#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/core/error.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <cublas_v2.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace cuvs::preprocessing::quantize::rabitq::detail {

namespace {

inline int64_t rd_up(int64_t dim, int64_t mult) { return ((dim + mult - 1) / mult) * mult; }

inline int64_t floor_log2(int64_t x)
{
  int64_t r = 0;
  while (x >>= 1) {
    ++r;
  }
  return r;
}

/// Derived fht_kac parameters. Recomputed on every launch from the PADDED
/// dimension, matching the standalone rotator (see its commit "Derive the
/// FHT-Kac segment size from the padded dim": the segment size follows the
/// padded width, which is also the width the flip bits and the kernels work on).
/// A useful side effect: log_n then always lands in [5, 15], so small dims can
/// never fall below fht::dispatch_log_n's supported range.
struct fht_kac_derived {
  int64_t trunc_dim;
  int log_n;
  float fac;
};

inline fht_kac_derived derive_fht_kac(int64_t D)
{
  const int64_t bottom_log = floor_log2(D);
  const int64_t trunc_dim  = int64_t{1} << bottom_log;
  return fht_kac_derived{
    trunc_dim, static_cast<int>(bottom_log), 1.0f / std::sqrt(static_cast<float>(trunc_dim))};
}

}  // namespace

// ---------------------------------------------------------------------------
// Sizes
// ---------------------------------------------------------------------------

int64_t padded_dim(int64_t dim) { return rd_up(dim, 32); }

int64_t flip_bytes(int64_t D) { return 4 * D / 8; }

// ---------------------------------------------------------------------------
// State generation
// ---------------------------------------------------------------------------

void init_state_matmul(raft::resources const& res, uint64_t seed, int64_t D, float* d_P_out)
{
  RAFT_EXPECTS(D > 0 && D <= std::numeric_limits<uint16_t>::max(),
               "rotator: matmul padded dimension must be in [1, 65535]");
  cuvs::preprocessing::detail::make_rotation_matrix(res,
                                                    true,
                                                    static_cast<uint32_t>(D),
                                                    static_cast<uint32_t>(D),
                                                    d_P_out,
                                                    raft::random::RngState(seed));
}

void init_state_fht_kac(cudaStream_t stream, uint64_t seed, int64_t D, uint8_t* d_flip_out)
{
  RAFT_EXPECTS(D > 0, "rotator: padded dimension must be positive");
  std::vector<uint8_t> h_flip(static_cast<size_t>(flip_bytes(D)));

  std::mt19937 gen(static_cast<uint32_t>(seed));
  std::uniform_int_distribution<int> dist(0, 255);
  for (auto& b : h_flip)
    b = static_cast<uint8_t>(dist(gen));

  RAFT_CUDA_TRY(cudaMemcpyAsync(
    d_flip_out, h_flip.data(), h_flip.size(), cudaMemcpyHostToDevice, stream));
  // h_flip is plain host memory and dies with this scope.
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
}

// ---------------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------------

void rotate_matmul(raft::resources const& res,
                   const float* d_P,
                   const float* in,
                   float* out,
                   int64_t N,
                   int64_t D)
{
  if (N <= 0) { return; }
  const float alpha = 1.0f;
  const float beta  = 0.0f;
  auto cublas_h     = raft::resource::get_cublas_handle(res);
  RAFT_CUBLAS_TRY(cublasSetStream(cublas_h, raft::resource::get_cuda_stream(res)));
  // Row-major row-of-A (size D) is reinterpreted as column-of-A in column-major,
  // and row-major P reads as P^T; compute C = P · A by transposing it back.
  RAFT_CUBLAS_TRY(cublasSgemm(cublas_h,
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              static_cast<int>(D),
                              static_cast<int>(N),
                              static_cast<int>(D),
                              &alpha,
                              d_P,
                              static_cast<int>(D),
                              in,
                              static_cast<int>(D),
                              &beta,
                              out,
                              static_cast<int>(D)));
}

void rotate_fht_kac(cudaStream_t stream,
                    const uint8_t* d_flip,
                    const float* in,
                    float* out,
                    int64_t N,
                    int64_t D)
{
  if (N <= 0) { return; }
  const auto p = derive_fht_kac(D);
  if (p.trunc_dim == D) {
    const float total_scale = p.fac * p.fac * p.fac * p.fac;
    fht::dispatch_fused_rotate(
      in, out, d_flip, static_cast<int>(N), p.log_n, total_scale, stream);
  } else {
    fht::dispatch_fused_rotate_nonpow2(in,
                                       out,
                                       d_flip,
                                       static_cast<int>(N),
                                       p.log_n,
                                       static_cast<int>(D),
                                       p.fac,
                                       0.25f,
                                       stream);
  }
}

void rotate(
  raft::resources const& res, rotator_ref const& rot, const float* in, float* out, int64_t N)
{
  if (rot.use_fht_kac) {
    rotate_fht_kac(
      raft::resource::get_cuda_stream(res), rot.flip_bits, in, out, N, rot.padded_dim);
  } else {
    rotate_matmul(res, rot.rotation_matrix, in, out, N, rot.padded_dim);
  }
}

}  // namespace cuvs::preprocessing::quantize::rabitq::detail
