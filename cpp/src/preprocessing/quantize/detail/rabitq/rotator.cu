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
#include <raft/core/error.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cuda_dev_essentials.cuh>
#include <raft/util/cuda_rt_essentials.hpp>

#include <cublas_v2.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace cuvs::preprocessing::quantize::rabitq::detail {

namespace {

/// Derived fht_kac parameters. Recomputed on every launch from the PADDED
/// dimension, matching the standalone rotator (see its commit "Derive the
/// FHT-Kac segment size from the padded dim": the segment size follows the
/// padded width, which is also the width the flip bits and the kernels work on).
/// A useful side effect: log_n is then always at least 5, so small dims can
/// never fall below fht::dispatch_log_n's supported range.
struct fht_kac_derived {
  int64_t trunc_dim;
  int log_n;
  float fac;
};

inline fht_kac_derived derive_fht_kac(int64_t D)
{
  const int64_t bottom_log = std::bit_width(static_cast<uint64_t>(D)) - 1;
  const int64_t trunc_dim  = int64_t{1} << bottom_log;
  return fht_kac_derived{
    trunc_dim, static_cast<int>(bottom_log), 1.0f / std::sqrt(static_cast<float>(trunc_dim))};
}

}  // namespace

// ---------------------------------------------------------------------------
// Sizes
// ---------------------------------------------------------------------------

int64_t padded_dim(int64_t dim) { return raft::alignTo(dim, int64_t{32}); }

int64_t flip_bytes(int64_t D) { return 4 * D / 8; }

// ---------------------------------------------------------------------------
// State generation
// ---------------------------------------------------------------------------

void init_state_matmul(raft::resources const& res,
                       uint64_t seed,
                       raft::device_matrix_view<float, int64_t, raft::row_major> P_out)
{
  const int64_t D = P_out.extent(0);
  RAFT_EXPECTS(D > 0 && D <= std::numeric_limits<uint16_t>::max(),
               "rotator: matmul padded dimension must be in [1, 65535]");
  cuvs::preprocessing::detail::make_rotation_matrix(res,
                                                    true,
                                                    static_cast<uint32_t>(D),
                                                    static_cast<uint32_t>(D),
                                                    P_out.data_handle(),
                                                    raft::random::RngState(seed));
}

void init_state_fht_kac(cudaStream_t stream,
                        uint64_t seed,
                        raft::device_vector_view<uint8_t, int64_t> flip_out)
{
  RAFT_EXPECTS(flip_out.extent(0) > 0, "rotator: padded dimension must be positive");
  std::vector<uint8_t> h_flip(static_cast<size_t>(flip_out.extent(0)));

  std::mt19937 gen(static_cast<uint32_t>(seed));
  std::uniform_int_distribution<int> dist(0, 255);
  for (auto& b : h_flip)
    b = static_cast<uint8_t>(dist(gen));

  RAFT_CUDA_TRY(cudaMemcpyAsync(
    flip_out.data_handle(), h_flip.data(), h_flip.size(), cudaMemcpyHostToDevice, stream));
  // h_flip is plain host memory and dies with this scope.
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
}

// ---------------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------------

void rotate_matmul(raft::resources const& res,
                   raft::device_matrix_view<const float, int64_t, raft::row_major> P,
                   raft::device_matrix_view<const float, int64_t, raft::row_major> in,
                   raft::device_matrix_view<float, int64_t, raft::row_major> out)
{
  const int64_t N = in.extent(0);
  const int64_t D = P.extent(0);
  if (N <= 0) { return; }
  const float alpha = 1.0f;
  const float beta  = 0.0f;
  auto cublas_h     = raft::resource::get_cublas_handle(res);
  RAFT_CUBLAS_TRY(cublasSetStream(cublas_h, raft::resource::get_cuda_stream(res).get()));
  // Row-major row-of-A (size D) is reinterpreted as column-of-A in column-major,
  // and row-major P reads as P^T; compute C = P · A by transposing it back.
  RAFT_CUBLAS_TRY(cublasSgemm(cublas_h,
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              static_cast<int>(D),
                              static_cast<int>(N),
                              static_cast<int>(D),
                              &alpha,
                              P.data_handle(),
                              static_cast<int>(D),
                              in.data_handle(),
                              static_cast<int>(D),
                              &beta,
                              out.data_handle(),
                              static_cast<int>(D)));
}

void rotate_fht_kac(cudaStream_t stream,
                    raft::device_vector_view<const uint8_t, int64_t> flip,
                    raft::device_matrix_view<const float, int64_t, raft::row_major> in,
                    raft::device_matrix_view<float, int64_t, raft::row_major> out)
{
  const int64_t N = in.extent(0);
  const int64_t D = in.extent(1);
  if (N <= 0) { return; }
  const auto p = derive_fht_kac(D);
  if (p.trunc_dim == D) {
    const float total_scale = p.fac * p.fac * p.fac * p.fac;
    fht::dispatch_fused_rotate(in.data_handle(),
                               out.data_handle(),
                               flip.data_handle(),
                               static_cast<int>(N),
                               p.log_n,
                               total_scale,
                               stream);
  } else {
    fht::dispatch_fused_rotate_nonpow2(in.data_handle(),
                                       out.data_handle(),
                                       flip.data_handle(),
                                       static_cast<int>(N),
                                       p.log_n,
                                       static_cast<int>(D),
                                       p.fac,
                                       0.25f,
                                       stream);
  }
}

void rotate(raft::resources const& res,
            rotator_view<const float> rot,
            raft::device_matrix_view<const float, int64_t, raft::row_major> in,
            raft::device_matrix_view<float, int64_t, raft::row_major> out)
{
  if (rot.kind == rotator_kind::fht_kac) {
    rotate_fht_kac(raft::resource::get_cuda_stream(res).get(), rot.flip_bits, in, out);
  } else {
    rotate_matmul(res, rot.rotation_matrix, in, out);
  }
}

}  // namespace cuvs::preprocessing::quantize::rabitq::detail
