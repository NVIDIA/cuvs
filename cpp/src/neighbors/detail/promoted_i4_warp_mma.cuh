/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/util/cudart_utils.hpp>

#include <mma.h>

#include <cstdint>
#include <type_traits>

// Warp-level tensor-core dot products of i4 codes (0..15, two per byte) staged in shared memory as
// rows of RowStride bytes. A warp owns a WarpTile x WarpTile block of the output: accumulate() adds
// the dot products over one staged K range, store() writes the raw sums as raw_t.

namespace cuvs::neighbors::nn_descent::detail {

// Masks one nibble half of a packed-i4 fragment into `dst`, normalised to 0..15. The high half is
// shifted down so both halves reach the MMA at the same scale and can share one accumulator.
template <bool High, typename FragT>
__device__ __forceinline__ void select_packed_nibble(const FragT& src, FragT& dst)
{
  constexpr int kWords = FragT::num_elements / 4;
  static_assert(FragT::num_elements % 4 == 0, "fragment must be a whole number of u32 words");
  const auto* w = reinterpret_cast<const uint32_t*>(src.x);
  auto* d       = reinterpret_cast<uint32_t*>(dst.x);
#pragma unroll
  for (int i = 0; i < kWords; ++i) {
    d[i] = High ? ((w[i] >> 4) & 0x0f0f0f0fu) : (w[i] & 0x0f0f0f0fu);
  }
}

// The staged bytes are packed i4; the MMA reads them as u8 and splits the nibbles here rather than
// leaving it to ptxas. u4 MMA is native only through sm_89, and past that ptxas emits this same
// split out of line, once per MMA. Inlining it keeps the IMMAs in the loop body on every arch, and
// the 16x16x16 tile u8 allows drops SUB_M/SUB_N to 1.
template <int WarpTile, int RowStride>
struct i4_warp_mma_u8 {
  using raw_t = int;

  static constexpr int MMA_M = 16;
  static constexpr int MMA_N = 16;
  // A k-step reads 16 packed bytes = 32 codes; each nibble half is a separate MMA over MMA_K.
  static constexpr int MMA_K       = 16;
  static constexpr int KSTEP_BYTES = MMA_K;
  static constexpr int SUB_M       = WarpTile / MMA_M;
  static constexpr int SUB_N       = WarpTile / MMA_N;
  static_assert(WarpTile % MMA_M == 0 && WarpTile % MMA_N == 0,
                "the warp tile must be a whole number of native MMA tiles");
  static_assert(RowStride % 16 == 0, "row stride must preserve 16-byte IMMA row alignment");

  using a_frag_t = nvcuda::wmma::
    fragment<nvcuda::wmma::matrix_a, MMA_M, MMA_N, MMA_K, uint8_t, nvcuda::wmma::row_major>;
  using b_frag_t = nvcuda::wmma::
    fragment<nvcuda::wmma::matrix_b, MMA_M, MMA_N, MMA_K, uint8_t, nvcuda::wmma::col_major>;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, MMA_M, MMA_N, MMA_K, int> c[SUB_M][SUB_N];

  __device__ __forceinline__ i4_warp_mma_u8()
  {
    for (int msub = 0; msub < SUB_M; ++msub) {
      for (int nsub = 0; nsub < SUB_N; ++nsub) {
        nvcuda::wmma::fill_fragment(c[msub][nsub], 0);
      }
    }
  }

  // a_frag depends only on (msub, kk); b_frag depends only on (nsub, kk) -- load each once per kk
  // and reuse across the other sub-tile index, instead of reloading redundantly inside a full
  // msub x nsub x kk cross product.
  // Kept rolled: unrolling the k-steps keeps more fragment live ranges simultaneous and costs a
  // block per SM in registers, for some intra-warp ILP.
  template <int RowBytes>
  __device__ __forceinline__ void accumulate(const uint8_t (*a)[RowStride],
                                             const uint8_t (*b)[RowStride],
                                             int row0,
                                             int col0)
  {
#pragma unroll 1
    for (int kk = 0; kk < RowBytes / KSTEP_BYTES; ++kk) {
      a_frag_t a_frag[SUB_M];
      b_frag_t b_frag[SUB_N];
      for (int msub = 0; msub < SUB_M; ++msub) {
        nvcuda::wmma::load_matrix_sync(
          a_frag[msub], a[row0 + msub * MMA_M] + kk * KSTEP_BYTES, RowStride);
      }
      for (int nsub = 0; nsub < SUB_N; ++nsub) {
        nvcuda::wmma::load_matrix_sync(
          b_frag[nsub], b[col0 + nsub * MMA_N] + kk * KSTEP_BYTES, RowStride);
      }
      // One scratch fragment per operand, reused across the halves.
      a_frag_t a_half[SUB_M];
      b_frag_t b_half[SUB_N];
      auto accumulate_half = [&](auto high_tag) {
        constexpr bool High = decltype(high_tag)::value;
        for (int msub = 0; msub < SUB_M; ++msub) {
          select_packed_nibble<High>(a_frag[msub], a_half[msub]);
        }
        for (int nsub = 0; nsub < SUB_N; ++nsub) {
          select_packed_nibble<High>(b_frag[nsub], b_half[nsub]);
        }
        for (int msub = 0; msub < SUB_M; ++msub) {
          for (int nsub = 0; nsub < SUB_N; ++nsub) {
            nvcuda::wmma::mma_sync(c[msub][nsub], a_half[msub], b_half[nsub], c[msub][nsub]);
          }
        }
      };
      accumulate_half(std::false_type{});
      accumulate_half(std::true_type{});
    }
  }

  template <int Stride>
  __device__ __forceinline__ void store(raw_t* dst, int row0, int col0) const
  {
    for (int msub = 0; msub < SUB_M; ++msub) {
      for (int nsub = 0; nsub < SUB_N; ++nsub) {
        nvcuda::wmma::store_matrix_sync(dst + (row0 + msub * MMA_M) * Stride + col0 + nsub * MMA_N,
                                        c[msub][nsub],
                                        Stride,
                                        nvcuda::wmma::mem_row_major);
      }
    }
  }
};

}  // namespace cuvs::neighbors::nn_descent::detail
