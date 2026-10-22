/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common.cuh"

#include <raft/util/arch.cuh>
#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda_fp16.h>

#include <vector>

namespace cuvs::neighbors::experimental::gpipnn::detail {

/**
 * One kernel for the whole of a leaf self-join: gather, dot products, scoring and selection, with
 * the score matrix never leaving registers. Output is bit-identical to the unfused path, which
 * materializes the score matrix and reads it back to select from.
 */

/** Padding in halves on each shared-memory row, to break the stride on operand reads. */
inline constexpr int kFusedSmemPadH = 8;

/** Threads per CTA. */
inline constexpr int kFusedThreads = 1024;

/**
 * Geometry of one fused tile, parameterized on the query rows it covers.
 *
 * Per candidate chunk the operand traffic is `BlockN * dim * 2` bytes against
 * `BlockM * BlockN * dim * 2` FLOP of maths, so arithmetic intensity is `BlockM` alone: the tile
 * wants to be tall and narrow rather than square, unlike a normal GEMM. That holds only while the Q
 * tile stays shared-memory resident across every chunk, so `BlockM` trades against `dim` for one
 * shared budget -- hence two instantiations, with `fused_leaf_tile_m` picking between them.
 * `BlockN = 32` is the narrowest that keeps every warp holding an n8 tile.
 */
template <int BlockM>
struct fused_tile_geometry {
  static constexpr int kBlockM    = BlockM;
  static constexpr int kBlockN    = 32;
  static constexpr int kWarps     = kFusedThreads / 32;
  static constexpr int kRowGroups = kBlockM / 16;
  static constexpr int kColGroups = kWarps / kRowGroups;
  static constexpr int kNTiles    = kBlockN / kColGroups / 8;

  static_assert(kBlockM % 16 == 0, "a row group is one 16-row mma tile");
  static_assert(kRowGroups * kColGroups == kWarps, "warps must tile the output exactly");
  static_assert(kNTiles >= 1, "every warp must own at least one n8 tile");
};

/** One unit of work: the `m_begin`-th row block of a segment. Built on the host. */
struct fused_tile {
  int32_t seg_id;
  int32_t m_begin;
};

/**
 * Bytes of the candidate region, which doubles as the drain's scratch for the per-column-group
 * partial lists and so is sized for whichever is larger. The two never overlap in time: the partial
 * lists go live after the last chunk, once the candidate buffers are dead.
 */
template <typename Geom, int K>
__host__ __device__ inline auto fused_cand_region_halves(int64_t dim_pad) -> int64_t
{
  constexpr int64_t part = static_cast<int64_t>(Geom::kBlockM) * Geom::kColGroups * K *
                           sizeof(uint64_t) / static_cast<int64_t>(sizeof(half));
  const int64_t buffers = 2 * Geom::kBlockN * dim_pad;
  return buffers > part ? buffers : part;
}

/** Dynamic shared memory one fused tile needs. */
template <typename Geom, int K>
inline auto fused_smem_bytes(int64_t dim) -> size_t
{
  const int64_t dim_pad = dim + kFusedSmemPadH;
  const size_t meta     = (Geom::kBlockM + Geom::kBlockN) * (sizeof(float) + sizeof(id_t)) +
                      Geom::kBlockM * sizeof(uint32_t);
  return (Geom::kBlockM * dim_pad + fused_cand_region_halves<Geom, K>(dim_pad)) * sizeof(half) +
         meta;
}

/** Ascending insert into a short sorted register list, dropping what falls off the end. */
template <int K>
__device__ inline void fused_insert_sorted(uint64_t (&best)[K], uint64_t x)
{
  if (x >= best[K - 1]) { return; }
#pragma unroll
  for (int i = 0; i < K; i++) {
    const uint64_t b = best[i];
    const bool take  = x < b;
    best[i]          = take ? x : b;
    x                = take ? b : x;
  }
}

/**
 * Whether the mainloop can be built for the architecture currently being targeted.
 *
 * `mma.sync.m16n8k16`, `ldmatrix` and `cp.async` are sm_80 instructions, and ptxas rejects the
 * whole translation unit rather than the one instruction when asked for an older target, so the
 * guard has to be a preprocessor one rather than a runtime branch.
 *
 * Defined as 1 during the host pass, where `__CUDA_ARCH__` is undefined, so the launcher and the
 * kernel signature stay visible to host code whatever the target set is; only the bodies vary. The
 * stub writes no output, so a launch built this way would produce an empty graph rather than
 * fail -- `check_gpipnn_arch` is what keeps that from happening.
 */
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define GPIPNN_FUSED_LEAF_DEVICE 1
#else
#define GPIPNN_FUSED_LEAF_DEVICE 0
#endif

/** `mma.sync.m16n8k16` on fp16 operands with fp32 accumulate. */
__device__ inline void fused_mma_m16n8k16(float (&d)[4],
                                          const uint32_t (&a)[4],
                                          const uint32_t (&b)[2])
{
#if GPIPNN_FUSED_LEAF_DEVICE
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

/**
 * `ldmatrix` fragment loads.
 *
 * For x4 the four 8x8 tiles come out as (rows 0-7, k 0-7), (rows 8-15, k 0-7), (rows 0-7, k 8-15),
 * (rows 8-15, k 8-15) -- exactly the order the mma A operand wants, so the destination registers
 * need no shuffling.
 */
__device__ inline void fused_ldmatrix_x4(uint32_t (&d)[4], const half* addr)
{
#if GPIPNN_FUSED_LEAF_DEVICE
  uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(addr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(a));
#else
  // Zeroed so the stub has no uninitialized outputs to warn about.
  d[0] = d[1] = d[2] = d[3] = 0u;
#endif
}

__device__ inline void fused_ldmatrix_x2(uint32_t (&d)[2], const half* addr)
{
#if GPIPNN_FUSED_LEAF_DEVICE
  uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(addr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(d[0]), "=r"(d[1])
               : "r"(a));
#else
  d[0] = d[1] = 0u;
#endif
}

/** 16-byte async global-to-shared copy. */
__device__ inline void fused_cp_async_16(half* dst, const half* src)
{
#if GPIPNN_FUSED_LEAF_DEVICE
  uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src));
#endif
}

__device__ inline void fused_cp_async_commit()
{
#if GPIPNN_FUSED_LEAF_DEVICE
  asm volatile("cp.async.commit_group;\n");
#endif
}

template <int N>
__device__ inline void fused_cp_async_wait()
{
#if GPIPNN_FUSED_LEAF_DEVICE
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

/** Everything the epilogue needs about one of a lane's two rows, read once per tile. */
struct fused_row_consts {
  id_t id;
  float norm;
  uint32_t salt;
  bool live;  // false for a row past the end of the segment
};

/**
 * Folds the mma accumulator straight into per-lane running lists, with no score tile.
 *
 * The `m16n8k16` C fragment gives lane `4g + p` the values for rows `g` and `g + 8` of the warp's
 * sixteen, at columns `8j + 2p` and `8j + 2p + 1` of each n8 tile. So a row's columns are spread
 * over exactly one quad of four lanes, and a lane carries running lists for exactly two rows.
 */
template <typename Geom, int K>
__device__ inline void fused_accumulate(uint64_t (&best)[2][K],
                                        const float (&acc)[Geom::kNTiles][4],
                                        const fused_row_consts (&rc)[2],
                                        const id_t* __restrict__ sCandId,
                                        const float* __restrict__ sCandNorm,
                                        int64_t n_cand,
                                        int64_t n0,
                                        bool exclude_self,
                                        int n_off,
                                        int tig)
{
  // A lane's two rows share a column, so the candidate's id and norm are read once and used
  // twice. Running the row as the inner loop is what makes that possible.
#pragma unroll
  for (int j = 0; j < Geom::kNTiles; j++) {
#pragma unroll
    for (int p = 0; p < 2; p++) {
      const int local_c = n_off + 8 * j + tig * 2 + p;
      const int64_t c   = n0 + local_c;
      if (c >= n_cand) { continue; }
      const id_t c_id      = sCandId[local_c];
      const float c_norm   = sCandNorm[local_c];
      const uint32_t c_low = static_cast<uint32_t>(c);
#pragma unroll
      for (int h = 0; h < 2; h++) {
        if (!rc[h].live) { continue; }
        const bool masked = exclude_self && c_id == rc[h].id;
        const float score = masked ? kMaskedScore : rc[h].norm + c_norm - 2.0f * acc[j][2 * h + p];
        fused_insert_sorted<K>(best[h],
                               (static_cast<uint64_t>(order_preserving_u32(score)) << 32) |
                                 static_cast<uint64_t>(c_low ^ rc[h].salt));
      }
    }
  }
}

/**
 * Reduce each row's quad to a K-list, publish the per-column-group partial lists, then merge and
 * emit. Runs once per tile, so its cost amortizes over every chunk. `sPart` aliases the candidate
 * buffers, which are dead by now.
 *
 * The shuffles are masked to the quad rather than the warp: whether a row exists is decided per
 * quad, since `m_rows` can land inside a CTA's rows, so quads diverge here and a full-warp mask
 * would be undefined.
 */
template <typename Geom, int K>
__device__ inline void fused_drain(uint64_t (&best)[2][K],
                                   uint64_t* __restrict__ sPart,
                                   const segment_dev& s,
                                   int64_t m_begin,
                                   const id_t* __restrict__ sQId,
                                   const uint32_t* __restrict__ sSalt,
                                   const id_t* __restrict__ cand_ids,
                                   int64_t m_rows,
                                   int64_t k,
                                   bool exclude_self,
                                   int m_off,
                                   int group,
                                   int tig,
                                   int cg,
                                   int tid,
                                   id_t* __restrict__ out_ids,
                                   float* __restrict__ out_dists,
                                   id_t* __restrict__ out_cand_slots)
{
  const uint32_t qmask = 0xfu << (4 * group);
#pragma unroll
  for (int h = 0; h < 2; h++) {
    const int r = m_off + group + 8 * h;
    if (r >= m_rows) { continue; }
    for (int j = 0; j < K; j++) {
      uint64_t m = best[h][0];
      m          = min(m, __shfl_xor_sync(qmask, m, 1));
      m          = min(m, __shfl_xor_sync(qmask, m, 2));
      if (tig == 0) { sPart[(r * Geom::kColGroups + cg) * K + j] = m; }
      // Keys carry their column and each column lives in exactly one lane of the quad, so at most
      // one lane matches and the pop needs no further arbitration.
      if (m != kEmptyKey && best[h][0] == m) {
#pragma unroll
        for (int x = 0; x < K - 1; x++) {
          best[h][x] = best[h][x + 1];
        }
        best[h][K - 1] = kEmptyKey;
      }
    }
  }
  __syncthreads();

  for (int64_t r = tid; r < m_rows; r += kFusedThreads) {
    uint64_t fin[K];
#pragma unroll
    for (int i = 0; i < K; i++) {
      fin[i] = kEmptyKey;
    }
#pragma unroll
    for (int c = 0; c < Geom::kColGroups; c++) {
#pragma unroll
      for (int j = 0; j < K; j++) {
        fused_insert_sorted<K>(fin, sPart[(r * Geom::kColGroups + c) * K + j]);
      }
    }
    for (int64_t j = 0; j < k; j++) {
      emit_rank(fin[j],
                sSalt[r],
                sQId[r],
                cand_ids,
                s.cand_begin,
                exclude_self,
                (s.query_begin + m_begin + r) * k + j,
                out_ids,
                out_dists,
                out_cand_slots);
    }
  }
}

template <typename Geom, int K>
RAFT_KERNEL __launch_bounds__(kFusedThreads)
  fused_leaf_topk_kernel(const half* __restrict__ dataset,
                         const float* __restrict__ norms,
                         const id_t* __restrict__ query_ids,
                         const id_t* __restrict__ cand_ids,
                         const segment_dev* __restrict__ segs,
                         const fused_tile* __restrict__ tiles,
                         int64_t dim,
                         bool exclude_self,
                         int64_t k,
                         id_t* __restrict__ out_ids,
                         float* __restrict__ out_dists,
                         id_t* __restrict__ out_cand_slots)
{
#if !GPIPNN_FUSED_LEAF_DEVICE
  // Stub for pre-sm_80 targets; see GPIPNN_FUSED_LEAF_DEVICE. Reaching this would write no output
  // at all, so `check_gpipnn_arch` rejects the build before any launch can get here.
  return;
#else
  constexpr int kBlockM = Geom::kBlockM;
  constexpr int kBlockN = Geom::kBlockN;

  const fused_tile t    = tiles[blockIdx.x];
  const segment_dev s   = segs[t.seg_id];
  const int64_t dim_pad = dim + kFusedSmemPadH;
  const int tid         = static_cast<int>(threadIdx.x);
  const int warp        = tid / 32;
  const int lane        = tid % 32;
  const int64_t n_cand  = s.cand_count;
  const int64_t m_rows =
    min(static_cast<int64_t>(kBlockM), static_cast<int64_t>(s.query_count) - t.m_begin);

  const int group = lane / 4;  // the fragment's row within an 8-wide group
  const int tig   = lane % 4;  // which pair of columns this lane holds
  const int m_off = 16 * (warp % Geom::kRowGroups);
  const int cg    = warp / Geom::kRowGroups;
  const int n_off = (kBlockN / Geom::kColGroups) * cg;

  extern __shared__ char fused_smem_raw[];
  auto* sQ = reinterpret_cast<half*>(fused_smem_raw);
  // Two candidate tiles, so the next chunk's `cp.async` is in flight while this one's mma runs.
  auto* sC        = sQ + kBlockM * dim_pad;
  auto* sQNorm    = reinterpret_cast<float*>(sC + fused_cand_region_halves<Geom, K>(dim_pad));
  auto* sCandNorm = sQNorm + kBlockM;
  auto* sQId      = reinterpret_cast<id_t*>(sCandNorm + kBlockN);
  auto* sCandId   = sQId + kBlockM;
  auto* sSalt     = reinterpret_cast<uint32_t*>(sCandId + kBlockN);

  // A tall tile leaves padding on the last tile of a segment. Row groups that lie wholly past the
  // end skip the mma below, so only rows that the surviving groups actually read are staged here.
  const int64_t m_active = min(static_cast<int64_t>(kBlockM), (m_rows + 15) / 16 * 16);
  for (int64_t idx = tid; idx < m_active * dim; idx += kFusedThreads) {
    const int64_t r = idx / dim;
    const int64_t c = idx - r * dim;
    half v          = __float2half(0.0f);
    if (r < m_rows) { v = dataset[query_ids[s.query_begin + t.m_begin + r] * dim + c]; }
    sQ[r * dim_pad + c] = v;
  }
  // A lane owns rows `m_off + group` and `m_off + group + 8`, and nothing else.
  uint64_t best[2][K];
#pragma unroll
  for (int i = 0; i < 2; i++) {
#pragma unroll
    for (int j = 0; j < K; j++) {
      best[i][j] = kEmptyKey;
    }
  }
  for (int64_t r = tid; r < m_rows; r += kFusedThreads) {
    const id_t q = query_ids[s.query_begin + t.m_begin + r];
    sQId[r]      = q;
    sQNorm[r]    = norms[q];
    sSalt[r]     = row_salt(q);
  }

  // A chunk is `kBlockN` rows of `dim` halves, copied 16 bytes at a time, which needs `dim` to be
  // a multiple of 8 halves -- guaranteed by `fused_leaf_eligible`'s multiple-of-16 test.
  const int64_t vec_per_row = dim / 8;
  const int64_t vecs        = kBlockN * vec_per_row;
  auto issue_chunk          = [&](int64_t base, half* dst) {
    for (int64_t v = tid; v < vecs; v += kFusedThreads) {
      const int64_t r = v / vec_per_row;
      const int64_t c = (v - r * vec_per_row) * 8;
      if (base + r < n_cand) {
        fused_cp_async_16(dst + r * dim_pad + c,
                          dataset + cand_ids[s.cand_begin + base + r] * dim + c);
      } else {
        *reinterpret_cast<uint4*>(dst + r * dim_pad + c) = uint4{0, 0, 0, 0};
      }
    }
    fused_cp_async_commit();
  };

  issue_chunk(0, sC);
  __syncthreads();

  // The barrier above also publishes the per-row metadata, so the two rows this lane owns can be
  // pulled into registers now and never re-read.
  fused_row_consts rc[2];
#pragma unroll
  for (int h = 0; h < 2; h++) {
    const int r = m_off + group + 8 * h;
    rc[h].live  = r < m_rows;
    rc[h].id    = rc[h].live ? sQId[r] : kInvalidId;
    rc[h].norm  = rc[h].live ? sQNorm[r] : 0.0f;
    rc[h].salt  = rc[h].live ? sSalt[r] : 0u;
  }

  int buf = 0;
  for (int64_t n0 = 0; n0 < n_cand; n0 += kBlockN, buf ^= 1) {
    const int64_t n_rows = min(static_cast<int64_t>(kBlockN), n_cand - n0);
    const bool has_next  = n0 + kBlockN < n_cand;
    if (has_next) { issue_chunk(n0 + kBlockN, sC + (buf ^ 1) * kBlockN * dim_pad); }
    // Ordinary loads, issued here so their latency hides behind the wait below.
    for (int64_t r = tid; r < n_rows; r += kFusedThreads) {
      const id_t c = cand_ids[s.cand_begin + n0 + r];
      sCandId[r]   = c;
      sCandNorm[r] = norms[c];
    }
    if (has_next) {
      fused_cp_async_wait<1>();
    } else {
      fused_cp_async_wait<0>();
    }
    __syncthreads();
    const half* sCur = sC + buf * kBlockN * dim_pad;

    const int ld_row  = lane % 16;
    const int ld_k    = (lane / 16) * 8;
    const int ldb_row = lane % 8;
    const int ldb_k   = ((lane % 16) / 8) * 8;

    // `m_off` is warp-uniform, so a warp whose 16 rows are all padding drops the whole mma chain.
    // It still loads and still reaches the barriers, since those are CTA-wide.
    float acc[Geom::kNTiles][4] = {};
    if (m_off < m_rows) {
      for (int64_t k0 = 0; k0 < dim; k0 += 16) {
        uint32_t a[4];
        fused_ldmatrix_x4(a, sQ + (m_off + ld_row) * dim_pad + k0 + ld_k);
#pragma unroll
        for (int j = 0; j < Geom::kNTiles; j++) {
          uint32_t b[2];
          fused_ldmatrix_x2(b, sCur + (n_off + 8 * j + ldb_row) * dim_pad + k0 + ldb_k);
          fused_mma_m16n8k16(acc[j], a, b);
        }
      }
    }

    fused_accumulate<Geom, K>(
      best, acc, rc, sCandId, sCandNorm, n_cand, n0, exclude_self, n_off, tig);
    // Only the operand buffers are shared, so the one barrier before the next chunk's tile is
    // written is all that is needed.
    __syncthreads();
  }

  fused_drain<Geom, K>(best,
                       reinterpret_cast<uint64_t*>(sC),
                       s,
                       t.m_begin,
                       sQId,
                       sSalt,
                       cand_ids,
                       m_rows,
                       k,
                       exclude_self,
                       m_off,
                       group,
                       tig,
                       cg,
                       tid,
                       out_ids,
                       out_dists,
                       out_cand_slots);
#endif
}

/**
 * Query rows per tile for `dim`, or 0 if neither instantiation fits. Takes the tallest tile the
 * device will accept the shared-memory request for, since arithmetic intensity is `BlockM`.
 */
inline auto fused_leaf_tile_m(int64_t dim, int64_t k) -> int
{
  if (k > 4) { return 0; }
  int device = 0;
  RAFT_CUDA_TRY(cudaGetDevice(&device));
  int smem_max = 0;
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&smem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  if (fused_smem_bytes<fused_tile_geometry<256>, 4>(dim) <= static_cast<size_t>(smem_max)) {
    return 256;
  }
  if (fused_smem_bytes<fused_tile_geometry<128>, 4>(dim) <= static_cast<size_t>(smem_max)) {
    return 128;
  }
  return 0;
}

/**
 * Whether this build of the library can run the leaf join on the current device.
 *
 * The runtime half of `GPIPNN_FUSED_LEAF_DEVICE`. Two independent ways to fail:
 *
 *  - the device predates sm_80, so `mma.sync.m16n8k16` does not exist on it;
 *  - the device is new enough but libcuvs carries no sm_80-or-newer image, so the driver JITs older
 *    PTX, which was generated with the mainloop compiled out.
 *
 * The second is why the kernel image is interrogated rather than just the hardware: checking the
 * device alone passes a library built only for older architectures, and the failure that follows is
 * an empty graph rather than an error.
 *
 * Non-throwing, so a caller with a fallback can ask without having to catch. `check_gpipnn_arch` is
 * the throwing form.
 */
inline auto gpipnn_arch_supported() -> bool
{
  const auto cc = raft::getComputeCapability();
  if (cc.first * 10 + cc.second < 80) { return false; }
  const auto image = raft::util::arch::kernel_virtual_arch(
    reinterpret_cast<void*>(fused_leaf_topk_kernel<fused_tile_geometry<256>, 4>));
  return raft::util::arch::SM_range(raft::util::arch::SM_80(), raft::util::arch::SM_future())
    .contains(image);
}

/** `gpipnn_arch_supported` as a precondition. Called once per build, not once per launch. */
inline void check_gpipnn_arch()
{
  const auto cc = raft::getComputeCapability();
  RAFT_EXPECTS(cc.first * 10 + cc.second >= 80,
               "gpipnn: requires a GPU of compute capability 8.0 (Ampere) or newer; this device "
               "reports %d.%d. The leaf join is built on mma.sync.m16n8k16, ldmatrix and cp.async, "
               "none of which exist before sm_80.",
               cc.first,
               cc.second);
  RAFT_EXPECTS(gpipnn_arch_supported(),
               "gpipnn: this build of libcuvs carries no sm_80-or-newer image for the leaf join "
               "kernel, so it was compiled out and would produce an empty graph. Rebuild with "
               "CMAKE_CUDA_ARCHITECTURES including 80 or later.");
}

/**
 * Whether a call's segments can go to the fused kernel. Deliberately narrow; anything outside it
 * keeps the batched path, which stays the general implementation rather than a fallback in name
 * only. Split from `fused_leaf_eligible` because this half queries the device, which should not
 * happen once per segment.
 *
 * - Self-join, since the kernel reads one id array as both operands.
 * - `k <= 4`, the only instantiation. Larger `k` is a register-pressure question rather than a
 *   correctness one, and the leaf rounds all use 4.
 * - `dim` a multiple of 16: the mainloop steps the mma k-dimension by 16 and the chunk copy moves
 *   8 halves.
 * - `dim` small enough for a resident Q tile. Past that the design's premise fails outright rather
 *   than degrading, so those calls are excluded rather than slowed.
 *
 * Note there is no width floor. A segment narrower than a tile wastes most of it, but handing those
 * to the batched path measures worse -- its per-batch launch count costs more than the fused kernel
 * loses to padding -- and the output is bit-identical either way.
 */
inline auto fused_leaf_usable(int64_t dim, int64_t k, bool self_join) -> bool
{
  return self_join && k >= 1 && k <= 4 && dim % 16 == 0 && fused_leaf_tile_m(dim, k) > 0;
}

/** The per-segment half. Only meaningful once `fused_leaf_usable` has passed. */
inline auto fused_leaf_eligible(const segment& s) -> bool
{
  return s.cand_count > 0 && s.query_begin == s.cand_begin && s.query_count == s.cand_count;
}

/** Builds the tile list for `segs` and runs one fused launch over all of them. */
template <int BlockM, int K>
inline void launch_fused_leaf_topk(const std::vector<segment>& segs,
                                   const half* dataset,
                                   const float* norms,
                                   const id_t* query_ids,
                                   const id_t* cand_ids,
                                   int64_t dim,
                                   int64_t k,
                                   bool exclude_self,
                                   id_t* out_ids,
                                   float* out_dists,
                                   id_t* out_cand_slots,
                                   cudaStream_t stream)
{
  using geom = fused_tile_geometry<BlockM>;
  if (segs.empty()) { return; }

  std::vector<segment_dev> h_segs(segs.size());
  std::vector<fused_tile> h_tiles;
  for (size_t i = 0; i < segs.size(); i++) {
    const auto& s = segs[i];
    h_segs[i]     = segment_dev{s.query_begin,
                            s.cand_begin,
                            static_cast<int32_t>(s.query_count),
                            static_cast<int32_t>(s.cand_count)};
    for (int64_t m = 0; m < s.query_count; m += geom::kBlockM) {
      h_tiles.push_back(fused_tile{static_cast<int32_t>(i), static_cast<int32_t>(m)});
    }
  }

  rmm::device_uvector<segment_dev> d_segs(h_segs.size(), stream);
  rmm::device_uvector<fused_tile> d_tiles(h_tiles.size(), stream);
  raft::update_device(d_segs.data(), h_segs.data(), h_segs.size(), stream);
  raft::update_device(d_tiles.data(), h_tiles.data(), h_tiles.size(), stream);

  const size_t smem = fused_smem_bytes<geom, K>(dim);
  RAFT_CUDA_TRY(cudaFuncSetAttribute(fused_leaf_topk_kernel<geom, K>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     static_cast<int>(smem)));
  fused_leaf_topk_kernel<geom, K><<<h_tiles.size(), kFusedThreads, smem, stream>>>(dataset,
                                                                                   norms,
                                                                                   query_ids,
                                                                                   cand_ids,
                                                                                   d_segs.data(),
                                                                                   d_tiles.data(),
                                                                                   dim,
                                                                                   exclude_self,
                                                                                   k,
                                                                                   out_ids,
                                                                                   out_dists,
                                                                                   out_cand_slots);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/** Dispatches to the tallest tile that fits `dim`. */
inline void launch_fused_leaf_topk(const std::vector<segment>& segs,
                                   const half* dataset,
                                   const float* norms,
                                   const id_t* query_ids,
                                   const id_t* cand_ids,
                                   int64_t dim,
                                   int64_t k,
                                   bool exclude_self,
                                   id_t* out_ids,
                                   float* out_dists,
                                   id_t* out_cand_slots,
                                   cudaStream_t stream)
{
  const int block_m = fused_leaf_tile_m(dim, k);
  RAFT_EXPECTS(block_m == 256 || block_m == 128,
               "gpipnn: no fused leaf tile fits dim %zu",
               static_cast<size_t>(dim));
  if (block_m == 256) {
    launch_fused_leaf_topk<256, 4>(segs,
                                   dataset,
                                   norms,
                                   query_ids,
                                   cand_ids,
                                   dim,
                                   k,
                                   exclude_self,
                                   out_ids,
                                   out_dists,
                                   out_cand_slots,
                                   stream);
  } else {
    launch_fused_leaf_topk<128, 4>(segs,
                                   dataset,
                                   norms,
                                   query_ids,
                                   cand_ids,
                                   dim,
                                   k,
                                   exclude_self,
                                   out_ids,
                                   out_dists,
                                   out_cand_slots,
                                   stream);
  }
}

}  // namespace cuvs::neighbors::experimental::gpipnn::detail
