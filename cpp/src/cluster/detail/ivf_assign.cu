/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ivf_assign.hpp"
#include <cuvs/cluster/kmeans.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_type.cuh>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace cuvs::cluster::kmeans::detail::ivf_assign {

namespace {

using KV = cub::KeyValuePair<int, float>;

struct ArgMaxOp {
  __device__ KV operator()(const KV& a, const KV& b) const
  {
    return (b.value > a.value || (b.value == a.value && b.key < a.key)) ? b : a;
  }
};

constexpr int kBlock = 256;

// score(r, c) = scores[r * stride + c] - (bias ? bias[c] : 0). One block per row.
__global__ void argmax_rows_kernel(
  const float* scores, int cols, int stride, const float* bias, int* arg)
{
  const float* row = scores + (size_t)blockIdx.x * stride;
  KV loc(0, -INFINITY);
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float v = row[c] - (bias ? bias[c] : 0.f);
    if (v > loc.value) loc = KV(c, v);
  }
  using BR = cub::BlockReduce<KV, kBlock>;
  __shared__ typename BR::TempStorage t;
  KV r = BR(t).Reduce(loc, ArgMaxOp());
  if (threadIdx.x == 0) arg[blockIdx.x] = r.key;
}

// Like argmax_rows_kernel, but writes the (id_map'd) winner of pair `pairs[row]`.
__global__ void argmax_scatter_kernel(const float* scores,
                                      int cols,
                                      int stride,
                                      const float* bias,
                                      const int* pairs,
                                      const int* id_map,
                                      float* res_score,
                                      int* res_id)
{
  const float* row = scores + (size_t)blockIdx.x * stride;
  KV loc(0, -INFINITY);
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float v = row[c] - (bias ? bias[c] : 0.f);
    if (v > loc.value) loc = KV(c, v);
  }
  using BR = cub::BlockReduce<KV, kBlock>;
  __shared__ typename BR::TempStorage t;
  KV r = BR(t).Reduce(loc, ArgMaxOp());
  if (threadIdx.x == 0) {
    int slot        = pairs[blockIdx.x];
    res_score[slot] = r.value;
    res_id[slot]    = id_map[r.key];
  }
}

// Each row's `p` best columns (after the bias), best first.
__global__ void top_p_kernel(
  const float* scores, int cols, int stride, const float* bias, int p, int* out)
{
  extern __shared__ float row[];
  for (int c = threadIdx.x; c < cols; c += blockDim.x)
    row[c] = scores[(size_t)blockIdx.x * stride + c] - (bias ? bias[c] : 0.f);
  __syncthreads();
  using BR = cub::BlockReduce<KV, kBlock>;
  __shared__ typename BR::TempStorage t;
  for (int j = 0; j < p; j++) {
    KV loc(0, -INFINITY);
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
      if (row[c] > loc.value) loc = KV(c, row[c]);
    KV r = BR(t).Reduce(loc, ArgMaxOp());
    if (threadIdx.x == 0) {
      out[(size_t)blockIdx.x * p + j] = r.key;
      row[r.key]                      = -INFINITY;
    }
    __syncthreads();
  }
}

__global__ void init_pairs_kernel(const int* lists, int n, int* keys, int* vals)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    keys[i] = lists[i];
    vals[i] = i;
  }
}

__global__ void count_keys_kernel(const int* keys, int n, int* counts)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(counts + keys[i], 1);
}

// Pair `sorted_vals[off + r]` is (query * p + probe); copy that query's row into out[r].
__global__ void gather_queries_kernel(
  const float* q, int dim, const int* sorted_vals, int off, int p, float* out)
{
  int query         = sorted_vals[off + blockIdx.x] / p;
  const float4* src = reinterpret_cast<const float4*>(q + (size_t)query * dim);
  float4* dst       = reinterpret_cast<float4*>(out + (size_t)blockIdx.x * dim);
  for (int c = threadIdx.x; c < dim / 4; c += blockDim.x)
    dst[c] = src[c];
}

__global__ void merge_pairs_kernel(
  const float* res_score, const int* res_id, int p, int nq, uint32_t* labels)
{
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= nq) return;
  float best = -INFINITY;
  int arg    = 0;
  for (int j = 0; j < p; j++) {
    float s = res_score[(size_t)q * p + j];
    if (s > best) {
      best = s;
      arg  = res_id[(size_t)q * p + j];
    }
  }
  labels[q] = static_cast<uint32_t>(arg);
}

// Row r of x scaled to unit norm (a zero row is left alone).
__global__ void normalize_rows_kernel(float* x, int dim)
{
  float* row = x + (size_t)blockIdx.x * dim;
  float s    = 0;
  for (int c = threadIdx.x; c < dim; c += blockDim.x)
    s += row[c] * row[c];
  using BR = cub::BlockReduce<float, kBlock>;
  __shared__ typename BR::TempStorage t;
  __shared__ float inv;
  float tot = BR(t).Sum(s);
  if (threadIdx.x == 0) inv = tot > 0 ? rsqrtf(tot) : 1.0f;
  __syncthreads();
  for (int c = threadIdx.x; c < dim; c += blockDim.x)
    row[c] *= inv;
}

__global__ void half_sq_norm_kernel(const float* x, int dim, float* out)
{
  const float* row = x + (size_t)blockIdx.x * dim;
  float s          = 0;
  for (int c = threadIdx.x; c < dim; c += blockDim.x)
    s += row[c] * row[c];
  using BR = cub::BlockReduce<float, kBlock>;
  __shared__ typename BR::TempStorage t;
  float tot = BR(t).Sum(s);
  if (threadIdx.x == 0) out[blockIdx.x] = 0.5f * tot;
}

// C (column-major [n, m], leading dimension ldc) = A^T B for row-major A [n, dim] and
// B [m, dim]; i.e. row-major C[m, n] = B A^T. TF32 tensor cores, fp32 accumulate.
void gemm_tf32(raft::resources const& res,
               int n,
               int m,
               int dim,
               const float* A,
               const float* B,
               float* C,
               int ldc)
{
  const float one = 1.f, zero = 0.f;
  cublasHandle_t h = raft::resource::get_cublas_handle(res);
  RAFT_CUBLAS_TRY(cublasSetStream(h, raft::resource::get_cuda_stream(res).get()));
  RAFT_CUBLAS_TRY(cublasGemmEx(h,
                               CUBLAS_OP_T,
                               CUBLAS_OP_N,
                               n,
                               m,
                               dim,
                               &one,
                               A,
                               CUDA_R_32F,
                               dim,
                               B,
                               CUDA_R_32F,
                               dim,
                               &zero,
                               C,
                               CUDA_R_32F,
                               ldc,
                               CUBLAS_COMPUTE_32F_FAST_TF32,
                               CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

inline int round_up(int x, int a) { return (x + a - 1) / a * a; }

// Default number of probed lists when the caller does not choose. A rule of thumb, not an accuracy
// guarantee: with about sqrt(K) lists, 16 probes matched the exhaustive assignment on at least 99%
// of the vectors for 4096 to 131072 centroids on every dataset measured (dimensions 96 to 2048).
constexpr int kDefaultProbes = 16;

inline int resolve_n_probes(uint32_t requested, int n_lists)
{
  const int p = requested != 0
                  ? static_cast<int>(std::min<uint32_t>(requested, static_cast<uint32_t>(n_lists)))
                  : kDefaultProbes;
  return std::clamp(p, 1, n_lists);
}

// ---------------------------------------------------------------------------------------------
// Fused assignment: one persistent kernel replaces the per-list gather + GEMM + argmax launches.
//
// Work items are (list, tile of 128 queries) in order of decreasing list size, claimed through an
// atomic counter. An item streams its queries (gathered through cp.async, no copy) against the
// list's centroids with TF32 tensor-core MMAs and keeps a running argmax, so no score matrix is
// materialized. The centroid tile covers `BN` columns (a whole list in one pass when it fits), so
// the query tile is read from memory once per list instead of once per 64 centroids; this is what
// makes the kernel compute-bound instead of memory-bound.
// ---------------------------------------------------------------------------------------------
constexpr int kFusedBM      = 128;  // queries per tile
constexpr int kFusedBK      = 32;   // k extent of a pipeline stage
constexpr int kFusedStages  = 3;
constexpr int kFusedLds     = kFusedBK + 4;  // padded shared-memory row, avoids bank conflicts
constexpr int kFusedThreads = 256;           // 8 warps: kFusedWM x kFusedWN
constexpr int kFusedWM      = 2;
constexpr int kFusedWN      = 4;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#define CUVS_IVF_ASSIGN_FUSED_ARCH 1
#endif

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool pred)
{
#ifdef CUVS_IVF_ASSIGN_FUSED_ARCH
  unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  int sz     = pred ? 16 : 0;  // 0 bytes: zero-fill
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(sz));
#endif
}

__device__ __forceinline__ void cp_commit()
{
#ifdef CUVS_IVF_ASSIGN_FUSED_ARCH
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int N>
__device__ __forceinline__ void cp_wait()
{
#ifdef CUVS_IVF_ASSIGN_FUSED_ARCH
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

// Round to nearest (ties away from zero), the rounding cuBLAS applies for TF32 inputs.
__device__ __forceinline__ unsigned to_tf32_rna(float x)
{
  unsigned r = 0;
#ifdef CUVS_IVF_ASSIGN_FUSED_ARCH
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
#endif
  return r;
}

__device__ __forceinline__ void mma_tf32(float* c, const unsigned* a, const unsigned* b)
{
#ifdef CUVS_IVF_ASSIGN_FUSED_ARCH
  asm volatile(
    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
    "{%0,%1,%2,%3};\n"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

// Rounds x to TF32 in place (the centroids are rounded once at build so the fused kernel can feed
// them to the tensor cores without converting).
__global__ void round_tf32_kernel(float* x, size_t n)
{
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    uint32_t u = __float_as_uint(x[i]);
    u          = (u + 0x1000u) & 0xFFFFE000u;
    x[i]       = __uint_as_float(u);
  }
}

// Marks every (query, probe) result as "no candidate" so lists without centroids cannot win.
__global__ void init_results_kernel(float* res_score, int* res_id, int64_t n)
{
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i < n) {
    res_score[i] = -INFINITY;
    res_id[i]    = 0;
  }
}

// From the per-list pair counts: pair offsets by list id, and the prefix sum of the number of
// 128-query tiles over the lists in `order` (widest first); also resets the work counter.
__global__ void build_work_table_kernel(const int* __restrict__ pair_counts,
                                        const int* __restrict__ list_size,
                                        const int* __restrict__ order,
                                        int n_lists,
                                        int* __restrict__ pair_offset,
                                        int* __restrict__ tile_prefix,
                                        int* __restrict__ counter)
{
  using BS = cub::BlockScan<int, 256>;
  __shared__ typename BS::TempStorage tmp;
  int carry_pairs = 0, carry_tiles = 0;
  for (int base = 0; base < n_lists; base += 256) {
    const int i = base + threadIdx.x;
    int c       = i < n_lists ? pair_counts[i] : 0, ex, tot;
    BS(tmp).ExclusiveSum(c, ex, tot);
    if (i < n_lists) pair_offset[i] = carry_pairs + ex;
    carry_pairs += tot;
    __syncthreads();
    int t = 0;
    if (i < n_lists) {
      const int m = order[i];
      t           = list_size[m] > 0 ? (pair_counts[m] + kFusedBM - 1) / kFusedBM : 0;
    }
    BS(tmp).ExclusiveSum(t, ex, tot);
    if (i < n_lists) tile_prefix[i] = carry_tiles + ex;
    carry_tiles += tot;
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    tile_prefix[n_lists] = carry_tiles;
    *counter             = 0;
  }
}

struct fused_args {
  const float* queries;
  int dim;
  int n_probes;
  const int* sorted_pairs;  // (query * n_probes + probe), bucketed by list
  const float* perm;        // [n_centroids + slack, dim] TF32-rounded, each list contiguous
  const float* perm_bias;   // |c|^2/2 in the same order, or nullptr (cosine, inner product)
  const int* perm_id;       // original centroid id of each row of `perm`
  const int* list_offset;   // first row of each list in `perm`
  const int* list_size;     // centroids per list
  const int* pair_offset;   // first pair of each list in `sorted_pairs`
  const int* pair_count;    // pairs (probing queries) per list
  int* counter;             // work counter
  float* res_score;         // [n_queries * n_probes] best score per (query, probe)
  int* res_id;              // its centroid id
  const int* order;         // lists by decreasing size
  const int* tile_prefix;   // tiles of the lists in `order`, prefix sum
  int n_lists;
};

template <int BN>
constexpr size_t fused_smem_bytes()
{
  return ((size_t)kFusedStages * (kFusedBM + BN) * kFusedLds + 2 * (size_t)kFusedWN * kFusedBM) *
         sizeof(float);
}

template <int BN>
__global__ void __launch_bounds__(kFusedThreads, 1) fused_assign_kernel(fused_args a)
{
#ifdef CUVS_IVF_ASSIGN_FUSED_ARCH
  constexpr int BM = kFusedBM, BK = kFusedBK, ST = kFusedStages, LDS = kFusedLds;
  constexpr int WM = kFusedWM, WN = kFusedWN;
  constexpr int MI  = BM / (16 * WM);
  constexpr int NI  = BN / (8 * WN);
  constexpr int NBL = BN / 32;  // centroid rows each thread loads per stage
  extern __shared__ __align__(16) float smem[];
  float* sA  = smem;                // [ST][BM][LDS]
  float* sB  = sA + ST * BM * LDS;  // [ST][BN][LDS]
  float* sRs = sB + ST * BN * LDS;  // [WN][BM]
  int* sRi   = reinterpret_cast<int*>(sRs + WN * BM);
  __shared__ int s_list, s_tile;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp % WM, wn = warp / WM;
  const int g = lane >> 2, t = lane & 3;
  const int dim = a.dim, P = a.n_probes;
  const int KT = (dim + BK - 1) / BK;  // the last stage is zero-filled past dim (dim % 4 == 0)
  for (;;) {
    if (tid == 0) {
      const int it = atomicAdd(a.counter, 1);
      int mm = -1, tt = 0;
      if (it < a.tile_prefix[a.n_lists]) {
        int lo = 0, hi = a.n_lists;  // tile_prefix[lo] <= it < tile_prefix[hi]
        while (hi - lo > 1) {
          const int mid = (lo + hi) >> 1;
          if (a.tile_prefix[mid] <= it)
            lo = mid;
          else
            hi = mid;
        }
        mm = a.order[lo];
        tt = it - a.tile_prefix[lo];
      }
      s_list = mm;
      s_tile = tt;
    }
    __syncthreads();
    const int m = s_list, tile = s_tile;
    __syncthreads();
    if (m < 0) break;
    const int n         = a.list_size[m];
    const int row0      = tile * BM;
    const int nrows     = min(BM, a.pair_count[m] - row0);
    const int base_pair = a.pair_offset[m] + row0;
    const int list_row  = a.list_offset[m];
    const float* bbase  = a.perm + (size_t)list_row * dim;
    const float* arow[4];
    bool aok[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int r = (tid >> 3) + 32 * i;
      aok[i]      = r < nrows;
      arow[i] = aok[i] ? a.queries + (size_t)(a.sorted_pairs[base_pair + r] / P) * dim : a.queries;
    }
    float bs[MI][2];
    int bi[MI][2];
#pragma unroll
    for (int x = 0; x < MI; x++)
#pragma unroll
      for (int y = 0; y < 2; y++) {
        bs[x][y] = -INFINITY;
        bi[x][y] = 0;
      }

    const int ntiles = (n + BN - 1) / BN;
    for (int nt = 0; nt < ntiles; nt++) {
      float acc[MI][NI][4];
#pragma unroll
      for (int x = 0; x < MI; x++)
#pragma unroll
        for (int y = 0; y < NI; y++)
#pragma unroll
          for (int z = 0; z < 4; z++)
            acc[x][y][z] = 0.f;
      const float* brow[NBL];
      bool bok[NBL];
#pragma unroll
      for (int i = 0; i < NBL; i++) {
        const int col = nt * BN + (tid >> 3) + 32 * i;
        bok[i]        = col < n;
        brow[i]       = bok[i] ? bbase + (size_t)col * dim : a.perm;
      }
      auto load_stage = [&](int stage, int kt) {
        const int kc = kt * BK + (tid & 7) * 4;
        // The last stage may extend past dim: those chunks are zero-filled, which adds nothing to
        // the dot products. dim % 4 == 0, so a chunk is either inside the row or entirely outside.
        const bool kok = kc < dim;
        const int koff = kok ? kc : 0;
#pragma unroll
        for (int i = 0; i < 4; i++)
          cp_async16(&sA[(stage * BM + (tid >> 3) + 32 * i) * LDS + (tid & 7) * 4],
                     arow[i] + koff,
                     aok[i] && kok);
#pragma unroll
        for (int i = 0; i < NBL; i++)
          cp_async16(&sB[(stage * BN + (tid >> 3) + 32 * i) * LDS + (tid & 7) * 4],
                     brow[i] + koff,
                     bok[i] && kok);
      };
      // A warp whose whole column range lies past the end of the list skips its MMAs.
      const bool warp_active = nt * BN + wn * (NI * 8) < n;
#pragma unroll
      for (int s = 0; s < ST - 1; s++) {
        if (s < KT) load_stage(s, s);
        cp_commit();
      }
      for (int kt = 0; kt < KT; kt++) {
        cp_wait<ST - 2>();
        __syncthreads();
        const int nk = kt + ST - 1;
        if (nk < KT) load_stage(nk % ST, nk);
        cp_commit();
        if (warp_active) {
          const float* A = sA + (kt % ST) * BM * LDS;
          const float* B = sB + (kt % ST) * BN * LDS;
#pragma unroll
          for (int kk = 0; kk < BK / 8; kk++) {
            unsigned af[MI][4], bf[NI][2];
#pragma unroll
            for (int mi = 0; mi < MI; mi++) {
              const int r = wm * (MI * 16) + mi * 16 + g;
              af[mi][0]   = to_tf32_rna(A[r * LDS + kk * 8 + t]);
              af[mi][1]   = to_tf32_rna(A[(r + 8) * LDS + kk * 8 + t]);
              af[mi][2]   = to_tf32_rna(A[r * LDS + kk * 8 + t + 4]);
              af[mi][3]   = to_tf32_rna(A[(r + 8) * LDS + kk * 8 + t + 4]);
            }
#pragma unroll
            for (int ni = 0; ni < NI; ni++) {
              const int nn = wn * (NI * 8) + ni * 8 + g;
              // the centroids were rounded to TF32 when the index was built
              bf[ni][0] = __float_as_uint(B[nn * LDS + kk * 8 + t]);
              bf[ni][1] = __float_as_uint(B[nn * LDS + kk * 8 + t + 4]);
            }
#pragma unroll
            for (int mi = 0; mi < MI; mi++)
#pragma unroll
              for (int ni = 0; ni < NI; ni++)
                mma_tf32(acc[mi][ni], af[mi], bf[ni]);
          }
        }
      }
      cp_wait<0>();
      __syncthreads();
#pragma unroll
      for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int ni = 0; ni < NI; ni++)
#pragma unroll
          for (int e = 0; e < 4; e++) {
            const int col = nt * BN + wn * (NI * 8) + ni * 8 + 2 * t + (e & 1);
            if (col < n) {
              float sc = acc[mi][ni][e];
              if (a.perm_bias != nullptr) sc -= a.perm_bias[list_row + col];
              if (sc > bs[mi][e >> 1]) {
                bs[mi][e >> 1] = sc;
                bi[mi][e >> 1] = col;
              }
            }
          }
    }
    // Reduce across the quad of lanes sharing a row, then across the WN column warps.
#pragma unroll
    for (int mi = 0; mi < MI; mi++)
#pragma unroll
      for (int h = 0; h < 2; h++) {
        float sc = bs[mi][h];
        int id   = bi[mi][h];
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
          const float s2 = __shfl_xor_sync(0xffffffffu, sc, off);
          const int i2   = __shfl_xor_sync(0xffffffffu, id, off);
          if (s2 > sc || (s2 == sc && i2 < id)) {
            sc = s2;
            id = i2;
          }
        }
        const int row = wm * (MI * 16) + mi * 16 + g + 8 * h;
        if (t == 0) {
          sRs[wn * BM + row] = sc;
          sRi[wn * BM + row] = id;
        }
      }
    __syncthreads();
    if (tid < nrows) {
      float sc = sRs[tid];
      int id   = sRi[tid];
#pragma unroll
      for (int w = 1; w < WN; w++) {
        const float s2 = sRs[w * BM + tid];
        const int i2   = sRi[w * BM + tid];
        if (s2 > sc || (s2 == sc && i2 < id)) {
          sc = s2;
          id = i2;
        }
      }
      const int slot    = a.sorted_pairs[base_pair + tid];
      a.res_score[slot] = sc;
      a.res_id[slot]    = a.perm_id[list_row + id];
    }
    __syncthreads();
  }
#endif  // CUVS_IVF_ASSIGN_FUSED_ARCH
}

struct fused_plan {
  int bn      = 0;  // centroid columns per pass; 0: the fused kernel cannot be used
  size_t smem = 0;
  int grid    = 0;
};

// Picks the widest tile whose shared memory fits and checks that the kernel really was compiled
// for sm_80 or newer (the body is empty for older architectures).
template <int BN>
bool try_fused_plan(int smem_optin, int num_sms, fused_plan& plan)
{
  const size_t smem = fused_smem_bytes<BN>() + 16;  // + static shared variables
  if (smem > static_cast<size_t>(smem_optin)) return false;
  cudaFuncAttributes attr;
  if (cudaFuncGetAttributes(&attr, fused_assign_kernel<BN>) != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  if (attr.ptxVersion < 80) return false;
  RAFT_CUDA_TRY(cudaFuncSetAttribute(fused_assign_kernel<BN>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     static_cast<int>(fused_smem_bytes<BN>())));
  int per_sm = 0;
  RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &per_sm, fused_assign_kernel<BN>, kFusedThreads, fused_smem_bytes<BN>()));
  if (per_sm < 1) return false;
  plan.bn   = BN;
  plan.smem = fused_smem_bytes<BN>();
  plan.grid = num_sms * per_sm;
  return true;
}

inline fused_plan plan_fused(int dim)
{
  fused_plan plan;
  if (dim % 4 != 0) return plan;  // 16-byte cp.async rows; the k-tail is zero-filled
  int dev = 0, major = 0, smem_optin = 0, num_sms = 0;
  RAFT_CUDA_TRY(cudaGetDevice(&dev));
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
  if (major < 8) return plan;
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev));
  // Widest first; a wider tile reads the queries fewer times per list.
  if (try_fused_plan<256>(smem_optin, num_sms, plan)) return plan;
  if (try_fused_plan<192>(smem_optin, num_sms, plan)) return plan;
  if (try_fused_plan<128>(smem_optin, num_sms, plan)) return plan;
  (void)try_fused_plan<64>(smem_optin, num_sms, plan);
  return plan;
}

inline void launch_fused(const fused_plan& plan, const fused_args& args, cudaStream_t stream)
{
  switch (plan.bn) {
    case 256:
      fused_assign_kernel<256><<<plan.grid, kFusedThreads, plan.smem, stream>>>(args);
      break;
    case 192:
      fused_assign_kernel<192><<<plan.grid, kFusedThreads, plan.smem, stream>>>(args);
      break;
    case 128:
      fused_assign_kernel<128><<<plan.grid, kFusedThreads, plan.smem, stream>>>(args);
      break;
    default: fused_assign_kernel<64><<<plan.grid, kFusedThreads, plan.smem, stream>>>(args); break;
  }
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace

index::index(raft::resources const& res)
  : list_centers(raft::make_device_matrix<float, int64_t>(res, 0, 0)),
    list_bias(raft::make_device_vector<float, int64_t>(res, 0)),
    perm_centroids(raft::make_device_matrix<float, int64_t>(res, 0, 0)),
    perm_bias(raft::make_device_vector<float, int64_t>(res, 0)),
    perm_id(raft::make_device_vector<int32_t, int64_t>(res, 0)),
    list_offset_d(raft::make_device_vector<int32_t, int64_t>(res, 0)),
    list_size_d(raft::make_device_vector<int32_t, int64_t>(res, 0)),
    list_order_d(raft::make_device_vector<int32_t, int64_t>(res, 0))
{
}

auto build(raft::resources const& res,
           const index_params& params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> centroids) -> index
{
  using cuvs::distance::DistanceType;
  const int K   = static_cast<int>(centroids.extent(0));
  const int dim = static_cast<int>(centroids.extent(1));
  RAFT_EXPECTS(K > 0 && dim > 0 && dim % 4 == 0, "ivf_assign needs dim divisible by 4");
  RAFT_EXPECTS(params.tile_align >= 1, "tile_align must be positive");
  const bool l2 =
    params.metric == DistanceType::L2Expanded || params.metric == DistanceType::L2SqrtExpanded;
  const bool cosine = params.metric == DistanceType::CosineExpanded;
  RAFT_EXPECTS(l2 || cosine || params.metric == DistanceType::InnerProduct,
               "ivf_assign supports L2, cosine and inner-product metrics");
  const int M =
    std::clamp<int>(params.n_lists ? static_cast<int>(params.n_lists)
                                   : static_cast<int>(std::lround(std::sqrt((double)K))),
                    1,
                    K);
  const int align     = static_cast<int>(params.tile_align);
  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();

  index idx(res);
  idx.n_centroids = K;
  idx.dim         = dim;
  idx.n_lists     = M;
  idx.tile_align  = align;
  idx.l2          = l2;

  // Working copy of the centroids (normalized for cosine).
  auto cents = raft::make_device_matrix<float, int64_t>(res, K, dim);
  raft::copy(cents.data_handle(), centroids.data_handle(), (size_t)K * dim, stream);
  if (cosine) normalize_rows_kernel<<<K, kBlock, 0, stream>>>(cents.data_handle(), dim);

  // ---- Group the centroids into M lists with cuVS balanced k-means. ----
  auto centers = raft::make_device_matrix<float, int64_t>(res, M, dim);
  auto cbias   = raft::make_device_vector<float, int64_t>(res, M);
  auto scores  = raft::make_device_matrix<float, int64_t>(res, K, M);
  auto assign  = raft::make_device_vector<int, int64_t>(res, K);
  {
    cuvs::cluster::kmeans::balanced_params kmeans_params;
    kmeans_params.metric  = params.metric;
    kmeans_params.n_iters = params.kmeans_n_iters;
    cuvs::cluster::kmeans::fit(
      res,
      kmeans_params,
      raft::make_device_matrix_view<const float, int64_t>(cents.data_handle(), K, dim),
      centers.view());
  }
  // Label the centroids against the final list centers with the same GEMM + argmax that
  // assign() uses to score lists, so the lists and the search agree exactly.
  if (l2)
    half_sq_norm_kernel<<<M, kBlock, 0, stream>>>(centers.data_handle(), dim, cbias.data_handle());
  gemm_tf32(res, M, K, dim, centers.data_handle(), cents.data_handle(), scores.data_handle(), M);
  argmax_rows_kernel<<<K, kBlock, 0, stream>>>(
    scores.data_handle(), M, M, l2 ? cbias.data_handle() : nullptr, assign.data_handle());

  std::vector<int> h_assign(K);
  raft::update_host(h_assign.data(), assign.data_handle(), K, stream);
  raft::resource::sync_stream(res);

  // ---- Lists in CSR form, centroids permuted so each list is contiguous. ----
  idx.list_size.assign(M, 0);
  idx.list_offset.assign(M + 1, 0);
  for (int a : h_assign)
    idx.list_size[a]++;
  for (int m = 0; m < M; m++)
    idx.list_offset[m + 1] = idx.list_offset[m] + idx.list_size[m];
  std::vector<int> fill(idx.list_offset.begin(), idx.list_offset.end() - 1), h_perm(K);
  for (int c = 0; c < K; c++) {
    int pos     = fill[h_assign[c]]++;
    h_perm[pos] = c;
  }

  idx.list_centers = raft::make_device_matrix<float, int64_t>(res, M, dim);
  raft::copy(idx.list_centers.data_handle(), centers.data_handle(), (size_t)M * dim, stream);
  idx.list_bias = raft::make_device_vector<float, int64_t>(res, l2 ? M : 0);
  if (l2) {
    half_sq_norm_kernel<<<M, kBlock, 0, stream>>>(
      centers.data_handle(), dim, idx.list_bias.data_handle());
  }
  idx.perm_id = raft::make_device_vector<int32_t, int64_t>(res, K);
  raft::update_device(idx.perm_id.data_handle(), h_perm.data(), K, stream);
  // Slack rows: a list's GEMM may run up to `align` columns past its end.
  const int64_t rows = (int64_t)K + align;
  idx.perm_centroids = raft::make_device_matrix<float, int64_t>(res, rows, dim);
  RAFT_CUDA_TRY(
    cudaMemsetAsync(idx.perm_centroids.data_handle(), 0, (size_t)rows * dim * 4, stream));
  // Rows of `cents` in list order (cosine: normalized). Row gather via per-row copies.
  for (int p = 0; p < K; p++)
    raft::copy(idx.perm_centroids.data_handle() + (size_t)p * dim,
               cents.data_handle() + (size_t)h_perm[p] * dim,
               dim,
               stream);
  idx.perm_bias = raft::make_device_vector<float, int64_t>(res, l2 ? rows : 0);
  if (l2) {
    half_sq_norm_kernel<<<rows, kBlock, 0, stream>>>(
      idx.perm_centroids.data_handle(), dim, idx.perm_bias.data_handle());
  }
  // Round the centroids to TF32 once (after the biases, which use the exact values), so the fused
  // kernel can use them as tensor-core operands directly.
  round_tf32_kernel<<<1024, kBlock, 0, stream>>>(idx.perm_centroids.data_handle(),
                                                 static_cast<size_t>(rows) * dim);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Device copies of the list layout for the fused kernel; lists ordered by decreasing size.
  std::vector<int32_t> h_order(M);
  std::iota(h_order.begin(), h_order.end(), 0);
  std::stable_sort(h_order.begin(), h_order.end(), [&](int32_t x, int32_t y) {
    return idx.list_size[x] > idx.list_size[y];
  });
  idx.list_offset_d = raft::make_device_vector<int32_t, int64_t>(res, M + 1);
  idx.list_size_d   = raft::make_device_vector<int32_t, int64_t>(res, M);
  idx.list_order_d  = raft::make_device_vector<int32_t, int64_t>(res, M);
  raft::update_device(idx.list_offset_d.data_handle(), idx.list_offset.data(), M + 1, stream);
  raft::update_device(idx.list_size_d.data_handle(), idx.list_size.data(), M, stream);
  raft::update_device(idx.list_order_d.data_handle(), h_order.data(), M, stream);
  raft::resource::sync_stream(res);
  return idx;
}

namespace {

// Fallback for GPUs and dimensions the fused kernel does not support: for every probed list,
// gather the queries probing it, run one dense TF32 GEMM against the list's centroids and take
// the row argmax. Synchronizes the stream once to read the per-list query counts.
void assign_gemm_loop(raft::resources const& res,
                      const index& idx,
                      const float* queries,
                      int P,
                      const int* vals_sorted,
                      const int* d_counts,
                      float* res_score,
                      int* res_id)
{
  const int M         = static_cast<int>(idx.n_lists);
  const int dim       = static_cast<int>(idx.dim);
  const int align     = static_cast<int>(idx.tile_align);
  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
  std::vector<int> counts(M);
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    counts.data(), d_counts, (size_t)M * sizeof(int), cudaMemcpyDeviceToHost, stream));
  raft::resource::sync_stream(res);

  int max_g = 0, max_cols = 0;
  for (int m = 0; m < M; m++) {
    if (counts[m] > 0 && idx.list_size[m] > 0) {
      max_g    = std::max(max_g, counts[m]);
      max_cols = std::max(max_cols, round_up(idx.list_size[m], align));
    }
  }
  if (max_g == 0) return;
  auto qg = raft::make_device_matrix<float, int64_t>(res, max_g, dim);
  auto sc = raft::make_device_matrix<float, int64_t>(res, max_g, max_cols);
  int off = 0;
  for (int m = 0; m < M; m++) {
    const int g = counts[m], n = idx.list_size[m];
    if (g > 0 && n > 0) {
      const int npad = round_up(n, align);
      const int row0 = idx.list_offset[m];
      gather_queries_kernel<<<g, kBlock, 0, stream>>>(
        queries, dim, vals_sorted, off, P, qg.data_handle());
      gemm_tf32(res,
                npad,
                g,
                dim,
                idx.perm_centroids.data_handle() + (size_t)row0 * dim,
                qg.data_handle(),
                sc.data_handle(),
                npad);
      argmax_scatter_kernel<<<g, kBlock, 0, stream>>>(
        sc.data_handle(),
        n,
        npad,
        idx.l2 ? idx.perm_bias.data_handle() + row0 : nullptr,
        vals_sorted + off,
        idx.perm_id.data_handle() + row0,
        res_score,
        res_id);
    }
    off += g;
  }
}

}  // namespace

void assign(raft::resources const& res,
            const search_params& params,
            const index& idx,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_vector_view<uint32_t, int64_t> labels)
{
  const int B   = static_cast<int>(queries.extent(0));
  const int dim = static_cast<int>(idx.dim);
  RAFT_EXPECTS(queries.extent(1) == dim, "query dimension does not match the index");
  // Both the fused kernel (cp.async) and the gather of the GEMM loop load rows 16 bytes at a time.
  RAFT_EXPECTS(reinterpret_cast<uintptr_t>(queries.data_handle()) % 16 == 0,
               "the queries must be 16-byte aligned");
  RAFT_EXPECTS(labels.extent(0) >= B, "labels is too small");
  if (B == 0) return;
  const int M      = static_cast<int>(idx.n_lists);
  const int P      = resolve_n_probes(params.n_probes, M);
  const int64_t NP = (int64_t)B * P;
  RAFT_EXPECTS(NP < (int64_t{1} << 31), "batch * n_probes is too large");
  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();

  // (1) list scores and each query's top-P lists
  const int Mpad = round_up(M, 8);
  auto s1        = raft::make_device_matrix<float, int64_t>(res, B, Mpad);
  auto probes    = raft::make_device_vector<int, int64_t>(res, NP);
  gemm_tf32(
    res, M, B, dim, idx.list_centers.data_handle(), queries.data_handle(), s1.data_handle(), Mpad);
  top_p_kernel<<<B, kBlock, (size_t)M * 4, stream>>>(s1.data_handle(),
                                                     M,
                                                     Mpad,
                                                     idx.l2 ? idx.list_bias.data_handle() : nullptr,
                                                     P,
                                                     probes.data_handle());

  // (2) bucket the (list, query) pairs by list
  auto keys        = raft::make_device_vector<int, int64_t>(res, NP);
  auto vals        = raft::make_device_vector<int, int64_t>(res, NP);
  auto keys_sorted = raft::make_device_vector<int, int64_t>(res, NP);
  auto vals_sorted = raft::make_device_vector<int, int64_t>(res, NP);
  auto d_counts    = raft::make_device_vector<int, int64_t>(res, M);
  init_pairs_kernel<<<(NP + 255) / 256, 256, 0, stream>>>(
    probes.data_handle(), (int)NP, keys.data_handle(), vals.data_handle());
  RAFT_CUDA_TRY(cudaMemsetAsync(d_counts.data_handle(), 0, (size_t)M * 4, stream));
  count_keys_kernel<<<(NP + 255) / 256, 256, 0, stream>>>(
    keys.data_handle(), (int)NP, d_counts.data_handle());
  const int end_bit = std::max(1, (int)std::ceil(std::log2((double)M)));
  size_t sort_bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceRadixSort::SortPairs(nullptr,
                                                sort_bytes,
                                                keys.data_handle(),
                                                keys_sorted.data_handle(),
                                                vals.data_handle(),
                                                vals_sorted.data_handle(),
                                                (int)NP,
                                                0,
                                                end_bit,
                                                stream));
  auto sort_tmp = raft::make_device_vector<uint8_t, int64_t>(res, sort_bytes);
  RAFT_CUDA_TRY(cub::DeviceRadixSort::SortPairs(sort_tmp.data_handle(),
                                                sort_bytes,
                                                keys.data_handle(),
                                                keys_sorted.data_handle(),
                                                vals.data_handle(),
                                                vals_sorted.data_handle(),
                                                (int)NP,
                                                0,
                                                end_bit,
                                                stream));
  auto res_score = raft::make_device_vector<float, int64_t>(res, NP);
  auto res_id    = raft::make_device_vector<int, int64_t>(res, NP);
  // A list without centroids produces no candidate; make sure a stale value cannot win.
  init_results_kernel<<<(NP + 255) / 256, 256, 0, stream>>>(
    res_score.data_handle(), res_id.data_handle(), NP);

  // (3) per list: the queries probing it against its centroids, row argmax
  fused_plan plan = params.algo == assign_algo::GEMM_LOOP ? fused_plan{} : plan_fused(dim);
  // The fused kernel is used whenever it can run. With large lists (about 1000 or more centroids)
  // and large batches the per-list cuBLAS GEMMs of the fallback can be up to ~1.4x faster on an
  // H100, because cuBLAS uses wgmma; the fused kernel's launch-free scheduling wins for small
  // batches and small lists.
  RAFT_EXPECTS(params.algo != assign_algo::FUSED || plan.bn != 0,
               "assign_algo::FUSED needs compute capability >= 8.0 and enough shared memory");
  if (plan.bn != 0) {
    auto pair_offset = raft::make_device_vector<int, int64_t>(res, M);
    auto tile_prefix = raft::make_device_vector<int, int64_t>(res, M + 1);
    auto counter     = raft::make_device_vector<int, int64_t>(res, 1);
    build_work_table_kernel<<<1, 256, 0, stream>>>(d_counts.data_handle(),
                                                   idx.list_size_d.data_handle(),
                                                   idx.list_order_d.data_handle(),
                                                   M,
                                                   pair_offset.data_handle(),
                                                   tile_prefix.data_handle(),
                                                   counter.data_handle());
    fused_args args;
    args.queries      = queries.data_handle();
    args.dim          = dim;
    args.n_probes     = P;
    args.sorted_pairs = vals_sorted.data_handle();
    args.perm         = idx.perm_centroids.data_handle();
    args.perm_bias    = idx.l2 ? idx.perm_bias.data_handle() : nullptr;
    args.perm_id      = idx.perm_id.data_handle();
    args.list_offset  = idx.list_offset_d.data_handle();
    args.list_size    = idx.list_size_d.data_handle();
    args.pair_offset  = pair_offset.data_handle();
    args.pair_count   = d_counts.data_handle();
    args.counter      = counter.data_handle();
    args.res_score    = res_score.data_handle();
    args.res_id       = res_id.data_handle();
    args.order        = idx.list_order_d.data_handle();
    args.tile_prefix  = tile_prefix.data_handle();
    args.n_lists      = M;
    launch_fused(plan, args, stream);
  } else {
    assign_gemm_loop(res,
                     idx,
                     queries.data_handle(),
                     P,
                     vals_sorted.data_handle(),
                     d_counts.data_handle(),
                     res_score.data_handle(),
                     res_id.data_handle());
  }

  // (4) merge each query's P partial winners
  merge_pairs_kernel<<<(B + 255) / 256, 256, 0, stream>>>(
    res_score.data_handle(), res_id.data_handle(), P, B, labels.data_handle());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace cuvs::cluster::kmeans::detail::ivf_assign
