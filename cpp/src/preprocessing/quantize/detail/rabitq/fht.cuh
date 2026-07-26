/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Imported from the standalone RaBitQ GPU quantizer. Behaviour unchanged;
 * only include paths, error macros and the enclosing namespace differ.
 */

/******************************************************************************
 * Fast Hadamard Transform + Kac's-walk rotation kernels (CUDA).
 *
 * Two execution strategies share one butterfly network. Both apply the
 * transform levels in ascending element-bit order with identical arithmetic
 * forms, so they produce bit-identical results:
 *
 *  - Warp-per-vector kernels (large batches, padded dim <= 2048): one warp
 *    holds the whole vector in registers, lane-strided; levels 0-4 are warp
 *    shuffles, higher levels are in-register butterflies. No shared memory,
 *    no block barriers.
 *
 *  - Cooperative kernels (small batches, and dims beyond the warp kernels'
 *    register budget): one block per vector; warp w owns the contiguous
 *    slice [w*S, (w+1)*S), lane-strided in registers. Levels below log2(S)
 *    stay inside the warp; each of the top log2(W) levels publishes the
 *    slices to a shared-memory mirror and combines with the partner
 *    slice's value at the same position. Threads never exchange element
 *    ownership.
 ******************************************************************************/

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <type_traits>
#include <raft/core/error.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

namespace cuvs::preprocessing::quantize::rabitq::detail {

namespace fht {

// ============================================================================
// Compile-time helpers
// ============================================================================

constexpr __host__ __device__ int clog2(int val) { return val > 1 ? 1 + clog2(val >> 1) : 0; }

/// Threads used by the cooperative kernels for a transform of 2^log_len
/// elements: length/4 clamped to [32, 256], except 512 for the largest size
/// (keeps registers per lane at <= 64 and avoids spills).
constexpr int coop_log_threads(int log_len) {
    int t = log_len - 2;
    if (t < 5) t = 5;
    if (t > 8) t = 8;
    if (log_len >= 15) t = 9;
    return t;
}

/// Generic compile-time dispatch: converts a runtime log_N (3..15) into a
/// template parameter by recursive if-constexpr.
/// Usage: dispatch_log_n(log_N, [&](auto K) { launch<K.value>(...); });
template<int kCur = 3, int kMax = 15, typename Func>
inline void dispatch_log_n(int log_n, Func&& func) {
    if (log_n == kCur) {
        func(std::integral_constant<int, kCur>{});
    } else if constexpr (kCur < kMax) {
        dispatch_log_n<kCur + 1, kMax>(log_n, std::forward<Func>(func));
    } else {
        std::cerr << "fht::dispatch_log_n: unsupported log_N=" << log_n << std::endl;
        exit(EXIT_FAILURE);
    }
}

template<int kCur, int kMax, typename Func>
inline void dispatch_p32(int p32, Func&& func) {
    if (p32 == kCur) {
        func(std::integral_constant<int, kCur>{});
    } else if constexpr (kCur < kMax) {
        dispatch_p32<kCur + 1, kMax>(p32, std::forward<Func>(func));
    } else {
        std::cerr << "fht::dispatch_p32: unsupported p32=" << p32 << std::endl;
        exit(EXIT_FAILURE);
    }
}

// ============================================================================
// Device helpers
// ============================================================================

__device__ __forceinline__ float flip_sign_bit(float v, uint32_t bit) {
    return __uint_as_float(__float_as_uint(v) ^ (bit << 31));
}

// ============================================================================
// Warp-per-vector fused rotation kernels (large-batch path)
//
// Each warp holds one full (padded) vector in registers, so all butterfly
// levels are either in-thread ops or warp shuffles: no shared memory and no
// __syncthreads(). Used when N is large enough that one-warp-per-vector
// saturates the GPU (see kWarpPathMinN); small batches use the cooperative
// kernels below, which spread a single vector across more threads.
// ============================================================================

// ----------------------------------------------------------------------------
// Power-of-2 path. Lane `l` of the warp owns float4 chunks at element offsets
// 128*c + 4*l, i.e. element-index bits [1:0] select within the float4,
// bits [6:2] the lane, bits [7+] the chunk register.
// ----------------------------------------------------------------------------
template<int kLogN>
__global__ void fht_kac_rotate_warp_kernel(
    const float* input,          // no __restrict__: in-place rotation allowed
    float* output,
    const uint8_t* __restrict__ flip,
    int N, float total_scale)
{
    constexpr int kDim  = 1 << kLogN;
    constexpr int kC    = kDim / 128;   // float4 registers per lane
    constexpr int kLogC = clog2(kC);
    constexpr int kWordsPerRound = kDim / 32;
    static_assert(kLogN >= 7, "warp kernel needs at least one float4 per lane");

    const int lane   = threadIdx.x & 31;
    const int vec_id = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (vec_id >= N) return;

    const uint32_t* flip32 = reinterpret_cast<const uint32_t*>(flip);

    const float4* vin = reinterpret_cast<const float4*>(input + (size_t)vec_id * kDim);
    float4 x[kC];
    #pragma unroll
    for (int c = 0; c < kC; ++c) x[c] = vin[c * 32 + lane];

    const int bit_shift = 4 * (lane & 7);   // this lane's 4 flip bits inside the word
    const int word_off  = lane >> 3;

    #pragma unroll
    for (int round = 0; round < 4; ++round) {
        // Sign flip (element e uses bit e%32 of word e/32; all four of this
        // lane's bits sit in one word).
        #pragma unroll
        for (int c = 0; c < kC; ++c) {
            uint32_t w = __ldg(&flip32[round * kWordsPerRound + 4 * c + word_off]) >> bit_shift;
            x[c].x = flip_sign_bit(x[c].x,  w        & 1u);
            x[c].y = flip_sign_bit(x[c].y, (w >> 1)  & 1u);
            x[c].z = flip_sign_bit(x[c].z, (w >> 2)  & 1u);
            x[c].w = flip_sign_bit(x[c].w, (w >> 3)  & 1u);
        }

        // Levels 0-1: inside each float4.
        #pragma unroll
        for (int c = 0; c < kC; ++c) {
            float a, b;
            a = x[c].x; b = x[c].y; x[c].x = a + b; x[c].y = a - b;
            a = x[c].z; b = x[c].w; x[c].z = a + b; x[c].w = a - b;
            a = x[c].x; b = x[c].z; x[c].x = a + b; x[c].z = a - b;
            a = x[c].y; b = x[c].w; x[c].y = a + b; x[c].w = a - b;
        }

        // Levels 2-6: element bit (lv+2) == lane bit lv -> warp shuffles.
        #pragma unroll
        for (int lv = 0; lv < 5; ++lv) {
            const int mask  = 1 << lv;
            const float sgn = (lane & mask) ? -1.f : 1.f;
            #pragma unroll
            for (int c = 0; c < kC; ++c) {
                float o;
                o = __shfl_xor_sync(0xffffffff, x[c].x, mask); x[c].x = sgn * x[c].x + o;
                o = __shfl_xor_sync(0xffffffff, x[c].y, mask); x[c].y = sgn * x[c].y + o;
                o = __shfl_xor_sync(0xffffffff, x[c].z, mask); x[c].z = sgn * x[c].z + o;
                o = __shfl_xor_sync(0xffffffff, x[c].w, mask); x[c].w = sgn * x[c].w + o;
            }
        }

        // Levels 7+: across chunk registers, in-thread. Pair p of a level
        // sits at offset (p % stride) inside butterfly group (p / stride).
        #pragma unroll
        for (int lv = 0; lv < kLogC; ++lv) {
            const int stride = 1 << lv;
            #pragma unroll
            for (int p = 0; p < kC / 2; ++p) {
                const int lo = (p / stride) * (2 * stride) + (p % stride);
                const int hi = lo + stride;
                float4 a = x[lo], b = x[hi];
                x[lo].x = a.x + b.x; x[lo].y = a.y + b.y;
                x[lo].z = a.z + b.z; x[lo].w = a.w + b.w;
                x[hi].x = a.x - b.x; x[hi].y = a.y - b.y;
                x[hi].z = a.z - b.z; x[hi].w = a.w - b.w;
            }
        }
    }

    float4* vout = reinterpret_cast<float4*>(output + (size_t)vec_id * kDim);
    #pragma unroll
    for (int c = 0; c < kC; ++c) {
        float4 v;
        v.x = __fmul_rn(x[c].x, total_scale); v.y = __fmul_rn(x[c].y, total_scale);
        v.z = __fmul_rn(x[c].z, total_scale); v.w = __fmul_rn(x[c].w, total_scale);
        vout[c * 32 + lane] = v;
    }
}

// ----------------------------------------------------------------------------
// Non-power-of-2 path. Lane-strided layout: lane `l` owns elements l + 32*j
// in register x[j]. Fully templated on (kLogT, kP32 = padded_dim/32) so all
// register indices are compile-time:
//   - padded dim  P = 32*kP32, registers per lane E = kP32
//   - FHT segment of T = 2^kLogT elements at offset 0 (even rounds) or
//     P - T (odd rounds); both are multiples of 32, so a segment is a
//     contiguous register window [base, base + T/32).
//   - segment element bits [4:0] = lane -> levels 0-4 via shuffles,
//     bits [5+] = register -> levels 5+ in-thread.
//   - Kac's walk pairs (e, e + P/2). For even kP32, P/2 is a multiple of 32
//     and the partner is register j + kP32/2 in the same lane. For odd
//     kP32, P/2 = 32*kHF + 16, so the partner lives in lane^16; the pairs
//     form independent 2-cycles between register j of lanes < 16 and
//     register m(j) of lanes >= 16, each resolved by one shfl_xor(16) with
//     both register indices compile-time.
// ----------------------------------------------------------------------------
template<int kLogT, int kP32>
__global__ void fht_kac_rotate_warp_nonpow2_kernel(
    const float* input,          // no __restrict__: in-place rotation allowed
    float* output,
    const uint8_t* __restrict__ flip,
    int N, float fac, float final_scale)
{
    static_assert(kLogT >= 5 && kLogT <= 10, "warp nonpow2 kernel needs T in [32, 1024]");
    constexpr int kE    = kP32;                                // registers = P/32
    constexpr int kP    = 32 * kP32;                           // padded dim
    constexpr int kT32  = 1 << (kLogT - 5);
    constexpr int kB    = kE - kT32;                           // (P - T)/32
    constexpr int kShflLevels = 5;
    constexpr int kRegLevels  = (kLogT > 5) ? (kLogT - 5) : 0;
    static_assert(kB >= 0 && kB < kE, "padded dim must be in (T, 2T]");

    const int lane   = threadIdx.x & 31;
    const int vec_id = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (vec_id >= N) return;

    const uint32_t* flip32 = reinterpret_cast<const uint32_t*>(flip);

    const float* vin = input + (size_t)vec_id * kP;
    float x[kE];
    #pragma unroll
    for (int j = 0; j < kE; ++j) x[j] = vin[lane + 32 * j];

    #pragma unroll
    for (int round = 0; round < 4; ++round) {
        // Sign flip on all P elements (element l + 32j -> bit l of word j).
        #pragma unroll
        for (int j = 0; j < kE; ++j) {
            uint32_t w = __ldg(&flip32[round * kE + j]);
            x[j] = flip_sign_bit(x[j], (w >> lane) & 1u);
        }

        const int base = (round & 1) ? kB : 0;   // constant per unrolled round

        // FHT levels 0-4 on the segment: segment bit lv == lane bit lv.
        #pragma unroll
        for (int lv = 0; lv < kShflLevels; ++lv) {
            const int mask  = 1 << lv;
            const float sgn = (lane & mask) ? -1.f : 1.f;
            #pragma unroll
            for (int j = 0; j < kT32; ++j) {
                float o = __shfl_xor_sync(0xffffffff, x[base + j], mask);
                x[base + j] = sgn * x[base + j] + o;
            }
        }

        // FHT levels 5+ across the segment's register window, in-thread.
        #pragma unroll
        for (int lv = 0; lv < kRegLevels; ++lv) {
            const int stride = 1 << lv;
            #pragma unroll
            for (int p = 0; p < kT32 / 2; ++p) {
                const int lo = base + (p / stride) * (2 * stride) + (p % stride);
                const int hi = lo + stride;
                float a = x[lo], b = x[hi];
                x[lo] = a + b; x[hi] = a - b;
            }
        }

        // Per-round fac on the FHT segment only (kacs walk mixes scales).
        // __fmul_rn keeps this a standalone multiply (never FMA-contracted).
        #pragma unroll
        for (int j = 0; j < kT32; ++j) x[base + j] = __fmul_rn(x[base + j], fac);

        // Kac's walk.
        if constexpr ((kP32 & 1) == 0) {
            // P/2 = 32*(kP32/2): partner is the same lane, register j + half.
            constexpr int kHalfR = kP32 / 2;
            #pragma unroll
            for (int j = 0; j < kHalfR; ++j) {
                float a = x[j], b = x[j + kHalfR];
                x[j] = a + b; x[j + kHalfR] = a - b;
            }
        } else {
            // P/2 = 32*kHF + 16: partner of element l + 32j is in lane^16.
            // Register j of lanes < 16 pairs with register m(j) of lanes
            // >= 16 (a mutual 2-cycle), where lanes-<16 slots j <= kHF and
            // lanes->=16 slots m < kHF hold the first-half element of their
            // pair. Exchange pre-values via one shuffle, then combine.
            constexpr int kHF = (kP32 - 1) / 2;
            #pragma unroll
            for (int j = 0; j < kP32; ++j) {
                const int m = (j < kHF) ? (j + kHF)
                            : (j == kHF) ? (kP32 - 1) : (j - kHF - 1);
                float send = (lane < 16) ? x[j] : x[m];
                float o = __shfl_xor_sync(0xffffffff, send, 16);
                if (lane < 16) x[j] = (j <= kHF) ? (x[j] + o) : (o - x[j]);
                else           x[m] = (m <  kHF) ? (x[m] + o) : (o - x[m]);
            }
        }
    }

    float* vout = output + (size_t)vec_id * kP;
    #pragma unroll
    for (int j = 0; j < kE; ++j) vout[lane + 32 * j] = __fmul_rn(x[j], final_scale);
}

// ============================================================================
// Cooperative fused rotation kernels (small-batch / large-dim path)
//
// One block per vector, W = threads/32 warps. Warp w owns the contiguous
// slice [w*S, (w+1)*S) of the transform, lane-strided within the slice:
// element(w, l, j) = w*S + l + 32*j. Levels below log2(S) are handled inside
// the warp exactly like the warp kernels above. Each of the top log2(W)
// levels publishes the slices to a shared-memory mirror and combines with
// the partner slice's value at the same (lane, register) position —
// conflict-free stride-1 accesses, no ownership exchange.
// ============================================================================

template<int kLogN>
__global__ void __launch_bounds__(1 << coop_log_threads(kLogN)) fht_kac_rotate_coop_kernel(
    const float* input,          // no __restrict__: in-place rotation allowed
    float* output,
    const uint8_t* __restrict__ flip,
    int N, float total_scale)
{
    constexpr int kDim  = 1 << kLogN;
    constexpr int kLogT = coop_log_threads(kLogN);
    constexpr int kLogW = kLogT - 5;               // log2(warps per block)
    constexpr int kLogS = kLogN - kLogW;           // slice = 2^kLogS elements
    constexpr int kS    = 1 << kLogS;
    constexpr int kE    = (kS >= 32) ? (kS / 32) : 1;   // registers per lane
    constexpr int kShfl = (kLogS < 5) ? kLogS : 5;
    constexpr int kRegLv = (kLogS > 5) ? (kLogS - 5) : 0;
    static_assert(kLogS >= 3, "slice must hold at least 8 elements");

    const int lane = threadIdx.x & 31;
    const int w    = threadIdx.x >> 5;
    const int vec_id = blockIdx.x;
    if (vec_id >= N) return;

    extern __shared__ float s_vec[];               // kDim floats when kW > 1

    const uint32_t* flip32 = reinterpret_cast<const uint32_t*>(flip);
    const float* vin = input  + (size_t)vec_id * kDim;
    float* vout      = output + (size_t)vec_id * kDim;
    const int sbase  = w * kS;

    float x[kE];
    #pragma unroll
    for (int j = 0; j < kE; ++j)
        x[j] = (kS >= 32 || lane < kS) ? vin[sbase + lane + 32 * j] : 0.f;

    for (int round = 0; round < 4; ++round) {
        // Sign flip in registers; flip bit index of element e is
        // round*kDim + e within the packed bit stream.
        #pragma unroll
        for (int j = 0; j < kE; ++j) {
            if (kS >= 32 || lane < kS) {
                const int b = round * kDim + sbase + 32 * j + lane;
                const uint32_t word = __ldg(&flip32[b >> 5]);
                x[j] = flip_sign_bit(x[j], (word >> (b & 31)) & 1u);
            }
        }

        // Levels 0..kShfl-1: lane shuffles.
        #pragma unroll
        for (int lv = 0; lv < kShfl; ++lv) {
            const int mask  = 1 << lv;
            const float sgn = (lane & mask) ? -1.f : 1.f;
            #pragma unroll
            for (int j = 0; j < kE; ++j) {
                float o = __shfl_xor_sync(0xffffffff, x[j], mask);
                x[j] = sgn * x[j] + o;
            }
        }

        // Levels 5..kLogS-1: register butterflies inside the slice.
        #pragma unroll
        for (int lv = 0; lv < kRegLv; ++lv) {
            const int stride = 1 << lv;
            #pragma unroll
            for (int g = 0; g < kE; g += 2 * stride) {
                #pragma unroll
                for (int m = 0; m < stride; ++m) {
                    const int lo = g + m, hi = lo + stride;
                    float a = x[lo], b = x[hi];
                    x[lo] = a + b; x[hi] = a - b;
                }
            }
        }

        // Levels kLogS..kLogN-1: cross-slice, one level at a time. Publish
        // this slice to the shared mirror, then combine with the partner
        // slice's value at the same (lane, register) position.
        if constexpr (kLogW > 0) {
            for (int m = 0; m < kLogW; ++m) {
                #pragma unroll
                for (int j = 0; j < kE; ++j)
                    s_vec[sbase + lane + 32 * j] = x[j];
                __syncthreads();
                const float sgn = (w & (1 << m)) ? -1.f : 1.f;
                const int pbase = (w ^ (1 << m)) * kS;
                #pragma unroll
                for (int j = 0; j < kE; ++j) {
                    float o = s_vec[pbase + lane + 32 * j];
                    x[j] = sgn * x[j] + o;
                }
                __syncthreads();
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < kE; ++j) {
        if (kS >= 32 || lane < kS)
            vout[sbase + lane + 32 * j] = __fmul_rn(x[j], total_scale);
    }
}

// ----------------------------------------------------------------------------
// Cooperative non-power-of-2 kernel. The padded vector lives in shared
// memory; sign flips and the Kac walk operate on it directly with
// block-strided loops, while each round's FHT segment (2^kLogTrunc elements
// at offset 0 / P-T) is pulled into registers slice-per-warp and transformed
// with the same machinery as the pow2 cooperative kernel.
// ----------------------------------------------------------------------------
template<int kLogTrunc>
__global__ void __launch_bounds__(1 << coop_log_threads(kLogTrunc)) fht_kac_rotate_coop_nonpow2_kernel(
    const float* input,          // no __restrict__: in-place rotation allowed
    float* output,
    const uint8_t* __restrict__ flip,
    int N, int padded_dim, float fac, float final_scale)
{
    constexpr int kTrunc = 1 << kLogTrunc;
    constexpr int kLogT  = coop_log_threads(kLogTrunc);
    constexpr int kNThreads = 1 << kLogT;
    constexpr int kLogW  = kLogT - 5;              // log2(warps per block)
    constexpr int kLogS  = kLogTrunc - kLogW;
    constexpr int kS     = 1 << kLogS;
    constexpr int kE     = (kS >= 32) ? (kS / 32) : 1;
    constexpr int kShfl  = (kLogS < 5) ? kLogS : 5;
    constexpr int kRegLv = (kLogS > 5) ? (kLogS - 5) : 0;
    static_assert(kLogS >= 3, "slice must hold at least 8 elements");

    const int lane = threadIdx.x & 31;
    const int w    = threadIdx.x >> 5;
    const int tid  = threadIdx.x;
    const int vec_id = blockIdx.x;
    if (vec_id >= N) return;

    // padded_dim floats for the vector, kTrunc scratch floats used as the
    // alternate publish target of the cross-slice levels (one barrier per
    // level instead of two), then the 4 rounds of flip words staged once so
    // the per-round flip reads stay in shared memory.
    extern __shared__ float s_vec[];

    const uint32_t* flip32 = reinterpret_cast<const uint32_t*>(flip);
    const int P     = padded_dim;
    const int words = P / 32;                      // flip words per round
    const int half  = P / 2;
    const int start = P - kTrunc;

    uint32_t* s_flip = reinterpret_cast<uint32_t*>(s_vec + P + kTrunc);
    for (int i = tid; i < 4 * words; i += kNThreads) s_flip[i] = flip32[i];

    const float* vin = input + (size_t)vec_id * P;
    for (int i = tid; i < P; i += kNThreads) s_vec[i] = vin[i];
    __syncthreads();

    for (int round = 0; round < 4; ++round) {
        // Sign flip on the full padded vector.
        for (int i = tid; i < P; i += kNThreads) {
            const uint32_t word = s_flip[round * words + (i >> 5)];
            s_vec[i] = flip_sign_bit(s_vec[i], (word >> (i & 31)) & 1u);
        }
        __syncthreads();

        // Pull this warp's slice of the segment into registers.
        const int off   = (round & 1) ? start : 0;
        const int sbase = off + w * kS;
        float x[kE];
        #pragma unroll
        for (int j = 0; j < kE; ++j)
            x[j] = (kS >= 32 || lane < kS) ? s_vec[sbase + lane + 32 * j] : 0.f;

        // Levels 0..kShfl-1: lane shuffles.
        #pragma unroll
        for (int lv = 0; lv < kShfl; ++lv) {
            const int mask  = 1 << lv;
            const float sgn = (lane & mask) ? -1.f : 1.f;
            #pragma unroll
            for (int j = 0; j < kE; ++j) {
                float o = __shfl_xor_sync(0xffffffff, x[j], mask);
                x[j] = sgn * x[j] + o;
            }
        }

        // Levels 5..kLogS-1: register butterflies inside the slice.
        #pragma unroll
        for (int lv = 0; lv < kRegLv; ++lv) {
            const int stride = 1 << lv;
            #pragma unroll
            for (int g = 0; g < kE; g += 2 * stride) {
                #pragma unroll
                for (int m = 0; m < stride; ++m) {
                    const int lo = g + m, hi = lo + stride;
                    float a = x[lo], b = x[hi];
                    x[lo] = a + b; x[hi] = a - b;
                }
            }
        }

        // Levels kLogS..kLogTrunc-1: cross-slice, one level at a time.
        // Publishes alternate between the scratch area at s_vec[P..P+trunc)
        // and the segment itself, so each level needs a single barrier (the
        // next level writes the other buffer; per-warp program order makes
        // its barrier also cover this level's reads). Parity is chosen so
        // the LAST level publishes to scratch, keeping the fac writeback
        // below race-free against partner reads.
        if constexpr (kLogW > 0) {
            for (int m = 0; m < kLogW; ++m) {
                const bool to_scratch = ((kLogW - 1 - m) & 1) == 0;
                const int pub = to_scratch ? (P + w * kS) : sbase;
                #pragma unroll
                for (int j = 0; j < kE; ++j)
                    s_vec[pub + lane + 32 * j] = x[j];
                __syncthreads();
                const float sgn = (w & (1 << m)) ? -1.f : 1.f;
                const int pbase = (to_scratch ? P : off) + (w ^ (1 << m)) * kS;
                #pragma unroll
                for (int j = 0; j < kE; ++j) {
                    float o = s_vec[pbase + lane + 32 * j];
                    x[j] = sgn * x[j] + o;
                }
            }
        }

        // Write the segment back with the per-round fac (kacs walk mixes
        // scales, so it cannot be deferred). Safe without a leading barrier:
        // the segment's mirror was last read at cross-slice level kLogW-2,
        // and the final level's publish barrier ordered those reads before
        // this write (for kLogW == 0 the segment was only read by its own
        // warp's slice load above).
        #pragma unroll
        for (int j = 0; j < kE; ++j) {
            if (kS >= 32 || lane < kS)
                s_vec[sbase + lane + 32 * j] = __fmul_rn(x[j], fac);
        }
        __syncthreads();

        // Kac's walk.
        for (int i = tid; i < half; i += kNThreads) {
            float a = s_vec[i], b = s_vec[i + half];
            s_vec[i] = a + b; s_vec[i + half] = a - b;
        }
        __syncthreads();
    }

    float* vout = output + (size_t)vec_id * P;
    for (int i = tid; i < P; i += kNThreads) vout[i] = __fmul_rn(s_vec[i], final_scale);
}

// ============================================================================
// Launch helpers + dispatch policy
// ============================================================================

/// Batch size at which the warp-per-vector kernels overtake the cooperative
/// kernels. Below this, one-block-per-vector wins by spreading a vector
/// across more threads (better latency at tiny batches); above it, warp
/// kernels win on removed barriers/smem traffic. Measured on RTX PRO 6000
/// Blackwell: warp path is >=1.16x at N=1000 for every dim in [96, 2048];
/// the cooperative path is faster for D >= 512 at N <= 100. Override with
/// env RABITQ_FHT_WARP_MIN_N (testing/tuning only).
constexpr int kWarpPathMinN = 1000;

inline int warp_path_min_n() {
    const char* e = std::getenv("RABITQ_FHT_WARP_MIN_N");
    return e ? std::atoi(e) : kWarpPathMinN;
}

constexpr int kWarpKernelBlock = 256;   // 8 vectors per block

template<int kLogN>
inline void launch_warp_rotate(const float* input, float* output,
                                const uint8_t* flip, int N,
                                float total_scale, cudaStream_t stream) {
    const int vecs_per_block = kWarpKernelBlock / 32;
    const int grid = (N + vecs_per_block - 1) / vecs_per_block;
    fht_kac_rotate_warp_kernel<kLogN><<<grid, kWarpKernelBlock, 0, stream>>>(
        input, output, flip, N, total_scale);
    RAFT_CUDA_TRY(cudaGetLastError());
}

template<int kLogN>
inline void launch_coop_rotate(const float* input, float* output,
                                const uint8_t* flip, int N,
                                float total_scale, cudaStream_t stream) {
    constexpr int kThreads = 1 << coop_log_threads(kLogN);
    constexpr int kLogW    = coop_log_threads(kLogN) - 5;
    const int smem = (kLogW > 0) ? (1 << kLogN) * (int)sizeof(float) : 0;
    auto kernel = &fht_kac_rotate_coop_kernel<kLogN>;
    if (smem >= 48 * 1024)
        RAFT_CUDA_TRY(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    kernel<<<N, kThreads, smem, stream>>>(input, output, flip, N, total_scale);
    RAFT_CUDA_TRY(cudaGetLastError());
}

template<int kLogTrunc>
inline void launch_coop_rotate_nonpow2(const float* input, float* output,
                                        const uint8_t* flip, int N,
                                        int padded_dim, float fac,
                                        float final_scale, cudaStream_t stream) {
    constexpr int kThreads = 1 << coop_log_threads(kLogTrunc);
    const int smem = (padded_dim + (1 << kLogTrunc)) * (int)sizeof(float)
                   + 4 * (padded_dim / 8);   // staged flip words
    auto kernel = &fht_kac_rotate_coop_nonpow2_kernel<kLogTrunc>;
    if (smem >= 48 * 1024)
        RAFT_CUDA_TRY(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    kernel<<<N, kThreads, smem, stream>>>(input, output, flip, N, padded_dim, fac, final_scale);
    RAFT_CUDA_TRY(cudaGetLastError());
}

inline bool warp_nonpow2_eligible(int log_trunc, int padded_dim) {
    if (log_trunc < 5 || log_trunc > 10) return false;
    const int T = 1 << log_trunc;
    return padded_dim > T && padded_dim <= 2 * T &&
           padded_dim % 32 == 0 && padded_dim <= 2048;
}

// ============================================================================
// Runtime dispatch
// ============================================================================

inline void dispatch_fused_rotate(const float* input, float* output,
                                   const uint8_t* flip, int N, int log_N,
                                   float total_scale, cudaStream_t stream) {
    if (N <= 0) return;
    if (log_N >= 7 && log_N <= 11 && N >= warp_path_min_n()) {
        dispatch_log_n<7, 11>(log_N, [&](auto K) {
            launch_warp_rotate<K.value>(input, output, flip, N, total_scale, stream);
        });
        return;
    }
    dispatch_log_n(log_N, [&](auto K) {
        launch_coop_rotate<K.value>(input, output, flip, N, total_scale, stream);
    });
}

inline void dispatch_fused_rotate_nonpow2(const float* input, float* output,
                                           const uint8_t* flip, int N,
                                           int log_trunc, int padded_dim,
                                           float fac, float final_scale,
                                           cudaStream_t stream) {
    if (N <= 0) return;
    if (warp_nonpow2_eligible(log_trunc, padded_dim) && N >= warp_path_min_n()) {
        dispatch_log_n<5, 10>(log_trunc, [&](auto KT) {
            constexpr int kLogT = KT.value;
            constexpr int kMinP32 = (1 << kLogT) / 32 + 1;
            constexpr int kMaxP32 = (1 << kLogT) / 16;
            dispatch_p32<kMinP32, kMaxP32>(padded_dim / 32, [&](auto KP) {
                const int vecs_per_block = kWarpKernelBlock / 32;
                const int grid = (N + vecs_per_block - 1) / vecs_per_block;
                fht_kac_rotate_warp_nonpow2_kernel<kLogT, KP.value>
                    <<<grid, kWarpKernelBlock, 0, stream>>>(
                        input, output, flip, N, fac, final_scale);
                RAFT_CUDA_TRY(cudaGetLastError());
            });
        });
        return;
    }
    dispatch_log_n(log_trunc, [&](auto K) {
        launch_coop_rotate_nonpow2<K.value>(input, output, flip, N,
                                             padded_dim, fac, final_scale, stream);
    });
}

}  // namespace fht

}  // namespace cuvs::preprocessing::quantize::rabitq::detail
