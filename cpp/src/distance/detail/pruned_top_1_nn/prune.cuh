/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Leonardo Xavier Kuffo Rivero
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/*
 * Copyright (c) 2025 Leonardo Xavier Kuffo Rivero
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file thirdparty/LICENSES/LICENSE.superkmeans
 */
#pragma once

#include "../../../core/nvtx.hpp"

#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_properties.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/integer_utils.hpp>
#include <raft/util/kernel_launch.hpp>
#include <raft/util/reduction.cuh>
#include <raft/util/vectorized.cuh>

#include <cub/device/device_scan.cuh>
#include <cub/warp/warp_reduce_batched.cuh>
#include <cub/warp/warp_scan.cuh>

#include <cuda_fp16.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cuvs::distance::detail::pruned_nn {

// Default filter prefix (d').
constexpr uint32_t kPartialDim = 256;
// FP16 staging pitch for float inputs, shared by all supported prefixes.
constexpr uint32_t kFilterCopyDim = kPartialDim;

/** ADSampling's margin, the eps0 of adsampling_ratio. */
constexpr float kAdsamplingEps0 = 1.5f;
// Centroids per batch: one fused filter and pruning launch per row batch and centroid batch
constexpr uint32_t kCentroidBatchSize = 1024;

constexpr std::size_t kAlignment = 16;

/**
 * ADSampling's pruning ratio for a partial distance over the first i of dim dims:
 * (i / dim) * (1 + eps0 / sqrt(i))^2, with eps0 ADSampling's margin. A partial distance over i
 * dims is compared against the ratio times the row's current best distance. The ratios assume
 * every dim carries the same share of a distance, which ADSampling ensures with a random rotation
 * of the data; without one they are a heuristic.
 */
__host__ __device__ inline float adsampling_ratio(uint32_t i, uint32_t dim, float eps0)
{
  const float f = 1.0f + eps0 / sqrtf(static_cast<float>(i));
  return static_cast<float>(i) / static_cast<float>(dim) * f * f;
}

/**
 * Squared norm of each of n rows (row stride `stride`) over their first `cols` dims; a warp per
 * row. Not raft::linalg::rowNorm: raft's row reductions take whole contiguous rows (row length =
 * stride), and masking the other columns would still read them all. When copying, pack the first
 * `copy_cols` dims as FP16 and compute the norm from those rounded values.
 */
template <typename T, typename Copy = std::nullptr_t>
RAFT_KERNEL squared_norms_kernel(const T* __restrict__ rows,
                                 int64_t n,
                                 int64_t stride,
                                 int64_t cols,
                                 float* __restrict__ out,
                                 Copy copy,
                                 int64_t copy_cols)
{
  const int64_t row = (blockIdx.x * int64_t{blockDim.x} + threadIdx.x) / raft::warp_size();
  if (row >= n) { return; }
  float s              = 0.f;
  constexpr bool kCopy = !std::is_same_v<Copy, std::nullptr_t>;
  for (int64_t j = threadIdx.x % raft::warp_size(); j < (kCopy ? copy_cols : cols);
       j += raft::warp_size()) {
    float v = static_cast<float>(rows[row * stride + j]);
    if constexpr (kCopy) {
      const half h              = static_cast<half>(v);
      copy[row * copy_cols + j] = h;
      v                         = __half2float(h);
    }
    if (j < cols) { s += v * v; }
  }
  s = raft::warpReduce(s);
  if (threadIdx.x % raft::warp_size() == 0) { out[row] = s; }
}

template <typename T, typename Copy = std::nullptr_t>
void squared_norms(const raft::resources& handle,
                   const T* rows,
                   int64_t n,
                   int64_t stride,
                   int64_t cols,
                   float* out,
                   Copy copy         = nullptr,
                   int64_t copy_cols = 0)
{
  constexpr uint32_t kThreads = 256;
  const auto blocks           = raft::ceildiv<int64_t>(n * raft::warp_size(), kThreads);
  raft::launch_kernel(handle,
                      blocks,
                      kThreads,
                      squared_norms_kernel<T, Copy>,
                      rows,
                      n,
                      stride,
                      cols,
                      out,
                      copy,
                      copy_cols);
}

using packed_winner = long long;

// Squared distances are nonnegative, so their FP32 bit patterns sort by distance.
__device__ __forceinline__ packed_winner pack_winner(float distance, uint32_t label)
{
  return (static_cast<packed_winner>(__float_as_uint(distance)) << 32) | label;
}

__device__ __forceinline__ float winner_distance(packed_winner winner)
{
  return __uint_as_float(static_cast<uint32_t>(winner >> 32));
}

/** Each row's squared distance to its assigned centroid; optionally pack its filter dims as FP16.
 */
template <bool Copy, typename T, typename C, typename Distance>
RAFT_KERNEL distance_to_assigned_and_copy_kernel(const T* __restrict__ rows,
                                                 int64_t n,
                                                 const C* __restrict__ centroids,
                                                 int64_t dim,
                                                 const uint32_t* __restrict__ labels,
                                                 Distance* __restrict__ dist,
                                                 half* __restrict__ filter_rows)
{
  const int64_t row = (blockIdx.x * int64_t{blockDim.x} + threadIdx.x) / raft::warp_size();
  if (row >= n) { return; }
  const T* row_data          = rows + row * dim;
  const C* assigned_centroid = centroids + static_cast<int64_t>(labels[row]) * dim;
  float distance             = 0.f;
  for (int64_t j = threadIdx.x % raft::warp_size(); j < dim; j += raft::warp_size()) {
    const float value = static_cast<float>(row_data[j]);
    const float diff  = value - static_cast<float>(assigned_centroid[j]);
    distance += diff * diff;
    if constexpr (Copy) {
      if (j < kFilterCopyDim) { filter_rows[row * kFilterCopyDim + j] = static_cast<half>(value); }
    }
  }
  distance = raft::warpReduce(distance);
  if (threadIdx.x % raft::warp_size() == 0) {
    if constexpr (std::is_same_v<Distance, packed_winner>) {
      dist[row] = pack_winner(distance, labels[row]);
    } else {
      dist[row] = distance;
    }
  }
}

template <bool Copy = false, typename T, typename C, typename Distance>
void distance_to_assigned(const raft::resources& handle,
                          const T* rows,
                          int64_t n,
                          const C* centroids,
                          int64_t dim,
                          const uint32_t* labels,
                          Distance* dist,
                          half* filter_rows = nullptr)
{
  constexpr uint32_t kThreads = 256;
  const auto blocks           = raft::ceildiv<int64_t>(n * raft::warp_size(), kThreads);
  raft::launch_kernel(handle,
                      blocks,
                      kThreads,
                      distance_to_assigned_and_copy_kernel<Copy, T, C, Distance>,
                      rows,
                      n,
                      centroids,
                      dim,
                      labels,
                      dist,
                      filter_rows);
}

/**
 * One dim tile in 16-byte loads. Dimensions that are not tile-aligned still have a whole-vector
 * tail; zero-fill vectors past its end. Aligned instantiations omit the bounds checks.
 */
template <uint32_t Dims, bool Aligned, typename T>
__device__ __forceinline__ void load_tile(const T* __restrict__ values,
                                          uint32_t valid_dims,
                                          T (&tile)[Dims / raft::warp_size()])
{
  constexpr uint32_t kVec = 16 / sizeof(T);
  static_assert(Dims % (raft::warp_size() * kVec) == 0,
                "a dim tile is whole 16-byte loads per lane");
  const uint32_t lane = threadIdx.x % raft::warp_size();
#pragma unroll
  for (uint32_t j = 0; j < Dims / (raft::warp_size() * kVec); j++) {
    const uint32_t offset = (j * raft::warp_size() + lane) * kVec;
    if constexpr (!Aligned) {
      if (offset >= valid_dims) {
#pragma unroll
        for (uint32_t k = 0; k < kVec; k++) {
          tile[j * kVec + k] = T{0};
        }
        continue;
      }
    }
    raft::TxN_t<T, kVec> v;
    v.load(values, offset);
#pragma unroll
    for (uint32_t k = 0; k < kVec; k++) {
      tile[j * kVec + k] = v.val.data[k];
    }
  }
}

// Packed pair IDs used within the pruning CTA
using pair_id                    = uint32_t;
constexpr uint32_t kPairsPerWarp = 16;
static_assert(std::has_single_bit(kPairsPerWarp) && kPairsPerWarp <= raft::warp_size());
static_assert(std::has_single_bit(kCentroidBatchSize));
constexpr uint32_t kPairCentroidBits = std::countr_zero(kCentroidBatchSize);
// The terminal offset must hold the sum even when every bitmap bit is set.
constexpr uint32_t kMaxPairRows = UINT32_MAX >> kPairCentroidBits;

__device__ __forceinline__ uint32_t pair_row(pair_id pair) { return pair >> kPairCentroidBits; }

__device__ __forceinline__ uint32_t pair_centroid(pair_id pair)
{
  return pair & (kCentroidBatchSize - 1);
}

// Survivor counts and bound snapshots
constexpr int kCountWarps             = 4;
constexpr int kCountLanesPerRow       = 8;
constexpr int kCountThreads           = kCountWarps * raft::warp_size();
constexpr int64_t kCountRowsPerBlock  = kCountThreads / kCountLanesPerRow;
constexpr uint32_t kBitmapWordBits    = 32;
constexpr uint32_t kBitmapWordsPerRow = kCentroidBatchSize / kBitmapWordBits;
constexpr int kCountWordsPerLane      = kBitmapWordsPerRow / kCountLanesPerRow;
static_assert(kCentroidBatchSize == kBitmapWordBits * raft::warp_size());
static_assert(kCountLanesPerRow * kCountWordsPerLane == kBitmapWordsPerRow);
static_assert(kCountWordsPerLane * sizeof(uint32_t) == kAlignment);

inline auto row_scan_workspace_size(int64_t n_rows) -> std::size_t
{
  std::size_t bytes = 0;
  auto* counts      = static_cast<uint32_t*>(nullptr);
  RAFT_CUDA_TRY(cub::DeviceScan::ExclusiveSum(nullptr, bytes, counts, counts, n_rows + 1));
  return bytes;
}

RAFT_KERNEL count_bitmap_survivors_kernel(int64_t n_rows,
                                          int bitmap_words,
                                          const uint32_t* bitmap,
                                          const packed_winner* winners,
                                          float* bound,
                                          uint32_t* offsets)
{
  const int lane       = raft::laneId() % kCountLanesPerRow;
  const uint32_t row   = blockIdx.x * kCountRowsPerBlock + threadIdx.x / kCountLanesPerRow;
  const int first_word = lane * kCountWordsPerLane;
  uint32_t count       = 0;
  if (row < n_rows) {
    const auto* row_bitmap = bitmap + row * kBitmapWordsPerRow;
    if (first_word + kCountWordsPerLane <= bitmap_words) {
      raft::TxN_t<uint32_t, kCountWordsPerLane> words;
      words.load(row_bitmap, first_word);
#pragma unroll
      for (int i = 0; i < kCountWordsPerLane; ++i) {
        count += __popc(words.val.data[i]);
      }
    } else {
      // Only read valid words in the final centroid batch.
#pragma unroll
      for (int i = 0; i < kCountWordsPerLane; ++i) {
        if (first_word + i < bitmap_words) { count += __popc(row_bitmap[first_word + i]); }
      }
    }
  }

  count = raft::logicalWarpReduce<kCountLanesPerRow>(count, raft::add_op{});
  if (lane == 0 && row < n_rows) {
    offsets[row] = count;
    // Keep the pruning bound fixed while pruning updates winners.
    if (count != 0) { bound[row] = winner_distance(winners[row]); }
    if (row == 0) { offsets[n_rows] = 0; }
  }
}

// Pair pruning
constexpr uint32_t kPruneWarps    = 4;
constexpr uint32_t kPairThreads   = kPruneWarps * raft::warp_size();
constexpr uint32_t kPairsPerBlock = kPruneWarps * kPairsPerWarp;
constexpr int kPruneBlocksPerSm   = 16;

template <typename T>
constexpr uint32_t kPruneTileSize = std::is_same_v<T, float> ? 128 : 256;

struct bitmap_row_span {
  uint32_t row;
  uint32_t row_begin;
};

/** Each warp compacts its live pairs locally and submits full distances to each row's winner. */
template <typename T, bool Aligned, uint32_t PartialDim>
__device__ __forceinline__ void prune_warp(const T* rows,
                                           const T* centroids,
                                           uint32_t dim,
                                           const float* bound,
                                           uint32_t centroid_offset,
                                           packed_winner* winners,
                                           pair_id pair,
                                           uint32_t live,
                                           pair_id* kept_pair,
                                           float* kept_distance,
                                           float* kept_bound)
{
  constexpr uint32_t kTileDims     = kPruneTileSize<T>;
  constexpr uint32_t kFirstTileDim = PartialDim + kTileDims;
  static_assert(PartialDim % kTileDims == 0);
  const uint32_t lane        = raft::laneId();
  float best                 = lane < live ? bound[pair_row(pair)] : 0.f;
  const auto accumulate_tile = [&](pair_id ids, uint32_t d, float sum) {
    const uint32_t r = pair_row(ids);
    const uint32_t c = pair_centroid(ids);
    T row_tile[kTileDims / raft::warp_size()];
    T centroid_tile[kTileDims / raft::warp_size()];
    load_tile<kTileDims, Aligned>(rows + static_cast<int64_t>(r) * dim + d, dim - d, row_tile);
    load_tile<kTileDims, Aligned>(
      centroids + static_cast<int64_t>(c) * dim + d, dim - d, centroid_tile);
#pragma unroll
    for (uint32_t j = 0; j < kTileDims / raft::warp_size(); ++j) {
      const float diff = static_cast<float>(row_tile[j]) - static_cast<float>(centroid_tile[j]);
      sum              = fmaf(diff, diff, sum);
    }
    return sum;
  };

  float first_partial[kPairsPerWarp] = {};
  // Keep the partial sums in registers and reduce once at the first bound check.
#pragma unroll 1
  for (uint32_t d = 0; d < kFirstTileDim; d += kTileDims) {
#pragma unroll
    for (uint32_t q = 0; q < kPairsPerWarp; ++q) {
      first_partial[q] = accumulate_tile(raft::shfl(pair, q), d, first_partial[q]);
    }
  }
  using FirstReduce = cub::WarpReduceBatched<float, kPairsPerWarp>;
  typename FirstReduce::TempStorage first_storage;
  float distance = FirstReduce(first_storage).Sum(first_partial);

  uint32_t d = kFirstTileDim;
  for (;;) {
    const bool last   = d >= dim;
    const float ratio = last ? 1.f : adsampling_ratio(d, dim, kAdsamplingEps0);
    const bool keep   = lane < live && distance < best * ratio;
    if (last) {
      if (keep) {
        atomicMin(winners + pair_row(pair),
                  pack_winner(distance, centroid_offset + pair_centroid(pair)));
      }
      break;
    }
    const uint32_t mask = __ballot_sync(raft::warp_full_mask(), keep);
    if (keep) {
      const uint32_t position = __popc(mask & ((1u << lane) - 1));
      kept_pair[position]     = pair;
      kept_distance[position] = distance;
      kept_bound[position]    = best;
    }
    __syncwarp();
    live = __popc(mask);
    if (lane < live) {
      pair     = kept_pair[lane];
      distance = kept_distance[lane];
      best     = kept_bound[lane];
    }
    __syncwarp();
    if (live <= 1) { break; }
    const auto process_batch = [&](auto tile, uint32_t i) {
      constexpr uint32_t kTile = decltype(tile)::value;
      float partial[kTile];
#pragma unroll
      for (uint32_t q = 0; q < kTile; ++q) {
        partial[q] = accumulate_tile(raft::shfl(pair, i + q), d, 0.f);
      }
      if constexpr (kTile == 1) {
        const float tail = raft::warpReduce(partial[0]);
        if (lane == i) { distance += tail; }
      } else {
        using Reduce = cub::WarpReduceBatched<float, kTile>;
        typename Reduce::TempStorage storage;
        const uint32_t chunk_lane = (lane - i) % raft::warp_size();
        const float tail          = raft::shfl(Reduce(storage).Sum(partial), chunk_lane);
        if (lane >= i && lane < i + kTile) { distance += tail; }
      }
    };
    uint32_t i                      = 0;
    const auto process_live_batches = [&](auto tile) {
      constexpr uint32_t kTile = decltype(tile)::value;
      for (; i + kTile <= live; i += kTile) {
        process_batch(tile, i);
      }
    };
    process_live_batches(std::integral_constant<uint32_t, 8>{});
    process_live_batches(std::integral_constant<uint32_t, 4>{});
    process_live_batches(std::integral_constant<uint32_t, 1>{});
    d += kTileDims;
  }
  if (live == 1 && d < dim) {
    // The remaining pair is in lane zero. Its distance no longer needs a batched reduction.
    const pair_id ids      = raft::shfl(pair, 0);
    const auto* row_values = rows + static_cast<int64_t>(pair_row(ids)) * dim;
    const auto* centroid   = centroids + static_cast<int64_t>(pair_centroid(ids)) * dim;
    distance               = raft::shfl(distance, 0);
    best                   = raft::shfl(best, 0);
    for (; d < dim; d += kTileDims) {
      T row_tile[kTileDims / raft::warp_size()];
      T centroid_tile[kTileDims / raft::warp_size()];
      load_tile<kTileDims, Aligned>(row_values + d, dim - d, row_tile);
      load_tile<kTileDims, Aligned>(centroid + d, dim - d, centroid_tile);
      float partial = 0.f;
#pragma unroll
      for (uint32_t j = 0; j < kTileDims / raft::warp_size(); ++j) {
        const float diff = static_cast<float>(row_tile[j]) - static_cast<float>(centroid_tile[j]);
        partial          = fmaf(diff, diff, partial);
      }
      distance += raft::warpReduce(partial);
      const bool last   = d + kTileDims >= dim;
      const float ratio = last ? 1.f : adsampling_ratio(d + kTileDims, dim, kAdsamplingEps0);
      const bool keep   = distance < best * ratio;
      if (last || !keep) {
        if (lane == 0 && keep) {
          atomicMin(winners + pair_row(ids),
                    pack_winner(distance, centroid_offset + pair_centroid(ids)));
        }
        break;
      }
    }
  }
}

template <typename T, bool Aligned, uint32_t PartialDim>
RAFT_KERNEL prune_kernel(const T* rows,
                         const T* centroids,
                         uint32_t dim,
                         const uint32_t* bitmap,
                         int bitmap_words,
                         const uint32_t* offsets,
                         uint32_t n_rows,
                         const float* bound,
                         uint32_t centroid_offset,
                         packed_winner* winners)
{
  __shared__ bitmap_row_span spans[kPairsPerBlock];
  __shared__ uint32_t span_count;
  __shared__ pair_id decoded_pairs[kPairsPerBlock];
  __shared__ pair_id kept_pair[kPruneWarps][kPairsPerWarp];
  __shared__ float kept_distance[kPruneWarps][kPairsPerWarp];
  __shared__ float kept_bound[kPruneWarps][kPairsPerWarp];

  const uint32_t total      = offsets[n_rows];
  const uint32_t lane       = raft::laneId();
  const uint32_t local_warp = threadIdx.x / raft::warp_size();
  const uint32_t warp_begin = local_warp * kPairsPerWarp;
  const uint32_t chunks     = raft::ceildiv(total, kPairsPerBlock);
  for (uint32_t chunk = blockIdx.x; chunk < chunks; chunk += gridDim.x) {
    const uint32_t begin = chunk * kPairsPerBlock;
    const uint32_t end   = min(begin + kPairsPerBlock, total);
    if (threadIdx.x == 0) {
      // Find the first row whose end is past begin, skipping duplicate offsets for empty rows.
      uint32_t lo = 0, hi = n_rows;
      while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        if (offsets[mid + 1] <= begin) {
          lo = mid + 1;
        } else {
          hi = mid;
        }
      }

      uint32_t count     = 0;
      uint32_t row_begin = offsets[lo];
      for (uint32_t row = lo; row_begin < end; ++row) {
        const uint32_t row_end = offsets[row + 1];
        if (row_end > row_begin) { spans[count++] = {row, row_begin}; }
        row_begin = row_end;
      }
      span_count = count;
    }
    __syncthreads();

    // Each warp decodes one row span at a time into the CTA's shared pair buffer.
    using BitmapScan = cub::WarpScan<uint32_t, raft::warp_size()>;
    BitmapScan::TempStorage scan_storage;
    for (uint32_t i = local_warp; i < span_count; i += kPruneWarps) {
      const auto span           = spans[i];
      const uint32_t span_begin = max(begin, span.row_begin);
      const uint32_t span_end   = i + 1 < span_count ? spans[i + 1].row_begin : end;
      uint32_t word = lane < bitmap_words ? bitmap[span.row * kBitmapWordsPerRow + lane] : 0;
      uint32_t word_offset;
      BitmapScan(scan_storage).ExclusiveSum(__popc(word), word_offset);

      const uint32_t first = span_begin - span.row_begin;
      const uint32_t last  = span_end - span.row_begin;
      for (uint32_t index = word_offset; word != 0 && index < last; ++index) {
        if (index >= first) {
          const uint32_t centroid = lane * kBitmapWordBits + (__ffs(word) - 1);
          decoded_pairs[span.row_begin + index - begin] =
            (span.row << kPairCentroidBits) | centroid;
        }
        word &= word - 1;
      }
    }
    __syncthreads();

    if (warp_begin < end - begin) {
      const uint32_t live = min(kPairsPerWarp, end - begin - warp_begin);
      // The first distance tile always processes a full warp's group of pairs.
      const pair_id pair = decoded_pairs[warp_begin + (lane < live ? lane : 0)];
      prune_warp<T, Aligned, PartialDim>(rows,
                                         centroids,
                                         dim,
                                         bound,
                                         centroid_offset,
                                         winners,
                                         pair,
                                         live,
                                         kept_pair[local_warp],
                                         kept_distance[local_warp],
                                         kept_bound[local_warp]);
    }
    // Finish consuming the decoded pairs before the next chunk reuses the shared buffers.
    __syncthreads();
  }
}

/** Publish the final labels and distances after the last centroid batch. */
RAFT_KERNEL unpack_winners_kernel(int64_t n_rows,
                                  const packed_winner* winners,
                                  uint32_t* labels,
                                  float* dist)
{
  const int64_t row = blockIdx.x * int64_t{blockDim.x} + threadIdx.x;
  if (row < n_rows) {
    const packed_winner winner = winners[row];
    labels[row]                = static_cast<uint32_t>(winner);
    dist[row]                  = winner_distance(winner);
  }
}

template <uint32_t PartialDim, typename T>
void prune(const raft::resources& handle,
           const T* rows,
           int64_t n_rows,
           const T* centroids,
           uint32_t centroid_offset,
           int64_t dim,
           const uint32_t* bitmap,
           int bitmap_words,
           uint32_t* offsets,
           void* scan_workspace,
           std::size_t scan_bytes,
           float* bound,
           packed_winner* winners)
{
  const bool aligned = (dim - PartialDim) % kPruneTileSize<T> == 0;
  {
    common::nvtx::range<common::nvtx::domain::cuvs> range("pruned_1nn::count_survivors");
    raft::launch_kernel(handle,
                        raft::ceildiv(n_rows, kCountRowsPerBlock),
                        kCountThreads,
                        count_bitmap_survivors_kernel,
                        n_rows,
                        bitmap_words,
                        bitmap,
                        winners,
                        bound,
                        offsets);
  }
  {
    common::nvtx::range<common::nvtx::domain::cuvs> range("pruned_1nn::scan_survivors");
    RAFT_CUDA_TRY(cub::DeviceScan::ExclusiveSum(scan_workspace,
                                                scan_bytes,
                                                offsets,
                                                offsets,
                                                n_rows + 1,
                                                raft::resource::get_cuda_stream(handle).get()));
  }
  {
    common::nvtx::range<common::nvtx::domain::cuvs> range("pruned_1nn::prune");
    const auto kernel =
      aligned ? prune_kernel<T, true, PartialDim> : prune_kernel<T, false, PartialDim>;
    const auto blocks =
      kPruneBlocksPerSm * raft::resource::get_device_properties(handle).multiProcessorCount;
    raft::launch_kernel(handle,
                        blocks,
                        kPairThreads,
                        kernel,
                        rows,
                        centroids,
                        static_cast<uint32_t>(dim),
                        bitmap,
                        bitmap_words,
                        offsets,
                        static_cast<uint32_t>(n_rows),
                        bound,
                        centroid_offset,
                        winners);
  }
}

}  // namespace cuvs::distance::detail::pruned_nn
