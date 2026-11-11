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

#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/reduction.cuh>
#include <raft/util/vectorized.cuh>

#include <cub/warp/warp_reduce_batched.cuh>

#include <cuda_fp16.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cuvs::distance::detail::pruned_nn {

// Dims used by the first filter (d')
constexpr uint32_t kPartialDim = 256;
// Dims between the survivors' re-filters
constexpr uint32_t kDimTileSize = 256;

/** ADSampling's margin, the eps0 of adsampling_ratio. */
constexpr float kAdsamplingEps0 = 1.5f;
// Centroids per batch: one fused filter and pruning launch per row batch and centroid batch
constexpr uint32_t kCentroidBatchSize = 1024;
// Survivors whose loads are batched over a dim tile
constexpr uint32_t kSurvivorTileSize = 16;
// fused_filter_kernel.py's ID_SHIFT
constexpr int kSurvivorIdShift = 32;

constexpr uint32_t kWarpSize = 32;
constexpr uint32_t kFullMask = 0xffffffff;

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
  const int64_t row = (blockIdx.x * int64_t{blockDim.x} + threadIdx.x) / kWarpSize;
  if (row >= n) { return; }
  float s              = 0.f;
  constexpr bool kCopy = !std::is_same_v<Copy, std::nullptr_t>;
  for (int64_t j = threadIdx.x % kWarpSize; j < (kCopy ? copy_cols : cols); j += kWarpSize) {
    float v = static_cast<float>(rows[row * stride + j]);
    if constexpr (kCopy) {
      const half h              = static_cast<half>(v);
      copy[row * copy_cols + j] = h;
      v                         = __half2float(h);
    }
    if (j < cols) { s += v * v; }
  }
  s = raft::warpReduce(s);
  if (threadIdx.x % kWarpSize == 0) { out[row] = s; }
}

template <typename T, typename Copy = std::nullptr_t>
void squared_norms(const T* rows,
                   int64_t n,
                   int64_t stride,
                   int64_t cols,
                   float* out,
                   cudaStream_t stream,
                   Copy copy         = nullptr,
                   int64_t copy_cols = 0)
{
  constexpr uint32_t kThreads = 256;
  const auto blocks           = raft::ceildiv<int64_t>(n * kWarpSize, kThreads);
  squared_norms_kernel<<<blocks, kThreads, 0, stream>>>(
    rows, n, stride, cols, out, copy, copy_cols);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/** Each row's squared distance to its assigned centroid; optionally pack its filter dims as FP16.
 */
template <bool Copy, typename T, typename C>
RAFT_KERNEL distance_to_assigned_and_copy_kernel(const T* __restrict__ rows,
                                                 int64_t n,
                                                 const C* __restrict__ centroids,
                                                 int64_t dim,
                                                 const uint32_t* __restrict__ labels,
                                                 float* __restrict__ dist,
                                                 half* __restrict__ filter_rows)
{
  const int64_t row = (blockIdx.x * int64_t{blockDim.x} + threadIdx.x) / kWarpSize;
  if (row >= n) { return; }
  const T* a     = rows + row * dim;
  const C* b     = centroids + static_cast<int64_t>(labels[row]) * dim;
  float distance = 0.f;
  for (int64_t j = threadIdx.x % kWarpSize; j < dim; j += kWarpSize) {
    const float value = static_cast<float>(a[j]);
    const float diff  = value - static_cast<float>(b[j]);
    distance += diff * diff;
    if constexpr (Copy) {
      if (j < kPartialDim) { filter_rows[row * kPartialDim + j] = static_cast<half>(value); }
    }
  }
  distance = raft::warpReduce(distance);
  if (threadIdx.x % kWarpSize == 0) { dist[row] = distance; }
}

template <bool Copy = false, typename T, typename C>
void distance_to_assigned(const T* rows,
                          int64_t n,
                          const C* centroids,
                          int64_t dim,
                          const uint32_t* labels,
                          float* dist,
                          cudaStream_t stream,
                          half* filter_rows = nullptr)
{
  constexpr uint32_t kThreads = 256;
  const auto blocks           = raft::ceildiv<int64_t>(n * kWarpSize, kThreads);
  distance_to_assigned_and_copy_kernel<Copy>
    <<<blocks, kThreads, 0, stream>>>(rows, n, centroids, dim, labels, dist, filter_rows);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * The first filter's survivor lists of one row batch x centroid batch (see launch_fused_filter):
 * row r's survivors at packed[r * row_stride], count[r] of them. The pruning kernel compacts them
 * in place.
 */
struct first_filter_lists {
  int64_t* packed;
  const int32_t* count;
  uint32_t row_stride;
};

/**
 * One dim tile of a vector (16-byte aligned) into the warp's registers, in 16-byte loads:
 * lane l holds dims (32 j + l) * V + [0, V) at tile[j * V + [0, V)], V = 16 / sizeof(T).
 */
template <uint32_t Dims, typename T>
__device__ __forceinline__ void load_tile(const T* __restrict__ values, T (&tile)[Dims / kWarpSize])
{
  constexpr uint32_t kVec = 16 / sizeof(T);
  static_assert(Dims % (kWarpSize * kVec) == 0, "a dim tile is whole 16-byte loads per lane");
  const uint32_t lane = threadIdx.x % kWarpSize;
#pragma unroll
  for (uint32_t j = 0; j < Dims / (kWarpSize * kVec); j++) {
    raft::TxN_t<T, kVec> v;
    v.load(values, (j * kWarpSize + lane) * kVec);
#pragma unroll
    for (uint32_t k = 0; k < kVec; k++) {
      tile[j * kVec + k] = v.val.data[k];
    }
  }
}

/**
 * Distances over one dim tile between the row (`row_tile`, laid out by load_tile) and the N
 * survivors at window positions [first, first + N), their ids held one per lane in `ids`; the lane
 * holding each survivor adds its distance to `distance`. The N centroids' loads are issued back to
 * back.
 */
template <uint32_t Dims, uint32_t N, typename T>
__device__ __forceinline__ void add_tile_distances(const T (&row_tile)[Dims / kWarpSize],
                                                   const T* __restrict__ centroids,
                                                   uint32_t stride,
                                                   uint32_t ids,
                                                   uint32_t first,
                                                   float& distance)
{
  const uint32_t lane = threadIdx.x % kWarpSize;
  float partial[N];
#pragma unroll
  for (uint32_t i = 0; i < N; i++) {
    const T* centroid =
      centroids + static_cast<int64_t>(__shfl_sync(kFullMask, ids, first + i)) * stride;
    T centroid_tile[Dims / kWarpSize];
    load_tile<Dims>(centroid, centroid_tile);
    partial[i] = 0;
#pragma unroll
    for (uint32_t j = 0; j < Dims / kWarpSize; j++) {
      const float diff = static_cast<float>(row_tile[j]) - static_cast<float>(centroid_tile[j]);
      partial[i]       = fmaf(diff, diff, partial[i]);
    }
  }
  // Lane i ends with survivor first + i's sum; move it to that survivor's lane
  using batched_reduce = cub::WarpReduceBatched<float, N>;
  typename batched_reduce::TempStorage temp_storage;
  const float sum = __shfl_sync(kFullMask, batched_reduce{temp_storage}.Sum(partial), lane - first);
  if (lane >= first && lane < first + N) { distance += sum; }
}

/** Calls f(integral_constant<N>), then with N / 2, ..., 1. */
template <uint32_t N, typename F>
__device__ __forceinline__ void for_each_halving(F&& f)
{
  f(std::integral_constant<uint32_t, N>{});
  if constexpr (N > 1) { for_each_halving<N / 2>(f); }
}

__device__ __forceinline__ int64_t pack_survivor(uint32_t id, float distance)
{
  return (static_cast<int64_t>(id) << kSurvivorIdShift) |
         static_cast<uint32_t>(__float_as_int(distance));
}

/**
 * A warp per block and row, against one centroid batch: the row's first-filter survivors (partial
 * distances over the first kPartialDim dims), the tail dims in dim tiles with a re-filter after
 * each but the last, then the argmin, kept if closer than the row's current centroid. Block b
 * handles row row_order[b] when given (heaviest rows first), and each row's first-filter survivors
 * are added to `survivors`.
 *
 * The survivor list stays in the row's region of `lists`, but for its first kWarpSize entries,
 * copied to shared memory; the warp reads it a window of kWarpSize entries at a time, lane l
 * holding entry l, and each re-filter compacts it in place.
 */
template <typename T>
RAFT_KERNEL prune_kernel(const T* __restrict__ rows,
                         const uint32_t* __restrict__ row_order,
                         const T* __restrict__ centroids,
                         uint32_t centroid_offset,
                         uint32_t dim,
                         first_filter_lists lists,
                         uint32_t* __restrict__ labels,
                         float* __restrict__ dist,
                         uint32_t* __restrict__ survivors)
{
  __shared__ int64_t head[kWarpSize];
  const uint32_t lane        = threadIdx.x;
  const uint32_t lanes_below = (1u << lane) - 1;
  const int64_t row          = row_order != nullptr ? row_order[blockIdx.x] : blockIdx.x;
  uint32_t n_left            = static_cast<uint32_t>(lists.count[row]);
  // Nothing survived: nothing else to read or write
  if (n_left == 0) { return; }
  const T* row_values = rows + row * dim;
  uint32_t best_index = labels[row];
  // The row's distance to its current centroid until the last tile's argmin
  float best = dist[row];

  int64_t* list = lists.packed + row * lists.row_stride;
  if (lane < n_left) { head[lane] = list[lane]; }
  __syncwarp();
  if (lane == 0) { survivors[row] += n_left; }
  // Only the current centroid survived
  if (n_left == 1 &&
      centroid_offset + static_cast<uint32_t>(head[0] >> kSurvivorIdShift) == best_index) {
    return;
  }

  // One pass per remaining dim tile; the last one ends with the argmin
  for (uint32_t d = kPartialDim; n_left; d += kDimTileSize) {
    const bool last = d + kDimTileSize >= dim;
    // The re-filter's ratio
    const float ratio = last ? 0.0f : adsampling_ratio(d + kDimTileSize, dim, kAdsamplingEps0);
    T row_tile[kDimTileSize / kWarpSize];
    load_tile<kDimTileSize>(row_values + d, row_tile);
    uint32_t kept = 0;
    for (uint32_t base = 0; base < n_left; base += kWarpSize) {
      const uint32_t w    = min(kWarpSize, n_left - base);
      const int64_t entry = lane >= w ? 0 : base == 0 ? head[lane] : list[base + lane];
      const auto ids      = static_cast<uint32_t>(entry >> kSurvivorIdShift);
      float distance      = __int_as_float(static_cast<int32_t>(entry));
      // Survivor tiles of kSurvivorTileSize, then halving sizes down to 1 for the ones left
      uint32_t i = 0;
      for_each_halving<kSurvivorTileSize>([&](auto tile) {
        constexpr uint32_t kTile = decltype(tile)::value;
        for (; i + kTile <= w; i += kTile) {
          add_tile_distances<kDimTileSize, kTile>(row_tile, centroids + d, dim, ids, i, distance);
        }
      });
      if (!last) {
        // The kept survivors go below this window's end, never into a later window
        const bool keep       = lane < w && distance < best * ratio;
        const uint32_t ballot = __ballot_sync(kFullMask, keep);
        if (keep) {
          const uint32_t slot                          = kept + __popc(ballot & lanes_below);
          (slot < kWarpSize ? head[slot] : list[slot]) = pack_survivor(ids, distance);
        }
        kept += __popc(ballot);
      } else if (lane < w && distance < best) {  // after the last tile the distances are full
        best       = distance;
        best_index = centroid_offset + ids;
      }
    }
    __syncwarp();  // the compacted list is visible to the warp
    n_left = last ? 0 : kept;
  }

  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    const uint32_t other_index = __shfl_down_sync(kFullMask, best_index, offset);
    const float other          = __shfl_down_sync(kFullMask, best, offset);
    if (other < best) {
      best       = other;
      best_index = other_index;
    }
  }
  if (lane == 0) {
    labels[row] = best_index;
    dist[row]   = best;
  }
}

/**
 * The pruning kernel for n_rows rows against one centroid batch (its first centroid
 * centroid_offset), after the fused first filter wrote `lists`. `labels` and `dist` hold each
 * row's current centroid and distance, and get the closer survivor if there is one; `survivors`
 * gets each row's first-filter survivors added. row_order (optional): the rows' order.
 */
template <typename T>
void prune(const T* rows,
           int64_t n_rows,
           const uint32_t* row_order,
           const T* centroids,
           uint32_t centroid_offset,
           int64_t dim,
           first_filter_lists lists,
           uint32_t* labels,
           float* dist,
           uint32_t* survivors,
           cudaStream_t stream)
{
  RAFT_EXPECTS(dim > kPartialDim && (dim - kPartialDim) % kDimTileSize == 0,
               "Pruned 1-NN needs a positive tail after %u dims in tiles of %u",
               kPartialDim,
               kDimTileSize);
  prune_kernel<<<n_rows, kWarpSize, 0, stream>>>(rows,
                                                 row_order,
                                                 centroids,
                                                 centroid_offset,
                                                 static_cast<uint32_t>(dim),
                                                 lists,
                                                 labels,
                                                 dist,
                                                 survivors);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace cuvs::distance::detail::pruned_nn
