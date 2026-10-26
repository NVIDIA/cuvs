/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/reduction.cuh>

#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace cuvs::distance::detail::pruned_nn {

// Dims between the survivors' re-filters, and d' of the first filter
constexpr uint32_t kDimTileSize = 256;
// Centroids per batch: a row's survivor list of one batch lives in the pruning kernel's shared
// memory
constexpr uint32_t kCentroidBatchSize = 1024;
// Rows per batch: bounds the first filter's survivor lists (row batch x centroid batch slots)
constexpr int64_t kRowBatchSize = 128 * 1024;
// Survivors whose loads are batched over a dim tile
constexpr uint32_t kSurvivorTileSize = 16;
// fused_filter_kernel.py's ID_SHIFT
constexpr int kSurvivorIdShift = 32;

constexpr uint32_t kWarpSize = 32;
constexpr uint32_t kFullMask = 0xffffffff;

using survivor_id_t = std::conditional_t<(kCentroidBatchSize <= 65536), uint16_t, uint32_t>;

/**
 * ADSampling's pruning ratios: ratios[i] = (i / d) * (1 + eps0 / sqrt(i))^2 for i dims seen,
 * i = 0..d (1 at both ends). A partial distance over the first i dims is compared against
 * ratios[i] times the row's current best distance. The ratios assume every dim carries the same
 * share of a distance, which ADSampling ensures with a random rotation of the data; without one
 * they are a heuristic.
 */
inline auto adsampling_ratios(uint32_t dim, float eps0 = 1.5f) -> std::vector<float>
{
  std::vector<float> ratios(dim + 1);
  for (uint32_t i = 0; i <= dim; i++) {
    if (i == 0 || i == dim) {
      ratios[i] = 1.0f;
      continue;
    }
    const double f = 1.0 + eps0 / std::sqrt(static_cast<double>(i));
    ratios[i]      = static_cast<float>(static_cast<double>(i) / dim * f * f);
  }
  return ratios;
}

/**
 * Sums N independent per-lane values across the warp (valid in the first lane), the N shuffles of
 * each step issued together so they overlap.
 */
template <uint32_t N, typename T>
__device__ __forceinline__ void warp_reduce_sum_n(T* values)
{
  T shifted[N];
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
#pragma unroll
    for (uint32_t i = 0; i < N; i++) {
      shifted[i] = __shfl_down_sync(kFullMask, values[i], offset);
    }
    asm volatile("" ::: "memory");
#pragma unroll
    for (uint32_t i = 0; i < N; i++) {
      values[i] += shifted[i];
    }
  }
}

/** Squared norm of each of n rows (row stride `stride`) over their first `cols` dims; a warp per
 * row. */
template <typename T>
RAFT_KERNEL squared_norms_kernel(
  const T* __restrict__ rows, int64_t n, int64_t stride, int64_t cols, float* __restrict__ out)
{
  const int64_t row = (blockIdx.x * int64_t{blockDim.x} + threadIdx.x) / kWarpSize;
  if (row >= n) { return; }
  float s = 0.f;
  for (int64_t j = threadIdx.x % kWarpSize; j < cols; j += kWarpSize) {
    const float v = static_cast<float>(rows[row * stride + j]);
    s += v * v;
  }
  s = raft::warpReduce(s);
  if (threadIdx.x % kWarpSize == 0) { out[row] = s; }
}

template <typename T>
void squared_norms(
  const T* rows, int64_t n, int64_t stride, int64_t cols, float* out, cudaStream_t stream)
{
  constexpr uint32_t kThreads = 256;
  const auto blocks           = raft::ceildiv<int64_t>(n * kWarpSize, kThreads);
  squared_norms_kernel<<<blocks, kThreads, 0, stream>>>(rows, n, stride, cols, out);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/** Each row's squared distance to its assigned centroid over all dims; a warp per row. */
template <typename T, typename C>
RAFT_KERNEL distance_to_assigned_kernel(const T* __restrict__ rows,
                                        int64_t n,
                                        const C* __restrict__ centroids,
                                        int64_t dim,
                                        const uint32_t* __restrict__ labels,
                                        float* __restrict__ dist)
{
  const int64_t row = (blockIdx.x * int64_t{blockDim.x} + threadIdx.x) / kWarpSize;
  if (row >= n) { return; }
  const T* a     = rows + row * dim;
  const C* b     = centroids + static_cast<int64_t>(labels[row]) * dim;
  float distance = 0.f;
  for (int64_t j = threadIdx.x % kWarpSize; j < dim; j += kWarpSize) {
    const float diff = static_cast<float>(a[j]) - static_cast<float>(b[j]);
    distance += diff * diff;
  }
  distance = raft::warpReduce(distance);
  if (threadIdx.x % kWarpSize == 0) { dist[row] = distance; }
}

template <typename T, typename C>
void distance_to_assigned(const T* rows,
                          int64_t n,
                          const C* centroids,
                          int64_t dim,
                          const uint32_t* labels,
                          float* dist,
                          cudaStream_t stream)
{
  constexpr uint32_t kThreads = 256;
  const auto blocks           = raft::ceildiv<int64_t>(n * kWarpSize, kThreads);
  distance_to_assigned_kernel<<<blocks, kThreads, 0, stream>>>(
    rows, n, centroids, dim, labels, dist);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * The first filter's survivor lists of one row batch x centroid batch (see launch_fused_filter):
 * per (row, centroid tile) a segment of tile_n slots in `packed`, its count in `seg_count`.
 */
struct first_filter_lists {
  const int64_t* packed;
  const int32_t* seg_count;
  uint32_t n_col_tiles;
  uint32_t tile_n;
};

/**
 * Distances over one dim tile between the row (`row_tile`, lane l holding dims l + 32 j) and N of
 * its survivors, added to `distances` by lane 0. The N centroids' loads are issued back to back.
 */
template <uint32_t Dims, uint32_t N, typename T>
__device__ __forceinline__ void add_tile_distances(const float (&row_tile)[Dims / kWarpSize],
                                                   const T* __restrict__ centroids,
                                                   uint32_t stride,
                                                   const survivor_id_t* __restrict__ ids,
                                                   float* __restrict__ distances)
{
  const uint32_t lane = threadIdx.x % kWarpSize;
  float partial[N];
  for (uint32_t i = 0; i < N; i++) {
    const T* centroid = centroids + static_cast<int64_t>(ids[i]) * stride;
    partial[i]        = 0;
#pragma unroll
    for (uint32_t j = 0; j < Dims / kWarpSize; j++) {
      const float diff = row_tile[j] - static_cast<float>(centroid[lane + j * kWarpSize]);
      partial[i]       = fmaf(diff, diff, partial[i]);
    }
  }
  warp_reduce_sum_n<N>(partial);
  if (lane == 0) {
    for (uint32_t i = 0; i < N; i++) {
      distances[i] += partial[i];
    }
  }
}

/** Calls f(integral_constant<N>), then with N / 2, ..., 1. */
template <uint32_t N, typename F>
__device__ __forceinline__ void for_each_halving(F&& f)
{
  f(std::integral_constant<uint32_t, N>{});
  if constexpr (N > 1) { for_each_halving<N / 2>(f); }
}

/** A row's first-filter survivors, in centroid order, into the warp's list; returns how many. */
__device__ inline uint32_t load_survivors(first_filter_lists lists,
                                          int64_t row,
                                          survivor_id_t* __restrict__ ids,
                                          float* __restrict__ distances)
{
  const uint32_t lane     = threadIdx.x % kWarpSize;
  const int64_t* segments = lists.packed + row * lists.n_col_tiles * lists.tile_n;
  const int32_t* counts   = lists.seg_count + row * lists.n_col_tiles;
  uint32_t n              = 0;
  for (uint32_t t = 0; t < lists.n_col_tiles; t++) {
    const auto count = static_cast<uint32_t>(counts[t]);
    for (uint32_t j = lane; j < count; j += kWarpSize) {
      const int64_t value = segments[t * lists.tile_n + j];
      ids[n + j]          = static_cast<survivor_id_t>(value >> kSurvivorIdShift);
      distances[n + j]    = __int_as_float(static_cast<int32_t>(value));
    }
    n += count;
  }
  __syncwarp();
  return n;
}

/** Keeps the n survivors under `threshold`, compacting the list in place; returns how many. */
__device__ inline uint32_t refilter(uint32_t n,
                                    float threshold,
                                    survivor_id_t* __restrict__ ids,
                                    float* __restrict__ distances)
{
  const uint32_t lane        = threadIdx.x % kWarpSize;
  const uint32_t lanes_below = (1u << lane) - 1;
  uint32_t kept              = 0;
  for (uint32_t base = 0; base < n; base += kWarpSize) {
    const uint32_t i = base + lane;
    uint32_t id      = 0;
    float distance   = 0.f;
    bool keep        = false;
    if (i < n) {
      id       = ids[i];
      distance = distances[i];
      keep     = distance < threshold;
    }
    const uint32_t ballot = __ballot_sync(kFullMask, keep);
    __syncwarp();  // all lanes have read their entry before the compaction writes
    if (keep) {
      const uint32_t slot = kept + __popc(ballot & lanes_below);
      ids[slot]           = static_cast<survivor_id_t>(id);
      distances[slot]     = distance;
    }
    kept += __popc(ballot);
  }
  return kept;
}

/**
 * A warp per block and row, against one centroid batch: the row's first-filter survivors (partial
 * distances over the first partial_dim dims), the tail dims in dim tiles with a re-filter after
 * each but the last, then the argmin, kept if closer than the row's current centroid. Block b
 * handles row row_order[b] when given (heaviest rows first), and each row's first-filter survivors
 * are added to `survivors`.
 */
template <typename T>
RAFT_KERNEL prune_kernel(const T* __restrict__ rows,
                         const uint32_t* __restrict__ row_order,
                         const T* __restrict__ centroids,
                         uint32_t centroid_offset,
                         uint32_t dim,
                         uint32_t partial_dim,
                         const float* __restrict__ ratios,
                         first_filter_lists lists,
                         uint32_t* __restrict__ labels,
                         float* __restrict__ dist,
                         uint32_t* __restrict__ survivors)
{
  __shared__ survivor_id_t ids[kCentroidBatchSize];
  __shared__ float distances[kCentroidBatchSize];
  const uint32_t lane = threadIdx.x;
  const int64_t row   = row_order != nullptr ? row_order[blockIdx.x] : blockIdx.x;
  const T* row_values = rows + row * dim;
  uint32_t best_index = labels[row];
  float best          = dist[row];
  const float bound   = best;

  uint32_t n_left = load_survivors(lists, row, ids, distances);
  if (lane == 0) { survivors[row] += n_left; }
  if (n_left == 0 || (n_left == 1 && centroid_offset + ids[0] == best_index)) { return; }

  for (uint32_t d = partial_dim; d < dim && n_left; d += kDimTileSize) {
    float row_tile[kDimTileSize / kWarpSize];
#pragma unroll
    for (uint32_t j = 0; j < kDimTileSize / kWarpSize; j++) {
      row_tile[j] = static_cast<float>(row_values[d + lane + j * kWarpSize]);
    }
    // Survivor tiles of kSurvivorTileSize, then halving sizes down to 1 for the ones left
    uint32_t i = 0;
    for_each_halving<kSurvivorTileSize>([&](auto tile) {
      constexpr uint32_t kTile = decltype(tile)::value;
      for (; i + kTile <= n_left; i += kTile) {
        add_tile_distances<kDimTileSize, kTile>(
          row_tile, centroids + d, dim, ids + i, distances + i);
      }
    });
    __syncwarp();  // lane 0's distance updates are visible to the warp
    // After the last tile the distances are complete: the argmin compares them with the best
    if (d + kDimTileSize < dim) {
      n_left = refilter(n_left, bound * ratios[d + kDimTileSize], ids, distances);
    }
  }

  for (uint32_t i = lane; i < n_left; i += kWarpSize) {
    if (distances[i] < best) {
      best       = distances[i];
      best_index = centroid_offset + ids[i];
    }
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
           uint32_t partial_dim,
           const float* ratios,
           first_filter_lists lists,
           uint32_t* labels,
           float* dist,
           uint32_t* survivors,
           cudaStream_t stream)
{
  RAFT_EXPECTS(partial_dim <= dim && (dim - partial_dim) % kDimTileSize == 0,
               "Pruned 1-NN needs dim - partial_dim to be a multiple of %u",
               kDimTileSize);
  prune_kernel<<<n_rows, kWarpSize, 0, stream>>>(rows,
                                                 row_order,
                                                 centroids,
                                                 centroid_offset,
                                                 static_cast<uint32_t>(dim),
                                                 partial_dim,
                                                 ratios,
                                                 lists,
                                                 labels,
                                                 dist,
                                                 survivors);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace cuvs::distance::detail::pruned_nn
