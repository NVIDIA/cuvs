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

#include "detail/pruned_top_1_nn/cutile/fused_filter_tile.hpp"
#include "detail/pruned_top_1_nn/prune.cuh"
#include "top_1_nn.cuh"

#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/core/resource/device_properties.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/integer_utils.hpp>

#include <cub/block/block_radix_sort.cuh>

#include <cuda_fp16.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <vector>

namespace cuvs::distance {

namespace detail::pruned_nn {

constexpr int64_t kMaxStreams    = 2;
constexpr std::size_t kAlignment = 16;

/** The fused first filter's tile on this device, or nullopt when the build or device has none. */
inline auto filter_tile() -> std::optional<fused_filter_tile>
{
#if CUVS_CUTILE_ENABLED
  return fused_filter_tile_config();
#else
  return std::nullopt;
#endif
}

/** The workspace's buffers, as byte offsets, for m rows on the handle's streams. */
struct workspace_layout {
  int64_t n_row_batches;
  int64_t n_streams;
  int64_t row_batch_size;
  int max_blocks_per_sm;
  std::vector<std::size_t> packed, count, filter_rows, row_order;
  std::size_t total_bytes;

  workspace_layout(raft::resources const& handle, int64_t m, bool copy_rows, int64_t batch_size)
    : row_batch_size(batch_size)
  {
    max_blocks_per_sm = raft::resource::get_device_properties(handle).maxBlocksPerMultiProcessor;
    RAFT_EXPECTS(filter_tile().has_value(), "Pruned 1-NN needs the fused filter's cuTile kernel");
    RAFT_EXPECTS(row_batch_size > 0 &&
                   (row_batch_size >= m || row_batch_size % (kAlignment / sizeof(float)) == 0),
                 "Pruned 1-NN row batches must preserve 16-byte alignment of per-row buffers");
    n_row_batches        = raft::ceildiv(m, row_batch_size);
    const auto pool_size = raft::resource::is_stream_pool_initialized(handle)
                             ? static_cast<int64_t>(raft::resource::get_stream_pool_size(handle))
                             : 0;
    n_streams            = std::min({1 + pool_size, kMaxStreams, n_row_batches});

    std::size_t end = 0;
    auto take       = [&](std::size_t bytes) {
      const std::size_t offset = raft::round_up_safe(end, kAlignment);
      end                      = offset + bytes;
      return offset;
    };
    const int64_t batch_rows = std::min(m, row_batch_size);
    // Streams reuse their buffers only after the previous row batch finishes.
    for (int64_t s = 0; s < n_streams; s++) {
      packed.push_back(take(batch_rows * kCentroidBatchSize * sizeof(int64_t)));
      count.push_back(take(batch_rows * sizeof(int32_t)));
      row_order.push_back(take(batch_rows * sizeof(uint32_t)));
      if (copy_rows) { filter_rows.push_back(take(batch_rows * kPartialDim * sizeof(half))); }
    }
    total_bytes = end;
  }
};

/**
 * Order the rows of one row batch for pruning against the current centroid batch. count[r] is
 * the number of centroids that survived the fused first filter for row r. Each sorting block
 * sorts (count, row id) pairs within one balanced, contiguous segment, largest counts first.
 * Only row ids are written; the counts and input data stay in their original positions.
 *
 * Interleave the sorted segments by rank to submit heavier rows from every segment early:
 *
 *   sorted segment 0: a(12), b(10), c(8)   // row ids, with counts in parentheses
 *   sorted segment 1: d(7),  e(4),  f(1)
 *   row_order:        a, d, b, e, c, f
 *
 * This approximates longest-processing-time-first (LPT) scheduling without a device-wide sort.
 * LPT applies across pruning blocks; similar-cost rows should share a block when multiple
 * row-processing warps are grouped together. Prune currently launches one warp (one row) per
 * block, with block b processing row_order[b], so there is no within-block row grouping.
 */
template <int BlockSize, int ItemsPerThread>
RAFT_KERNEL order_rows_by_survivors_kernel(const int32_t* __restrict__ count,
                                           uint32_t* __restrict__ row_order,
                                           int64_t n_rows)
{
  using Sort = cub::BlockRadixSort<uint32_t, BlockSize, ItemsPerThread, uint32_t>;
  __shared__ typename Sort::TempStorage scratch;
  const int64_t n_segments       = gridDim.x;
  const int64_t rows_per_segment = n_rows / n_segments;
  const int64_t extra_rows       = n_rows % n_segments;
  const int64_t base         = blockIdx.x * rows_per_segment + min(int64_t{blockIdx.x}, extra_rows);
  const int64_t segment_rows = rows_per_segment + (blockIdx.x < extra_rows);
  uint32_t keys[ItemsPerThread];
  uint32_t ids[ItemsPerThread];
#pragma unroll
  for (int j = 0; j < ItemsPerThread; j++) {
    const int64_t rank = threadIdx.x * ItemsPerThread + j;
    const int64_t row  = base + rank;
    ids[j]             = static_cast<uint32_t>(row);
    // Zero is reserved for padding, so it follows valid rows with no survivors.
    keys[j] = rank < segment_rows ? static_cast<uint32_t>(count[row]) + 1 : 0;
  }
  constexpr int kCountBits = std::bit_width(kCentroidBatchSize + 1);
  Sort(scratch).SortDescendingBlockedToStriped(keys, ids, 0, kCountBits);
#pragma unroll
  for (int j = 0; j < ItemsPerThread; j++) {
    const int64_t rank = threadIdx.x + j * BlockSize;
    // Longer segments come first, so the final partial rank leaves no holes.
    if (rank < segment_rows) { row_order[rank * n_segments + blockIdx.x] = ids[j]; }
  }
}

inline void order_rows_by_survivors(raft::resources const& handle,
                                    const int32_t* count,
                                    uint32_t* row_order,
                                    int64_t n_rows,
                                    int max_blocks_per_sm)
{
  constexpr int64_t kMaxSegmentRows = 256 * 32;
  // Use the resident-block limit as the interleaved rank width, adding segments only when
  // needed to keep each segment within one sorting block's capacity.
  const int64_t n_segments =
    std::min(n_rows, std::max(int64_t{max_blocks_per_sm}, raft::ceildiv(n_rows, kMaxSegmentRows)));
  const int64_t segment_rows = raft::ceildiv(n_rows, n_segments);
  // CUB requires both sizes at compile time. Keep a compact 2048-entry variant to avoid
  // excessive padding on short segments; the 8192-entry variant uses more threads to limit
  // per-thread storage for larger segments.
  if (segment_rows <= 128 * 16) {
    raft::launch_kernel(
      handle, n_segments, 128, order_rows_by_survivors_kernel<128, 16>, count, row_order, n_rows);
  } else {
    raft::launch_kernel(
      handle, n_segments, 256, order_rows_by_survivors_kernel<256, 32>, count, row_order, n_rows);
  }
}

}  // namespace detail::pruned_nn

/**
 * Whether pruned_top_1_nn runs for k features on this build and device: it needs the fused
 * filter's cuTile kernel, the first detail::pruned_nn::kPartialDim dims for filtering, and a
 * positive tail in whole detail::pruned_nn::kDimTileSize tiles to prune on.
 */
inline bool is_pruned_top_1_nn_available(int64_t k)
{
  return k > detail::pruned_nn::kPartialDim &&
         (k - detail::pruned_nn::kPartialDim) % detail::pruned_nn::kDimTileSize == 0 &&
         detail::pruned_nn::filter_tile().has_value();
}

/** The workspace pruned_top_1_nn needs for m rows on `handle` (the same handle as the call). */
template <typename DataT>
inline std::size_t pruned_top_1_nn_workspace_size(raft::resources const& handle,
                                                  int64_t m,
                                                  int64_t row_batch_size)
{
  return detail::pruned_nn::workspace_layout(
           handle, m, !std::is_same_v<DataT, half>, row_batch_size)
    .total_bytes;
}

/**
 * Updates each row of x to its nearest row of y in L2, starting from its current one: the pruned
 * assignment of SuperKMeans (Kuffo, Hepkema, Boncz, "A Super Fast K-means for Indexing Vector
 * Embeddings", https://arxiv.org/abs/2603.20009). With b a row's squared distance to its current
 * neighbor and ratio(i) = (i / k) * (1 + eps0 / sqrt(i))^2 (ADSampling's bound, see
 * detail::pruned_nn::adsampling_ratio):
 *
 *  1. a fused FP16 GEMM over the first detail::pruned_nn::kPartialDim dims keeps the rows of y
 *     whose partial distance is under b * ratio(kPartialDim);
 *  2. the pruning kernel adds the survivors' remaining dims a tile at a time, keeping those under
 *     b * ratio(dims so far);
 *  3. the closest survivor over all k dims replaces the current neighbor, and b, if closer.
 *
 * Not exact: a neighbor pruned away on a partial distance is missed.
 *
 * Each first-filter tile orders its rows by the newly produced survivor counts before pruning.
 * Counts are sorted within segments and their ranks interleaved, approximating heavy-first order.
 * The row order and survivor lists are scratch buffers private to each stream.
 *
 * Rows go in batches of row_batch_size against batches of kCentroidBatchSize rows of y, each
 * centroid batch tightening b for the next. The row batches round-robin over the main stream and,
 * with a stream pool on the handle, its first stream, which waits on and is synchronized with the
 * main stream.
 *
 * @param output      in: each row's current neighbor (nearest_idx); out: its nearest neighbor and
 *                    squared distance
 * @param x           [m, k] row-major
 * @param y           [n, k] row-major
 * @param y_fp16      [n, kPartialDim]: y's first dims in FP16, for float; ignored for half
 * @param xn          [m] squared norms of the original x over the first kPartialDim dims
 * @param yn          [n] squared norms of the FP16 filter centroids over the first kPartialDim dims
 * @param row_batch_size rows per batch, matching workspace sizing
 * @param workspace   pruned_top_1_nn_workspace_size<DataT>(handle, m, row_batch_size) bytes,
 *                    16-byte aligned
 */
template <typename DataT>
void pruned_top_1_nn(raft::resources const& handle,
                     Top1nnOutput<uint32_t, float> output,
                     const DataT* x,
                     const DataT* y,
                     const half* y_fp16,
                     const float* xn,
                     const float* yn,
                     int64_t m,
                     int64_t n,
                     int64_t k,
                     int64_t row_batch_size,
                     void* workspace,
                     std::size_t workspace_bytes)
{
  static_assert(std::is_same_v<DataT, float> || std::is_same_v<DataT, half>,
                "Pruned 1-NN supports float and half");
#if CUVS_CUTILE_ENABLED
  using namespace detail::pruned_nn;
  constexpr bool kHalf = std::is_same_v<DataT, half>;
  RAFT_EXPECTS(is_pruned_top_1_nn_available(k),
               "Pruned 1-NN needs the fused filter's cuTile kernel and a positive tail after %u "
               "dims in tiles of %u",
               kPartialDim,
               kDimTileSize);
  RAFT_EXPECTS(m > 0 && n > 0, "Pruned 1-NN needs rows in x and y");
  RAFT_EXPECTS(kHalf || y_fp16 != nullptr,
               "Pruned 1-NN on float needs an FP16 copy of y's first dims");
  RAFT_EXPECTS(xn != nullptr && yn != nullptr, "Pruned 1-NN needs row and centroid norms");
  const workspace_layout layout(handle, m, !kHalf, row_batch_size);
  RAFT_EXPECTS(workspace_bytes >= layout.total_bytes, "Pruned 1-NN's workspace is too small");
  auto* ws = static_cast<char*>(workspace);

  // The filter reads FP16 rows: half data whole, float data's FP16 copies of its first dims
  const int64_t filter_dim = kHalf ? k : kPartialDim;
  const half* y_filter     = nullptr;
  if constexpr (kHalf) {
    y_filter = y;
  } else {
    y_filter = y_fp16;
  }
  const bool pooled = layout.n_streams > 1;
  if (pooled) { raft::resource::wait_stream_pool_on_stream(handle); }

  std::vector<raft::resources> stream_handles(layout.n_streams, handle);
  for (int64_t s = 1; s < layout.n_streams; s++) {
    raft::resource::set_cuda_stream(stream_handles[s],
                                    raft::resource::get_stream_from_stream_pool(handle, s - 1));
  }

  for (int64_t b = 0; b < layout.n_row_batches; b++) {
    const int64_t i          = b * layout.row_batch_size;
    const int64_t n_rows     = std::min(layout.row_batch_size, m - i);
    const int64_t s          = b % layout.n_streams;
    const auto& batch_handle = stream_handles[s];
    auto stream              = raft::resource::get_cuda_stream(batch_handle).get();
    const DataT* rows        = x + i * k;
    uint32_t* row_labels     = output.nearest_idx + i;
    float* row_dist          = output.nearest_dist + i;
    auto* order              = reinterpret_cast<uint32_t*>(ws + layout.row_order[s]);
    auto* packed             = reinterpret_cast<int64_t*>(ws + layout.packed[s]);
    auto* count              = reinterpret_cast<int32_t*>(ws + layout.count[s]);
    // Each row's bound: its distance to its current neighbor
    const half* x_filter;
    const float* row_norms = xn + i;
    if constexpr (kHalf) {
      x_filter = rows;
      distance_to_assigned(batch_handle, rows, n_rows, y, k, row_labels, row_dist);
    } else {
      // Each stream reuses its filter inputs only after its previous row batch finishes.
      auto* batch_rows = reinterpret_cast<half*>(ws + layout.filter_rows[s]);
      distance_to_assigned<true>(
        batch_handle, rows, n_rows, y, k, row_labels, row_dist, batch_rows);
      x_filter = batch_rows;
    }
    for (int64_t j = 0; j < n; j += kCentroidBatchSize) {
      const int64_t n_centroids = std::min<int64_t>(kCentroidBatchSize, n - j);
      launch_fused_filter(x_filter,
                          static_cast<int>(n_rows),
                          y_filter + j * filter_dim,
                          static_cast<int>(n_centroids),
                          static_cast<int>(filter_dim),
                          row_norms,
                          yn + j,
                          row_dist,
                          adsampling_ratio(kPartialDim, static_cast<uint32_t>(k), kAdsamplingEps0),
                          static_cast<int>(kPartialDim),
                          packed,
                          count,
                          static_cast<int>(kCentroidBatchSize),
                          stream);
      order_rows_by_survivors(batch_handle, count, order, n_rows, layout.max_blocks_per_sm);
      prune(batch_handle,
            rows,
            n_rows,
            order,
            y + j * k,
            static_cast<uint32_t>(j),
            k,
            first_filter_lists{packed, count, kCentroidBatchSize},
            row_labels,
            row_dist);
    }
  }
  if (pooled) { raft::resource::sync_stream_pool(handle, std::vector<std::size_t>{0}); }
#else
  RAFT_FAIL("Pruned 1-NN needs a build with cuTile");
#endif
}

}  // namespace cuvs::distance
