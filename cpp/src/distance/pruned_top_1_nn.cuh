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
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/integer_utils.hpp>

#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <type_traits>

namespace cuvs::distance {

namespace detail::pruned_nn {

/** The fused first filter's tile on this device, or nullopt when the build or device has none. */
inline auto filter_tile() -> std::optional<fused_filter_tile>
{
#if CUVS_CUTILE_ENABLED
  return fused_filter_tile_config();
#else
  return std::nullopt;
#endif
}

/** Byte offsets for the buffers reused by each row batch on the handle's stream. */
struct workspace_layout {
  std::size_t bitmap = 0, filter_rows = 0, offsets = 0, winners = 0, scan = 0;
  std::size_t scan_bytes  = 0;
  std::size_t total_bytes = 0;

  workspace_layout(int64_t m, bool copy_rows, int64_t row_batch_size)
  {
    RAFT_EXPECTS(filter_tile().has_value(), "Pruned 1-NN needs the fused filter's cuTile kernel");
    RAFT_EXPECTS(row_batch_size > 0 &&
                   (row_batch_size >= m || row_batch_size % (kAlignment / sizeof(float)) == 0),
                 "Pruned 1-NN row batches must preserve 16-byte alignment of per-row buffers");
    if (m == 0) { return; }

    const auto take = [&](std::size_t bytes) {
      const std::size_t offset = raft::round_up_safe(total_bytes, kAlignment);
      total_bytes              = offset + bytes;
      return offset;
    };
    const int64_t batch_rows = std::min(m, row_batch_size);
    bitmap                   = take(batch_rows * kBitmapWordsPerRow * sizeof(uint32_t));
    offsets                  = take((batch_rows + 1) * sizeof(uint32_t));
    winners                  = take(batch_rows * sizeof(packed_winner));
    scan_bytes               = row_scan_workspace_size(batch_rows);
    scan                     = take(scan_bytes);
    if (copy_rows) { filter_rows = take(batch_rows * kFilterCopyDim * sizeof(half)); }
  }
};

}  // namespace detail::pruned_nn

/**
 * Whether pruned_top_1_nn runs for k features on this build and device: it needs the fused
 * filter's cuTile kernel, the first detail::pruned_nn::kPartialDim dims for filtering, and a
 * positive tail to prune on.
 */
inline bool is_pruned_top_1_nn_available(int64_t k)
{
  return k > detail::pruned_nn::kPartialDim && detail::pruned_nn::filter_tile().has_value();
}

/** The workspace pruned_top_1_nn needs for m rows at the given batch size. */
template <typename DataT>
inline std::size_t pruned_top_1_nn_workspace_size(int64_t m, int64_t row_batch_size)
{
  return detail::pruned_nn::workspace_layout(m, !std::is_same_v<DataT, half>, row_batch_size)
    .total_bytes;
}

/**
 * Updates each row of x to its nearest row of y in L2, starting from its current one: the pruned
 * assignment of SuperKMeans (Kuffo, Hepkema, Boncz, "A Super Fast K-means for Indexing Vector
 * Embeddings", https://arxiv.org/abs/2603.20009). With b a row's squared distance to its current
 * neighbor and ratio(i) = (i / k) * (1 + eps0 / sqrt(i))^2 (ADSampling's bound, see
 * detail::pruned_nn::adsampling_ratio):
 *
 *  1. a fused FP16 GEMM over the first PartialDim dims keeps the rows of y
 *     whose partial distance is under b * ratio(PartialDim);
 *  2. the pruning kernel reconstructs the survivors' prefix distances from the original data and
 *     adds the remaining dims a tile at a time, keeping those under b * ratio(dims so far);
 *  3. the closest survivor over all k dims replaces the current neighbor, and b, if closer.
 *
 * Not exact: a neighbor pruned away on a partial distance is missed.
 *
 * After each first filter, a scan of survivor counts locates each row in the logical list of
 * survivors. Pruning CTAs decode consecutive spans of that list from the bitmap and atomically
 * update row winners with completed distances.
 *
 * Rows go in batches of row_batch_size against batches of kCentroidBatchSize rows of y, each
 * centroid batch tightening b for the next. All batches reuse the workspace on the handle's stream.
 *
 * @param output      in: each row's current neighbor (nearest_idx); out: its nearest neighbor and
 *                    squared distance
 * @param x           [m, k] row-major
 * @param y           [n, k] row-major
 * @param y_fp16      [n, kFilterCopyDim]: y's first dims in FP16, for float; ignored for half
 * @param xn          [m] squared norms of the original x over the first PartialDim dims
 * @param yn          [n] squared norms of the FP16 filter centroids over the first PartialDim dims
 * @param row_batch_size rows per batch, matching workspace sizing
 * @param workspace   pruned_top_1_nn_workspace_size<DataT>(m, row_batch_size) bytes,
 *                    16-byte aligned
 */
template <typename DataT, uint32_t PartialDim = detail::pruned_nn::kPartialDim>
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
  static_assert(PartialDim > 0 && PartialDim <= kFilterCopyDim);
  RAFT_EXPECTS(is_pruned_top_1_nn_available(k),
               "Pruned 1-NN needs the fused filter's cuTile kernel and a positive tail after %u "
               "dims",
               kPartialDim);
  RAFT_EXPECTS(m > 0 && n > 0, "Pruned 1-NN needs rows in x and y");
  RAFT_EXPECTS(kHalf || y_fp16 != nullptr,
               "Pruned 1-NN on float needs an FP16 copy of y's first dims");
  RAFT_EXPECTS(xn != nullptr && yn != nullptr, "Pruned 1-NN needs row and centroid norms");
  RAFT_EXPECTS(std::min(m, row_batch_size) <= kMaxPairRows,
               "Pruned 1-NN row batch is too large for packed pair IDs");
  const workspace_layout layout(m, !kHalf, row_batch_size);
  RAFT_EXPECTS(workspace_bytes >= layout.total_bytes, "Pruned 1-NN's workspace is too small");
  auto* ws = static_cast<char*>(workspace);

  // The filter reads FP16 rows: half data whole, float data's FP16 copies of its first dims
  const int64_t filter_dim = kHalf ? k : kFilterCopyDim;
  const half* y_filter     = nullptr;
  if constexpr (kHalf) {
    y_filter = y;
  } else {
    y_filter = y_fp16;
  }
  auto* offsets = reinterpret_cast<uint32_t*>(ws + layout.offsets);
  auto* winners = reinterpret_cast<packed_winner*>(ws + layout.winners);
  auto* bitmap  = reinterpret_cast<uint32_t*>(ws + layout.bitmap);
  for (int64_t i = 0; i < m; i += row_batch_size) {
    const int64_t n_rows = std::min(row_batch_size, m - i);
    const DataT* rows    = x + i * k;
    uint32_t* row_labels = output.nearest_idx + i;
    float* row_dist      = output.nearest_dist + i;
    // Each row's bound: its distance to its current neighbor
    const half* x_filter;
    const float* row_norms = xn + i;
    if constexpr (kHalf) {
      x_filter = rows;
      distance_to_assigned(handle, rows, n_rows, y, k, row_labels, winners);
    } else {
      auto* batch_rows = reinterpret_cast<half*>(ws + layout.filter_rows);
      distance_to_assigned<true>(handle, rows, n_rows, y, k, row_labels, winners, batch_rows);
      x_filter = batch_rows;
    }
    for (int64_t j = 0; j < n; j += kCentroidBatchSize) {
      const int64_t n_centroids = std::min<int64_t>(kCentroidBatchSize, n - j);
      launch_fused_filter(handle,
                          x_filter,
                          static_cast<int>(n_rows),
                          y_filter + j * filter_dim,
                          static_cast<int>(n_centroids),
                          static_cast<int>(filter_dim),
                          row_norms,
                          yn + j,
                          winners,
                          adsampling_ratio(PartialDim, static_cast<uint32_t>(k), kAdsamplingEps0),
                          static_cast<int>(PartialDim),
                          bitmap,
                          static_cast<int>(kBitmapWordsPerRow));
      prune<PartialDim>(handle,
                        rows,
                        n_rows,
                        y + j * k,
                        static_cast<uint32_t>(j),
                        k,
                        bitmap,
                        static_cast<int>(raft::ceildiv(n_centroids, int64_t{kBitmapWordBits})),
                        offsets,
                        ws + layout.scan,
                        layout.scan_bytes,
                        row_dist,
                        winners);
    }
    {
      common::nvtx::range<common::nvtx::domain::cuvs> range("pruned_1nn::unpack_winners");
      constexpr int64_t kThreads = 256;
      raft::launch_kernel(handle,
                          raft::ceildiv(n_rows, kThreads),
                          kThreads,
                          unpack_winners_kernel,
                          n_rows,
                          winners,
                          row_labels,
                          row_dist);
    }
  }
#else
  RAFT_FAIL("Pruned 1-NN needs a build with cuTile");
#endif
}

}  // namespace cuvs::distance
