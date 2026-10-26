/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "detail/pruned_top_1_nn/cutile/fused_filter_tile.hpp"
#include "detail/pruned_top_1_nn/prune.cuh"
#include "top_1_nn.cuh"

#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/integer_utils.hpp>

#include <cub/device/device_segmented_radix_sort.cuh>

#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>
#include <type_traits>
#include <vector>

namespace cuvs::distance {

/** The pruned 1-NN's bounds, and the order its pruning kernel takes the rows in. */
struct PrunedTop1nnParams {
  /** [k + 1] on device: ratios[i] scales a row's bound after i dims (see
   * detail::pruned_nn::adsampling_ratios). */
  const float* ratios = nullptr;
  /** ratios[detail::pruned_nn::kDimTileSize] on host: the first filter's. */
  float first_ratio = 1.0f;
  /** [m], optional: each row batch's rows in the order to prune them, heaviest first. */
  const uint32_t* row_order = nullptr;
  /** [m], optional: the order for the next call, from this call's first-filter survivors. May be
   * row_order. */
  uint32_t* next_row_order = nullptr;
};

namespace detail::pruned_nn {

constexpr int64_t kMaxStreams    = 2;
constexpr std::size_t kAlignment = 256;

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
  fused_filter_tile tile;
  std::size_t survivors, sorted_keys, local_ids, batch_offsets, sort_temp, sort_temp_bytes;
  std::vector<std::size_t> packed, seg_count;
  std::size_t total_bytes;

  workspace_layout(raft::resources const& handle, int64_t m)
  {
    const auto t = filter_tile();
    RAFT_EXPECTS(t.has_value(), "Pruned 1-NN needs the fused filter's cuTile kernel");
    tile                 = *t;
    n_row_batches        = raft::ceildiv(m, kRowBatchSize);
    const auto pool_size = raft::resource::is_stream_pool_initialized(handle)
                             ? static_cast<int64_t>(raft::resource::get_stream_pool_size(handle))
                             : 0;
    n_streams            = std::max<int64_t>(1, std::min({pool_size, kMaxStreams, n_row_batches}));

    std::size_t end = 0;
    auto take       = [&](std::size_t bytes) {
      const std::size_t offset = raft::round_up_safe(end, kAlignment);
      end                      = offset + bytes;
      return offset;
    };
    survivors     = take(m * sizeof(uint32_t));
    sorted_keys   = take(m * sizeof(uint32_t));
    local_ids     = take(m * sizeof(uint32_t));
    batch_offsets = take((n_row_batches + 1) * sizeof(int64_t));
    RAFT_CUDA_TRY(
      cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr,
                                                         sort_temp_bytes,
                                                         static_cast<const uint32_t*>(nullptr),
                                                         static_cast<uint32_t*>(nullptr),
                                                         static_cast<const uint32_t*>(nullptr),
                                                         static_cast<uint32_t*>(nullptr),
                                                         m,
                                                         n_row_batches,
                                                         static_cast<const int64_t*>(nullptr),
                                                         static_cast<const int64_t*>(nullptr)));
    sort_temp = take(sort_temp_bytes);
    // The first filter's survivor lists, one pair per stream, sized for a whole batch
    const int64_t batch_rows  = std::min(m, kRowBatchSize);
    const int64_t n_col_tiles = raft::ceildiv<int64_t>(kCentroidBatchSize, tile.tile_n);
    for (int64_t s = 0; s < n_streams; s++) {
      packed.push_back(take(batch_rows * n_col_tiles * tile.tile_n * sizeof(int64_t)));
      seg_count.push_back(take(batch_rows * n_col_tiles * sizeof(int32_t)));
    }
    total_bytes = end;
  }
};

/** Each row's id within its row batch, and the row batches' bounds (n_row_batches + 1). */
inline void row_batch_layout(raft::resources const& handle,
                             int64_t m,
                             raft::device_vector_view<uint32_t, int64_t> local_ids,
                             raft::device_vector_view<int64_t, int64_t> batch_offsets)
{
  raft::linalg::map_offset(handle, local_ids, [] __device__(int64_t i) {
    return static_cast<uint32_t>(i % kRowBatchSize);
  });
  raft::linalg::map_offset(handle, batch_offsets, [m] __device__(int64_t b) {
    return b * kRowBatchSize < m ? b * kRowBatchSize : m;
  });
}

/**
 * Each row batch's local row ids ordered by survivor count, descending, into `row_order`.
 * Approximate: one radix pass over the counts' top 8 significant bits (the counts are at most n),
 * rows of a bucket keeping their order.
 */
inline void order_rows_by_survivors(raft::resources const& handle,
                                    const workspace_layout& layout,
                                    char* workspace,
                                    int64_t m,
                                    int64_t n,
                                    uint32_t* row_order)
{
  auto* local_ids     = reinterpret_cast<uint32_t*>(workspace + layout.local_ids);
  auto* batch_offsets = reinterpret_cast<int64_t*>(workspace + layout.batch_offsets);
  row_batch_layout(handle,
                   m,
                   raft::make_device_vector_view(local_ids, m),
                   raft::make_device_vector_view(batch_offsets, layout.n_row_batches + 1));
  int end_bit = 1;
  while (end_bit < 32 && (int64_t{1} << end_bit) <= n) {
    end_bit++;
  }
  const int begin_bit   = std::max(0, end_bit - 8);
  std::size_t temp_size = layout.sort_temp_bytes;
  RAFT_CUDA_TRY(cub::DeviceSegmentedRadixSort::SortPairsDescending(
    workspace + layout.sort_temp,
    temp_size,
    reinterpret_cast<const uint32_t*>(workspace + layout.survivors),
    reinterpret_cast<uint32_t*>(workspace + layout.sorted_keys),
    local_ids,
    row_order,
    m,
    layout.n_row_batches,
    batch_offsets,
    batch_offsets + 1,
    begin_bit,
    end_bit,
    raft::resource::get_cuda_stream(handle).get()));
}

}  // namespace detail::pruned_nn

/**
 * Whether pruned_top_1_nn runs for k features on this build and device: it needs the fused
 * filter's cuTile kernel, and k a positive multiple of detail::pruned_nn::kDimTileSize.
 */
inline bool is_pruned_top_1_nn_available(int64_t k)
{
  return k > 0 && k % detail::pruned_nn::kDimTileSize == 0 &&
         detail::pruned_nn::filter_tile().has_value();
}

/** The workspace pruned_top_1_nn needs for m rows on `handle` (the same handle as the call). */
inline std::size_t pruned_top_1_nn_workspace_size(raft::resources const& handle, int64_t m)
{
  return detail::pruned_nn::workspace_layout(handle, m).total_bytes;
}

/**
 * Updates each row of x to its nearest row of y in L2, starting from its current one. Not exact:
 * from the distance to the current neighbor (the bound), a fused partial GEMM over the first
 * detail::pruned_nn::kDimTileSize dims (FP16) keeps the candidates under the bound times
 * params.first_ratio; the pruning kernel adds the candidates' tail dims a tile at a time, keeping
 * those under the bound times params.ratios[dims so far], and takes the closest left if it beats
 * the current neighbor. A neighbor pruned away early can be missed.
 *
 * Rows go in batches of kRowBatchSize against batches of kCentroidBatchSize rows of y. The row
 * batches round-robin over up to two streams of the handle's stream pool (the main stream without
 * one), which wait on and are synchronized with the main stream.
 *
 * @param output      in: each row's current neighbor (nearest_idx); out: its nearest neighbor and
 *                    squared distance
 * @param x           [m, k] row-major
 * @param y           [n, k] row-major
 * @param x_fp16      [m, kDimTileSize]: x's first dims in FP16, for float; ignored for half
 * @param y_fp16      [n, kDimTileSize]: y's first dims in FP16, for float; ignored for half
 * @param xn          [m] squared norms of x over the first kDimTileSize dims
 * @param yn          [n] squared norms of y over the first kDimTileSize dims
 * @param workspace   pruned_top_1_nn_workspace_size(handle, m) bytes, 256-byte aligned
 */
template <typename DataT>
void pruned_top_1_nn(raft::resources const& handle,
                     Top1nnOutput<uint32_t, float> output,
                     const DataT* x,
                     const DataT* y,
                     const half* x_fp16,
                     const half* y_fp16,
                     const float* xn,
                     const float* yn,
                     int64_t m,
                     int64_t n,
                     int64_t k,
                     const PrunedTop1nnParams& params,
                     void* workspace,
                     std::size_t workspace_bytes)
{
  static_assert(std::is_same_v<DataT, float> || std::is_same_v<DataT, half>,
                "Pruned 1-NN supports float and half");
#if CUVS_CUTILE_ENABLED
  using namespace detail::pruned_nn;
  constexpr bool kHalf = std::is_same_v<DataT, half>;
  RAFT_EXPECTS(is_pruned_top_1_nn_available(k),
               "Pruned 1-NN needs the fused filter's cuTile kernel and k a multiple of %u",
               kDimTileSize);
  RAFT_EXPECTS(m > 0 && n > 0, "Pruned 1-NN needs rows in x and y");
  RAFT_EXPECTS(params.ratios != nullptr, "Pruned 1-NN needs the pruning ratios");
  RAFT_EXPECTS(kHalf || (x_fp16 != nullptr && y_fp16 != nullptr),
               "Pruned 1-NN on float needs FP16 copies of x's and y's first dims");
  const workspace_layout layout(handle, m);
  RAFT_EXPECTS(workspace_bytes >= layout.total_bytes, "Pruned 1-NN's workspace is too small");
  auto* ws = static_cast<char*>(workspace);

  // The filter reads FP16 rows: half data whole, float data's FP16 copies of its first dims
  const int64_t filter_dim = kHalf ? k : kDimTileSize;
  const half* x_filter     = nullptr;
  const half* y_filter     = nullptr;
  if constexpr (kHalf) {
    x_filter = x;
    y_filter = y;
  } else {
    x_filter = x_fp16;
    y_filter = y_fp16;
  }
  auto main_stream = raft::resource::get_cuda_stream(handle).get();
  auto* survivors  = reinterpret_cast<uint32_t*>(ws + layout.survivors);
  RAFT_CUDA_TRY(cudaMemsetAsync(survivors, 0, m * sizeof(uint32_t), main_stream));
  const bool pooled = layout.n_streams > 1;
  if (pooled) { raft::resource::wait_stream_pool_on_stream(handle); }

  for (int64_t b = 0; b < layout.n_row_batches; b++) {
    const int64_t i      = b * kRowBatchSize;
    const int64_t n_rows = std::min(kRowBatchSize, m - i);
    const int64_t s      = b % layout.n_streams;
    cudaStream_t stream =
      pooled ? raft::resource::get_stream_from_stream_pool(handle, s).get() : main_stream;
    const DataT* rows     = x + i * k;
    uint32_t* row_labels  = output.nearest_idx + i;
    float* row_dist       = output.nearest_dist + i;
    const uint32_t* order = params.row_order != nullptr ? params.row_order + i : nullptr;
    auto* packed          = reinterpret_cast<int64_t*>(ws + layout.packed[s]);
    auto* seg_count       = reinterpret_cast<int32_t*>(ws + layout.seg_count[s]);
    // Each row's bound: its distance to its current neighbor
    distance_to_assigned(rows, n_rows, y, k, row_labels, row_dist, stream);
    for (int64_t j = 0; j < n; j += kCentroidBatchSize) {
      const int64_t n_centroids = std::min<int64_t>(kCentroidBatchSize, n - j);
      launch_fused_filter(x_filter + i * filter_dim,
                          static_cast<int>(n_rows),
                          y_filter + j * filter_dim,
                          static_cast<int>(n_centroids),
                          static_cast<int>(filter_dim),
                          xn + i,
                          yn + j,
                          row_dist,
                          params.first_ratio,
                          static_cast<int>(kDimTileSize),
                          packed,
                          seg_count,
                          stream);
      prune(rows,
            n_rows,
            order,
            y + j * k,
            static_cast<uint32_t>(j),
            k,
            kDimTileSize,
            params.ratios,
            first_filter_lists{
              packed,
              seg_count,
              static_cast<uint32_t>(raft::ceildiv<int64_t>(n_centroids, layout.tile.tile_n)),
              static_cast<uint32_t>(layout.tile.tile_n)},
            row_labels,
            row_dist,
            survivors + i,
            stream);
    }
  }
  if (pooled) {
    std::vector<std::size_t> used(layout.n_streams);
    std::iota(used.begin(), used.end(), std::size_t{0});
    raft::resource::sync_stream_pool(handle, used);
  }
  if (params.next_row_order != nullptr) {
    order_rows_by_survivors(handle, layout, ws, m, n, params.next_row_order);
  }
#else
  RAFT_FAIL("Pruned 1-NN needs a build with cuTile");
#endif
}

}  // namespace cuvs::distance
