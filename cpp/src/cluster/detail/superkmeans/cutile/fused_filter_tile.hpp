/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>
#include <cuvs/detail/jit_lto/cutile_compat.hpp>

#include <cuda_fp16.h>

#include <cstdint>
#include <optional>

namespace cuvs::cluster::kmeans::detail::superkmeans {

struct fused_filter_tile {
  int tile_m;
  int tile_n;
  int tile_k;
};

/** The tile of the fused first-filter cubin for this device, or nullopt when there is none. */
auto fused_filter_tile_config() -> std::optional<fused_filter_tile>;

/**
 * SuperKMeans' first pruning filter: the partial distances of rows to centroids over the first
 * `partial_dim` dims, keeping the pairs below `best[row] * ratio`. FP16 only: the caller filters
 * float data on half copies of its first dims.
 *
 * Survivors go to `packed`, one segment of tile_n slots per (row, centroid tile) at
 * (row * n_col_tiles + tile) * tile_n, each `centroid << 32 | float bits of the partial distance`,
 * in centroid order; `seg_count` [n_rows, n_col_tiles] holds how many slots of each segment are
 * filled. n_col_tiles = ceil(n_centroids / tile_n).
 *
 * @param rows            [n_rows, dim] row-major, 16-byte aligned, 16-byte row pitch
 * @param centroids       [n_centroids, dim] same layout as rows
 * @param row_norms       [n_rows] squared norms over the first partial_dim dims
 * @param centroid_norms  [n_centroids] squared norms over the first partial_dim dims
 * @param best            [n_rows] each row's distance to its current centroid
 * @param ratio           the pruning bound's scale at partial_dim
 * @param partial_dim     a multiple of tile_k, at most dim
 * @param packed          [n_rows * n_col_tiles * tile_n] survivors
 * @param seg_count       [n_rows, n_col_tiles] survivors per segment
 * @param stream          the stream to launch on
 */
void launch_fused_filter(const half* rows,
                         int n_rows,
                         const half* centroids,
                         int n_centroids,
                         int dim,
                         const float* row_norms,
                         const float* centroid_norms,
                         const float* best,
                         float ratio,
                         int partial_dim,
                         int64_t* packed,
                         int* seg_count,
                         cudaStream_t stream);

}  // namespace cuvs::cluster::kmeans::detail::superkmeans
