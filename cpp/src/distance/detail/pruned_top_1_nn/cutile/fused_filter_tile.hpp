/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>
#include <cuvs/detail/jit_lto/cutile_compat.hpp>

#include <raft/core/resources.hpp>

#include <cuda_fp16.h>

#include <cstdint>
#include <optional>

namespace cuvs::distance::detail::pruned_nn {

struct fused_filter_tile {
  int tile_m;
  int tile_n;
  int tile_k;
};

/** The tile of the fused first-filter cubin for this device, or nullopt when there is none. */
auto fused_filter_tile_config() -> std::optional<fused_filter_tile>;

/**
 * The pruned 1-NN's first filter: the partial distances of rows to centroids over the first
 * `partial_dim` dims, keeping the pairs below `best[row] * ratio`. FP16 only: the caller filters
 * float data on half copies of its first dims.
 *
 * Set one bit per surviving centroid in the batch, in row-major order. Words past the last
 * centroid tile are left unwritten.
 *
 * @param handle          RAFT resources providing the stream and dry-run state
 * @param rows            [n_rows, dim] row-major, 16-byte aligned, 16-byte row pitch
 * @param centroids       [n_centroids, dim] same layout as rows
 * @param row_norms       [n_rows] squared norms over the first partial_dim dims
 * @param centroid_norms  [n_centroids] squared norms over the first partial_dim dims
 * @param best            [n_rows] packed winner containing each row's distance and centroid ID
 * @param ratio           the pruning bound's scale at partial_dim
 * @param partial_dim     a multiple of tile_k, at most dim
 * @param bitmap          [n_rows, bitmap_stride] survivor bitmap
 * @param bitmap_stride   per-row capacity in words, including padding to a centroid tile
 */
void launch_fused_filter(const raft::resources& handle,
                         const half* rows,
                         int n_rows,
                         const half* centroids,
                         int n_centroids,
                         int dim,
                         const float* row_norms,
                         const float* centroid_norms,
                         const long long* best,
                         float ratio,
                         int partial_dim,
                         uint32_t* bitmap,
                         int bitmap_stride);

}  // namespace cuvs::distance::detail::pruned_nn
