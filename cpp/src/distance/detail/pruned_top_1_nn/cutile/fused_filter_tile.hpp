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
 * Row r's survivors go to `packed` at r * row_stride + 0 .. count[r] - 1, each
 * `centroid << 32 | float bits of the partial distance`, in centroid order.
 *
 * @param rows            [n_rows, dim] row-major, 16-byte aligned, 16-byte row pitch
 * @param centroids       [n_centroids, dim] same layout as rows
 * @param row_norms       [n_rows] squared norms over the first partial_dim dims
 * @param centroid_norms  [n_centroids] squared norms over the first partial_dim dims
 * @param best            [n_rows] each row's distance to its current centroid
 * @param ratio           the pruning bound's scale at partial_dim
 * @param partial_dim     a multiple of tile_k, at most dim
 * @param packed          [n_rows * row_stride] survivors
 * @param count           [n_rows] survivors per row
 * @param row_stride      each row's region in `packed`, at least n_centroids
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
                         int* count,
                         int row_stride,
                         cudaStream_t stream);

}  // namespace cuvs::distance::detail::pruned_nn
