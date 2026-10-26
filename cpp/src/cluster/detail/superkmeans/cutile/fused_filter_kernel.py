# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
"""SuperKMeans' first pruning filter: the partial GEMM over the first d' dims fused with the filter.

Each CTA owns TM rows and loops over the centroid tiles (TN wide). For each it computes the partial
distance ||row||^2 + ||centroid||^2 - 2 row.centroid over the first d' dims (FP16 tensor cores) and
keeps the pairs under the row's threshold best[row] * ratio, where best is the row's
distance to its current centroid.

Output, for n_rows rows against n_centroids centroids (n_col_tiles = ceil(n_centroids / TN)):

  packed     int64, one segment of TN slots per (row, centroid tile), at (row * n_col_tiles + tile) * TN:
               slots 0 .. count-1   that tile's survivors for that row, in centroid order
               slots count .. TN-1  not written
             a survivor: (centroid index) << 32 | (the partial distance's float32 bits)
  seg_count  int32 (n_rows, n_col_tiles): count, how many slots of each segment are filled

A segment holds a whole tile, so it cannot overflow, and a survivor's slot is just its rank among
the row's survivors in that tile: no prefix sum across rows.

d' is a runtime argument and must be a multiple of TK.
"""

import cuda.tile as ct

ConstInt = ct.Constant[int]

# A survivor packed in one int64: the centroid index in the high 32 bits, the partial distance's
# float32 bits in the low 32. One scatter per survivor is the costly part of the kernel; separate
# 32-bit scatters per field are much slower.
ID_SHIFT = 32
LOW_32_BITS = (1 << 32) - 1


def kernel_symbol(data_type: str) -> str:
    return {"half": "superkmeans_fused_filter_h"}[data_type]


def make_kernel(*, gpu_code: str, occupancy: int | None = None):
    kernel_options = {}
    if occupancy is not None:
        kernel_options["occupancy"] = ct.ByTarget(**{gpu_code: occupancy})

    @ct.kernel(**kernel_options)
    def fused_filter(
        rows,
        centroids,
        row_norms,
        centroid_norms,
        best,
        packed,
        seg_count,
        n_rows,
        n_centroids,
        ratio,
        dp,
        tm: ConstInt,
        tn: ConstInt,
        tk: ConstInt,
    ):
        row_tile = ct.bid(0)
        zero_pad = ct.PaddingMode.ZERO
        n_col_tiles = ct.num_tiles(centroids, axis=0, shape=(tn, tk))
        row_ids = row_tile * tm + ct.arange(tm, dtype=ct.int32)[:, None]
        row_norm = ct.load(
            row_norms, index=(row_tile,), shape=(tm,), padding_mode=zero_pad
        )[:, None]
        threshold = (
            ct.load(best, index=(row_tile,), shape=(tm,), padding_mode=zero_pad)[
                :, None
            ]
            * ratio
        )
        for centroid_tile in range(n_col_tiles):
            acc = ct.full((tm, tn), 0, dtype=ct.float32)
            for kk in range(dp // tk):
                a = ct.load(
                    rows, index=(row_tile, kk), shape=(tm, tk), padding_mode=zero_pad
                )
                b = ct.load(
                    centroids,
                    index=(kk, centroid_tile),
                    shape=(tk, tn),
                    padding_mode=zero_pad,
                    order=(1, 0),
                )
                acc = ct.mma(a, b, acc)
            centroid_norm = ct.load(
                centroid_norms,
                index=(centroid_tile,),
                shape=(tn,),
                padding_mode=zero_pad,
            )[None, :]
            dist = -2.0 * acc + row_norm + centroid_norm
            centroid_ids = centroid_tile * tn + ct.arange(tn, dtype=ct.int32)[None, :]
            keep = (dist < threshold) & (centroid_ids < n_centroids) & (row_ids < n_rows)
            # Slot: the start of the (row, tile) segment plus the survivor's rank in the row
            keep_i = keep.astype(ct.int32)
            per_row = ct.sum(keep_i, axis=1, keepdims=True)
            slot = (row_ids * n_col_tiles + centroid_tile) * tn + ct.cumsum(keep_i, 1) - 1
            # The distance's bits are masked: widening a negative int32 (a negative float, e.g. a
            # rounded-down 0) to int64 would sign-extend into the index
            index = ct.broadcast_to(centroid_ids, (tm, tn)).astype(ct.int64)
            dist_bits = ct.bitcast(dist, ct.int32).astype(ct.int64) & LOW_32_BITS
            value = ct.bitwise_lshift(index, ID_SHIFT) | dist_bits
            ct.scatter(packed, slot, value, mask=keep)
            ct.store(seg_count, index=(row_tile, centroid_tile), tile=per_row)

    return fused_filter
