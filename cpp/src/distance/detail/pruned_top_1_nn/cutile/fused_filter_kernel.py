# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
"""The pruned 1-NN's first filter: the partial GEMM over the first d' dims fused with the filter.

Each CTA owns TM rows and loops over the centroid tiles (TN wide). For each it computes the partial
distance ||row||^2 + ||centroid||^2 - 2 row.centroid over the first d' dims (FP16 tensor cores) and
keeps the pairs under the row's threshold best[row] * ratio, where best is the row's
distance to its current centroid.

The output is a row-major bitmap. Bit b of word w marks centroid 32*w+b in the batch.
The pair builder enumerates these bits and recomputes partial distances for the pairs it enqueues.

d' is a runtime argument and must be a multiple of TK.
"""

import cuda.tile as ct

ConstInt = ct.Constant[int]


def kernel_symbol(data_type: str) -> str:
    return {"half": "pruned_1nn_fused_filter_h"}[data_type]


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
        bitmap,
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
        row_norm = ct.load(
            row_norms, index=(row_tile,), shape=(tm,), padding_mode=zero_pad
        )[:, None]
        winners = ct.load(best, index=(row_tile,), shape=(tm,), padding_mode=zero_pad)
        bits = ct.bitwise_rshift(winners, 32).astype(ct.uint32)
        threshold = ct.bitcast(bits, ct.float32)[:, None] * ratio
        shifts = ct.arange(32, dtype=ct.uint32)[None, None, :]
        for centroid_tile in range(n_col_tiles):
            acc = ct.full((tm, tn), 0, dtype=ct.float32)
            for kk in range(dp // tk):
                a = ct.load(
                    rows,
                    index=(row_tile, kk),
                    shape=(tm, tk),
                    padding_mode=zero_pad,
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
            # Expanded L2 can round below zero when its terms nearly cancel.
            dist = ct.where(dist <= 0.0, 0.0, dist)
            centroid_ids = (
                centroid_tile * tn + ct.arange(tn, dtype=ct.int32)[None, :]
            )
            keep = (dist < threshold) & (centroid_ids < n_centroids)
            # Each group of 32 predicates contributes distinct bits to one bitmap word.
            mask = ct.reshape(keep, (tm, tn // 32, 32)).astype(ct.uint32)
            words = ct.sum(ct.bitwise_lshift(mask, shifts), axis=2).astype(ct.uint32)
            ct.store(bitmap, index=(row_tile, centroid_tile), tile=words)

    return fused_filter
