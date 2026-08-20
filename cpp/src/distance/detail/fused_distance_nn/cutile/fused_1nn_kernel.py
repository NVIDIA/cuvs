# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cuTile fused GEMM + 1-NN kernel with runtime metric selection."""

from __future__ import annotations

import cuda.tile as ct

ConstInt = ct.Constant[int]

# Default tile geometry; overridden per export via make_kernel(..., tile_m, tile_n, tile_k).
DEFAULT_TILE_M = 128
DEFAULT_TILE_N = 128
DEFAULT_TILE_K = 32

METRICS = ("runtime",)
INDEX_TYPES = ("int32", "int64")
METRIC_L2_EXPANDED = 0
METRIC_COSINE_EXPANDED = 2
METRIC_INNER_PRODUCT = 6


def _idx_dtype(index_type: str):
    if index_type == "int32":
        return ct.int32
    if index_type == "int64":
        return ct.int64
    raise ValueError(f"Unsupported index_type {index_type!r}")


def make_kernel(
    data_type: str,
    metric: str,
    tile_m: int = DEFAULT_TILE_M,
    tile_n: int = DEFAULT_TILE_N,
    tile_k: int = DEFAULT_TILE_K,
    *,
    index_type: str = "int32",
    gpu_code: str = "sm_80",
    matrix_layout: str = "strict",
):
    """Build the flat-reduction runtime-metric cuTile kernel."""
    if data_type not in ("half", "float"):
        raise ValueError(f"Unsupported data_type {data_type!r}")
    if metric not in METRICS:
        raise ValueError(f"Unsupported metric {metric!r}")
    if index_type not in INDEX_TYPES:
        raise ValueError(f"Unsupported index_type {index_type!r}")
    if matrix_layout not in ("strict", "relaxed"):
        raise ValueError(f"Unsupported matrix_layout {matrix_layout!r}")

    acc_dtype = ct.float32
    idx_dtype = _idx_dtype(index_type)
    out_dist_dtype = ct.float16 if data_type == "half" else ct.float32
    core_shape = (tile_m, tile_n)
    best_shape = (tile_m, 1)

    @ct.kernel(occupancy=ct.ByTarget(sm_120=2))
    def fused_1nn_kernel(
        A,
        B,
        A_norm,
        B_norm,
        OutIdx,
        OutDist,
        M,
        N,
        K,
        apply_sqrt,
        store_idx,
        metric_code,
        tm: ConstInt,
        tn: ConstInt,
        tk: ConstInt,
    ):
        bidm = ct.bid(0)
        best_dist = ct.full(best_shape, 3.4e38, acc_dtype)
        best_idx = ct.zeros(best_shape, idx_dtype)
        num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))
        num_tiles_n = ct.num_tiles(B, axis=0, shape=(tn, tk))
        zero_pad = ct.PaddingMode.ZERO

        def reduce_scores(dists, indices):
            def red_op(a_score, a_idx, b_score, b_idx):
                cond = (a_score < b_score) | (
                    (a_score == b_score) & (a_idx < b_idx)
                )
                return (
                    ct.where(cond, a_score, b_score),
                    ct.where(cond, a_idx, b_idx),
                )

            return ct.reduce(
                (dists, indices),
                1,
                red_op,
                (3.4e38, -1),
                keepdims=True,
            )

        local_indices = ct.arange(tn, dtype=ct.int16)[None, :]
        for n in range(num_tiles_n):
            accumulator = ct.full((tm, tn), 0, dtype=acc_dtype)
            for k in range(num_tiles_k):
                dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype
                a = ct.load(
                    A, index=(bidm, k), shape=(tm, tk), padding_mode=zero_pad
                ).astype(dtype)
                b_T = ct.load(
                    B,
                    index=(k, n),
                    shape=(tk, tn),
                    padding_mode=zero_pad,
                    order=(1, 0),
                ).astype(dtype)
                accumulator = ct.mma(a, b_T, accumulator)

            if metric_code == METRIC_INNER_PRODUCT:
                score = -accumulator
            else:
                b_norm = ct.load(
                    B_norm, index=(n,), shape=(tn,), padding_mode=zero_pad
                )
                if metric_code == METRIC_L2_EXPANDED:
                    # L2 receives squared row norms; cosine receives L2 magnitudes.
                    # The A norm is constant across centroids. Reduce
                    # 0.5 * ||y||^2 - dot(x, y), then recover full L2 once.
                    score = (0.5 * b_norm)[None, :] - accumulator
                else:
                    # Defer the A-norm division until after selecting the
                    # winning centroid.
                    score = accumulator / (-b_norm)[None, :]

            if n == num_tiles_n - 1:
                col = ct.arange(tn, dtype=ct.int16)
                score = ct.where((n * tn + col)[None, :] < N, score, 3.4e38)

            curr_best, curr_idx = reduce_scores(
                score.reshape(core_shape), local_indices
            )
            update = curr_best < best_dist
            best_dist = ct.where(update, curr_best, best_dist)
            best_idx = ct.where(update, n * tn + curr_idx, best_idx)

        if metric_code == METRIC_INNER_PRODUCT:
            out_dist = -best_dist
        else:
            a_norm = ct.load(
                A_norm, index=(bidm,), shape=(tm,), padding_mode=zero_pad
            )[:, None]
            if metric_code == METRIC_L2_EXPANDED:
                out_dist = a_norm + 2.0 * best_dist
                out_dist = ct.where(
                    apply_sqrt != 0, ct.sqrt(out_dist), out_dist
                )
            else:
                out_dist = 1.0 + best_dist / a_norm

        if store_idx != 0:
            ct.store(OutIdx, index=(bidm,), tile=best_idx.reshape((tm,)))
        ct.store(
            OutDist,
            index=(bidm,),
            tile=out_dist.reshape((tm,)).astype(out_dist_dtype),
        )

    return fused_1nn_kernel


def kernel_symbol(
    data_abbrev: str,
    index_abbrev: str,
    matrix_layout: str = "strict",
) -> str:
    """Must stay in sync with fused_1nn_kernel_entrypoint() in fused_1nn_planner.hpp."""
    base = f"fused_1nn_{data_abbrev}_{index_abbrev}"
    if matrix_layout == "strict":
        return base
    if matrix_layout == "relaxed":
        return f"{base}_relaxed"
    raise ValueError(f"Unsupported matrix layout {matrix_layout!r}")


def index_abbrev(index_type: str) -> str:
    return {"int32": "i32", "int64": "i64"}[index_type]
