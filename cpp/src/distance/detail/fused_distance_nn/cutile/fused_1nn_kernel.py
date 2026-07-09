# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""cuTile fused GEMM + 1-NN kernels (InnerProduct, L2Expanded, CosineExpanded)."""

from __future__ import annotations

import cuda.tile as ct

ConstInt = ct.Constant[int]

# Default tile geometry; overridden per export via make_kernel(..., tile_m, tile_n, tile_k).
DEFAULT_TILE_M = 128
DEFAULT_TILE_N = 128
DEFAULT_TILE_K = 64

METRICS = ("inner_product", "l2_expanded", "cosine_expanded")
INDEX_TYPES = ("int32", "int64")


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
):
    """Build a cuTile kernel with metric, index width, and tile sizes baked in at compile time."""
    if data_type not in ("half", "float"):
        raise ValueError(f"Unsupported data_type {data_type!r}")
    if metric not in METRICS:
        raise ValueError(f"Unsupported metric {metric!r}")
    if index_type not in INDEX_TYPES:
        raise ValueError(f"Unsupported index_type {index_type!r}")

    acc_dtype = ct.float32
    idx_dtype = _idx_dtype(index_type)
    out_dist_dtype = ct.float16 if data_type == "half" else ct.float32
    is_ip = metric == "inner_product"
    is_l2 = metric == "l2_expanded"
    is_cos = metric == "cosine_expanded"

    if gpu_code == "sm_100":
        # Blackwell tcgen05 maps one logical row of the output tile to the reduction owner.
        core_shape = (tile_m, tile_n)
        best_shape = (tile_m, 1)
        inner_reduction_axes = (1,)
        outer_reduction_axes = (1,)
    else:
        # SM80/86/90/110/120 distribute eight consecutive N values across four threads,
        # two per thread.
        # Reducing in this physical layout keeps the score/index pair reduction local and avoids
        # the slower generic argmin/argmax lowering over the full N dimension.
        core_shape = (tile_m, tile_n // 8, 4, 2)
        best_shape = (tile_m, 1, 4, 1)
        inner_reduction_axes = (1, 3)
        outer_reduction_axes = (2,)

    @ct.kernel
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
        tm: ConstInt,
        tn: ConstInt,
        tk: ConstInt,
    ):
        bidm = ct.bid(0)

        if is_ip:
            best_dist = ct.full(best_shape, -3.4e38, acc_dtype)
            neutral_dist = -3.4e38
        else:
            best_dist = ct.full(best_shape, 3.4e38, acc_dtype)
            neutral_dist = 3.4e38
        best_idx = ct.zeros(best_shape, idx_dtype)

        num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))
        num_tiles_n = ct.num_tiles(B, axis=0, shape=(tn, tk))
        zero_pad = ct.PaddingMode.ZERO

        def reduce_scores(best, best_idx, axes):
            def red_op(a_score, a_idx, b_score, b_idx):
                if is_ip:
                    cond = a_score > b_score
                else:
                    cond = a_score < b_score

                return (
                    ct.where(cond, a_score, b_score),
                    ct.where(cond, a_idx, b_idx),
                )

            if len(axes) >= 1:
                best, best_idx = ct.reduce(
                    (best, best_idx),
                    axes[0],
                    red_op,
                    (neutral_dist, -1),
                    keepdims=True,
                )
            if len(axes) >= 2:
                best, best_idx = ct.reduce(
                    (best, best_idx),
                    axes[1],
                    red_op,
                    (neutral_dist, -1),
                    keepdims=True,
                )

            return best, best_idx

        for n in range(num_tiles_n):
            accumulator = ct.full((tm, tn), 0, dtype=acc_dtype)

            for k in range(num_tiles_k):
                dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

                a = ct.load(
                    A, index=(bidm, k), shape=(tm, tk), padding_mode=zero_pad
                ).astype(dtype)
                b_T = ct.load(
                    B, index=(n, k), shape=(tn, tk), padding_mode=zero_pad
                ).astype(dtype)

                accumulator = ct.mma(a, ct.transpose(b_T), accumulator)

            if is_ip:
                score = accumulator
            elif is_l2 or is_cos:
                a_norm = ct.load(
                    A_norm, index=(bidm,), shape=(tm,), padding_mode=zero_pad
                )
                b_norm = ct.load(
                    B_norm, index=(n,), shape=(tn,), padding_mode=zero_pad
                )
                if is_l2:
                    # L2 expanded: ||x||^2 + ||y||^2 - 2 * dot(x, y); norms are squared.
                    score = (
                        a_norm[:, None] + b_norm[None, :] - (2.0 * accumulator)
                    )
                elif is_cos:
                    # Cosine expanded distance: 1 - dot / (||x|| * ||y||); norms are L2 (not squared).
                    denom = a_norm[:, None] * b_norm[None, :]
                    score = 1.0 - (accumulator / denom)

            # Only the final N-tile can include zero-padded centroid columns.
            if n == num_tiles_n - 1:
                col = ct.arange(tn, dtype=idx_dtype)
                global_col = (n * tn + col).astype(idx_dtype)
                valid = global_col < N
                if is_ip:
                    score = ct.where(valid[None, :], score, -3.4e38)
                else:
                    score = ct.where(valid[None, :], score, 3.4e38)

            if is_ip:
                curr_idx = ct.arange(tn, dtype=idx_dtype).reshape(
                    core_shape[1:]
                )[None, ...]
                curr_best, curr_idx = reduce_scores(
                    score.reshape(core_shape), curr_idx, inner_reduction_axes
                )
                update = curr_best > best_dist
                best_dist = ct.where(update, curr_best, best_dist)
            else:
                curr_idx = ct.arange(tn, dtype=idx_dtype).reshape(
                    core_shape[1:]
                )[None, ...]
                curr_best, curr_idx = reduce_scores(
                    score.reshape(core_shape), curr_idx, inner_reduction_axes
                )
                update = curr_best < best_dist
                best_dist = ct.where(update, curr_best, best_dist)

            best_idx = ct.where(
                update, (n * tn + curr_idx).astype(idx_dtype), best_idx
            )

        best_dist, best_idx = reduce_scores(
            best_dist, best_idx, outer_reduction_axes
        )

        out_dist = best_dist
        if is_l2:
            out_dist = ct.where(apply_sqrt != 0, ct.sqrt(best_dist), best_dist)
        if store_idx != 0:
            ct.store(OutIdx, index=(bidm,), tile=best_idx.reshape((tm,)))
        ct.store(
            OutDist,
            index=(bidm,),
            tile=out_dist.reshape((tm,)).astype(out_dist_dtype),
        )

    return fused_1nn_kernel


def kernel_symbol(
    data_abbrev: str, metric_abbrev: str, index_abbrev: str
) -> str:
    """Must stay in sync with fused_1nn_kernel_entrypoint() in fused_1nn_planner.hpp."""
    return f"fused_1nn_{data_abbrev}_{metric_abbrev}_{index_abbrev}"


def metric_abbrev(metric: str) -> str:
    return {
        "inner_product": "ip",
        "l2_expanded": "l2",
        "cosine_expanded": "cos",
    }[metric]


def index_abbrev(index_type: str) -> str:
    return {"int32": "i32", "int64": "i64"}[index_type]
