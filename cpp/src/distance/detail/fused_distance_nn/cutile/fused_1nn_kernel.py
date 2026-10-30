# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cuTile fused GEMM + 1-NN kernel with runtime metric selection."""

from __future__ import annotations

import inspect

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
METRIC_BITWISE_HAMMING = 20

# ~FLT_MAX (cuTile has no float limits): larger than any score, so it marks "no candidate yet" and
# the masked columns of the last tile in the argmin.
FLT_MAX = 3.4e38

# The binary variant needs block-scaled fp4 MMA, which Blackwell and later have (tcgen05 on
# sm_100 / sm_107, warp-level mma on sm_120). Older GPUs keep the SIMT popc kernel.
BINARY_GPU_CODES = ("sm_100", "sm_107", "sm_120")


def _idx_dtype(index_type: str):
    if index_type == "int32":
        return ct.int32
    if index_type == "int64":
        return ct.int64
    raise ValueError(f"Unsupported index_type {index_type!r}")


def _supports_rounding_mode() -> bool:
    """cuda-tile >= 1.6 has RoundingMode.RA and astype(..., rounding_mode=...)."""
    return hasattr(getattr(ct, "RoundingMode", None), "RA") and (
        "rounding_mode" in inspect.signature(ct.astype).parameters
    )


def _mma_rounding_mode(gpu_code: str):
    """Rounding of the operand conversion to the MMA dtype; None selects the cuTile default.

    sm_8x has no hardware round-to-nearest-even float32 -> tfloat32 conversion, so the default
    is emulated with integer code that nearly halves fused-1NN throughput on A100. Round to
    nearest with ties away (RA) is much cheaper there, and it is also how compute_tf32_norms()
    rounds the norms (cvt.rna). Casts that keep the dtype (half -> half) are unaffected. RA needs
    tileiras >= 13.4; export_fused_1nn.py falls back to the default on older tileiras.
    """
    if gpu_code.startswith("sm_8") and _supports_rounding_mode():
        return ct.RoundingMode.RA
    return None


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
    occupancy: int | None = None,
    num_ctas: int | None = None,
    mma_rounding: bool = True,
):
    """Build the flat-reduction runtime-metric cuTile kernel."""
    if data_type not in ("half", "float", "binary"):
        raise ValueError(f"Unsupported data_type {data_type!r}")
    if data_type == "binary" and gpu_code not in BINARY_GPU_CODES:
        raise ValueError(
            f"The binary kernel needs block-scaled fp4 MMA; {gpu_code} is not supported"
        )
    if metric not in METRICS:
        raise ValueError(f"Unsupported metric {metric!r}")
    if index_type not in INDEX_TYPES:
        raise ValueError(f"Unsupported index_type {index_type!r}")
    if matrix_layout not in ("strict", "relaxed", "high_dim"):
        raise ValueError(f"Unsupported matrix_layout {matrix_layout!r}")

    mma_dtype = ct.tfloat32 if data_type == "float" else ct.float16
    rmode = _mma_rounding_mode(gpu_code) if mma_rounding else None
    if rmode is None:

        def to_mma_dtype(tile):
            return tile.astype(mma_dtype)
    else:

        def to_mma_dtype(tile):
            return tile.astype(mma_dtype, rounding_mode=rmode)

    acc_dtype = ct.float32
    idx_dtype = _idx_dtype(index_type)
    core_shape = (tile_m, tile_n)
    best_shape = (tile_m, 1)
    kernel_options = {}
    if occupancy is not None:
        kernel_options["occupancy"] = ct.ByTarget(**{gpu_code: occupancy})
    if num_ctas is not None:
        kernel_options["num_ctas"] = ct.ByTarget(**{gpu_code: num_ctas})

    if data_type == "binary":
        return _make_binary_kernel(kernel_options, idx_dtype, tile_m)

    @ct.kernel(**kernel_options)
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
        best_dist = ct.full(best_shape, FLT_MAX, acc_dtype)
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
                (FLT_MAX, -1),
                keepdims=True,
            )

        local_indices = ct.arange(tn, dtype=ct.int16)[None, :]
        for n in range(num_tiles_n):
            accumulator = ct.full((tm, tn), 0, dtype=acc_dtype)
            for k in range(num_tiles_k):
                a = to_mma_dtype(
                    ct.load(
                        A,
                        index=(bidm, k),
                        shape=(tm, tk),
                        padding_mode=zero_pad,
                    )
                )
                b_T = to_mma_dtype(
                    ct.load(
                        B,
                        index=(k, n),
                        shape=(tk, tn),
                        padding_mode=zero_pad,
                        order=(1, 0),
                    )
                )
                accumulator = ct.mma(a, b_T, accumulator)

            if metric_code == METRIC_INNER_PRODUCT:
                score = -accumulator
            else:
                b_norm = ct.load(
                    B_norm, index=(n,), shape=(tn,), padding_mode=zero_pad
                )
                if metric_code == METRIC_L2_EXPANDED:
                    # The A norm is constant across centroids. Excluding it
                    # avoids cancellation in the score used by argmin.
                    score = (0.5 * b_norm)[None, :] - accumulator
                else:
                    # Cosine distance involving any zero-norm vector is defined as 1.
                    # A zero B norm therefore contributes a normalized dot score of 0.
                    positive_b_norm = b_norm > 0.0
                    safe_b_norm = ct.where(positive_b_norm, b_norm, 1.0)
                    score = accumulator / (-safe_b_norm)[None, :]
                    score = ct.where(positive_b_norm[None, :], score, 0.0)

            if n == num_tiles_n - 1:
                col = ct.arange(tn, dtype=ct.int16)
                score = ct.where((n * tn + col)[None, :] < N, score, FLT_MAX)

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
                # Separately reduced norms and the MMA can reconstruct a
                # slightly negative distance; clamp before an optional sqrt.
                out_dist = ct.where(out_dist > 0.0, out_dist, 0.0)
                out_dist = ct.where(
                    apply_sqrt != 0, ct.sqrt(out_dist), out_dist
                )
            else:
                positive_a_norm = a_norm > 0.0
                safe_a_norm = ct.where(positive_a_norm, a_norm, 1.0)
                out_dist = 1.0 + best_dist / safe_a_norm
                out_dist = ct.where(positive_a_norm, out_dist, 1.0)

        if store_idx != 0:
            ct.store(OutIdx, index=(bidm,), tile=best_idx.reshape((tm,)))
        ct.store(
            OutDist,
            index=(bidm,),
            tile=out_dist.reshape((tm,)),
        )

    return fused_1nn_kernel


def _make_binary_kernel(kernel_options, idx_dtype, tile_m):
    """Bitwise-Hamming 1-NN on bit-packed rows, on the tensor core.

    hamming(x, y) = popc(x) + popc(y) - 2 * <bits(x), bits(y)>, so the K loop is a 0/1 dot
    product. The launcher promotes every bit to an e2m1 nibble beforehand (0x0 is 0.0, 0x2 is
    1.0, both exact; bit i is nibble i), so A and B hold two bits per byte, rows padded with zero
    nibbles to a 16-byte pitch, and tk counts those bytes. With unit block scales the dot product is
    exact for any width that fits the fp32 accumulator. A_norm and B_norm are the per-row
    popcounts. The query popcount is constant across centroids, so it is left out of the argmin.
    """
    best_shape = (tile_m, 1)

    @ct.kernel(**kernel_options)
    def fused_1nn_binary_kernel(
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
        zero_pad = ct.PaddingMode.ZERO
        num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))
        num_tiles_n = ct.num_tiles(B, axis=0, shape=(tn, tk))
        bits = 2 * tk
        a_scale = ct.full((tm, bits // 32), 1.0, ct.float8_e8m0fnu)
        b_scale = ct.full((bits // 32, tn), 1.0, ct.float8_e8m0fnu)

        def as_e2m1(t, rows):
            return ct.unpack_from_bytes(t.reshape((rows * tk,)), ct.float4_e2m1fn).reshape(
                (rows, bits)
            )

        def red_op(a_score, a_idx, b_score, b_idx):
            cond = (a_score < b_score) | ((a_score == b_score) & (a_idx < b_idx))
            return ct.where(cond, a_score, b_score), ct.where(cond, a_idx, b_idx)

        best_dist = ct.full(best_shape, FLT_MAX, ct.float32)
        best_idx = ct.zeros(best_shape, idx_dtype)
        local_indices = ct.arange(tn, dtype=ct.int16)[None, :]
        for n in range(num_tiles_n):
            accumulator = ct.full((tm, tn), 0, dtype=ct.float32)
            for k in range(num_tiles_k):
                a = ct.load(A, index=(bidm, k), shape=(tm, tk), padding_mode=zero_pad)
                b = ct.load(B, index=(n, k), shape=(tn, tk), padding_mode=zero_pad)
                accumulator = ct.mma_scaled(
                    as_e2m1(a, tm),
                    a_scale,
                    ct.permute(as_e2m1(b, tn), (1, 0)),
                    b_scale,
                    accumulator,
                )

            b_popc = ct.load(B_norm, index=(n,), shape=(tn,), padding_mode=zero_pad)
            score = b_popc[None, :] - 2.0 * accumulator
            if n == num_tiles_n - 1:
                col = ct.arange(tn, dtype=ct.int16)
                score = ct.where((n * tn + col)[None, :] < N, score, FLT_MAX)

            curr_best, curr_idx = ct.reduce(
                (score, local_indices), 1, red_op, (FLT_MAX, -1), keepdims=True
            )
            update = curr_best < best_dist
            best_dist = ct.where(update, curr_best, best_dist)
            best_idx = ct.where(update, n * tn + curr_idx, best_idx)

        a_popc = ct.load(A_norm, index=(bidm,), shape=(tm,), padding_mode=zero_pad)[:, None]
        if store_idx != 0:
            ct.store(OutIdx, index=(bidm,), tile=best_idx.reshape((tm,)))
        ct.store(OutDist, index=(bidm,), tile=(a_popc + best_dist).reshape((tm,)))

    return fused_1nn_binary_kernel


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
    if matrix_layout == "high_dim":
        return f"{base}_hd"
    raise ValueError(f"Unsupported matrix layout {matrix_layout!r}")


def index_abbrev(index_type: str) -> str:
    return {"int32": "i32", "int64": "i64"}[index_type]
