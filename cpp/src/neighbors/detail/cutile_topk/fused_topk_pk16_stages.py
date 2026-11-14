# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM120 PK16 stages; every split is refined before native merge."""

from __future__ import annotations
import cuda.tile as ct
from fused_topk_kernel import (
    ConstInt,
    METRICS,
    INDEX_TYPES,
    PHYSICAL_KS,
    METRIC_L2_EXPANDED,
    METRIC_L2_SQRT_EXPANDED,
    METRIC_COSINE_EXPANDED,
    METRIC_INNER_PRODUCT,
    _INF,
    _INVALID_INDEX,
    _idx_dtype,
    _insert2,
    _insert4,
    _insert8,
    _insert16,
    _column,
    _take_left,
)


@ct.function
def _refine_network_stage(
    scores,
    indices,
    tm: ConstInt,
    pk: ConstInt,
    distance: ConstInt,
    sequence: ConstInt,
):
    groups = pk // (2 * distance)
    ss = scores.reshape((tm, groups, 2 * distance))
    ii = indices.reshape((tm, groups, 2 * distance))
    lo_s = ct.extract(ss, (0, 0, 0), shape=(tm, groups, distance))
    hi_s = ct.extract(ss, (0, 0, 1), shape=(tm, groups, distance))
    lo_i = ct.extract(ii, (0, 0, 0), shape=(tm, groups, distance))
    hi_i = ct.extract(ii, (0, 0, 1), shape=(tm, groups, distance))
    other_s = ct.cat((hi_s, lo_s), axis=2).reshape((tm, pk))
    other_i = ct.cat((hi_i, lo_i), axis=2).reshape((tm, pk))
    slots = ct.arange(pk, dtype=ct.int32)
    take_min = ((slots & distance) == 0) == ((slots & sequence) == 0)
    own_less = _take_left(scores, indices, other_s, other_i)
    keep = ct.where(take_min[None, :], own_less, ~own_less)
    return ct.where(keep, scores, other_s), ct.where(keep, indices, other_i)


@ct.function
def _refine_sort_network(scores, indices, tm: ConstInt, pk: ConstInt):
    scores, indices = _refine_network_stage(scores, indices, tm, pk, 1, 2)
    scores, indices = _refine_network_stage(scores, indices, tm, pk, 2, 4)
    scores, indices = _refine_network_stage(scores, indices, tm, pk, 1, 4)
    scores, indices = _refine_network_stage(scores, indices, tm, pk, 4, 8)
    scores, indices = _refine_network_stage(scores, indices, tm, pk, 2, 8)
    scores, indices = _refine_network_stage(scores, indices, tm, pk, 1, 8)
    if pk == 16:
        scores, indices = _refine_network_stage(scores, indices, tm, pk, 8, 16)
        scores, indices = _refine_network_stage(scores, indices, tm, pk, 4, 16)
        scores, indices = _refine_network_stage(scores, indices, tm, pk, 2, 16)
        scores, indices = _refine_network_stage(scores, indices, tm, pk, 1, 16)
    return scores, indices


@ct.function
def _merge_stage16(scores, indices, tm: ConstInt, distance: ConstInt):
    # Fixed tile permutations, not dynamic gathers or assumed lane placement.
    groups = 16 // (2 * distance)
    shaped_scores = scores.reshape((tm, groups, 2 * distance))
    shaped_indices = indices.reshape((tm, groups, 2 * distance))
    score_lo = ct.extract(
        shaped_scores, (0, 0, 0), shape=(tm, groups, distance)
    )
    score_hi = ct.extract(
        shaped_scores, (0, 0, 1), shape=(tm, groups, distance)
    )
    index_lo = ct.extract(
        shaped_indices, (0, 0, 0), shape=(tm, groups, distance)
    )
    index_hi = ct.extract(
        shaped_indices, (0, 0, 1), shape=(tm, groups, distance)
    )
    partner_scores = ct.cat((score_hi, score_lo), axis=2).reshape((tm, 16))
    partner_indices = ct.cat((index_hi, index_lo), axis=2).reshape((tm, 16))
    lower = (ct.arange(16, dtype=ct.int32) & distance) == 0
    own_is_less = _take_left(scores, indices, partner_scores, partner_indices)
    keep_own = ct.where(lower[None, :], own_is_less, ~own_is_less)
    return (
        ct.where(keep_own, scores, partner_scores),
        ct.where(keep_own, indices, partner_indices),
    )


@ct.function
def _merge_sorted16(
    scores, indices, descending_scores, descending_indices, tm: ConstInt
):
    # The first half of a bitonic 32-way merge contains the smallest 16.
    # Sort only that half: all valid corpus IDs are globally unique.
    keep_own = _take_left(
        scores, indices, descending_scores, descending_indices
    )
    scores = ct.where(keep_own, scores, descending_scores)
    indices = ct.where(keep_own, indices, descending_indices)
    scores, indices = _merge_stage16(scores, indices, tm, 8)
    scores, indices = _merge_stage16(scores, indices, tm, 4)
    scores, indices = _merge_stage16(scores, indices, tm, 2)
    return _merge_stage16(scores, indices, tm, 1)


@ct.function
def _extract_four(
    remaining,
    tiled_indices,
    tile_scores,
    tile_indices,
    tm: ConstInt,
    offset: ConstInt,
):
    def reduce_pair(a_score, a_idx, b_score, b_idx):
        take_a = _take_left(a_score, a_idx, b_score, b_idx)
        return (
            ct.where(take_a, a_score, b_score),
            ct.where(take_a, a_idx, b_idx),
        )

    slots = ct.arange(16, dtype=ct.int32)
    candidate_score = ct.full((tm,), _INF, ct.float32)
    for extracted in range(4):
        candidate_score, candidate_idx = ct.reduce(
            (remaining, tiled_indices),
            1,
            reduce_pair,
            (_INF, _INVALID_INDEX),
            keepdims=False,
        )
        candidate_idx = ct.where(
            candidate_score < _INF, candidate_idx, _INVALID_INDEX
        )
        position = slots == 15 - offset - extracted
        tile_scores = ct.where(
            position[None, :], candidate_score[:, None], tile_scores
        )
        tile_indices = ct.where(
            position[None, :], candidate_idx[:, None], tile_indices
        )
        remaining = ct.where(
            tiled_indices == candidate_idx[:, None], _INF, remaining
        )
    return remaining, tile_scores, tile_indices, candidate_score


@ct.function
def _group4_continue(last_score, worst_score, valid_rows):
    # Equality must continue: a smaller index can still win a score tie.
    # An incomplete retained list has +inf worst and also continues.
    return (
        ct.max(
            (valid_rows & (last_score <= worst_score)).astype(ct.int32), axis=0
        )
        != 0
    )


@ct.function
def _witness_update16(
    scores, tiled_indices, best_scores, best_indices, valid_rows, tm: ConstInt
):
    def reduce_pair(a_score, a_idx, b_score, b_idx):
        take_a = _take_left(a_score, a_idx, b_score, b_idx)
        return (
            ct.where(take_a, a_score, b_score),
            ct.where(take_a, a_idx, b_idx),
        )

    first_score, first_idx = ct.reduce(
        (scores, tiled_indices),
        1,
        reduce_pair,
        (_INF, _INVALID_INDEX),
        keepdims=False,
    )
    first_idx = ct.where(first_score < _INF, first_idx, _INVALID_INDEX)
    worst_score = _column(best_scores, 15, tm)
    if _group4_continue(first_score, worst_score, valid_rows):
        slots = ct.arange(16, dtype=ct.int32)
        tile_scores = ct.full((tm, 16), _INF, ct.float32)
        tile_indices = ct.full((tm, 16), _INVALID_INDEX, ct.int32)
        tile_scores = ct.where(
            (slots == 15)[None, :], first_score[:, None], tile_scores
        )
        tile_indices = ct.where(
            (slots == 15)[None, :], first_idx[:, None], tile_indices
        )
        remaining = ct.where(tiled_indices == first_idx[:, None], _INF, scores)
        last = first_score
        for extracted in range(3):
            last, idx = ct.reduce(
                (remaining, tiled_indices),
                1,
                reduce_pair,
                (_INF, _INVALID_INDEX),
                keepdims=False,
            )
            idx = ct.where(last < _INF, idx, _INVALID_INDEX)
            position = slots == 14 - extracted
            tile_scores = ct.where(
                position[None, :], last[:, None], tile_scores
            )
            tile_indices = ct.where(
                position[None, :], idx[:, None], tile_indices
            )
            remaining = ct.where(
                tiled_indices == idx[:, None], _INF, remaining
            )
        if _group4_continue(last, worst_score, valid_rows):
            remaining, tile_scores, tile_indices, last = _extract_four(
                remaining, tiled_indices, tile_scores, tile_indices, tm, 4
            )
            if _group4_continue(last, worst_score, valid_rows):
                remaining, tile_scores, tile_indices, last = _extract_four(
                    remaining, tiled_indices, tile_scores, tile_indices, tm, 8
                )
                if _group4_continue(last, worst_score, valid_rows):
                    remaining, tile_scores, tile_indices, last = _extract_four(
                        remaining,
                        tiled_indices,
                        tile_scores,
                        tile_indices,
                        tm,
                        12,
                    )
        best_scores, best_indices = _merge_sorted16(
            best_scores, best_indices, tile_scores, tile_indices, tm
        )
    return best_scores, best_indices


@ct.function
def _extract_group(
    remaining,
    tiled_indices,
    tile_scores,
    tile_indices,
    tm: ConstInt,
    offset: ConstInt,
    count: ConstInt,
):
    def reduce_pair(a_score, a_idx, b_score, b_idx):
        take_a = _take_left(a_score, a_idx, b_score, b_idx)
        return (
            ct.where(take_a, a_score, b_score),
            ct.where(take_a, a_idx, b_idx),
        )

    slots = ct.arange(16, dtype=ct.int32)
    candidate_score = ct.full((tm,), _INF, ct.float32)
    for extracted in range(count):
        candidate_score, candidate_idx = ct.reduce(
            (remaining, tiled_indices),
            1,
            reduce_pair,
            (_INF, _INVALID_INDEX),
            keepdims=False,
        )
        candidate_idx = ct.where(
            candidate_score < _INF, candidate_idx, _INVALID_INDEX
        )
        position = slots == 15 - offset - extracted
        tile_scores = ct.where(
            position[None, :], candidate_score[:, None], tile_scores
        )
        tile_indices = ct.where(
            position[None, :], candidate_idx[:, None], tile_indices
        )
        remaining = ct.where(
            tiled_indices == candidate_idx[:, None], _INF, remaining
        )
    return remaining, tile_scores, tile_indices, candidate_score


@ct.function
def _front_witness_update16(
    scores, tiled_indices, best_scores, best_indices, valid_rows, tm: ConstInt
):
    def reduce_pair(a_score, a_idx, b_score, b_idx):
        take_a = _take_left(a_score, a_idx, b_score, b_idx)
        return (
            ct.where(take_a, a_score, b_score),
            ct.where(take_a, a_idx, b_idx),
        )

    first_score, first_idx = ct.reduce(
        (scores, tiled_indices),
        1,
        reduce_pair,
        (_INF, _INVALID_INDEX),
        keepdims=False,
    )
    first_idx = ct.where(first_score < _INF, first_idx, _INVALID_INDEX)
    worst_score = _column(best_scores, 15, tm)
    if _group4_continue(first_score, worst_score, valid_rows):
        slots = ct.arange(16, dtype=ct.int32)
        tile_scores = ct.full((tm, 16), _INF, ct.float32)
        tile_indices = ct.full((tm, 16), _INVALID_INDEX, ct.int32)
        tile_scores = ct.where(
            (slots == 15)[None, :], first_score[:, None], tile_scores
        )
        tile_indices = ct.where(
            (slots == 15)[None, :], first_idx[:, None], tile_indices
        )
        remaining = ct.where(tiled_indices == first_idx[:, None], _INF, scores)
        remaining, tile_scores, tile_indices, last = _extract_group(
            remaining, tiled_indices, tile_scores, tile_indices, tm, 1, 1
        )
        if _group4_continue(last, worst_score, valid_rows):
            remaining, tile_scores, tile_indices, last = _extract_group(
                remaining, tiled_indices, tile_scores, tile_indices, tm, 2, 2
            )
            if _group4_continue(last, worst_score, valid_rows):
                remaining, tile_scores, tile_indices, last = _extract_group(
                    remaining,
                    tiled_indices,
                    tile_scores,
                    tile_indices,
                    tm,
                    4,
                    4,
                )
                if _group4_continue(last, worst_score, valid_rows):
                    remaining, tile_scores, tile_indices, last = (
                        _extract_group(
                            remaining,
                            tiled_indices,
                            tile_scores,
                            tile_indices,
                            tm,
                            8,
                            4,
                        )
                    )
                    if _group4_continue(last, worst_score, valid_rows):
                        remaining, tile_scores, tile_indices, last = (
                            _extract_group(
                                remaining,
                                tiled_indices,
                                tile_scores,
                                tile_indices,
                                tm,
                                12,
                                4,
                            )
                        )
        best_scores, best_indices = _merge_sorted16(
            best_scores, best_indices, tile_scores, tile_indices, tm
        )
    return best_scores, best_indices


def make_select_kernel(
    data_type: str,
    metric: str,
    physical_k: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    *,
    index_type: str = "int32",
    gpu_code: str = "sm_120",
    matrix_layout: str = "strict",
    occupancy: int | None = None,
    first_candidate_witness: bool = False,
):
    """Build native-half or TF32 selection with FP32 accumulation/refinement."""
    if physical_k != 16 or gpu_code != "sm_120" or index_type != "int32":
        raise ValueError(
            "Separated stages require SM120 PK16 int32 candidates"
        )
    if data_type not in ("float", "half"):
        raise ValueError(f"Unsupported data_type {data_type!r}")
    if metric not in METRICS:
        raise ValueError(f"Unsupported metric {metric!r}")
    if physical_k not in PHYSICAL_KS:
        raise ValueError(f"Unsupported physical K {physical_k}")
    if index_type not in INDEX_TYPES:
        raise ValueError(f"Unsupported index_type {index_type!r}")
    if matrix_layout not in ("strict", "relaxed"):
        raise ValueError(f"Unsupported matrix_layout {matrix_layout!r}")

    if index_type == "int64" and (physical_k != 8 or gpu_code != "sm_120"):
        raise ValueError(
            "Direct int64 storage is currently qualified only for SM120 PK8"
        )
    # Candidate arithmetic stays int32; only final storage may widen.
    idx_dtype = _idx_dtype("int32")
    mma_dtype = ct.float16 if data_type == "half" else ct.tfloat32
    kernel_options = {}
    if occupancy is not None:
        kernel_options["occupancy"] = ct.ByTarget(**{gpu_code: occupancy})

    @ct.kernel(**kernel_options)
    def fused_topk_kernel(
        queries,
        corpus,
        out_indices,
        out_distances,
        query_rows,
        corpus_rows,
        dimensions,
        metric_code,
        num_splits,
        tm: ConstInt,
        tn: ConstInt,
        tk: ConstInt,
        pk: ConstInt,
    ):
        bid_m = ct.bid(0)
        split_id = ct.bid(1)
        best_scores = ct.full((tm, physical_k), _INF, ct.float32)
        best_indices = ct.full((tm, physical_k), _INVALID_INDEX, idx_dtype)
        num_tiles_k = ct.num_tiles(queries, axis=1, shape=(tm, tk))
        num_tiles_n = ct.num_tiles(corpus, axis=0, shape=(tn, tk))
        # Quotient/remainder partitioning keeps every intermediate bounded by
        # num_tiles_n. The algebraically simpler split_id * num_tiles_n form
        # can overflow int32 for otherwise valid large corpora.
        base_tiles = num_tiles_n // num_splits
        extra_tiles = num_tiles_n - base_tiles * num_splits
        split_prefix = ct.where(split_id < extra_tiles, split_id, extra_tiles)
        split_begin = split_id * base_tiles + split_prefix
        split_end = (
            split_begin + base_tiles + ct.where(split_id < extra_tiles, 1, 0)
        )

        def reduce_best(scores, indices):
            def reduce_pair(a_score, a_idx, b_score, b_idx):
                take_a = _take_left(a_score, a_idx, b_score, b_idx)
                return (
                    ct.where(take_a, a_score, b_score),
                    ct.where(take_a, a_idx, b_idx),
                )

            return ct.reduce(
                (scores, indices),
                1,
                reduce_pair,
                (_INF, _INVALID_INDEX),
                keepdims=False,
            )

        for tile_n_index in range(split_begin, split_end):
            accumulator = ct.zeros((tm, tn), ct.float32)
            # Match the native-half or one-pass TF32 dot-product operands. Using cuVS's
            # canonical FP32 norms here would make the ranking score internally
            # inconsistent and increases candidate misses near the boundary.
            rounded_corpus_norms = ct.zeros((tn,), ct.float32)
            for tile_k_index in range(num_tiles_k):
                query_tile = ct.load(
                    queries,
                    index=(bid_m, tile_k_index),
                    shape=(tm, tk),
                    padding_mode=ct.PaddingMode.ZERO,
                ).astype(mma_dtype)
                corpus_t = ct.load(
                    corpus,
                    index=(tile_k_index, tile_n_index),
                    shape=(tk, tn),
                    order=(1, 0),
                    padding_mode=ct.PaddingMode.ZERO,
                ).astype(mma_dtype)
                accumulator = ct.mma(query_tile, corpus_t, accumulator)
                if metric_code != METRIC_INNER_PRODUCT:
                    corpus_f32 = corpus_t.astype(ct.float32)
                    rounded_corpus_norms = rounded_corpus_norms + ct.sum(
                        corpus_f32 * corpus_f32, axis=0
                    )

            global_indices = tile_n_index * tn + ct.arange(tn, dtype=idx_dtype)
            if metric_code == METRIC_INNER_PRODUCT:
                scores = -accumulator
            else:
                if metric_code == METRIC_COSINE_EXPANDED:
                    positive_norm = rounded_corpus_norms > 0.0
                    safe_norm = ct.where(
                        positive_norm, rounded_corpus_norms, 1.0
                    )
                    scores = accumulator / (-ct.sqrt(safe_norm))[None, :]
                    # A zero corpus vector has cosine distance one, hence rank score zero.
                    scores = ct.where(positive_norm[None, :], scores, 0.0)
                else:
                    scores = (0.5 * rounded_corpus_norms)[
                        None, :
                    ] - accumulator

            valid_columns = global_indices < corpus_rows
            scores = ct.where(valid_columns[None, :], scores, _INF)
            tiled_indices = ct.broadcast_to(global_indices[None, :], (tm, tn))
            if first_candidate_witness:
                valid_rows = (
                    bid_m * tm + ct.arange(tm, dtype=ct.int32) < query_rows
                )
                if data_type == "float":
                    best_scores, best_indices = _front_witness_update16(
                        scores,
                        tiled_indices,
                        best_scores,
                        best_indices,
                        valid_rows,
                        tm,
                    )
                else:
                    best_scores, best_indices = _witness_update16(
                        scores,
                        tiled_indices,
                        best_scores,
                        best_indices,
                        valid_rows,
                        tm,
                    )
            else:
                remaining = scores
                tile_scores = ct.full((tm, physical_k), _INF, ct.float32)
                tile_indices = ct.full(
                    (tm, physical_k), _INVALID_INDEX, idx_dtype
                )
                slots = ct.arange(physical_k, dtype=idx_dtype)
                for extracted in range(physical_k):
                    candidate_score, candidate_idx = reduce_best(
                        remaining, tiled_indices
                    )
                    # A physical slot with no remaining corpus candidate reduces
                    # to the score identity. Do not let the reduction's tie-broken
                    # column index turn that identity into a duplicate neighbor.
                    candidate_idx = ct.where(
                        candidate_score < _INF, candidate_idx, _INVALID_INDEX
                    )
                    position = slots == physical_k - 1 - extracted
                    tile_scores = ct.where(
                        position[None, :],
                        candidate_score[:, None],
                        tile_scores,
                    )
                    tile_indices = ct.where(
                        position[None, :], candidate_idx[:, None], tile_indices
                    )
                    remaining = ct.where(
                        tiled_indices == candidate_idx[:, None],
                        _INF,
                        remaining,
                    )
                best_scores, best_indices = _merge_sorted16(
                    best_scores, best_indices, tile_scores, tile_indices, tm
                )

        ct.store(out_indices, index=(bid_m, split_id), tile=best_indices)

    return fused_topk_kernel


def make_refine_kernel(
    data_type: str,
    metric: str,
    physical_k: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    *,
    index_type: str = "int32",
    gpu_code: str = "sm_120",
    matrix_layout: str = "strict",
    occupancy: int | None = None,
    sort_network: bool = False,
):
    """Recompute and order all retained candidates using direct FP32 arithmetic."""
    if sort_network and (
        data_type != "float"
        or matrix_layout != "strict"
        or gpu_code != "sm_120"
    ):
        raise ValueError(
            "Sorting-network refinement requires strict SM120 FP32"
        )
    if physical_k != 16 or gpu_code != "sm_120" or index_type != "int32":
        raise ValueError(
            "Separated stages require SM120 PK16 int32 candidates"
        )
    if data_type not in ("float", "half"):
        raise ValueError(f"Unsupported data_type {data_type!r}")
    if metric not in METRICS:
        raise ValueError(f"Unsupported metric {metric!r}")
    if physical_k not in PHYSICAL_KS:
        raise ValueError(f"Unsupported physical K {physical_k}")
    if index_type not in INDEX_TYPES:
        raise ValueError(f"Unsupported index_type {index_type!r}")
    if matrix_layout not in ("strict", "relaxed"):
        raise ValueError(f"Unsupported matrix_layout {matrix_layout!r}")

    if index_type == "int64" and (physical_k != 8 or gpu_code != "sm_120"):
        raise ValueError(
            "Direct int64 storage is currently qualified only for SM120 PK8"
        )
    # Candidate arithmetic stays int32; only final storage may widen.
    idx_dtype = _idx_dtype("int32")
    kernel_options = {}
    if occupancy is not None:
        kernel_options["occupancy"] = ct.ByTarget(**{gpu_code: occupancy})
    insert_candidate = (
        _insert2
        if physical_k == 2
        else _insert4
        if physical_k == 4
        else _insert8
        if physical_k == 8
        else _insert16
    )

    @ct.kernel(**kernel_options)
    def fused_topk_kernel(
        queries,
        corpus,
        out_indices,
        out_distances,
        query_rows,
        corpus_rows,
        dimensions,
        metric_code,
        num_splits,
        tm: ConstInt,
        tn: ConstInt,
        tk: ConstInt,
        pk: ConstInt,
    ):
        bid_m = ct.bid(0)
        split_id = ct.bid(1)
        num_tiles_k = ct.num_tiles(queries, axis=1, shape=(tm, tk))
        # A single read/write descriptor permits in-place update without aliasing
        # distinct descriptors. All IDs are loaded before this CTA stores them.
        best_indices = ct.load(
            out_indices,
            index=(bid_m, split_id),
            shape=(tm, physical_k),
            padding_mode=ct.PaddingMode.ZERO,
        )
        # Recompute only the retained candidates with IEEE FP32 arithmetic. This
        # keeps public distances internally consistent even though TF32 selects
        # the candidate set.
        rows = bid_m * tm + ct.arange(tm, dtype=idx_dtype)
        valid_rows = rows < query_rows
        valid_selected = (best_indices >= 0) & (best_indices < corpus_rows)
        safe_selected = ct.where(valid_selected, best_indices, 0)
        exact_scores = ct.zeros((tm, physical_k), ct.float32)
        dots = ct.zeros((tm, physical_k), ct.float32)
        query_norms = ct.zeros((tm,), ct.float32)
        selected_norms = ct.zeros((tm, physical_k), ct.float32)

        for tile_k_index in range(num_tiles_k):
            query_tile = ct.load(
                queries,
                index=(bid_m, tile_k_index),
                shape=(tm, tk),
                padding_mode=ct.PaddingMode.ZERO,
            ).astype(ct.float32)
            features = tile_k_index * tk + ct.arange(tk, dtype=idx_dtype)
            selected_values = ct.gather(
                corpus,
                (safe_selected[:, :, None], features[None, None, :]),
                padding_value=0,
            ).astype(ct.float32)
            if (
                metric_code == METRIC_L2_EXPANDED
                or metric_code == METRIC_L2_SQRT_EXPANDED
            ):
                delta = query_tile[:, None, :] - selected_values
                exact_scores = exact_scores + ct.sum(delta * delta, axis=2)
            else:
                dots = dots + ct.sum(
                    query_tile[:, None, :] * selected_values, axis=2
                )
                if metric_code == METRIC_COSINE_EXPANDED:
                    query_norms = query_norms + ct.sum(
                        query_tile * query_tile, axis=1
                    )
                    selected_norms = selected_norms + ct.sum(
                        selected_values * selected_values, axis=2
                    )

        if metric_code == METRIC_INNER_PRODUCT:
            exact_scores = -dots
        elif metric_code == METRIC_COSINE_EXPANDED:
            positive_norms = (query_norms[:, None] > 0.0) & (
                selected_norms > 0.0
            )
            safe_query_norms = ct.where(query_norms > 0.0, query_norms, 1.0)
            safe_selected_norms = ct.where(
                selected_norms > 0.0, selected_norms, 1.0
            )
            exact_scores = 1.0 - dots / (
                ct.sqrt(safe_query_norms)[:, None]
                * ct.sqrt(safe_selected_norms)
            )
            exact_scores = ct.where(positive_norms, exact_scores, 1.0)
        elif metric_code == METRIC_L2_SQRT_EXPANDED:
            # Sort the returned metric, not its squared precursor: distinct
            # squared distances can round to the same FP32 square root.
            # The split merge uses this same public (distance, index) key.
            exact_scores = ct.sqrt(
                ct.where(exact_scores < 0.0, 0.0, exact_scores)
            )
        exact_scores = ct.where(
            valid_rows[:, None] & valid_selected, exact_scores, _INF
        )

        if sort_network:
            sorted_scores, sorted_indices = _refine_sort_network(
                exact_scores, best_indices, tm, physical_k
            )
        else:
            sorted_scores = ct.full((tm, physical_k), _INF, ct.float32)
            sorted_indices = ct.full(
                (tm, physical_k), _INVALID_INDEX, idx_dtype
            )
            for column in range(physical_k):
                sorted_scores, sorted_indices = insert_candidate(
                    sorted_scores,
                    sorted_indices,
                    _column(exact_scores, column, tm),
                    _column(best_indices, column, tm),
                    tm,
                )

        valid_sorted = sorted_indices < _INVALID_INDEX
        sorted_indices = ct.where(valid_sorted, sorted_indices, -1)
        if metric_code == METRIC_INNER_PRODUCT:
            output_values = -sorted_scores
        else:
            output_values = sorted_scores
        output_values = ct.where(valid_sorted, output_values, _INF)
        # A single-split descriptor may have logical-K width: bounded stores
        # discard surplus only AFTER all physical candidates were refined/sorted.
        # Store converts int32 candidates to the output descriptor's index type.
        ct.store(out_indices, index=(bid_m, split_id), tile=sorted_indices)
        ct.store(out_distances, index=(bid_m, split_id), tile=output_values)

    return fused_topk_kernel
