# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cuTile fused GEMM + small physical-top-k kernels."""

from __future__ import annotations

import cuda.tile as ct

ConstInt = ct.Constant[int]

METRICS = ("runtime",)
INDEX_TYPES = ("int32",)
PHYSICAL_KS = (2, 4)

METRIC_L2_EXPANDED = 0
METRIC_L2_SQRT_EXPANDED = 1
METRIC_COSINE_EXPANDED = 2
METRIC_INNER_PRODUCT = 6

_INF = float("inf")
_INVALID_INDEX = 2**31 - 1


@ct.function
def _column(tile, column: ConstInt, tile_m: ConstInt):
    return ct.extract(tile, (0, column), shape=(tile_m, 1)).reshape((tile_m,))


@ct.function
def _pack2(value0, value1):
    return ct.cat((value0[:, None], value1[:, None]), axis=1)


@ct.function
def _pack4(value0, value1, value2, value3):
    return ct.cat((_pack2(value0, value1), _pack2(value2, value3)), axis=1)


@ct.function
def _take_left(left_score, left_idx, right_score, right_idx):
    return (left_score < right_score) | (
        (left_score == right_score) & (left_idx < right_idx)
    )


@ct.function
def _insert2(
    scores, indices, candidate_score, candidate_idx, tile_m: ConstInt
):
    score0 = _column(scores, 0, tile_m)
    score1 = _column(scores, 1, tile_m)
    index0 = _column(indices, 0, tile_m)
    index1 = _column(indices, 1, tile_m)

    take = _take_left(candidate_score, candidate_idx, score0, index0)
    displaced_score, displaced_idx = score0, index0
    score0 = ct.where(take, candidate_score, score0)
    index0 = ct.where(take, candidate_idx, index0)
    candidate_score = ct.where(take, displaced_score, candidate_score)
    candidate_idx = ct.where(take, displaced_idx, candidate_idx)

    take = _take_left(candidate_score, candidate_idx, score1, index1)
    score1 = ct.where(take, candidate_score, score1)
    index1 = ct.where(take, candidate_idx, index1)
    return _pack2(score0, score1), _pack2(index0, index1)


@ct.function
def _insert4(
    scores, indices, candidate_score, candidate_idx, tile_m: ConstInt
):
    score0 = _column(scores, 0, tile_m)
    score1 = _column(scores, 1, tile_m)
    score2 = _column(scores, 2, tile_m)
    score3 = _column(scores, 3, tile_m)
    index0 = _column(indices, 0, tile_m)
    index1 = _column(indices, 1, tile_m)
    index2 = _column(indices, 2, tile_m)
    index3 = _column(indices, 3, tile_m)

    take = _take_left(candidate_score, candidate_idx, score0, index0)
    displaced_score, displaced_idx = score0, index0
    score0 = ct.where(take, candidate_score, score0)
    index0 = ct.where(take, candidate_idx, index0)
    candidate_score = ct.where(take, displaced_score, candidate_score)
    candidate_idx = ct.where(take, displaced_idx, candidate_idx)

    take = _take_left(candidate_score, candidate_idx, score1, index1)
    displaced_score, displaced_idx = score1, index1
    score1 = ct.where(take, candidate_score, score1)
    index1 = ct.where(take, candidate_idx, index1)
    candidate_score = ct.where(take, displaced_score, candidate_score)
    candidate_idx = ct.where(take, displaced_idx, candidate_idx)

    take = _take_left(candidate_score, candidate_idx, score2, index2)
    displaced_score, displaced_idx = score2, index2
    score2 = ct.where(take, candidate_score, score2)
    index2 = ct.where(take, candidate_idx, index2)
    candidate_score = ct.where(take, displaced_score, candidate_score)
    candidate_idx = ct.where(take, displaced_idx, candidate_idx)

    take = _take_left(candidate_score, candidate_idx, score3, index3)
    score3 = ct.where(take, candidate_score, score3)
    index3 = ct.where(take, candidate_idx, index3)
    return (
        _pack4(score0, score1, score2, score3),
        _pack4(index0, index1, index2, index3),
    )


def _idx_dtype(index_type: str):
    if index_type == "int32":
        return ct.int32
    raise ValueError(f"Unsupported index_type {index_type!r}")


def make_kernel(
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
    stage: str = "fused",
    witness: bool = False,
):
    """Build native-half or TF32 selection with FP32 accumulation/refinement."""
    if stage not in ("fused", "select", "refine"):
        raise ValueError(f"Unsupported stage {stage!r}")
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

    idx_dtype = _idx_dtype(index_type)
    mma_dtype = ct.float16 if data_type == "half" else ct.tfloat32
    kernel_options = {}
    if occupancy is not None:
        kernel_options["occupancy"] = ct.ByTarget(**{gpu_code: occupancy})
    insert_candidate = _insert2 if physical_k == 2 else _insert4

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
        if stage != "refine":
            best_scores = ct.full((tm, physical_k), _INF, ct.float32)
            best_indices = ct.full((tm, physical_k), _INVALID_INDEX, idx_dtype)
            num_tiles_k = ct.num_tiles(queries, axis=1, shape=(tm, tk))
            num_tiles_n = ct.num_tiles(corpus, axis=0, shape=(tn, tk))
            # Quotient/remainder partitioning keeps every intermediate bounded by
            # num_tiles_n. The algebraically simpler split_id * num_tiles_n form
            # can overflow int32 for otherwise valid large corpora.
            base_tiles = num_tiles_n // num_splits
            extra_tiles = num_tiles_n - base_tiles * num_splits
            split_prefix = ct.where(
                split_id < extra_tiles, split_id, extra_tiles
            )
            split_begin = split_id * base_tiles + split_prefix
            split_end = (
                split_begin
                + base_tiles
                + ct.where(split_id < extra_tiles, 1, 0)
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

                global_indices = tile_n_index * tn + ct.arange(
                    tn, dtype=idx_dtype
                )
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
                tiled_indices = ct.broadcast_to(
                    global_indices[None, :], (tm, tn)
                )
                remaining = scores
                if witness:
                    # Skip only when no real query row can admit a candidate.
                    # Equality and incomplete lists must still enter the loop.
                    first_score, first_idx = reduce_best(
                        remaining, tiled_indices
                    )
                    first_idx = ct.where(
                        first_score < _INF, first_idx, _INVALID_INDEX
                    )
                    worst_score = _column(best_scores, physical_k - 1, tm)
                    valid_rows = (
                        bid_m * tm + ct.arange(tm, dtype=idx_dtype)
                        < query_rows
                    )
                    need_more = (
                        ct.max(
                            (valid_rows & (first_score <= worst_score)).astype(
                                ct.int32
                            ),
                            axis=0,
                        )
                        != 0
                    )
                    if need_more:
                        best_scores, best_indices = insert_candidate(
                            best_scores,
                            best_indices,
                            first_score,
                            first_idx,
                            tm,
                        )
                        remaining = ct.where(
                            tiled_indices == first_idx[:, None],
                            _INF,
                            remaining,
                        )
                        for _ in range(physical_k - 1):
                            candidate_score, candidate_idx = reduce_best(
                                remaining, tiled_indices
                            )
                            # A physical slot with no remaining corpus candidate reduces
                            # to the score identity. Do not let the reduction's tie-broken
                            # column index turn that identity into a duplicate neighbor.
                            candidate_idx = ct.where(
                                candidate_score < _INF,
                                candidate_idx,
                                _INVALID_INDEX,
                            )
                            best_scores, best_indices = insert_candidate(
                                best_scores,
                                best_indices,
                                candidate_score,
                                candidate_idx,
                                tm,
                            )
                            remaining = ct.where(
                                tiled_indices == candidate_idx[:, None],
                                _INF,
                                remaining,
                            )
                else:
                    for _ in range(physical_k):
                        candidate_score, candidate_idx = reduce_best(
                            remaining, tiled_indices
                        )
                        # A physical slot with no remaining corpus candidate reduces
                        # to the score identity. Do not let the reduction's tie-broken
                        # column index turn that identity into a duplicate neighbor.
                        candidate_idx = ct.where(
                            candidate_score < _INF,
                            candidate_idx,
                            _INVALID_INDEX,
                        )
                        best_scores, best_indices = insert_candidate(
                            best_scores,
                            best_indices,
                            candidate_score,
                            candidate_idx,
                            tm,
                        )
                        remaining = ct.where(
                            tiled_indices == candidate_idx[:, None],
                            _INF,
                            remaining,
                        )

        else:
            num_tiles_k = ct.num_tiles(queries, axis=1, shape=(tm, tk))
            best_indices = ct.load(
                out_indices,
                index=(bid_m, split_id),
                shape=(tm, physical_k),
                padding_mode=ct.PaddingMode.ZERO,
            )

        if stage == "select":
            ct.store(out_indices, index=(bid_m, split_id), tile=best_indices)
        else:
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
                safe_query_norms = ct.where(
                    query_norms > 0.0, query_norms, 1.0
                )
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
            ct.store(out_indices, index=(bid_m, split_id), tile=sorted_indices)
            ct.store(
                out_distances, index=(bid_m, split_id), tile=output_values
            )

    return fused_topk_kernel


def kernel_symbol(
    physical_k: int, matrix_layout: str = "strict", data_type: str = "float"
) -> str:
    if data_type not in ("float", "half"):
        raise ValueError(f"Unsupported data_type {data_type!r}")
    abbrev = "h" if data_type == "half" else "f"
    base = f"fused_topk_pk{physical_k}_{abbrev}_i32"
    if matrix_layout == "strict":
        return base
    if matrix_layout == "relaxed":
        return f"{base}_relaxed"
    raise ValueError(f"Unsupported matrix layout {matrix_layout!r}")
