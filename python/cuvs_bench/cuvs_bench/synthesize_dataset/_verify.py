#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
"""Verify that the cluster-probing GT approximation is accurate enough.

Generates queries from a synthetic dataset, computes both the cheap
nprobe GT and the exact GT, and reports recall of the former
against the latter.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ._config import ClusterConfig
from ._ground_truth import (
    compute_groundtruth_exact,
    compute_groundtruth_nprobe,
    generate_queries,
)


def _recall_with_ties(
    pred_indices: np.ndarray,
    pred_distances: np.ndarray,
    gt_indices: np.ndarray,
    gt_distances: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> Tuple[float, List[int]]:
    """Recall@k that counts a tied distance as a match.

    Two indices at the same distance are interchangeable for ANN purposes, so
    we count a hit when either the index matches or there exists a GT hit
    with the same distance.
    """
    nqueries, k = gt_indices.shape
    matches = 0
    mismatched: List[int] = []

    for i in range(nqueries):
        gt_idx_set = set(gt_indices[i].tolist())
        gt_dists = gt_distances[i]

        per_query = 0
        for j in range(k):
            p_idx = int(pred_indices[i, j])
            p_dist = float(pred_distances[i, j])
            if p_idx in gt_idx_set:
                per_query += 1
            elif np.any(np.isclose(gt_dists, p_dist, rtol=rtol, atol=atol)):
                per_query += 1
        matches += per_query
        if per_query < k:
            mismatched.append(i)

    return matches / (nqueries * k), mismatched


def verify_groundtruth(
    config: ClusterConfig,
    total_rows: int,
    nqueries: int,
    k: int,
    nprobes: int,
    metric: str = "sqeuclidean",
) -> dict:
    """Compare nprobe GT vs exact GT at small scale.

    Parameters
    ----------
    config : ClusterConfig
    total_rows : int
        Synthetic dataset size to verify against.
    nqueries : int
        Number of queries to draw.
    k : int
        Number of neighbors.
    nprobes : int
        Number of clusters each query probes for the nprobe GT.

    Returns
    -------
    dict with:
        - ``recall``: tie-aware recall of nprobe GT against exact GT
        - ``simple_recall``: naive index-set recall (no tie awareness)
        - ``mismatched_queries``: indices of queries that didn't match
        - ``nprobe_timing``, ``exact_gt_timing``: per-step timing breakdowns
    """
    print(
        f"Verifying nprobe GT against exact GT at "
        f"total_rows={total_rows:,}, nqueries={nqueries}, "
        f"k={k}, nprobes={nprobes}..."
    )

    queries = generate_queries(
        nqueries=nqueries,
        total_rows=total_rows,
        config=config,
    )

    nprobe_idx, nprobe_dist, nprobe_timing = compute_groundtruth_nprobe(
        queries=queries,
        total_rows=total_rows,
        config=config,
        k=k,
        nprobes=nprobes,
        metric=metric,
    )

    exact_idx, exact_dist, exact_gt_timing = compute_groundtruth_exact(
        queries=queries,
        total_rows=total_rows,
        config=config,
        k=k,
        metric=metric,
    )

    recall, mismatched = _recall_with_ties(
        nprobe_idx, nprobe_dist, exact_idx, exact_dist
    )

    # Naive recall without tie awareness, useful as a sanity check.
    simple_hits = 0
    for i in range(nqueries):
        simple_hits += len(set(nprobe_idx[i]) & set(exact_idx[i]))
    simple_recall = simple_hits / (nqueries * k)

    print(
        f"  Recall (with ties): {recall:.4f}  "
        f"({len(mismatched)}/{nqueries} imperfect queries)"
    )
    print(f"  Recall (naive):     {simple_recall:.4f}")

    return {
        "recall": recall,
        "simple_recall": simple_recall,
        "mismatched_queries": mismatched,
        "nprobe_timing": nprobe_timing,
        "exact_gt_timing": exact_gt_timing,
        "total_rows": total_rows,
        "nqueries": nqueries,
        "k": k,
        "nprobes": nprobes,
    }
