#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
"""Verify that the cluster-probing GT approximation is accurate enough."""

from __future__ import annotations


from ..backends._utils import compute_recall
from ._fingerprint import Fingerprint
from ._generate import generate_queries
from ._ground_truth import (
    compute_groundtruth_exact,
    compute_groundtruth_nprobe,
)


def verify_groundtruth(
    config: Fingerprint,
    total_rows: int,
    nqueries: int,
    k: int,
    nprobes: int,
    metric: str = "sqeuclidean",
) -> dict:
    """Compare nprobe GT vs exact GT.

    Parameters
    ----------
    config : Fingerprint
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
        - ``recall``: recall of nprobe GT against exact GT
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

    nprobe_idx, _, nprobe_timing = compute_groundtruth_nprobe(
        queries=queries,
        total_rows=total_rows,
        config=config,
        k=k,
        nprobes=nprobes,
        metric=metric,
    )

    exact_idx, _, exact_gt_timing = compute_groundtruth_exact(
        queries=queries,
        total_rows=total_rows,
        config=config,
        k=k,
        metric=metric,
    )

    recall = compute_recall(nprobe_idx, exact_idx, k)

    print(f"  Recall: {recall:.4f}")

    return {
        "recall": recall,
        "nprobe_timing": nprobe_timing,
        "exact_gt_timing": exact_gt_timing,
        "total_rows": total_rows,
        "nqueries": nqueries,
        "k": k,
        "nprobes": nprobes,
    }
