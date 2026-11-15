# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate recall and projected production memory for the ANN prototype."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ann_service import (
    PRESETS,
    load_or_create_data,
    load_requirements,
    result_summary,
    search_embeddings,
)


def recall_at_k(expected: np.ndarray, observed: np.ndarray) -> float:
    recalls = []
    for exp, obs in zip(expected, observed, strict=True):
        recalls.append(len(set(exp.tolist()) & set(obs.tolist())) / len(exp))
    return float(np.mean(recalls))


def exact_ground_truth(
    corpus: np.ndarray, queries: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Independent exact cosine-search oracle for validation."""
    all_scores: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []
    corpus_t = np.ascontiguousarray(corpus.T)

    for start in range(0, queries.shape[0], 64):
        query_chunk = np.ascontiguousarray(queries[start : start + 64])
        scores = query_chunk @ corpus_t
        top = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        top_scores = np.take_along_axis(scores, top, axis=1)
        order = np.argsort(-top_scores, axis=1)
        all_indices.append(
            np.take_along_axis(top, order, axis=1).astype(np.int64)
        )
        all_scores.append(
            np.take_along_axis(top_scores, order, axis=1).astype(np.float32)
        )

    return np.vstack(all_scores), np.vstack(all_indices)


def projected_index_gb(
    algorithm: str, index_params: dict, requirements: dict
) -> float:
    """Independent production memory projection used by the acceptance check."""
    n = int(requirements["production_vectors"])
    dim = int(requirements["production_dimensions"])
    algorithm = algorithm.lower()

    if "ivf_pq" in algorithm or "ivf-pq" in algorithm:
        pq_dim = int(index_params.get("pq_dim", max(1, dim // 4)))
        pq_bits = int(index_params.get("pq_bits", 8))
        n_lists = int(index_params.get("n_lists", max(1, round(n**0.5))))
        return (n * pq_dim * pq_bits / 8 + n * 8 + n_lists * dim * 4) / 1e9

    if "compressed" in algorithm and "cagra" in algorithm:
        pq_dim = int(index_params.get("pq_dim", max(1, dim // 4)))
        pq_bits = int(index_params.get("pq_bits", 8))
        graph_degree = int(index_params.get("graph_degree", 64))
        return (n * pq_dim * pq_bits / 8 + n * graph_degree * 4) / 1e9

    if "cagra" in algorithm:
        graph_degree = int(index_params.get("graph_degree", 64))
        return (n * dim * 4 + n * graph_degree * 4) / 1e9

    if "ivf_flat" in algorithm or "ivf-flat" in algorithm:
        return (n * dim * 4 + n * 8) / 1e9

    return (n * dim * 4) / 1e9


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate ANN recall and memory budget"
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    parser.add_argument("--use-staged-data", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()

    data_dir = args.data_dir or Path(__file__).resolve().parent / "staged-data"
    requirements_path = (
        data_dir / "requirements.json" if args.use_staged_data else None
    )
    requirements = (
        load_requirements(requirements_path)
        if requirements_path
        else load_requirements()
    )
    k = int(requirements["k"])

    corpus, queries = load_or_create_data(
        args.preset, data_dir, args.use_staged_data
    )
    exact_scores, exact_indices = exact_ground_truth(corpus, queries, k)
    result = search_embeddings(corpus, queries, k, requirements)

    scores = np.asarray(result["scores"])
    indices = np.asarray(result["indices"])
    if (
        scores.shape != exact_scores.shape
        or indices.shape != exact_indices.shape
    ):
        raise AssertionError(
            f"expected scores/indices shape {exact_scores.shape}, got {scores.shape}/{indices.shape}"
        )
    if not np.isfinite(scores).all():
        raise AssertionError("scores contain non-finite values")

    recall = recall_at_k(exact_indices, indices)
    summary = result_summary(result, requirements)
    projected_gb = projected_index_gb(
        str(result.get("algorithm", "")),
        dict(result.get("index_params") or {}),
        requirements,
    )
    effective_budget_gb = float(requirements["gpu_memory_budget_gb"]) * float(
        requirements["gpu_memory_headroom_fraction"]
    )
    summary["projected_production_index_gb"] = round(projected_gb, 3)
    summary["effective_memory_budget_gb"] = round(effective_budget_gb, 3)
    summary["memory_budget_ok"] = projected_gb <= effective_budget_gb
    summary["recall_at_k"] = round(recall, 6)
    print(json.dumps(summary, indent=2, sort_keys=True))

    min_recall = float(requirements["recall_at_k_min"])
    if recall < min_recall:
        raise AssertionError(
            f"recall@{k} {recall:.4f} is below required {min_recall:.4f}"
        )
    if not summary["memory_budget_ok"]:
        raise AssertionError(
            "projected production index memory "
            f"{summary['projected_production_index_gb']} GB exceeds effective budget "
            f"{summary['effective_memory_budget_gb']} GB"
        )


if __name__ == "__main__":
    main()
