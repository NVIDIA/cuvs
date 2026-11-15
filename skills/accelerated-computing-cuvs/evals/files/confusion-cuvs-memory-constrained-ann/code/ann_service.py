# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Memory-constrained embedding retrieval prototype.

The current implementation uses exact CPU cosine search. It is simple and
accurate, but it is not a viable production ANN serving path for the requested
scale and memory budget.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
REQUIREMENTS_PATH = ROOT / "service_requirements.json"

PRESETS: dict[str, dict[str, int]] = {
    "smoke": {
        "n_corpus": 12_000,
        "n_queries": 128,
        "dim": 128,
        "n_topics": 512,
        "seed": 17,
    },
    "benchmark": {
        "n_corpus": 120_000,
        "n_queries": 512,
        "dim": 128,
        "n_topics": 2_048,
        "seed": 23,
    },
}


def load_requirements(path: Path = REQUIREMENTS_PATH) -> dict[str, Any]:
    return json.loads(path.read_text())


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def make_embeddings(
    *,
    n_corpus: int,
    n_queries: int,
    dim: int,
    n_topics: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic clustered embeddings with non-trivial neighbors."""
    rng = np.random.default_rng(seed)
    centers = normalize_rows(
        rng.normal(size=(n_topics, dim)).astype(np.float32)
    )

    corpus_topics = rng.integers(0, n_topics, size=n_corpus)
    query_topics = rng.choice(corpus_topics, size=n_queries, replace=False)

    corpus = centers[corpus_topics] + 0.045 * rng.normal(
        size=(n_corpus, dim)
    ).astype(np.float32)
    queries = centers[query_topics] + 0.035 * rng.normal(
        size=(n_queries, dim)
    ).astype(np.float32)
    return normalize_rows(corpus.astype(np.float32)), normalize_rows(
        queries.astype(np.float32)
    )


def stage_data(preset: str, data_dir: Path) -> None:
    cfg = PRESETS[preset]
    corpus, queries = make_embeddings(**cfg)
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "corpus.npy", corpus)
    np.save(data_dir / "queries.npy", queries)
    (data_dir / "requirements.json").write_text(
        json.dumps(load_requirements(), indent=2)
    )
    print(
        f"staged corpus={corpus.shape} queries={queries.shape} into {data_dir}"
    )


def load_or_create_data(
    preset: str, data_dir: Path | None, use_staged_data: bool
) -> tuple[np.ndarray, np.ndarray]:
    if use_staged_data:
        if data_dir is None:
            raise ValueError("--data-dir is required with --use-staged-data")
        return np.load(data_dir / "corpus.npy"), np.load(
            data_dir / "queries.npy"
        )
    return make_embeddings(**PRESETS[preset])


def exact_cosine_search(
    corpus: np.ndarray,
    queries: np.ndarray,
    k: int,
    *,
    chunk_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Return top-k cosine scores and indices using chunked CPU matrix multiply."""
    all_scores: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []

    corpus_t = np.ascontiguousarray(corpus.T)
    for start in range(0, queries.shape[0], chunk_size):
        query_chunk = np.ascontiguousarray(queries[start : start + chunk_size])
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


def estimate_production_index_gb(
    algorithm: str,
    index_params: dict[str, Any],
    requirements: dict[str, Any],
) -> float:
    """Estimate production index size from public constraints and index metadata."""
    n = int(requirements["production_vectors"])
    dim = int(requirements["production_dimensions"])
    algorithm = algorithm.lower()

    if "ivf_pq" in algorithm or "ivf-pq" in algorithm:
        pq_dim = int(index_params.get("pq_dim", max(1, dim // 4)))
        pq_bits = int(index_params.get("pq_bits", 8))
        n_lists = int(index_params.get("n_lists", max(1, round(n**0.5))))
        code_bytes = n * pq_dim * pq_bits / 8
        id_bytes = n * 8
        centroid_bytes = n_lists * dim * 4
        return (code_bytes + id_bytes + centroid_bytes) / 1e9

    if "compressed" in algorithm and "cagra" in algorithm:
        pq_dim = int(index_params.get("pq_dim", max(1, dim // 4)))
        pq_bits = int(index_params.get("pq_bits", 8))
        graph_degree = int(index_params.get("graph_degree", 64))
        code_bytes = n * pq_dim * pq_bits / 8
        graph_bytes = n * graph_degree * 4
        return (code_bytes + graph_bytes) / 1e9

    if "cagra" in algorithm:
        graph_degree = int(index_params.get("graph_degree", 64))
        vector_bytes = n * dim * 4
        graph_bytes = n * graph_degree * 4
        return (vector_bytes + graph_bytes) / 1e9

    if "ivf_flat" in algorithm or "ivf-flat" in algorithm:
        vector_bytes = n * dim * 4
        id_bytes = n * 8
        return (vector_bytes + id_bytes) / 1e9

    # Exact search, brute force, sklearn, and unreported algorithms are treated
    # as full-vector storage for production memory planning.
    return (n * dim * 4) / 1e9


def search_embeddings(
    corpus: np.ndarray,
    queries: np.ndarray,
    k: int,
    requirements: dict[str, Any],
) -> dict[str, Any]:
    """Search embeddings and report enough metadata for validation."""
    started = time.perf_counter()
    scores, indices = exact_cosine_search(corpus, queries, k)
    elapsed = time.perf_counter() - started
    algorithm = "exact_cpu"
    index_params: dict[str, Any] = {}

    return {
        "scores": scores,
        "indices": indices,
        "algorithm": algorithm,
        "index_params": index_params,
        "elapsed_seconds": elapsed,
        "projected_production_index_gb": estimate_production_index_gb(
            algorithm, index_params, requirements
        ),
    }


def result_summary(
    result: dict[str, Any], requirements: dict[str, Any]
) -> dict[str, Any]:
    budget = float(requirements["gpu_memory_budget_gb"]) * float(
        requirements["gpu_memory_headroom_fraction"]
    )
    return {
        "algorithm": result["algorithm"],
        "index_params": result["index_params"],
        "elapsed_seconds": round(float(result["elapsed_seconds"]), 6),
        "result_shape": list(result["indices"].shape),
        "projected_production_index_gb": round(
            float(result["projected_production_index_gb"]), 3
        ),
        "effective_memory_budget_gb": round(budget, 3),
        "memory_budget_ok": float(result["projected_production_index_gb"])
        <= budget,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embedding retrieval prototype"
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    parser.add_argument("--stage-data", action="store_true")
    parser.add_argument("--use-staged-data", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()

    data_dir = args.data_dir or ROOT / "staged-data"
    if args.stage_data:
        stage_data(args.preset, data_dir)
        return

    requirements = (
        load_requirements(data_dir / "requirements.json")
        if args.use_staged_data
        else load_requirements()
    )
    corpus, queries = load_or_create_data(
        args.preset, data_dir, args.use_staged_data
    )
    result = search_embeddings(
        corpus, queries, int(requirements["k"]), requirements
    )
    print(
        json.dumps(
            result_summary(result, requirements), indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
