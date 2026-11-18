# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact top-k search for a small catalog.

The workload file next to this script is the production shape. A GPU is
installed on the host. Neighbors must match an exact brute-force result.
"""

import json
from pathlib import Path

import numpy as np


def load_workload() -> dict:
    return json.loads(Path(__file__).with_name("workload.json").read_text())


def make_vectors(n_rows: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_rows, dim), dtype=np.float32)


def exact_topk(corpus: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    corpus_sq = np.einsum("nd,nd->n", corpus, corpus)
    query_sq = np.einsum("qd,qd->q", queries, queries)
    distances = (
        query_sq[:, None] + corpus_sq[None, :] - 2.0 * (queries @ corpus.T)
    )
    chosen = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(queries.shape[0])[:, None]
    order = np.argsort(distances[rows, chosen], axis=1)
    return chosen[rows, order]


def search(corpus: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    return exact_topk(corpus, queries, k)


def main() -> None:
    spec = load_workload()
    corpus = make_vectors(spec["n_vectors"], spec["dimensions"], seed=42)
    queries = make_vectors(
        spec["query_batch_size"], spec["dimensions"], seed=7
    )
    neighbors = search(corpus, queries, spec["k"])
    expected = exact_topk(corpus, queries, spec["k"])
    if not np.array_equal(
        np.sort(neighbors, axis=1), np.sort(expected, axis=1)
    ):
        raise SystemExit("results are not the exact top-k")
    print(
        f"exact top-{spec['k']} ok corpus={spec['n_vectors']} "
        f"dim={spec['dimensions']} batch={spec['query_batch_size']} "
        f"gpu_available={spec['gpu_available']}"
    )


if __name__ == "__main__":
    main()
