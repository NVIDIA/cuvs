# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Top-k search for a wide-vector, large-batch catalog.

workload.json is the production shape. The default run searches only
smoke_queries rows so the process finishes; production batches are larger.
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


def recall_at_k(expected: np.ndarray, actual: np.ndarray) -> float:
    matched = 0
    for row_expected, row_actual in zip(expected, actual):
        matched += len(set(row_expected.tolist()) & set(row_actual.tolist()))
    return matched / expected.size


def search(corpus: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    return exact_topk(corpus, queries, k)


def main() -> None:
    spec = load_workload()
    corpus = make_vectors(spec["n_vectors"], spec["dimensions"], seed=42)
    queries = make_vectors(spec["smoke_queries"], spec["dimensions"], seed=7)
    neighbors = search(corpus, queries, spec["k"])
    expected = exact_topk(corpus, queries, spec["k"])
    recall = recall_at_k(expected, neighbors)
    print(
        f"smoke recall@{spec['k']}={recall:.3f} corpus={spec['n_vectors']} "
        f"dim={spec['dimensions']} smoke_queries={spec['smoke_queries']} "
        f"production_batch={spec['query_batch_size']}"
    )
    if recall < spec["recall_at_k_min"]:
        raise SystemExit("recall below workload minimum")


if __name__ == "__main__":
    main()
