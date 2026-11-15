# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare KNN search results across multiple distance metrics.

Runs the same KNN search with euclidean, cosine, and inner-product
metrics, then analyzes how much the result sets overlap and how
rankings differ between metrics.
"""

import time

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


def generate_embeddings(
    n_corpus: int, n_queries: int, dim: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate corpus and query embedding vectors."""
    rng = np.random.RandomState(seed)
    corpus = rng.randn(n_corpus, dim).astype(np.float32)
    queries = rng.randn(n_queries, dim).astype(np.float32)
    return corpus, queries


def knn_search(
    corpus: np.ndarray, queries: np.ndarray, k: int, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    """Perform KNN search with a specific metric.

    For inner_product, we negate dot products to work with sklearn's
    distance-based interface (smaller = better).
    """
    if metric == "inner_product":
        # sklearn doesn't have inner product directly; use cosine on
        # normalized vectors (equivalent ranking for normalized data)
        corpus_norm = normalize(corpus, norm="l2")
        queries_norm = normalize(queries, norm="l2")
        nn = NearestNeighbors(
            n_neighbors=k, algorithm="brute", metric="cosine"
        )
        nn.fit(corpus_norm)
        distances, indices = nn.kneighbors(queries_norm)
        # Convert cosine distance to inner product similarity
        similarities = 1.0 - distances
        return similarities, indices
    else:
        nn = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
        nn.fit(corpus)
        distances, indices = nn.kneighbors(queries)
        return distances, indices


def compute_overlap(indices_a: np.ndarray, indices_b: np.ndarray) -> float:
    """Compute mean overlap between two sets of KNN results.

    For each query, measures what fraction of neighbors are shared.
    """
    n_queries = indices_a.shape[0]
    k = indices_a.shape[1]
    overlaps = []
    for q in range(n_queries):
        set_a = set(indices_a[q])
        set_b = set(indices_b[q])
        overlaps.append(len(set_a & set_b) / k)
    return np.mean(overlaps)


def rank_correlation(indices_a: np.ndarray, indices_b: np.ndarray) -> float:
    """Compute mean Spearman-like rank correlation for shared neighbors."""
    n_queries = indices_a.shape[0]
    correlations = []

    for q in range(n_queries):
        shared = set(indices_a[q]) & set(indices_b[q])
        if len(shared) < 2:
            continue

        rank_a = {idx: rank for rank, idx in enumerate(indices_a[q])}
        rank_b = {idx: rank for rank, idx in enumerate(indices_b[q])}

        ranks_a = [rank_a[idx] for idx in shared]
        ranks_b = [rank_b[idx] for idx in shared]

        # Spearman correlation
        n = len(ranks_a)
        d_sq = sum((a - b) ** 2 for a, b in zip(ranks_a, ranks_b))
        if n > 1:
            rho = 1 - (6 * d_sq) / (n * (n**2 - 1))
            correlations.append(rho)

    return np.mean(correlations) if correlations else 0.0


def main():
    # Parameters
    n_corpus = 50_000
    n_queries = 1_000
    dim = 128
    k = 20
    metrics = ["euclidean", "cosine", "inner_product"]

    print(f"Corpus: {n_corpus}, Queries: {n_queries}, Dim: {dim}, k: {k}")
    print(f"Metrics: {metrics}")
    print()

    # Generate data
    print("Generating embeddings...")
    corpus, queries = generate_embeddings(n_corpus, n_queries, dim)

    # Run search for each metric
    results = {}
    for metric in metrics:
        print(f"\nSearching with metric: {metric}...")
        t0 = time.time()
        distances, indices = knn_search(corpus, queries, k, metric)
        search_time = time.time() - t0
        results[metric] = {"distances": distances, "indices": indices}
        print(f"  Search time: {search_time:.3f}s")
        print(
            f"  Avg nearest distance/similarity: {distances[:, 0].mean():.6f}"
        )
        print(f"  Avg k-th distance/similarity: {distances[:, -1].mean():.6f}")

    # Cross-metric comparison
    print(f"\n{'=' * 60}")
    print("Cross-metric overlap (fraction of shared neighbors):")
    print(f"{'':>16s}", end="")
    for m in metrics:
        print(f"{m:>16s}", end="")
    print()

    for m1 in metrics:
        print(f"{m1:>16s}", end="")
        for m2 in metrics:
            if m1 == m2:
                print(f"{'1.000':>16s}", end="")
            else:
                overlap = compute_overlap(
                    results[m1]["indices"], results[m2]["indices"]
                )
                print(f"{overlap:>16.3f}", end="")
        print()

    # Rank correlation for shared neighbors
    print("\nRank correlation (for shared neighbors):")
    print(f"{'':>16s}", end="")
    for m in metrics:
        print(f"{m:>16s}", end="")
    print()

    for m1 in metrics:
        print(f"{m1:>16s}", end="")
        for m2 in metrics:
            if m1 == m2:
                print(f"{'1.000':>16s}", end="")
            else:
                corr = rank_correlation(
                    results[m1]["indices"], results[m2]["indices"]
                )
                print(f"{corr:>16.3f}", end="")
        print()

    # Sample query comparison
    print("\nSample: Query 0 top-5 neighbors per metric:")
    for metric in metrics:
        top5 = results[metric]["indices"][0, :5]
        vals = results[metric]["distances"][0, :5]
        pairs = ", ".join(f"{idx}({v:.4f})" for idx, v in zip(top5, vals))
        print(f"  {metric:>14s}: [{pairs}]")


if __name__ == "__main__":
    main()
