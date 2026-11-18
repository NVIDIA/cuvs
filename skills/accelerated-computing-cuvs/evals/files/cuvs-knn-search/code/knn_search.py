# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Basic k-nearest neighbors search using sklearn NearestNeighbors.

Generates random embeddings, builds a KNN index, queries it, and
reports recall against brute-force ground truth.
"""

import time

import numpy as np
from sklearn.neighbors import NearestNeighbors


def generate_embeddings(
    n_vectors: int, dim: int, seed: int = 42
) -> np.ndarray:
    """Generate random embedding vectors."""
    rng = np.random.RandomState(seed)
    return rng.random((n_vectors, dim)).astype(np.float32)


def brute_force_knn(
    corpus: np.ndarray, queries: np.ndarray, k: int
) -> np.ndarray:
    """Compute exact KNN using brute force for ground truth."""
    nn = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="euclidean")
    nn.fit(corpus)
    _distances, indices = nn.kneighbors(queries)
    return indices


def approximate_knn(
    corpus: np.ndarray, queries: np.ndarray, k: int
) -> np.ndarray:
    """Find k-nearest neighbors using sklearn ball_tree."""
    nn = NearestNeighbors(
        n_neighbors=k, algorithm="ball_tree", metric="euclidean"
    )
    nn.fit(corpus)
    _distances, indices = nn.kneighbors(queries)
    return indices


def compute_recall(ground_truth: np.ndarray, predictions: np.ndarray) -> float:
    """Compute recall@k: fraction of true neighbors found."""
    n_queries = ground_truth.shape[0]
    recall_sum = 0.0
    for i in range(n_queries):
        gt_set = set(ground_truth[i])
        pred_set = set(predictions[i])
        recall_sum += len(gt_set & pred_set) / len(gt_set)
    return recall_sum / n_queries


def main():
    # Parameters
    n_corpus = 50_000
    n_queries = 1_000
    dim = 128
    k = 10

    print(
        f"Generating {n_corpus} corpus vectors and {n_queries} query vectors "
        f"of dimension {dim}..."
    )
    corpus = generate_embeddings(n_corpus, dim, seed=42)
    queries = generate_embeddings(n_queries, dim, seed=123)

    # Ground truth via brute force
    print("Computing brute-force ground truth...")
    t0 = time.time()
    ground_truth = brute_force_knn(corpus, queries, k)
    brute_time = time.time() - t0
    print(f"Brute-force search took {brute_time:.3f}s")

    # Approximate KNN search
    print("Running approximate KNN search...")
    t0 = time.time()
    predictions = approximate_knn(corpus, queries, k)
    approx_time = time.time() - t0
    print(f"Approximate search took {approx_time:.3f}s")

    # Evaluate recall
    recall = compute_recall(ground_truth, predictions)
    print("\nResults:")
    print(f"  Recall@{k}: {recall:.4f}")
    print(f"  Speedup: {brute_time / approx_time:.2f}x")
    print(f"  Corpus size: {n_corpus}, Queries: {n_queries}, Dim: {dim}")


if __name__ == "__main__":
    main()
