# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Semantic search retrieval pipeline.

Simulates a retrieval system with corpus embeddings and query embeddings.
Builds an index on the corpus, retrieves top-k documents per query, and
evaluates precision@k against synthetic ground truth relevance labels.
"""

import time

import numpy as np
from sklearn.neighbors import NearestNeighbors


def generate_corpus_and_queries(
    n_corpus: int,
    n_queries: int,
    dim: int,
    n_relevant: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[set[int]]]:
    """Generate corpus embeddings, query embeddings, and ground truth relevance.

    Each query is constructed to be close to n_relevant corpus vectors
    (its "relevant documents"), so we have a known ground truth.
    """
    rng = np.random.RandomState(seed)
    corpus = rng.randn(n_corpus, dim).astype(np.float32)

    queries = np.empty((n_queries, dim), dtype=np.float32)
    ground_truth = []

    for q in range(n_queries):
        # Pick n_relevant random corpus vectors as relevant docs
        relevant_ids = set(
            rng.choice(n_corpus, size=n_relevant, replace=False)
        )
        ground_truth.append(relevant_ids)

        # Make query = mean of relevant docs + small noise
        relevant_vecs = corpus[list(relevant_ids)]
        queries[q] = (
            relevant_vecs.mean(axis=0)
            + rng.randn(dim).astype(np.float32) * 0.1
        )

    return corpus, queries, ground_truth


def build_retrieval_index(corpus: np.ndarray) -> NearestNeighbors:
    """Build a nearest-neighbors index on the corpus."""
    nn = NearestNeighbors(n_neighbors=50, algorithm="auto", metric="euclidean")
    nn.fit(corpus)
    return nn


def retrieve_top_k(
    index: NearestNeighbors, queries: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Retrieve top-k corpus documents for each query."""
    distances, indices = index.kneighbors(queries, n_neighbors=k)
    return distances, indices


def precision_at_k(
    retrieved: np.ndarray, ground_truth: list[set[int]], k: int
) -> float:
    """Compute mean precision@k across all queries."""
    n_queries = retrieved.shape[0]
    precisions = []
    for q in range(n_queries):
        top_k = set(retrieved[q, :k])
        relevant = ground_truth[q]
        precisions.append(len(top_k & relevant) / k)
    return np.mean(precisions)


def recall_at_k(
    retrieved: np.ndarray, ground_truth: list[set[int]], k: int
) -> float:
    """Compute mean recall@k across all queries."""
    n_queries = retrieved.shape[0]
    recalls = []
    for q in range(n_queries):
        top_k = set(retrieved[q, :k])
        relevant = ground_truth[q]
        recalls.append(len(top_k & relevant) / len(relevant))
    return np.mean(recalls)


def main():
    # Parameters
    n_corpus = 50_000
    n_queries = 500
    dim = 128
    n_relevant = 10
    k_values = [1, 5, 10, 20, 50]

    print(f"Corpus: {n_corpus} docs, Queries: {n_queries}, Dim: {dim}")
    print(f"Relevant docs per query: {n_relevant}")
    print()

    # Generate data
    print("Generating corpus, queries, and ground truth relevance...")
    corpus, queries, ground_truth = generate_corpus_and_queries(
        n_corpus, n_queries, dim, n_relevant
    )

    # Build index
    print("Building retrieval index...")
    t0 = time.time()
    index = build_retrieval_index(corpus)
    build_time = time.time() - t0
    print(f"Index build time: {build_time:.3f}s")

    # Retrieve
    max_k = max(k_values)
    print(f"Retrieving top-{max_k} documents for {n_queries} queries...")
    t0 = time.time()
    distances, retrieved = retrieve_top_k(index, queries, max_k)
    search_time = time.time() - t0
    print(f"Search time: {search_time:.3f}s")

    # Evaluate at multiple k values
    print("\nRetrieval metrics:")
    print(f"  {'k':>4s}  {'Precision@k':>12s}  {'Recall@k':>10s}")
    print(f"  {'-' * 4}  {'-' * 12}  {'-' * 10}")
    for k in k_values:
        p_at_k = precision_at_k(retrieved, ground_truth, k)
        r_at_k = recall_at_k(retrieved, ground_truth, k)
        print(f"  {k:4d}  {p_at_k:12.4f}  {r_at_k:10.4f}")

    # Summary
    print(
        f"\nAvg distance to nearest retrieved doc: {distances[:, 0].mean():.4f}"
    )
    print(f"Queries per second: {n_queries / search_time:.0f}")


if __name__ == "__main__":
    main()
