# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batched top-k nearest neighbor search over a large embedding corpus.

Splits queries into batches, searches for top-k neighbors per query,
and aggregates timing statistics across batches.
"""

import time

import numpy as np
from sklearn.neighbors import NearestNeighbors


def generate_corpus(n_vectors: int, dim: int, seed: int = 42) -> np.ndarray:
    """Generate a large corpus of embedding vectors."""
    rng = np.random.RandomState(seed)
    return rng.random((n_vectors, dim)).astype(np.float32)


def generate_queries(n_queries: int, dim: int, seed: int = 99) -> np.ndarray:
    """Generate query embedding vectors."""
    rng = np.random.RandomState(seed)
    return rng.random((n_queries, dim)).astype(np.float32)


def build_index(corpus: np.ndarray) -> NearestNeighbors:
    """Build a sklearn nearest neighbors index on the corpus."""
    nn = NearestNeighbors(n_neighbors=10, algorithm="auto", metric="euclidean")
    nn.fit(corpus)
    return nn


def batch_search(
    index: NearestNeighbors,
    queries: np.ndarray,
    k: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Search the index in batches, returning all distances and indices."""
    n_queries = queries.shape[0]
    all_distances = []
    all_indices = []

    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        batch = queries[start:end]
        distances, indices = index.kneighbors(batch, n_neighbors=k)
        all_distances.append(distances)
        all_indices.append(indices)

    return np.vstack(all_distances), np.vstack(all_indices)


def main():
    # Parameters
    n_corpus = 100_000
    n_queries = 5_000
    dim = 128
    k = 10
    batch_size = 500

    print(f"Corpus: {n_corpus} vectors, Queries: {n_queries}, Dim: {dim}")
    print(f"Top-k: {k}, Batch size: {batch_size}")
    print()

    # Generate data
    print("Generating corpus embeddings...")
    corpus = generate_corpus(n_corpus, dim)

    print("Generating query embeddings...")
    queries = generate_queries(n_queries, dim)

    # Build index
    print("Building index...")
    t0 = time.time()
    index = build_index(corpus)
    build_time = time.time() - t0
    print(f"Index build time: {build_time:.3f}s")

    # Batch search
    print(f"Searching {n_queries} queries in batches of {batch_size}...")
    t0 = time.time()
    distances, indices = batch_search(index, queries, k, batch_size)
    search_time = time.time() - t0
    print(f"Total search time: {search_time:.3f}s")

    # Report statistics
    n_batches = (n_queries + batch_size - 1) // batch_size
    print("\nResults:")
    print(f"  Total batches processed: {n_batches}")
    print(f"  Queries per second: {n_queries / search_time:.0f}")
    print(f"  Avg distance to nearest neighbor: {distances[:, 0].mean():.4f}")
    print(f"  Avg distance to {k}th neighbor: {distances[:, -1].mean():.4f}")
    print(f"  Result shape: {indices.shape}")


if __name__ == "__main__":
    main()
