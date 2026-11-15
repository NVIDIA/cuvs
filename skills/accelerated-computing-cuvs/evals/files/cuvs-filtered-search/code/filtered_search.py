# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vector search with metadata filtering.

Generates vectors with categorical metadata, performs nearest-neighbor
search with a category filter constraint, returning only results that
match the desired category.
"""

import time

import numpy as np
from sklearn.neighbors import NearestNeighbors


def generate_vectors_with_metadata(
    n_vectors: int, dim: int, n_categories: int = 5, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate vectors with random category labels.

    Returns (vectors, categories) where categories is an int array.
    """
    rng = np.random.RandomState(seed)
    vectors = rng.random((n_vectors, dim)).astype(np.float32)
    categories = rng.randint(0, n_categories, size=n_vectors)
    return vectors, categories


def generate_queries(n_queries: int, dim: int, seed: int = 99) -> np.ndarray:
    """Generate query vectors."""
    rng = np.random.RandomState(seed)
    return rng.random((n_queries, dim)).astype(np.float32)


def filtered_knn_search(
    corpus: np.ndarray,
    categories: np.ndarray,
    queries: np.ndarray,
    target_category: int,
    k: int,
    overfetch_factor: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Find k nearest neighbors that match the target category.

    Strategy: over-fetch k * overfetch_factor neighbors, then post-filter
    to keep only those matching the target category.
    """
    fetch_k = min(k * overfetch_factor, corpus.shape[0])

    nn = NearestNeighbors(
        n_neighbors=fetch_k, algorithm="auto", metric="euclidean"
    )
    nn.fit(corpus)
    distances, indices = nn.kneighbors(queries)

    # Post-filter: keep only results matching target category
    n_queries = queries.shape[0]
    filtered_distances = np.full((n_queries, k), np.inf, dtype=np.float32)
    filtered_indices = np.full((n_queries, k), -1, dtype=np.int64)

    for q in range(n_queries):
        count = 0
        for j in range(fetch_k):
            if count >= k:
                break
            idx = indices[q, j]
            if categories[idx] == target_category:
                filtered_distances[q, count] = distances[q, j]
                filtered_indices[q, count] = idx
                count += 1

    return filtered_distances, filtered_indices


def unfiltered_knn_search(
    corpus: np.ndarray, queries: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Standard KNN search without filtering (for comparison)."""
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
    nn.fit(corpus)
    return nn.kneighbors(queries)


def main():
    # Parameters
    n_corpus = 50_000
    n_queries = 500
    dim = 128
    n_categories = 5
    target_category = 2
    k = 10

    print(f"Corpus: {n_corpus} vectors, Dim: {dim}")
    print(
        f"Categories: {n_categories}, Target filter: category={target_category}"
    )
    print(f"Queries: {n_queries}, k: {k}")
    print()

    # Generate data
    print("Generating vectors with metadata...")
    corpus, categories = generate_vectors_with_metadata(
        n_corpus, dim, n_categories
    )
    queries = generate_queries(n_queries, dim)

    n_target = np.sum(categories == target_category)
    print(
        f"Vectors in target category: {n_target} "
        f"({100 * n_target / n_corpus:.1f}%)"
    )

    # Filtered search
    print(f"\nRunning filtered search (category={target_category})...")
    t0 = time.time()
    filt_dist, filt_idx = filtered_knn_search(
        corpus, categories, queries, target_category, k
    )
    filtered_time = time.time() - t0
    print(f"Filtered search time: {filtered_time:.3f}s")

    # Verify all results match the filter
    valid_results = filt_idx[filt_idx >= 0]
    all_match = np.all(categories[valid_results] == target_category)
    filled_ratio = np.mean(filt_idx[:, -1] >= 0)

    # Unfiltered search for comparison
    print("Running unfiltered search...")
    t0 = time.time()
    unfilt_dist, _unfilt_idx = unfiltered_knn_search(corpus, queries, k)
    unfiltered_time = time.time() - t0
    print(f"Unfiltered search time: {unfiltered_time:.3f}s")

    # Report
    print("\nResults:")
    print(f"  All filtered results match category: {all_match}")
    print(f"  Queries with full k results: {100 * filled_ratio:.1f}%")
    print(f"  Avg filtered distance (nearest): {filt_dist[:, 0].mean():.4f}")
    print(
        f"  Avg unfiltered distance (nearest): {unfilt_dist[:, 0].mean():.4f}"
    )
    print(
        f"  Distance penalty from filtering: "
        f"{filt_dist[:, 0].mean() / unfilt_dist[:, 0].mean():.2f}x"
    )

    # Show sample results
    print("\nSample filtered results (query 0):")
    for j in range(k):
        idx = filt_idx[0, j]
        if idx >= 0:
            print(
                f"  Rank {j + 1}: vector {idx}, "
                f"category={categories[idx]}, "
                f"distance={filt_dist[0, j]:.4f}"
            )


if __name__ == "__main__":
    main()
