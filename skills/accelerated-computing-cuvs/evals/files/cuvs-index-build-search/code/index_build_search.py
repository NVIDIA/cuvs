# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build, serialize, reload, and search a nearest-neighbors index.

Demonstrates the full index lifecycle: build an index on corpus data,
save it to disk, reload it in a fresh context, search, and verify that
results match the original index.
"""

import os
import pickle
import tempfile
import time

import numpy as np
from sklearn.neighbors import NearestNeighbors


def generate_data(
    n_corpus: int, n_queries: int, dim: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate corpus and query embeddings."""
    rng = np.random.RandomState(seed)
    corpus = rng.random((n_corpus, dim)).astype(np.float32)
    queries = rng.random((n_queries, dim)).astype(np.float32)
    return corpus, queries


def build_index(corpus: np.ndarray, k: int) -> NearestNeighbors:
    """Build a nearest-neighbors index."""
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
    nn.fit(corpus)
    return nn


def search_index(
    index: NearestNeighbors, queries: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Search the index for k nearest neighbors."""
    return index.kneighbors(queries, n_neighbors=k)


def save_index(index: NearestNeighbors, filepath: str) -> None:
    """Save the index to disk using pickle."""
    with open(filepath, "wb") as f:
        pickle.dump(index, f)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  Saved index to {filepath} ({size_mb:.1f} MB)")


def load_index(filepath: str) -> NearestNeighbors:
    """Load the index from disk."""
    with open(filepath, "rb") as f:
        index = pickle.load(f)
    print(f"  Loaded index from {filepath}")
    return index


def verify_results(
    distances1: np.ndarray,
    indices1: np.ndarray,
    distances2: np.ndarray,
    indices2: np.ndarray,
) -> bool:
    """Verify that two sets of search results are identical."""
    indices_match = np.array_equal(indices1, indices2)
    distances_close = np.allclose(distances1, distances2, atol=1e-6)
    return indices_match and distances_close


def main():
    # Parameters
    n_corpus = 50_000
    n_queries = 1_000
    dim = 128
    k = 10

    print(f"Corpus: {n_corpus}, Queries: {n_queries}, Dim: {dim}, k: {k}")
    print()

    # Generate data
    print("Generating embeddings...")
    corpus, queries = generate_data(n_corpus, n_queries, dim)

    # Step 1: Build index
    print("Step 1: Building index...")
    t0 = time.time()
    index = build_index(corpus, k)
    build_time = time.time() - t0
    print(f"  Build time: {build_time:.3f}s")

    # Step 2: Search original index
    print("Step 2: Searching original index...")
    t0 = time.time()
    distances_orig, indices_orig = search_index(index, queries, k)
    search_time_orig = time.time() - t0
    print(f"  Search time: {search_time_orig:.3f}s")

    # Step 3: Save index to disk
    print("Step 3: Saving index to disk...")
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = os.path.join(tmpdir, "nn_index.pkl")
        t0 = time.time()
        save_index(index, index_path)
        save_time = time.time() - t0
        print(f"  Save time: {save_time:.3f}s")

        # Step 4: Reload index
        print("Step 4: Reloading index from disk...")
        t0 = time.time()
        loaded_index = load_index(index_path)
        load_time = time.time() - t0
        print(f"  Load time: {load_time:.3f}s")

        # Step 5: Search reloaded index
        print("Step 5: Searching reloaded index...")
        t0 = time.time()
        distances_loaded, indices_loaded = search_index(
            loaded_index, queries, k
        )
        search_time_loaded = time.time() - t0
        print(f"  Search time: {search_time_loaded:.3f}s")

    # Step 6: Verify consistency
    print("Step 6: Verifying result consistency...")
    match = verify_results(
        distances_orig, indices_orig, distances_loaded, indices_loaded
    )
    print(f"  Results match: {match}")

    # Summary
    print("\nSummary:")
    print(f"  Index build:  {build_time:.3f}s")
    print(f"  Save to disk: {save_time:.3f}s")
    print(f"  Load from disk: {load_time:.3f}s")
    print(f"  Search (original):  {search_time_orig:.3f}s")
    print(f"  Search (reloaded):  {search_time_loaded:.3f}s")
    print(f"  Avg nearest distance: {distances_orig[:, 0].mean():.4f}")


if __name__ == "__main__":
    main()
