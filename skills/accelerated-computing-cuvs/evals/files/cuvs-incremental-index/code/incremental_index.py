# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Incremental index building and search.

Builds an initial nearest-neighbors index, then adds new vectors in
batches, searching after each update to verify that newly added vectors
are discoverable.
"""

import time

import numpy as np
from sklearn.neighbors import NearestNeighbors


def generate_initial_corpus(
    n_vectors: int, dim: int, seed: int = 42
) -> np.ndarray:
    """Generate the initial corpus of embeddings."""
    rng = np.random.RandomState(seed)
    return rng.random((n_vectors, dim)).astype(np.float32)


def generate_new_batch(n_vectors: int, dim: int, seed: int) -> np.ndarray:
    """Generate a batch of new vectors to add to the index."""
    rng = np.random.RandomState(seed)
    return rng.random((n_vectors, dim)).astype(np.float32)


def generate_queries(n_queries: int, dim: int, seed: int = 999) -> np.ndarray:
    """Generate query vectors."""
    rng = np.random.RandomState(seed)
    return rng.random((n_queries, dim)).astype(np.float32)


class IncrementalIndex:
    """Wrapper around sklearn NearestNeighbors that supports incremental adds."""

    def __init__(self, k: int = 10, metric: str = "euclidean"):
        self.k = k
        self.metric = metric
        self.corpus = None
        self._index = None

    def build(self, vectors: np.ndarray) -> None:
        """Build the initial index."""
        self.corpus = vectors.copy()
        self._rebuild_index()

    def add(self, new_vectors: np.ndarray) -> None:
        """Add new vectors to the index (rebuilds internally)."""
        self.corpus = np.vstack([self.corpus, new_vectors])
        self._rebuild_index()

    def search(
        self, queries: np.ndarray, k: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search the index for k nearest neighbors."""
        if k is None:
            k = self.k
        return self._index.kneighbors(queries, n_neighbors=k)

    def _rebuild_index(self) -> None:
        """Rebuild the sklearn index on the current corpus."""
        self._index = NearestNeighbors(
            n_neighbors=self.k, algorithm="auto", metric=self.metric
        )
        self._index.fit(self.corpus)

    @property
    def size(self) -> int:
        return self.corpus.shape[0] if self.corpus is not None else 0


def verify_new_vectors_found(
    indices: np.ndarray, new_start_idx: int, new_end_idx: int
) -> float:
    """Check what fraction of queries found at least one newly added vector."""
    found_new = 0
    for q in range(indices.shape[0]):
        if any(new_start_idx <= idx < new_end_idx for idx in indices[q]):
            found_new += 1
    return found_new / indices.shape[0]


def main():
    # Parameters
    initial_size = 50_000
    dim = 128
    k = 10
    n_queries = 500
    batch_sizes = [5_000, 10_000, 10_000]

    print(f"Initial corpus: {initial_size} vectors, Dim: {dim}, k: {k}")
    print(f"Batches to add: {batch_sizes}")
    print()

    # Generate initial corpus and queries
    print("Generating initial corpus...")
    corpus = generate_initial_corpus(initial_size, dim)

    print("Generating queries...")
    queries = generate_queries(n_queries, dim)

    # Build initial index
    print("Building initial index...")
    index = IncrementalIndex(k=k)
    t0 = time.time()
    index.build(corpus)
    build_time = time.time() - t0
    print(f"Initial build time: {build_time:.3f}s (size={index.size})")

    # Search initial index
    print("Searching initial index...")
    t0 = time.time()
    distances_init, _indices_init = index.search(queries)
    search_time = time.time() - t0
    print(f"Search time: {search_time:.3f}s")
    print(f"Avg nearest distance: {distances_init[:, 0].mean():.4f}")

    # Incrementally add batches
    for batch_num, batch_size in enumerate(batch_sizes):
        print(f"\n--- Adding batch {batch_num + 1}: {batch_size} vectors ---")

        new_start = index.size
        new_vectors = generate_new_batch(batch_size, dim, seed=100 + batch_num)

        t0 = time.time()
        index.add(new_vectors)
        add_time = time.time() - t0
        new_end = index.size
        print(f"Add time: {add_time:.3f}s (new size={index.size})")

        # Search updated index
        t0 = time.time()
        distances, indices = index.search(queries)
        search_time = time.time() - t0
        print(f"Search time: {search_time:.3f}s")

        # Verify new vectors are discoverable
        frac_found = verify_new_vectors_found(indices, new_start, new_end)
        print(f"Queries finding new vectors: {100 * frac_found:.1f}%")
        print(f"Avg nearest distance: {distances[:, 0].mean():.4f}")

    # Final comparison
    print("\nFinal summary:")
    print(f"  Initial index size: {initial_size}")
    print(f"  Final index size: {index.size}")
    print(f"  Initial avg nearest dist: {distances_init[:, 0].mean():.4f}")
    print(f"  Final avg nearest dist: {distances[:, 0].mean():.4f}")
    print(
        f"  Improvement: {distances_init[:, 0].mean() / distances[:, 0].mean():.2f}x closer"
    )


if __name__ == "__main__":
    main()
