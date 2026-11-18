# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Near-duplicate vector detection using distance threshold search.

Generates embeddings with injected near-duplicates, finds pairs within
a distance threshold, and groups them into duplicate clusters using
connected components.
"""

import time

import numpy as np
from sklearn.neighbors import NearestNeighbors


def generate_embeddings_with_duplicates(
    n_vectors: int, dim: int, n_dup_groups: int = 200, seed: int = 42
) -> tuple[np.ndarray, list[set[int]]]:
    """Generate embeddings with injected near-duplicate groups.

    Returns the embeddings and the ground-truth duplicate groups.
    """
    rng = np.random.RandomState(seed)
    embeddings = rng.random((n_vectors, dim)).astype(np.float32)

    # Inject near-duplicates: for each group, pick a base vector and create
    # 2-4 near-copies with small perturbations
    true_groups = []
    idx = n_vectors - n_dup_groups * 4  # reserve space at the end
    for g in range(n_dup_groups):
        base_idx = rng.randint(0, idx)
        base_vec = embeddings[base_idx].copy()
        group = {base_idx}
        n_copies = rng.randint(2, 5)
        for c in range(n_copies):
            dup_idx = idx + g * 4 + c
            if dup_idx < n_vectors:
                noise = rng.randn(dim).astype(np.float32) * 0.01
                embeddings[dup_idx] = base_vec + noise
                group.add(dup_idx)
        true_groups.append(group)

    return embeddings, true_groups


def find_duplicates_radius(
    embeddings: np.ndarray, threshold: float
) -> list[tuple[int, int]]:
    """Find all pairs of vectors within a distance threshold."""
    nn = NearestNeighbors(
        radius=threshold, algorithm="ball_tree", metric="euclidean"
    )
    nn.fit(embeddings)
    adjacency = nn.radius_neighbors(embeddings, return_distance=False)

    pairs = set()
    for i, neighbors in enumerate(adjacency):
        for j in neighbors:
            if i < j:
                pairs.add((i, j))

    return sorted(pairs)


def build_duplicate_clusters(
    pairs: list[tuple[int, int]], n_vectors: int
) -> list[set[int]]:
    """Group duplicate pairs into connected components (union-find)."""
    parent = list(range(n_vectors))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, j in pairs:
        union(i, j)

    clusters = {}
    for idx in range(n_vectors):
        root = find(idx)
        if root not in clusters:
            clusters[root] = set()
        clusters[root].add(idx)

    # Only return clusters with more than 1 member
    return [c for c in clusters.values() if len(c) > 1]


def main():
    # Parameters
    n_vectors = 50_000
    dim = 64
    threshold = 0.5
    n_dup_groups = 200

    print(
        f"Generating {n_vectors} embeddings with {n_dup_groups} injected "
        f"duplicate groups (dim={dim})..."
    )
    embeddings, true_groups = generate_embeddings_with_duplicates(
        n_vectors, dim, n_dup_groups
    )

    # Find near-duplicate pairs
    print(f"Searching for pairs within distance threshold {threshold}...")
    t0 = time.time()
    pairs = find_duplicates_radius(embeddings, threshold)
    search_time = time.time() - t0
    print(f"Radius search took {search_time:.3f}s")

    # Build clusters
    print("Building duplicate clusters...")
    clusters = build_duplicate_clusters(pairs, n_vectors)

    # Report
    print("\nResults:")
    print(f"  Duplicate pairs found: {len(pairs)}")
    print(f"  Duplicate clusters: {len(clusters)}")
    print(f"  Injected duplicate groups: {len(true_groups)}")
    print(f"  Avg cluster size: {np.mean([len(c) for c in clusters]):.1f}")
    print(f"  Largest cluster: {max(len(c) for c in clusters)}")

    # Show a few clusters
    print("\nSample clusters (first 5):")
    for i, cluster in enumerate(clusters[:5]):
        members = sorted(cluster)[:8]
        suffix = "..." if len(cluster) > 8 else ""
        print(f"  Cluster {i}: {members}{suffix} (size={len(cluster)})")


if __name__ == "__main__":
    main()
