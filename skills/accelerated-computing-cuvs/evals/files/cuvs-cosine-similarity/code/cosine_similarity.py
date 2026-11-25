# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Find most similar document pairs using cosine similarity.

Generates normalized document embeddings and finds the top most similar
pairs using sklearn pairwise_distances with cosine metric.
"""

import time

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize


def generate_document_embeddings(
    n_docs: int, dim: int, seed: int = 42
) -> np.ndarray:
    """Generate random document embeddings, L2-normalized."""
    rng = np.random.RandomState(seed)
    raw = rng.randn(n_docs, dim).astype(np.float32)
    return normalize(raw, norm="l2")


def find_top_similar_pairs(
    embeddings: np.ndarray, top_n: int = 20
) -> list[tuple[int, int, float]]:
    """Find the top-N most similar (closest cosine) document pairs.

    Returns list of (doc_i, doc_j, similarity_score) sorted by similarity desc.
    """
    # Compute full pairwise cosine distance matrix
    dist_matrix = pairwise_distances(embeddings, metric="cosine")

    # Zero out diagonal and lower triangle to avoid duplicates
    n = dist_matrix.shape[0]
    for i in range(n):
        dist_matrix[i, i] = np.inf
        for j in range(i):
            dist_matrix[i, j] = np.inf

    # Find top-N smallest distances (highest similarity)
    pairs = []
    flat_indices = np.argsort(dist_matrix, axis=None)
    for idx in flat_indices[:top_n]:
        i, j = divmod(idx, n)
        similarity = 1.0 - dist_matrix[i, j]
        pairs.append((int(i), int(j), float(similarity)))

    return pairs


def find_most_similar_per_doc(
    embeddings: np.ndarray, k: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """For each document, find its k most similar neighbors.

    Returns (indices, similarities) arrays of shape (n_docs, k).
    """
    dist_matrix = pairwise_distances(embeddings, metric="cosine")
    np.fill_diagonal(dist_matrix, np.inf)

    top_k_indices = np.argsort(dist_matrix, axis=1)[:, :k]
    top_k_similarities = np.take_along_axis(
        1.0 - dist_matrix, top_k_indices, axis=1
    )
    return top_k_indices, top_k_similarities


def main():
    # Parameters
    n_docs = 10_000
    dim = 256
    top_n_pairs = 20
    k_per_doc = 5

    print(
        f"Generating {n_docs} normalized document embeddings of dim {dim}..."
    )
    embeddings = generate_document_embeddings(n_docs, dim)

    # Find top similar pairs globally
    print(f"Finding top {top_n_pairs} most similar pairs...")
    t0 = time.time()
    top_pairs = find_top_similar_pairs(embeddings, top_n=top_n_pairs)
    pairs_time = time.time() - t0
    print(f"Pair search took {pairs_time:.3f}s")

    print(f"\nTop {top_n_pairs} most similar document pairs:")
    for doc_i, doc_j, sim in top_pairs:
        print(f"  Doc {doc_i:5d} <-> Doc {doc_j:5d}  similarity: {sim:.6f}")

    # Find k most similar per document
    print(f"\nFinding {k_per_doc} most similar neighbors per document...")
    t0 = time.time()
    neighbor_ids, neighbor_sims = find_most_similar_per_doc(
        embeddings, k=k_per_doc
    )
    per_doc_time = time.time() - t0
    print(f"Per-doc search took {per_doc_time:.3f}s")

    # Print a sample
    print("\nSample results (first 5 documents):")
    for i in range(5):
        neighbors = ", ".join(
            f"{neighbor_ids[i, j]}({neighbor_sims[i, j]:.4f})"
            for j in range(k_per_doc)
        )
        print(f"  Doc {i}: [{neighbors}]")

    print(
        f"\nAvg similarity to nearest neighbor: "
        f"{neighbor_sims[:, 0].mean():.6f}"
    )


if __name__ == "__main__":
    main()
