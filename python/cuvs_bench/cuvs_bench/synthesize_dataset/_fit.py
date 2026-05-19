#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
"""Fit a cluster fingerprint from a sample of real data.

Pipeline:
1. Detect whether the input is L2-normalized. The result is stored in the
   fingerprint as ``is_normalized_data`` and is used at generate/verify time.
2. Decide whether to L2-normalize the sample on-the-fly for KMeans (centroids,
   variances, and PCA are always computed in the original space, regardless).
   Governed by the ``normalize_for_clustering`` argument (CLI:
   ``--no-normalize-for-clustering``): when left at its default ``None``, it
   follows the step-1 detection (enable iff the input was not already
   L2-normalized); pass ``True``/``False`` (or set the CLI flag to force
   ``False``) to override.
3. Run KMeans (``cuvs.cluster.kmeans``) to partition the sample.
4. For each cluster, fit a rank-``ncomp`` PCA (``cuvs.preprocessing.pca``) on
   the cluster's residuals, capturing the dominant intra-cluster correlations.
5. Estimate a residual noise variance per cluster (the variance not captured
   by the top-``ncomp`` components).

The output dict can be saved with :func:`save_cluster_stats`.
"""

from __future__ import annotations

import time
from typing import Any, Dict

import cupy as cp
import numpy as np
from cuvs.cluster import kmeans as cuvs_kmeans
from cuvs.preprocessing import pca as cuvs_pca
from tqdm import tqdm

from ._io import is_l2_normalized


def _normalize_batched_gpu(
    data: np.ndarray, batch_size: int = 1_000_000
) -> np.ndarray:
    """L2-normalize host data in batches via cupy."""
    n_rows = len(data)
    n_batches = (n_rows + batch_size - 1) // batch_size
    out = np.empty_like(data)

    print(
        f"  Normalizing {n_rows:,} vectors on GPU "
        f"({n_batches} batches of {batch_size:,})..."
    )

    for b in range(n_batches):
        s = b * batch_size
        e = min(s + batch_size, n_rows)
        d_gpu = cp.asarray(data[s:e])
        norms = cp.linalg.norm(d_gpu, axis=1, keepdims=True)
        norms = cp.maximum(norms, 1e-12)
        d_gpu /= norms
        out[s:e] = cp.asnumpy(d_gpu)
        del d_gpu, norms
        cp.get_default_memory_pool().free_all_blocks()

    return out


def _run_kmeans(
    data: np.ndarray,
    n_clusters: int,
    seed: int,
    max_iter: int,
):
    """Run cuvs KMeans, returning ``(labels, centroids)`` as numpy arrays."""
    params = cuvs_kmeans.KMeansParams(
        n_clusters=n_clusters,
        max_iter=max_iter,
        init_size=len(data),
    )

    print(
        f"  Running cuvs KMeans (n_clusters={n_clusters}, "
        f"max_iter={max_iter}, init_size={len(data):,})..."
    )

    t0 = time.perf_counter()
    centroids_out, _, _ = cuvs_kmeans.fit(params, data)
    print(f"  KMeans fit time: {time.perf_counter() - t0:.2f}s")

    centroids_gpu = cp.asarray(centroids_out)
    data_gpu = cp.asarray(data, dtype=cp.float32)
    labels_out, _ = cuvs_kmeans.predict(params, data_gpu, centroids_gpu)

    centroids = cp.asnumpy(centroids_gpu).astype(np.float32)
    labels = cp.asnumpy(cp.asarray(labels_out)).astype(np.int64)

    del data_gpu, centroids_gpu
    cp.get_default_memory_pool().free_all_blocks()

    return labels, centroids


def _fit_cluster_pca(
    residuals: np.ndarray, n_components: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a single PCA via cuvs and return ``(components, explained_var)``."""
    residuals_gpu = cp.asarray(residuals, dtype=cp.float32)
    params = cuvs_pca.Params(n_components=n_components, copy=True)
    out = cuvs_pca.fit(params, residuals_gpu)
    components = cp.asnumpy(cp.asarray(out.components)).astype(np.float32)
    explained_var = cp.asnumpy(cp.asarray(out.explained_var)).astype(
        np.float32
    )
    return components, explained_var


def fit_cluster_stats(
    data: np.ndarray,
    n_clusters: int,
    pca_components: int,
    seed: int = 42,
    max_iter: int = 300,
    normalize_for_clustering: bool | None = None,
) -> Dict[str, Any]:
    """Fit a cluster fingerprint to a real dataset sample.

    Parameters
    ----------
    data : np.ndarray, shape (n, d)
        Real data to fit against.
    n_clusters : int
        Number of KMeans clusters.
    pca_components : int
        Number of principal directions per cluster.
    seed : int
        Random seed for KMeans initialization.
    max_iter : int
        KMeans max iterations.
    normalize_for_clustering : bool or None
        Whether to L2-normalize the sample before KMeans (centroids, variances,
        and PCA still computed in the original space). When ``None`` (default),
        this is decided automatically: enabled iff the input is detected as
        not-L2-normalized. Pass ``True``/``False`` to override. Exposed on the
        CLI as ``--no-normalize-for-clustering`` (sets this to ``False``).

    Returns
    -------
    dict with keys:
        ``centroids``               : (n_clusters, d) float32
        ``densities``               : (n_clusters,) float64
        ``variances_per_dim``       : (n_clusters, d) float32
        ``pca_components_list``     : list of (ncomp, d) float32 arrays or None
        ``pca_explained_var_list``  : list of (ncomp,) float32 arrays or None
        ``pca_noise_var``           : (n_clusters,) float32
        ``pca_n_components``        : int (the requested ncomp)
        ``is_normalized_data``     : bool — whether the fit input was
                                      already L2-unit-norm. ``generate`` and
                                      ``verify`` use this to default their
                                      ``normalize`` setting.
    """
    n_dim = data.shape[1]
    data_sample = np.ascontiguousarray(data.astype(np.float32))

    is_normalized_data = is_l2_normalized(data_sample)
    print(
        "Detected input as "
        + ("L2-normalized" if is_normalized_data else "non-L2-normalized")
        + " (sampled rows have norms "
        + ("≈ 1.0" if is_normalized_data else "≠ 1.0")
        + ")."
    )

    if normalize_for_clustering is None:
        normalize_for_clustering = not is_normalized_data
        print(
            f"  normalize_for_clustering=auto -> "
            f"{normalize_for_clustering} "
            f"(skipped for already-normalized data, "
            f"enabled otherwise)."
        )

    if normalize_for_clustering:
        print(
            "Normalizing sample for clustering (stats stay in original "
            "space)..."
        )
        clustering_data = _normalize_batched_gpu(data_sample)
    else:
        clustering_data = data_sample

    labels, centroids = _run_kmeans(
        clustering_data, n_clusters, seed, max_iter
    )

    # If we clustered on normalized data, recompute centroids from the
    # original-space points belonging to each cluster.
    if normalize_for_clustering:
        print("Recomputing centroids in original (unnormalized) space...")
        centroids = np.zeros((n_clusters, n_dim), dtype=np.float32)
        for i in tqdm(
            range(n_clusters),
            desc="Recomputing centroids",
        ):
            mask = labels == i
            if mask.any():
                cluster_gpu = cp.asarray(data_sample[mask])
                centroids[i] = cp.asnumpy(cp.mean(cluster_gpu, axis=0))
                del cluster_gpu
        cp.get_default_memory_pool().free_all_blocks()

    print(
        f"Fitting per-cluster PCA (ncomp={pca_components}) for "
        f"{n_clusters} clusters..."
    )

    densities = np.zeros(n_clusters, dtype=np.float64)
    variances_per_dim = np.zeros((n_clusters, n_dim), dtype=np.float32)
    pca_components_list: list = []
    pca_explained_var_list: list = []
    pca_noise_var = np.zeros(n_clusters, dtype=np.float32)

    for i in tqdm(range(n_clusters), desc="Per-cluster PCA"):
        mask = labels == i
        cluster_points = data_sample[mask]
        n_points = len(cluster_points)
        densities[i] = n_points / len(data_sample)

        if n_points < pca_components + 1:
            # Diagonal fallback: too few points for a rank-ncomp PCA.
            if n_points > 1:
                variances_per_dim[i] = np.var(cluster_points, axis=0)
            else:
                variances_per_dim[i] = np.full(n_dim, 0.01, dtype=np.float32)
            pca_components_list.append(None)
            pca_explained_var_list.append(None)
            pca_noise_var[i] = float(np.mean(variances_per_dim[i]))
            continue

        variances_per_dim[i] = np.var(cluster_points, axis=0)

        residuals = cluster_points - centroids[i]
        n_components = min(pca_components, n_points - 1, n_dim)
        components, explained_var = _fit_cluster_pca(residuals, n_components)

        pca_components_list.append(components)
        pca_explained_var_list.append(explained_var)

        # Residual variance not captured by the top-ncomp components.
        total_var = float(np.var(residuals))
        pca_noise_var[i] = max(total_var - float(explained_var.sum()), 1e-6)

    n_valid = sum(1 for c in pca_components_list if c is not None)
    print(
        f"  {n_valid}/{n_clusters} clusters have a rank-ncomp PCA; "
        f"{n_clusters - n_valid} fell back to diagonal (too few points)."
    )

    return {
        "centroids": centroids.astype(np.float32),
        "densities": densities,
        "variances_per_dim": variances_per_dim.astype(np.float32),
        "pca_components_list": pca_components_list,
        "pca_explained_var_list": pca_explained_var_list,
        "pca_noise_var": pca_noise_var.astype(np.float32),
        "pca_n_components": int(pca_components),
        "is_normalized_data": bool(is_normalized_data),
    }
