#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
"""Fit a cluster fingerprint from a sample of real data.

Pipeline:
1. Run KMeans (``cuvs.cluster.kmeans``) on the sample.
2. For each cluster, fit a rank-``ncomp`` PCA (``cuvs.preprocessing.pca``) on
   the cluster's residuals, capturing the dominant intra-cluster correlations.
3. Estimate a residual noise variance per cluster (the variance not captured
   by the top-``ncomp`` components).

The output dict can be saved with :func:`save_fingerprint`.
"""

from __future__ import annotations

import time
from typing import Any, Dict

import cupy as cp
import numpy as np
from cuvs.cluster import kmeans as cuvs_kmeans
from cuvs.preprocessing import pca as cuvs_pca
from tqdm import tqdm

from ..generate_groundtruth.utils import is_l2_normalized


def _run_kmeans(
    data: np.ndarray,
    n_clusters: int,
    seed: int,
):
    """Run cuvs KMeans, returning ``(labels, centroids)`` as numpy arrays."""
    params = cuvs_kmeans.KMeansParams(
        n_clusters=n_clusters,
        init_size=len(data),
    )

    print(
        f"  Running cuvs KMeans (n_clusters={n_clusters}, "
        f"init_size={len(data):,})..."
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

    labels, centroids = _run_kmeans(data_sample, n_clusters, seed)

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
