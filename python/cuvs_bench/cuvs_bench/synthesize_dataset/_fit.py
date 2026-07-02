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
4. Compute a per-cluster norm inverse-CDF (quantile grid) if needed, which is stored as
   ``norm_quantiles`` and used by the "percentile" norm scheme at generate time.

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

# Number of points in each per-cluster norm inverse-CDF quantile grid.
_NORM_QUANTILE_COUNT = 256

# Tolerance for checking if the real vectors are essentially unit-norm (mean ~1, tiny spread).
_NORM_UNIT_MEAN_TOL = 0.02
_NORM_UNIT_CV_TOL = 0.05

# Rows streamed to the GPU per batch on the cuVS host (out-of-core) KMeans
# fit/predict paths. Materializing the whole sample at once needs n*d*4 bytes on
# the GPU (e.g. 10M x 1024 f32 = 38 GiB) and OOMs; stream it in batches instead.
_KMEANS_STREAM_BATCH = 1_000_000


def _run_kmeans(
    data: np.ndarray,
    n_clusters: int,
):
    """Run cuvs KMeans, returning ``(labels, centroids)`` as numpy arrays."""
    params = cuvs_kmeans.KMeansParams(
        n_clusters=n_clusters,
        max_iter=300,
        streaming_batch_size=_KMEANS_STREAM_BATCH,
    )

    print(
        f"  Running cuvs KMeans (n_clusters={n_clusters}, "
        f"streaming_batch_size={_KMEANS_STREAM_BATCH:,})..."
    )

    t0 = time.perf_counter()
    centroids_out, _, _ = cuvs_kmeans.fit(params, data)
    print(f"  KMeans fit time: {time.perf_counter() - t0:.2f}s")

    centroids_gpu = cp.asarray(centroids_out)
    centroids = cp.asnumpy(centroids_gpu).astype(np.float32)

    # cuVS predict has no host/streaming path, so assign labels in batches to
    # avoid uploading the full sample (n*d*4 bytes) to the GPU at once.
    n = len(data)
    labels = np.empty(n, dtype=np.int64)
    for start in range(0, n, _KMEANS_STREAM_BATCH):
        stop = min(start + _KMEANS_STREAM_BATCH, n)
        chunk_gpu = cp.asarray(data[start:stop], dtype=cp.float32)
        labels_out, _ = cuvs_kmeans.predict(params, chunk_gpu, centroids_gpu)
        labels[start:stop] = cp.asnumpy(cp.asarray(labels_out)).astype(
            np.int64
        )
        del chunk_gpu
        cp.get_default_memory_pool().free_all_blocks()

    del centroids_gpu
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

    del residuals_gpu, out
    return components, explained_var


def fit_fingerprint(
    data: np.ndarray,
    n_clusters: int,
    pca_components: int,
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
        ``norm_quantiles``         : (n_clusters, 256) float32 — per-cluster
                                      empirical inverse-CDF of the real vector
                                      norms.
    """
    n_dim = data.shape[1]
    data_sample = np.ascontiguousarray(data.astype(np.float32))

    # Norm quantiles for the "percentile" norm scheme at generate time.
    quantile_levels = np.linspace(0.0, 1.0, _NORM_QUANTILE_COUNT)
    n = len(data_sample)
    data_sample_norms = np.empty(n, dtype=np.float32)
    for start in range(0, n, _KMEANS_STREAM_BATCH):
        stop = min(start + _KMEANS_STREAM_BATCH, n)
        chunk_gpu = cp.asarray(data_sample[start:stop], dtype=cp.float32)
        data_sample_norms[start:stop] = cp.asnumpy(
            cp.linalg.norm(chunk_gpu, axis=1)
        )
        del chunk_gpu

    norms64 = data_sample_norms.astype(np.float64)
    norm_mean = float(norms64.mean())
    norm_cv = float(norms64.std() / max(norm_mean, 1e-12))
    global_norm_quantiles = np.quantile(
        data_sample_norms, quantile_levels
    ).astype(np.float32)

    norm_unit = (
        abs(norm_mean - 1.0) <= _NORM_UNIT_MEAN_TOL
        and norm_cv <= _NORM_UNIT_CV_TOL
    )
    if norm_unit:
        print(
            f"  Real vector-norm distribution: mean={norm_mean:.4f}, "
            f"CV={norm_cv:.4f} -> ~unit-norm; storing norm_unit flag "
        )
    else:
        print(
            f"  Real vector-norm distribution: mean={norm_mean:.4f}, "
            f"CV={norm_cv:.4f} (stored as per-cluster "
            f"{_NORM_QUANTILE_COUNT}-point quantile grids)."
        )

    labels, centroids = _run_kmeans(data_sample, n_clusters)

    print(
        f"Fitting per-cluster PCA (ncomp={pca_components}) for "
        f"{n_clusters} clusters..."
    )

    densities = np.zeros(n_clusters, dtype=np.float64)
    variances_per_dim = np.zeros((n_clusters, n_dim), dtype=np.float32)
    pca_components_list: list = []
    pca_explained_var_list: list = []
    pca_noise_var = np.zeros(n_clusters, dtype=np.float32)
    norm_quantiles = (
        None
        if norm_unit
        else np.empty((n_clusters, _NORM_QUANTILE_COUNT), dtype=np.float32)
    )

    for i in tqdm(range(n_clusters), desc="Per-cluster PCA"):
        mask = labels == i
        cluster_points = data_sample[mask]
        n_points = len(cluster_points)
        densities[i] = n_points / len(data_sample)

        if not norm_unit:
            if n_points >= 1:
                norm_quantiles[i] = np.quantile(
                    data_sample_norms[mask], quantile_levels
                ).astype(np.float32)
            else:
                norm_quantiles[i] = global_norm_quantiles

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
        "norm_quantiles": norm_quantiles,
        "norm_unit": norm_unit,
    }
