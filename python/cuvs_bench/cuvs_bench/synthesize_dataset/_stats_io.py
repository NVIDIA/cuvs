#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
"""NPZ I/O for the cluster fingerprint.

The fingerprint is a NPZ file holding
the output of the ``fit`` step. It is the only artifact passed between the
fit and generate steps.

File schema
-----------
- ``centroids``           : float32, shape (nclusters, ncols)
- ``densities``           : float64, shape (nclusters,)
- ``variances_per_dim``   : float32, shape (nclusters, ncols)
- ``pca_components_arr``  : object array of length nclusters; entry i is
                            either a (ncomp, ncols) float32 array or an empty
                            float32 array if cluster i fell back to diagonal.
- ``pca_explained_var_arr`` : object array of length nclusters, parallel to
                              ``pca_components_arr``; each entry is a (ncomp,)
                              float32 array.
- ``pca_noise_var``       : float32, shape (nclusters,)
- ``pca_n_components``    : int, scalar (the requested ncomp; actual ncomp per
                            cluster may be smaller if it had too few points).
- ``is_normalized_data`` : bool, scalar — whether the fit-time input was
                            detected as L2-unit-norm. Drives the default
                            ``normalize`` setting at generate/verify time.
"""

import os
from typing import Any, Dict

import numpy as np

from ._config import ClusterConfig


def save_cluster_stats(filepath: str, stats: Dict[str, Any]) -> None:
    """Save a fitted cluster fingerprint to an NPZ file.

    Parameters
    ----------
    filepath : str
        Output path. Parent directories are created as needed.
    stats : dict
        Dict with keys produced by :func:`fit_cluster_stats` (see schema in
        the module docstring).
    """
    out_dir = os.path.dirname(filepath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    n_clusters = len(stats["centroids"])

    components_arr = np.empty(n_clusters, dtype=object)
    explained_var_arr = np.empty(n_clusters, dtype=object)

    for i in range(n_clusters):
        comp = stats["pca_components_list"][i]
        ev = stats["pca_explained_var_list"][i]
        if comp is not None:
            components_arr[i] = comp
            explained_var_arr[i] = ev
        else:
            # Diagonal-fallback marker: empty array.
            components_arr[i] = np.array([], dtype=np.float32)
            explained_var_arr[i] = np.array([], dtype=np.float32)

    np.savez(
        filepath,
        centroids=stats["centroids"],
        densities=stats["densities"],
        variances_per_dim=stats["variances_per_dim"],
        pca_components_arr=components_arr,
        pca_explained_var_arr=explained_var_arr,
        pca_noise_var=stats["pca_noise_var"],
        pca_n_components=np.array([stats["pca_n_components"]]),
        is_normalized_data=np.array([bool(stats["is_normalized_data"])]),
    )


def load_cluster_stats(filepath: str) -> Dict[str, Any]:
    """Load a fitted cluster fingerprint from an NPZ file.

    Returns a dict with the same shape as :func:`fit_cluster_stats`.
    """
    data = np.load(filepath, allow_pickle=True)

    n_clusters = len(data["centroids"])
    components_arr = data["pca_components_arr"]
    explained_var_arr = data["pca_explained_var_arr"]

    pca_components_list = []
    pca_explained_var_list = []
    for i in range(n_clusters):
        if len(components_arr[i]) > 0:
            pca_components_list.append(components_arr[i])
            pca_explained_var_list.append(explained_var_arr[i])
        else:
            pca_components_list.append(None)
            pca_explained_var_list.append(None)

    return {
        "centroids": data["centroids"],
        "densities": data["densities"],
        "variances_per_dim": data["variances_per_dim"],
        "pca_components_list": pca_components_list,
        "pca_explained_var_list": pca_explained_var_list,
        "pca_noise_var": data["pca_noise_var"],
        "pca_n_components": int(data["pca_n_components"][0]),
        "is_normalized_data": bool(data["is_normalized_data"][0]),
    }


def cluster_config_from_stats(
    stats: Dict[str, Any], seed: int
) -> ClusterConfig:
    """Build a :class:`ClusterConfig` from a loaded stats dict."""
    centroids = stats["centroids"]
    return ClusterConfig(
        nclusters=int(centroids.shape[0]),
        ncols=int(centroids.shape[1]),
        seed=seed,
        cluster_centers=centroids,
        cluster_variances=stats["variances_per_dim"],
        cluster_densities=stats["densities"].astype(np.float64),
        pca_components_list=stats["pca_components_list"],
        pca_explained_var_list=stats["pca_explained_var_list"],
        pca_noise_var=stats["pca_noise_var"],
        is_normalized_data=bool(stats["is_normalized_data"]),
    )
