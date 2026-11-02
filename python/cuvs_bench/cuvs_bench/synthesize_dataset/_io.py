#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Dataset loading and fingerprint NPZ I/O.

Supported dataset formats for :func:`load_dataset`: ``.fbin``, ``.npy``,
and ``.pkl``.

Fingerprint NPZ file schema (written by :func:`save_fingerprint`, read by
:func:`load_fingerprint`):

- ``centroids``             : float32, shape (nclusters, ncols)
- ``densities``             : float64, shape (nclusters,)
- ``variances_per_dim``     : float32, shape (nclusters, ncols)
- ``pca_components_arr``    : object array of length nclusters; entry i is
                              either a (ncomp, ncols) float32 array or an empty
                              float32 array if cluster i fell back to diagonal.
- ``pca_explained_var_arr`` : object array of length nclusters, parallel to
                              ``pca_components_arr``; each entry is a (ncomp,)
                              float32 array.
- ``pca_noise_var``         : float32, shape (nclusters,)
- ``pca_n_components``      : int, scalar (the requested ncomp; actual ncomp per
                              cluster may be smaller if it had too few points).
- ``norm_quantiles``        : float32, shape (nclusters, 256) — per-cluster
                              empirical inverse-CDF of the real vector norms.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict

import numpy as np

from ..generate_groundtruth.utils import memmap_bin_file
from ._fingerprint import Fingerprint


def _finalize_dataset(data: np.ndarray) -> np.ndarray:
    """Validate a loaded dataset and return it as contiguous float32."""
    if data.ndim != 2:
        raise ValueError(
            f"dataset must be 2D (n_rows, n_cols); got ndim={data.ndim}, "
            f"shape={data.shape}."
        )
    if data.shape[1] == 0:
        raise ValueError(
            f"dataset must have n_cols > 0; got shape={data.shape}."
        )
    return np.ascontiguousarray(data.astype(np.float32))


def load_dataset(
    path: str,
    sample_size: int | None = None,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Load a real dataset for fitting.

    Supported formats:
    - ``.fbin``: cuvs-bench binary header (``n_rows``, ``n_dim``; legacy
      uint32 or extended uint64, auto-detected) followed by raw data.
    - ``.npy``: standard numpy array file.
    - ``.pkl``: pickled numpy array (or anything ``np.array`` can convert).

    Parameters
    ----------
    path : str
        Path to the dataset file.
    sample_size : int or None
        If given and smaller than the on-disk row count, only the **first**
        ``sample_size`` rows are loaded (no shuffling). The caller is
        responsible for ensuring the head-of-file slice is representative.
    dtype : numpy dtype
        Element dtype, used only for ``.fbin`` (the other formats carry their
        own dtype). Defaults to float32.

    Returns
    -------
    np.ndarray, shape (n, d)
    """

    if sample_size is not None and sample_size <= 0:
        raise ValueError(
            f"sample_size must be > 0 when provided, got {sample_size}."
        )

    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        data = np.load(path)
        if sample_size is not None and sample_size < len(data):
            data = data[:sample_size]
        return _finalize_dataset(data)

    if ext == ".pkl":
        with open(path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        if sample_size is not None and sample_size < len(data):
            data = data[:sample_size]
        return _finalize_dataset(data)

    # Default: treat as fbin (covers ".fbin" and unknown extensions).
    # memmap_bin_file auto-detects the legacy uint32 / extended uint64 header.
    mm = memmap_bin_file(path, dtype, mode="r")
    if sample_size is not None:
        mm = mm[:sample_size]
    return _finalize_dataset(mm)


def save_fingerprint(filepath: str, fingerprint: Dict[str, Any]) -> None:
    """Save a fitted cluster fingerprint to an NPZ file.

    Parameters
    ----------
    filepath : str
        Output path. Parent directories are created as needed.
    fingerprint : dict
        Dict with keys produced by :func:`fit_fingerprint`.
    """
    out_dir = os.path.dirname(filepath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    n_clusters = len(fingerprint["centroids"])

    components_arr = np.empty(n_clusters, dtype=object)
    explained_var_arr = np.empty(n_clusters, dtype=object)

    for i in range(n_clusters):
        comp = fingerprint["pca_components_list"][i]
        ev = fingerprint["pca_explained_var_list"][i]
        if comp is not None:
            components_arr[i] = comp
            explained_var_arr[i] = ev
        else:
            # Diagonal-fallback marker: empty array.
            components_arr[i] = np.array([], dtype=np.float32)
            explained_var_arr[i] = np.array([], dtype=np.float32)

    to_save = dict(
        centroids=fingerprint["centroids"],
        densities=fingerprint["densities"],
        variances_per_dim=fingerprint["variances_per_dim"],
        pca_components_arr=components_arr,
        pca_explained_var_arr=explained_var_arr,
        pca_noise_var=fingerprint["pca_noise_var"],
        pca_n_components=np.array([fingerprint["pca_n_components"]]),
        norm_unit=np.array([bool(fingerprint.get("norm_unit", False))]),
    )
    # Unit-norm fingerprints carry no quantile grids (generation L2-normalizes).
    if fingerprint.get("norm_quantiles") is not None:
        to_save["norm_quantiles"] = np.asarray(
            fingerprint["norm_quantiles"], dtype=np.float32
        )
    np.savez(filepath, **to_save)


def load_fingerprint(filepath: str, seed: int) -> Fingerprint:
    """Load a fitted cluster fingerprint from an NPZ file into a Fingerprint."""
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

    centroids = data["centroids"]
    norm_quantiles = (
        np.asarray(data["norm_quantiles"], dtype=np.float32)
        if "norm_quantiles" in data.files
        else None
    )
    norm_unit = (
        bool(data["norm_unit"][0]) if "norm_unit" in data.files else False
    )
    return Fingerprint(
        nclusters=int(centroids.shape[0]),
        ncols=int(centroids.shape[1]),
        seed=seed,
        cluster_centers=centroids,
        cluster_variances=data["variances_per_dim"],
        cluster_densities=data["densities"].astype(np.float64),
        pca_components_list=pca_components_list,
        pca_explained_var_list=pca_explained_var_list,
        pca_noise_var=data["pca_noise_var"],
        pca_n_components=int(data["pca_n_components"][0]),
        norm_quantiles=norm_quantiles,
        norm_unit=norm_unit,
    )
