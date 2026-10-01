#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Fingerprint: in-memory representation of a fitted cluster fingerprint."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Fingerprint:
    """In-memory representation of a fitted cluster fingerprint.

    Produced by ``cuvs_bench.synthesize_dataset.fit``.

    Parameters
    ----------
    nclusters : int
        Number of KMeans clusters fitted on the real-data sample.
    ncols : int
        Dimensionality of the data.
    seed : int
        Random seed used for deterministic per-cluster generation.
    cluster_centers : np.ndarray, shape (nclusters, ncols)
        Cluster centroids.
    cluster_variances : np.ndarray, shape (nclusters, ncols)
        Per-dimension variance per cluster, used for variance matching when
        rescaling generated points.
    cluster_densities : np.ndarray, shape (nclusters,)
        Relative density per cluster. Normalized to sum to 1 in ``__post_init__``.
    pca_components_list : list[Optional[np.ndarray]]
        Per-cluster principal directions, each of shape ``(ncomp, ncols)``.
        Entries may be ``None`` for clusters that fell back to diagonal-only
        sampling because they had too few points to fit a rank-``ncomp`` PCA.
    pca_explained_var_list : list[Optional[np.ndarray]]
        Per-cluster variance along each principal direction, each of shape
        ``(ncomp,)``. ``None`` for diagonal-fallback clusters.
    pca_noise_var : np.ndarray, shape (nclusters,)
        Residual noise variance per cluster. Used as the variance of an isotropic
        Gaussian added to every generated point.
    pca_n_components : int
        The requested number of PCA components per cluster as specified at
        fit time.
    norm_quantiles : np.ndarray, optional
        Per-cluster empirical inverse-CDF of the real vector norms, shape
        ``(nclusters, 256)`` -- one 256-point quantile grid per cluster,
        produced at fit time. Used by the ``"percentile"`` norm scheme to draw
        a per-vector target norm via inverse-CDF sampling from that vector's
        cluster grid, so each cluster reproduces its own real radial spread.
        ``None`` for grid-less fingerprints (those fall back to ``"off"`` norm
        rescaling at generate time.
    norm_unit : bool
        Set at fit time when the real vectors are essentially unit-norm. Generation
        L2-normalizes instead of inverse-CDF sampling using norm_quantiles.
    """

    nclusters: int
    ncols: int
    seed: int
    cluster_centers: np.ndarray
    cluster_variances: np.ndarray
    cluster_densities: np.ndarray

    pca_components_list: List[Optional[np.ndarray]]
    pca_explained_var_list: List[Optional[np.ndarray]]
    pca_noise_var: np.ndarray
    pca_n_components: int

    norm_quantiles: Optional[np.ndarray] = None
    norm_unit: bool = False

    def __post_init__(self):
        if self.cluster_densities.shape != (self.nclusters,):
            raise ValueError(
                f"cluster_densities shape {self.cluster_densities.shape} does "
                f"not match expected ({self.nclusters},)."
            )
        if not np.isfinite(self.cluster_densities).all():
            raise ValueError(
                "cluster_densities must contain only finite values "
                "(no NaN/Inf)."
            )
        if (self.cluster_densities < 0).any():
            raise ValueError("cluster_densities must be non-negative.")
        total = float(self.cluster_densities.sum())
        if total <= 0.0:
            raise ValueError("cluster_densities must sum to a positive value")
        self.cluster_densities = self.cluster_densities / total
