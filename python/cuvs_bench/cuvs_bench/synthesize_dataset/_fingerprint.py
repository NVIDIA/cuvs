#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
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
    is_normalized_data : bool
        Whether the fit-time real-data slice was detected as L2-unit-norm.
        ``generate``, ``verify``, and the streaming
        GT routines use this to decide whether to re-normalize the synthetic output.
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

    is_normalized_data: bool

    def __post_init__(self):
        total = float(self.cluster_densities.sum())
        if total <= 0.0:
            raise ValueError("cluster_densities must sum to a positive value")
        self.cluster_densities = self.cluster_densities / total
