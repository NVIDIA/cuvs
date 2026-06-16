#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
"""Lightweight readers for the source datasets users want to fit against.

Supported formats: ``.fbin`` (with an optional nrows cap), ``.npy``, and
``.pkl``.
"""

from __future__ import annotations

import os
import pickle

import numpy as np

from ..generate_groundtruth.utils import memmap_bin_file


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
        responsible for ensuring the head-of-file slice is representative
        (e.g. by pre-shuffling the dataset on disk if it has any structural
        ordering).
    dtype : numpy dtype
        Element dtype, used only for ``.fbin`` (the other formats carry their
        own dtype). Defaults to float32.

    Returns
    -------
    np.ndarray, shape (n, d)
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        data = np.load(path)
        if sample_size is not None and sample_size < len(data):
            data = data[:sample_size]
        return np.ascontiguousarray(data.astype(np.float32))

    if ext == ".pkl":
        with open(path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        if sample_size is not None and sample_size < len(data):
            data = data[:sample_size]
        return np.ascontiguousarray(data.astype(np.float32))

    # Default: treat as fbin (covers ".fbin" and unknown extensions).
    # memmap_bin_file auto-detects the legacy uint32 / extended uint64 header.
    mm = memmap_bin_file(path, dtype, mode="r")
    if sample_size is not None:
        mm = mm[:sample_size]
    return np.ascontiguousarray(mm)


def is_l2_normalized(
    data: np.ndarray,
    sample_size: int = 10_000,
    tol: float = 1e-2,
    seed: int = 0,
) -> bool:
    """Cheaply check whether ``data`` rows are L2-unit-norm.

    Samples up to ``sample_size`` rows uniformly at random and returns ``True``
    iff every sampled row has ``|‖x‖ - 1| < tol``.

    Parameters
    ----------
    data : np.ndarray, shape (n, d)
        Real-data slice to inspect. Empty input returns ``False``.
    sample_size : int
        Number of random rows to sample (default: 10000). Capped at
        ``len(data)``.
    tol : float
        Allowed deviation of each row's L2 norm from 1.0 (default: 1e-2).
    seed : int
        RNG seed for the row sample (default: 0). Fixed so the detection is
        deterministic for a given input.
    """
    n = len(data)
    if n == 0:
        return False
    rng = np.random.default_rng(seed)
    take = min(sample_size, n)
    idx = rng.choice(n, size=take, replace=False)
    norms = np.linalg.norm(data[idx].astype(np.float32), axis=1)
    return bool(np.all(np.abs(norms - 1.0) < tol))
