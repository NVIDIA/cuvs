#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
"""Per-cluster Gaussian sampler with rank-``ncomp`` covariance.

Each cluster ``c`` defines a Gaussian:

    x ~ centroid_c + components_c.T @ z + n,
        z ~ N(0, diag(explained_var_c))
        n ~ N(0, noise_var_c * I)

After projection-plus-noise, we rescale per-dimension so the empirical
variance of the generated batch matches the cluster's per-dimension target.
"""

from __future__ import annotations

import os
import threading
from typing import Union

import cupy as cp
import numpy as np
from tqdm import tqdm

from .._bin_format import write_bin_header
from ..generate_groundtruth.utils import add_jitter
from ._fingerprint import Fingerprint

ArrayLike = Union[np.ndarray, cp.ndarray]


def get_cluster_seed(base_seed: int, cluster_id: int) -> int:
    """Map ``(base_seed, cluster_id)`` to a deterministic per-cluster seed."""
    return base_seed * 1_000_000 + int(cluster_id)


def resolve_norm_scheme(config: Fingerprint) -> str:
    """Norm-rescaling scheme for generation, derived purely from the fingerprint.

    - ``"unit"``: fit flagged the data as unit-norm -> L2-normalize (fast path).
    - ``"percentile"``: fit stored a per-cluster norm-quantile grid -> each cluster
      reproduces its own real radial spread via inverse-CDF sampling.
    - ``"off"``: neither -> vectors keep their natural generated magnitude.

    Not a user knob -- it follows the fit.
    """
    if getattr(config, "norm_unit", False):
        return "unit"
    has_quantiles = getattr(config, "norm_quantiles", None) is not None
    return "percentile" if has_quantiles else "off"


def _rescale_to_scheme(
    points: cp.ndarray,
    scheme: str,
    config: Fingerprint,
    cluster_id: int,
    rng: cp.random.RandomState,
) -> cp.ndarray:
    if scheme == "off" or points.shape[0] == 0:
        return points

    cur = cp.maximum(cp.linalg.norm(points, axis=1, keepdims=True), 1e-8)
    if scheme == "unit":
        return points / cur

    # "percentile": draw each vector's target norm from its cluster's inverse-CDF.
    quantiles = cp.asarray(config.norm_quantiles[cluster_id], dtype=cp.float32)
    u = rng.random_sample(size=points.shape[0], dtype=cp.float32)
    grid = cp.linspace(0.0, 1.0, len(quantiles), dtype=cp.float32)
    target = cp.interp(u, grid, quantiles).astype(cp.float32)[:, None]

    return points * (target / cur)


def gen_cluster_gpu(
    cluster_id: int,
    n_points: int,
    config: Fingerprint,
    return_cupy: bool = True,
) -> ArrayLike:
    """Generate ``n_points`` points for one cluster on the GPU."""
    center = cp.asarray(config.cluster_centers[cluster_id])
    seed = get_cluster_seed(config.seed, cluster_id)
    rng = cp.random.RandomState(seed)

    components_host = config.pca_components_list[cluster_id]
    explained_var_host = config.pca_explained_var_list[cluster_id]
    noise_var = float(config.pca_noise_var[cluster_id])

    if components_host is not None:
        components = cp.asarray(components_host)
        explained_var = cp.asarray(explained_var_host)
        k = len(explained_var)

        z = rng.standard_normal(
            size=(n_points, k), dtype=cp.float32
        ) * cp.sqrt(cp.maximum(explained_var, 0.0))
        projected = z @ components

        noise_std = float(cp.sqrt(max(noise_var, 0.0)))
        noise = (
            rng.standard_normal(
                size=(n_points, config.ncols), dtype=cp.float32
            )
            * noise_std
        )

        centered = projected + noise

        # Per-dimension scaling for variance matching, applied only when we
        # have enough points for a reliable empirical variance estimate.
        if n_points >= 10:
            target_var = cp.asarray(config.cluster_variances[cluster_id])
            actual_var = cp.var(centered, axis=0)
            # Floor the actual variance at noise_var (the minimum expected
            # variance) so we never divide by an unreliably small estimate.
            scale = cp.sqrt(target_var / cp.maximum(actual_var, noise_var))
            # Cap the scale at 5x to prevent blowup from noisy estimates.
            scale = cp.minimum(scale, 5.0)
            centered = centered * scale

        points = (center + centered).astype(cp.float32)
    else:
        # Diagonal fallback (in-place for memory efficiency at scale).
        scale = cp.sqrt(cp.asarray(config.cluster_variances[cluster_id]))
        points = rng.standard_normal(
            size=(n_points, config.ncols), dtype=cp.float32
        )
        points *= scale
        points += center

    scheme = resolve_norm_scheme(config)
    points = _rescale_to_scheme(points, scheme, config, cluster_id, rng)

    if return_cupy:
        return points
    return cp.asnumpy(points)


def get_num_points_per_cluster(
    total_points: int,
    config: Fingerprint,
    min_points_per_cluster: int = 1,
) -> np.ndarray:
    """Allocate ``total_points`` across clusters proportional to densities.

    Each cluster gets at least ``min_points_per_cluster``, then the remainder
    is distributed by density. Remaining points are spread round-robin so
    no single cluster absorbs the rounding error.
    """
    nclusters = config.nclusters
    min_required = nclusters * min_points_per_cluster
    if total_points < min_required:
        raise ValueError(
            f"total_points={total_points} is less than "
            f"nclusters * min_points_per_cluster = {min_required}. "
            f"Increase total_points or decrease nclusters."
        )

    points_per_cluster = np.full(
        nclusters, min_points_per_cluster, dtype=np.int64
    )
    remaining = total_points - min_required
    if remaining > 0:
        extra = (config.cluster_densities * remaining).astype(np.int64)
        points_per_cluster += extra

        diff = total_points - int(points_per_cluster.sum())
        if diff != 0:
            for i in range(abs(diff)):
                c = i % nclusters
                points_per_cluster[c] += 1 if diff > 0 else -1
    return points_per_cluster


def generate_synthetic_dataset(
    config: Fingerprint,
    total_points: int,
) -> np.ndarray:
    """Materialize a full ``(total_points, ncols)`` synthetic dataset.

    For very large ``total_points``, prefer
    :func:`generate_synthetic_dataset_to_file` to keep host RAM bounded.
    """
    points_per_cluster = get_num_points_per_cluster(total_points, config)

    ncols = config.cluster_centers.shape[1]
    result = np.empty((total_points, ncols), dtype=np.float32)
    write_idx = 0

    iterator = tqdm(
        range(config.nclusters), desc="Generating synthetic dataset"
    )

    for cluster_id in iterator:
        n_points = int(points_per_cluster[cluster_id])
        if n_points <= 0:
            continue
        cluster_points = gen_cluster_gpu(
            cluster_id,
            n_points,
            config,
            return_cupy=False,
        )
        result[write_idx : write_idx + n_points] = cluster_points
        write_idx += n_points

    cp.get_default_memory_pool().free_all_blocks()

    return result


def generate_synthetic_dataset_to_file(
    config: Fingerprint,
    total_points: int,
    output_path: str,
    batch_size: int = 1_000_000,
) -> None:
    """Stream a synthetic dataset directly to a fbin file.

    Parameters
    ----------
    config : Fingerprint
        Fitted cluster fingerprint.
    total_points : int
        Number of synthetic vectors to generate.
    output_path : str
        Destination ``.fbin`` path. Parent directories are created as needed.
    batch_size : int
        Number of rows for a flush threshold.
    """
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    points_per_cluster = get_num_points_per_cluster(total_points, config)
    ncols = int(config.cluster_centers.shape[1])

    # Buffer must be large enough to absorb the cluster that triggers a
    # flush without overflowing
    buf_capacity = batch_size + int(points_per_cluster.max())
    bufs = [
        np.empty((buf_capacity, ncols), dtype=np.float32) for _ in range(2)
    ]
    active_buf = 0
    buf_offset = 0
    rows_written = 0
    write_thread: threading.Thread | None = None

    def _wait_for_write() -> None:
        nonlocal write_thread
        if write_thread is not None:
            write_thread.join()
            write_thread = None

    def _flush_async(f, buf_view: np.ndarray) -> None:
        nonlocal write_thread
        write_thread = threading.Thread(
            target=lambda: buf_view.tofile(f), daemon=True
        )
        write_thread.start()

    iterator = tqdm(
        range(config.nclusters),
        desc="Streaming synthetic dataset",
        total=config.nclusters,
    )

    with open(output_path, "wb") as f:
        write_bin_header(f, total_points, ncols)

        for cluster_id in iterator:
            n_points = int(points_per_cluster[cluster_id])
            if n_points <= 0:
                continue

            cluster_points = gen_cluster_gpu(
                cluster_id,
                n_points,
                config,
                return_cupy=False,
            )

            bufs[active_buf][buf_offset : buf_offset + n_points] = (
                cluster_points
            )
            buf_offset += n_points
            del cluster_points

            if buf_offset >= batch_size:
                # Wait for the previous write to complete before reusing
                # its buffer
                _wait_for_write()
                _flush_async(f, bufs[active_buf][:buf_offset])
                rows_written += buf_offset
                active_buf = 1 - active_buf
                buf_offset = 0
                cp.get_default_memory_pool().free_all_blocks()

        # wait for any in-flight write, then flush whatever's left in the active buffer.
        _wait_for_write()
        if buf_offset > 0:
            bufs[active_buf][:buf_offset].tofile(f)
            rows_written += buf_offset

    cp.get_default_memory_pool().free_all_blocks()

    if rows_written != total_points:
        print(
            f"  Note: wrote {rows_written:,}/{total_points:,} rows "
            f"(some clusters yielded zero points)."
        )


def generate_queries(
    nqueries: int,
    total_rows: int,
    config: Fingerprint,
    query_seed_offset: int = 999_999,
) -> np.ndarray:
    """Sample ``nqueries`` query points from the synthetic distribution.

    For each query we pick a global index uniformly from ``range(total_rows)``,
    determine which cluster it falls into, regenerate that cluster's points
    deterministically, and grab the corresponding local index. Small Gaussian
    noise is added at the end to avoid exact-match recall artefacts.
    """
    scheme = resolve_norm_scheme(config)
    points_per_cluster = get_num_points_per_cluster(total_rows, config)
    cumsum = np.cumsum(points_per_cluster)

    rng = np.random.default_rng(config.seed + query_seed_offset)
    global_indices = rng.choice(total_rows, size=nqueries, replace=False)

    # ``searchsorted(side='right')`` maps a global index to the cluster_id of
    # the cluster whose cumulative range contains it.
    cluster_ids = np.searchsorted(cumsum, global_indices, side="right")

    # Group queries by cluster so we generate each cluster only once.
    cluster_to_queries: dict[int, list[tuple[int, int]]] = {}
    for q_idx, (g_idx, c_id) in enumerate(zip(global_indices, cluster_ids)):
        local_idx = g_idx if c_id == 0 else g_idx - cumsum[c_id - 1]
        cluster_to_queries.setdefault(int(c_id), []).append(
            (q_idx, int(local_idx))
        )

    queries = np.zeros((nqueries, config.ncols), dtype=np.float32)
    for c_id, qlist in tqdm(
        cluster_to_queries.items(),
        total=len(cluster_to_queries),
        desc="Generating query points",
    ):
        n_points = int(points_per_cluster[c_id])
        cluster_points = gen_cluster_gpu(
            c_id,
            n_points,
            config,
            return_cupy=False,
        )
        for q_idx, local_idx in qlist:
            queries[q_idx] = cluster_points[local_idx]

    return add_jitter(queries, rng, normalize=(scheme == "unit"))
