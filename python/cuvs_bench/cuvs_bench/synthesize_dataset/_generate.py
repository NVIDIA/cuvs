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

Generation is deterministic in ``(cluster_id, config.seed)`` so the same call
always produces the same points -- this is essential for the ``nprobe`` GT
trick where the same cluster gets regenerated from scratch on each query.
"""

from __future__ import annotations

import os
import threading
from typing import Union

import cupy as cp
import numpy as np
from tqdm import tqdm

from ._config import ClusterConfig

UINT32_MAX = (1 << 32) - 1

ArrayLike = Union[np.ndarray, cp.ndarray]


def get_cluster_seed(base_seed: int, cluster_id: int) -> int:
    """Deterministic per-cluster seed. Multiply the base seed by 1000000 for different sequences for different base_seeds."""
    return base_seed * 1_000_000 + int(cluster_id)


def gen_cluster_gpu(
    cluster_id: int,
    n_points: int,
    config: ClusterConfig,
    return_cupy: bool = True,
    normalize: bool = True,
) -> ArrayLike:
    """Generate ``n_points`` points for one cluster on the GPU"""
    center = cp.asarray(config.cluster_centers[cluster_id])

    if n_points == 1:
        points = center.reshape(1, -1)
        if normalize:
            norms = cp.linalg.norm(points, axis=1, keepdims=True)
            points = points / cp.maximum(norms, 1e-8)
        if return_cupy:
            return points.astype(cp.float32)
        return cp.asnumpy(points).astype(np.float32)

    seed = get_cluster_seed(config.seed, cluster_id)
    rng = cp.random.RandomState(seed)
    n_random = n_points - 1

    components_host = config.pca_components_list[cluster_id]
    explained_var_host = config.pca_explained_var_list[cluster_id]
    noise_var = float(config.pca_noise_var[cluster_id])

    if components_host is not None:
        components = cp.asarray(components_host)
        explained_var = cp.asarray(explained_var_host)
        k = len(explained_var)

        z = rng.standard_normal(
            size=(n_random, k), dtype=cp.float32
        ) * cp.sqrt(cp.maximum(explained_var, 0.0))
        projected = z @ components

        noise_std = float(cp.sqrt(max(noise_var, 0.0)))
        noise = (
            rng.standard_normal(
                size=(n_random, config.ncols), dtype=cp.float32
            )
            * noise_std
        )

        centered = projected + noise

        # Per-dimension scaling for variance matching
        # Only apply if we have enough points for reliable variance estimation
        if n_random >= 10:
            target_var = cp.asarray(config.cluster_variances[cluster_id])
            actual_var = cp.var(centered, axis=0)
            scale = cp.sqrt(target_var / cp.maximum(actual_var, noise_var))
            # Scale each dimension separately
            # Use noise_var as floor since that's the minimum expected variance
            # Cap scale_factors to 5x to prevent blowup from unreliable variance estimates
            scale = cp.minimum(scale, 5.0)
            centered = (
                centered * scale
            )  # broadcasts (n_random, ncols) * (ncols,)

        random_points = center + centered
    else:
        # Diagonal fallback (in-place for memory efficiency at scale).
        scale = cp.sqrt(cp.asarray(config.cluster_variances[cluster_id]))
        random_points = rng.standard_normal(
            size=(n_random, config.ncols), dtype=cp.float32
        )
        random_points *= scale
        random_points += center

    points = cp.vstack([center.reshape(1, -1), random_points]).astype(
        cp.float32
    )

    if normalize:
        norms = cp.linalg.norm(points, axis=1, keepdims=True)
        points = points / cp.maximum(norms, 1e-8)

    if return_cupy:
        return points
    return cp.asnumpy(points)


def get_num_points_per_cluster(
    total_points: int,
    config: ClusterConfig,
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
    config: ClusterConfig,
    total_points: int,
) -> np.ndarray:
    """Materialize a full ``(total_points, ncols)`` synthetic dataset.

    For very large ``total_points``, prefer
    :func:`generate_synthetic_dataset_to_file` to keep host RAM bounded.
    """
    normalize = bool(config.is_normalized_data)
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
            normalize=normalize,
        )
        result[write_idx : write_idx + n_points] = cluster_points
        write_idx += n_points

        # Periodically free GPU pool to keep fragmentation bounded across
        # the long-running per-cluster loop.
        if cluster_id % 500 == 499:
            cp.get_default_memory_pool().free_all_blocks()

    cp.get_default_memory_pool().free_all_blocks()

    return result


def generate_synthetic_dataset_to_file(
    config: ClusterConfig,
    total_points: int,
    output_path: str,
    batch_size: int = 1_000_000,
) -> int:
    """Stream a synthetic dataset directly to a fbin file.

    Generates clusters one at a time on the GPU, accumulates their points
    into a small CPU-side double-buffer, and flushes ``batch_size``-row
    chunks to disk in a background thread so disk I/O overlaps with the
    next cluster's generation. Peak host memory stays bounded by roughly
    ``2 * (batch_size + max_cluster_points) * ncols * 4`` bytes regardless
    of ``total_points`` -- prefer this over
    :func:`generate_synthetic_dataset` whenever the full target synthetic dataset
    wouldn't comfortably fit in host RAM.

    The output file uses the canonical cuvs-bench fbin layout (uint32
    ``n_rows``, uint32 ``n_dim`` header followed by row-major float32
    data), readable by ``cuvs_bench.run`` and the rest of the toolchain.

    Parameters
    ----------
    config : ClusterConfig
        Fitted cluster fingerprint.
    total_points : int
        Number of synthetic vectors to generate. Capped at ``UINT32_MAX``
        because the cuvs-bench fbin reader uses a uint32 row-count header.
    output_path : str
        Destination ``.fbin`` path. Parent directories are created as needed.
    batch_size : int
        Flush threshold in rows. Higher = better disk throughput but more
        host RAM. Default 1M.

    Returns
    -------
    int
        Number of rows written. Always equals ``total_points`` unless a
        cluster yielded zero points.
    """
    if total_points > UINT32_MAX:
        raise ValueError(
            f"total_points={total_points:,} exceeds the cuvs-bench fbin "
            f"uint32 row-count cap of {UINT32_MAX:,}. Generate multiple "
            f"shards if you need more rows."
        )

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    normalize = bool(config.is_normalized_data)
    points_per_cluster = get_num_points_per_cluster(total_points, config)
    ncols = int(config.cluster_centers.shape[1])

    # Buffer must be large enough to absorb the cluster that triggers a
    # flush without overflowing -- size it for batch_size + the largest
    # single cluster plus a small safety margin.
    max_cluster_pts = int(points_per_cluster.max())
    buf_capacity = max(int(batch_size * 1.1), batch_size + max_cluster_pts)
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
        # Canonical cuvs-bench fbin header
        np.asarray([total_points, ncols], dtype=np.uint32).tofile(f)

        for cluster_id in iterator:
            n_points = int(points_per_cluster[cluster_id])
            if n_points <= 0:
                continue

            cluster_points = gen_cluster_gpu(
                cluster_id,
                n_points,
                config,
                return_cupy=False,
                normalize=normalize,
            )

            bufs[active_buf][buf_offset : buf_offset + n_points] = (
                cluster_points
            )
            buf_offset += n_points
            del cluster_points

            if buf_offset >= batch_size:
                # Wait for the previous write to complete before reusing
                # *its* buffer (the one we're about to swap to).
                _wait_for_write()
                _flush_async(f, bufs[active_buf][:buf_offset])
                rows_written += buf_offset
                active_buf = 1 - active_buf
                buf_offset = 0
                cp.get_default_memory_pool().free_all_blocks()

        # Drain: wait for any in-flight write, then synchronously flush
        # whatever's left in the active buffer.
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

    return rows_written
