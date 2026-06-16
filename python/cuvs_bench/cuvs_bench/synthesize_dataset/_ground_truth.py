#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
"""Query generation and ground-truth computation for the synthetic dataset.

Two GT modes:

- :func:`compute_groundtruth_exact` checks every cluster, computes brute-force
  k-NN against the queries, and merges results across clusters.
- :func:`compute_groundtruth_nprobe` only checks the ``nprobes`` nearest
  clusters per query (IVF-style).
"""

from __future__ import annotations

import time
from typing import Tuple

import cupy as cp
import numpy as np
from cuvs.neighbors import brute_force as cuvs_brute_force
from tqdm import tqdm

from ._fingerprint import Fingerprint
from ._generate import gen_cluster_gpu, get_num_points_per_cluster


def _bf_knn_gpu(
    database_gpu: cp.ndarray,
    queries_gpu: cp.ndarray,
    k: int,
    metric: str = "sqeuclidean",
) -> Tuple[np.ndarray, np.ndarray]:
    """Brute-force k-NN on GPU using cuvs."""
    index = cuvs_brute_force.build(database_gpu, metric=metric)
    distances, indices = cuvs_brute_force.search(index, queries_gpu, k)
    return indices.copy_to_host(), distances.copy_to_host()


def _find_nearby_clusters(
    queries: np.ndarray,
    config: Fingerprint,
    nprobes: int,
    metric: str = "sqeuclidean",
) -> np.ndarray:
    """Return the ``nprobes`` nearest cluster ids for each query."""
    centers_gpu = cp.asarray(config.cluster_centers, dtype=cp.float32)
    queries_gpu = cp.asarray(queries, dtype=cp.float32)
    indices, _ = _bf_knn_gpu(centers_gpu, queries_gpu, nprobes, metric=metric)
    return indices  # (nqueries, nprobes)


def compute_groundtruth_exact(
    queries: np.ndarray,
    total_rows: int,
    config: Fingerprint,
    k: int,
    metric: str = "sqeuclidean",
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Exact brute-force GT, computed by streaming every cluster.
    Each cluster is regenerated, brute-forced against the queries, and the
    running top-k is merged.
    """
    normalize = bool(config.is_normalized_data)
    nqueries = len(queries)
    points_per_cluster = get_num_points_per_cluster(total_rows, config)
    cumsum = np.cumsum(points_per_cluster)

    gt_indices = np.full((nqueries, k), -1, dtype=np.int64)
    gt_distances = np.full((nqueries, k), np.inf, dtype=np.float32)

    queries_gpu = cp.asarray(queries, dtype=cp.float32)

    total_gen_time = 0.0
    total_bf_time = 0.0
    total_merge_time = 0.0

    iterator = tqdm(range(config.nclusters), desc="Exact GT (streaming)")

    for cluster_id in iterator:
        n_points = int(points_per_cluster[cluster_id])
        global_start = 0 if cluster_id == 0 else int(cumsum[cluster_id - 1])

        t0 = time.perf_counter()
        cluster_gpu = gen_cluster_gpu(
            cluster_id, n_points, config, return_cupy=True, normalize=normalize
        )
        total_gen_time += time.perf_counter() - t0

        t0 = time.perf_counter()
        k_cluster = min(k, n_points)
        local_indices, local_dists = _bf_knn_gpu(
            cluster_gpu, queries_gpu, k_cluster, metric=metric
        )
        total_bf_time += time.perf_counter() - t0

        global_indices = local_indices.astype(np.int64) + global_start

        t0 = time.perf_counter()
        merged_idx = np.concatenate([gt_indices, global_indices], axis=1)
        merged_dist = np.concatenate([gt_distances, local_dists], axis=1)
        order = np.argsort(merged_dist, axis=1)[:, :k]
        gt_indices = np.take_along_axis(merged_idx, order, axis=1)
        gt_distances = np.take_along_axis(merged_dist, order, axis=1)
        total_merge_time += time.perf_counter() - t0

        # Free cluster's points before the next iteration.
        del cluster_gpu

    timing = {
        "gen_time": total_gen_time,
        "bf_time": total_bf_time,
        "merge_time": total_merge_time,
    }
    return gt_indices, gt_distances, timing


def compute_groundtruth_nprobe(
    queries: np.ndarray,
    total_rows: int,
    config: Fingerprint,
    k: int,
    nprobes: int,
    metric: str = "sqeuclidean",
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Cheap GT via cluster probing: only the ``nprobes`` nearest clusters
    per query are regenerated and searched.
    """
    normalize = bool(config.is_normalized_data)
    nqueries = len(queries)
    points_per_cluster = get_num_points_per_cluster(total_rows, config)
    cumsum = np.cumsum(points_per_cluster)

    t0 = time.perf_counter()
    nearby = _find_nearby_clusters(queries, config, nprobes, metric=metric)
    find_clusters_time = time.perf_counter() - t0

    gt_indices = np.full((nqueries, k), -1, dtype=np.int64)
    gt_distances = np.full((nqueries, k), np.inf, dtype=np.float32)

    # Reverse mapping: cluster_id -> queries that probe it.
    t0 = time.perf_counter()
    cluster_to_queries: dict[int, list[int]] = {}
    for q_idx in range(nqueries):
        for c_id in nearby[q_idx]:
            cluster_to_queries.setdefault(int(c_id), []).append(q_idx)
    mapping_time = time.perf_counter() - t0

    total_gen_time = 0.0
    total_bf_time = 0.0
    total_merge_time = 0.0

    iterator = tqdm(
        cluster_to_queries.items(),
        desc="Nprobe GT",
        total=len(cluster_to_queries),
    )

    for cluster_id, q_list in iterator:
        n_points = int(points_per_cluster[cluster_id])
        global_start = 0 if cluster_id == 0 else int(cumsum[cluster_id - 1])
        q_idx_arr = np.asarray(q_list, dtype=np.int64)
        batch_queries = queries[q_idx_arr]
        batch_queries_gpu = cp.asarray(batch_queries, dtype=cp.float32)

        t0 = time.perf_counter()
        cluster_gpu = gen_cluster_gpu(
            cluster_id, n_points, config, return_cupy=True, normalize=normalize
        )
        total_gen_time += time.perf_counter() - t0

        t0 = time.perf_counter()
        k_cluster = min(k, n_points)
        local_indices, local_dists = _bf_knn_gpu(
            cluster_gpu, batch_queries_gpu, k_cluster, metric=metric
        )
        total_bf_time += time.perf_counter() - t0

        global_indices = local_indices.astype(np.int64) + global_start

        t0 = time.perf_counter()
        cur_idx = gt_indices[q_idx_arr]
        cur_dist = gt_distances[q_idx_arr]
        merged_idx = np.concatenate([cur_idx, global_indices], axis=1)
        merged_dist = np.concatenate([cur_dist, local_dists], axis=1)
        order = np.argsort(merged_dist, axis=1)[:, :k]
        gt_indices[q_idx_arr] = np.take_along_axis(merged_idx, order, axis=1)
        gt_distances[q_idx_arr] = np.take_along_axis(
            merged_dist, order, axis=1
        )
        total_merge_time += time.perf_counter() - t0

        del cluster_gpu

    timing = {
        "find_clusters_time": find_clusters_time,
        "mapping_time": mapping_time,
        "gen_time": total_gen_time,
        "bf_time": total_bf_time,
        "merge_time": total_merge_time,
    }
    return gt_indices, gt_distances, timing
