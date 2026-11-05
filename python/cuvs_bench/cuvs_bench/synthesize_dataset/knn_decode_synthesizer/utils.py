#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Shared helpers: I/O, logging, cuVS wrappers, and the per-cluster residual sampler."""

from __future__ import annotations

import os
import struct
import time

import numpy as np
import torch
from tqdm import tqdm

# Row-chunk for every streaming GPU pass (decode, residual fit, norms, mean/std,
# graph wiring, brute-force GT tiles).  Caps peak memory to ~one chunk's worth of
# rows.
CHUNK_SIZE = 1_000_000


def load_fbin(path):
    """Load an .fbin file (``[n, d]`` int32 header + ``n*d`` float32) as (n, d)."""
    with open(path, "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
    return np.fromfile(path, dtype=np.float32, offset=8).reshape(n, d)


def ts():
    return time.strftime("%H:%M:%S")


def section(title):
    bar = "=" * 70
    print(f"\n{bar}", flush=True)
    print(f"  {ts()}  {title}", flush=True)
    print(bar, flush=True)


def step(msg):
    print(f"[{ts()}] {msg}", flush=True)


def done(msg, t0):
    elapsed = time.perf_counter() - t0
    print(f"[{ts()}] DONE — {msg}  ({elapsed:.1f}s)", flush=True)


class PhaseTimer:
    """Accumulate per-phase wall time, grouped (e.g. 'fit' / 'generate'), and print
    a breakdown at the end.  Use `t0 = timer.start()` around a phase, then
    `timer.lap('fit', 'kmeans', t0)` (also prints the usual DONE line).
    """

    def __init__(self):
        self._laps = []  # list of (group, name, seconds)

    def start(self):
        return time.perf_counter()

    def lap(self, group, name, t0):
        el = time.perf_counter() - t0
        self._laps.append((group, name, el))
        print(f"[{ts()}] DONE — {name}  ({el:.1f}s)", flush=True)
        return el

    def summary(self):
        section("TIMING SUMMARY")
        groups, order = {}, []
        for g, n, s in self._laps:
            if g not in groups:
                groups[g] = []
                order.append(g)
            groups[g].append((n, s))
        grand = sum(s for _, _, s in self._laps)
        for g in order:
            items = groups[g]
            tot = sum(s for _, s in items)
            print(
                f"  {g.upper():9s} total: {tot:9.1f}s   ({100 * tot / grand:4.1f}% of run)",
                flush=True,
            )
            for n, s in items:
                bar = "#" * round(30 * s / max(tot, 1e-9))
                print(
                    f"      {n:26s} {s:9.1f}s  {100 * s / tot:5.1f}%  {bar}",
                    flush=True,
                )
        print(f"  {'TOTAL':9s}      : {grand:9.1f}s", flush=True)


def normalize_features(feats, mu, sd):
    """Z-score features with given per-dim stats: (feats - mu) / sd.
    Fit the stats once with batched_mean_std (GPU); this just applies them.
    """
    return ((feats - mu) / sd).astype(np.float32)


def batched_mean_std(X, chunk=CHUNK_SIZE):
    """Per-dim mean & std of a (n, D) array, computed in batches.
    Returns (mu (1, D), sd (1, D)) float32.
    """
    import cupy as cp

    n, D = X.shape
    s1 = cp.zeros(D, dtype=cp.float64)  # running sum(x)
    s2 = cp.zeros(D, dtype=cp.float64)  # running sum(x^2)
    for i in range(0, n, chunk):
        xb = cp.asarray(X[i : i + chunk])  # (b, D) on GPU
        s1 += xb.sum(0, dtype=cp.float64)
        s2 += (xb * xb).sum(0, dtype=cp.float64)
    mu = s1 / n
    var = cp.maximum(
        s2 / n - mu * mu, 0.0
    )  # guard tiny negative from round-off
    sd = cp.sqrt(var) + 1e-6
    return (
        cp.asnumpy(mu).astype(np.float32)[None, :],
        cp.asnumpy(sd).astype(np.float32)[None, :],
    )


def build_all_neighbors(data, k):
    """Approximate all-neighbors kNN via the cuVS all_neighbors API."""
    import cupy as cp
    from cuvs.neighbors import all_neighbors, nn_descent

    gd = max(128, k + 1)
    nnd = nn_descent.IndexParams(
        metric="sqeuclidean",
        graph_degree=gd,
        intermediate_graph_degree=2 * gd,
        max_iterations=100,
    )

    params = all_neighbors.AllNeighborsParams(
        algo="nn_descent",
        metric="sqeuclidean",
        nn_descent_params=nnd,
    )
    idx = all_neighbors.build(data, k + 1, params)  # (N, k+1) int64
    return cp.asnumpy(cp.asarray(idx)).astype(np.int64)[
        :, 1:
    ]  # drop self column


def build_kmeans(X, n_clusters, seed=42):
    """Kmeans on GPU via cuVS.  Returns (labels_np int32, centroids_np float32)."""
    import cupy as cp
    from cuvs.cluster import kmeans as cuvs_kmeans

    n = X.shape[0]
    batch = int(min(n, max(50_000, 1_000_000_000 // max(n_clusters, 1))))
    params = cuvs_kmeans.KMeansParams(
        n_clusters=n_clusters, max_iter=300, device_buffer_samples=batch
    )
    centroids_out, _, _ = cuvs_kmeans.fit(params, X)

    lab, _ = cuvs_kmeans.predict(
        params, cp.asarray(X), cp.asarray(centroids_out)
    )
    labels = cp.asnumpy(cp.asarray(lab)).astype(np.int32)
    centroids = cp.asnumpy(cp.asarray(centroids_out)).astype(np.float32)
    return labels, centroids


def sample_residuals(cluster_chunk, resid_params, D, device):
    """Sample one residual per node from its cluster's Gaussian.

    cluster_chunk : (B,) int64 tensor of cluster labels
    -> (B, D) float32 tensor (zeros for clusters with no fitted Gaussian)
    """
    B = cluster_chunk.shape[0]
    out = torch.zeros(B, D, device=device)
    for c in torch.unique(cluster_chunk).tolist():
        p = resid_params[c]
        if p is None:
            continue
        sel = (cluster_chunk == c).nonzero(as_tuple=True)[0]
        n = sel.shape[0]
        z = torch.randn(n, p["comps"].shape[0], device=device) * p["stds"]
        s = z @ p["comps"]  # (n, D)
        eps = torch.randn(n, D, device=device) * p["noise_std"]
        out[sel] = p["mean"] + s + eps
    return out


def write_fbin(path, A):
    """Write a 2-D array to a cuvs-bench .fbin (uint32 [n, d] header + float32 data)."""
    A = np.ascontiguousarray(A, dtype=np.float32)
    with open(path, "wb") as f:
        f.write(struct.pack("<II", A.shape[0], A.shape[1]))
        f.write(A.tobytes())


def write_fbin_header(f, n, d):
    """Write just the cuvs-bench .fbin header (uint32 n, uint32 d) to an open file,
    so the row data can be appended in streaming chunks afterwards.
    """
    f.write(struct.pack("<II", int(n), int(d)))


def read_fbin_rows(path, start, end, d):
    """Read rows [start, end) from a .fbin (8-byte header + row-major float32) as
    (end-start, d).  Lets ground truth stream the base back from disk in tiles.
    """
    a = np.fromfile(
        path,
        dtype=np.float32,
        count=(end - start) * d,
        offset=8 + start * d * 4,
    )
    return a.reshape(end - start, d)


def holdout_split(pool, n_q, seed):
    """Split `pool` into (base, queries): `n_q` random rows held out as queries."""
    n = len(pool)
    rng = np.random.default_rng(seed)
    qidx = rng.choice(n, min(n_q, n), replace=False)
    mask = np.ones(n, dtype=bool)
    mask[qidx] = False
    return np.ascontiguousarray(pool[mask]), np.ascontiguousarray(pool[qidx])


def _write_exact_gt(
    out_dir, queries, base_count, gt_k, get_tile, tile=CHUNK_SIZE
):
    """Exact brute-force GT over `base_count` base rows.  A running
    top-k is merged across tiles and written to groundtruth.neighbors.ibin +
    groundtruth.distances.fbin.
    """
    import cupy as cp
    from cuvs.neighbors import brute_force

    k = gt_k
    step(
        f"exact GT (brute force, k={k}) — batched over {base_count:,} base in "
        f"{tile:,}-row tiles ..."
    )
    q_d = cp.asarray(np.ascontiguousarray(queries, dtype=np.float32))
    gt = np.full((len(queries), k), -1, dtype=np.int64)
    gtd = np.full((len(queries), k), np.inf, dtype=np.float32)
    for s in tqdm(range(0, base_count, tile), desc="exact GT", unit="tile"):
        e = min(s + tile, base_count)
        tile_d = cp.asarray(get_tile(s, e))
        idx = brute_force.build(tile_d, metric="sqeuclidean")
        dd, ii = brute_force.search(idx, q_d, min(k, e - s))
        dd = cp.asnumpy(dd).astype(np.float32)
        ii = cp.asnumpy(ii).astype(np.int64) + s  # local -> global row id
        merged_i = np.concatenate([gt, ii], axis=1)
        merged_d = np.concatenate([gtd, dd], axis=1)
        order = np.argsort(merged_d, axis=1)[:, :k]
        gt = np.take_along_axis(merged_i, order, axis=1)
        gtd = np.take_along_axis(merged_d, order, axis=1)
        del tile_d, idx
    with open(os.path.join(out_dir, "groundtruth.neighbors.ibin"), "wb") as f:
        f.write(struct.pack("<II", gt.shape[0], gt.shape[1]))
        f.write(np.ascontiguousarray(gt, dtype=np.uint32).tobytes())
    with open(os.path.join(out_dir, "groundtruth.distances.fbin"), "wb") as f:
        f.write(struct.pack("<II", gtd.shape[0], gtd.shape[1]))
        f.write(np.ascontiguousarray(gtd, dtype=np.float32).tobytes())
    del q_d
    print(
        f"  wrote groundtruth.neighbors.ibin + groundtruth.distances.fbin "
        f"({gt.shape[0]} x {gt.shape[1]})",
        flush=True,
    )


def write_bundle(out_dir, base, queries, gt_k, tile=CHUNK_SIZE):
    """Write a benchmark bundle to out_dir: base.fbin, queries.fbin, and exact
    brute-force ground truth.  `base` is a full (N, D) array in host RAM.
    """
    write_fbin(os.path.join(out_dir, "base.fbin"), base)
    write_fbin(os.path.join(out_dir, "queries.fbin"), queries)
    print(
        f"  base {base.shape}  queries {queries.shape}  -> {out_dir}",
        flush=True,
    )
    _write_exact_gt(
        out_dir,
        queries,
        len(base),
        gt_k,
        lambda s, e: np.ascontiguousarray(base[s:e], dtype=np.float32),
        tile,
    )


def write_bundle_streamed(
    out_dir, base_path, base_count, queries, gt_k, d, tile=CHUNK_SIZE
):
    """Finalize a bundle whose base.fbin was already STREAMED to disk by the decoder
    (see decode_graph's streaming mode).  Writes queries.fbin and exact brute-force
    GT, reading the base back from disk one tile at a time — so the full base never
    has to be resident in host RAM nor on the GPU.
    """
    write_fbin(os.path.join(out_dir, "queries.fbin"), queries)
    print(
        f"  base {base_count:,} x {d} (streamed to disk)  "
        f"queries {queries.shape}  -> {out_dir}",
        flush=True,
    )
    _write_exact_gt(
        out_dir,
        queries,
        base_count,
        gt_k,
        lambda s, e: read_fbin_rows(base_path, s, e, d),
        tile,
    )
