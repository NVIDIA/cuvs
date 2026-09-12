#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""kNN upsampling: build the N-node synthetic graph from the sample kNN."""

from __future__ import annotations

import numpy as np
from tqdm import tqdm
from utils import CHUNK_SIZE


def _node_coherence(nbr, idx):
    """Triangle Density for the nodes in `idx`.  Returns (len(idx),) float32: fraction
    of each node's ordered neighbor-pairs (j, j') where j' is also its neighbor.
    """
    k = nbr.shape[1]
    nb = nbr[idx]  # (m, k) i.e. the k nbrs
    nnb = nbr[nb]  # (m, k, k) nbrs-of-nbrs (2-hop)
    match = (nnb[..., None] == nb[:, None, None, :]).any(
        -1
    )  # (m, k, k) j' in N(i)?
    return (match.sum((1, 2)) / (k * (k - 1))).astype(
        np.float32
    )  # (m) frac btw [0,1]


def _est_coherence(nbr, seed=0, m=4000):
    """Mean local coherence over a random sample of m nodes
    — a cheap scalar for the triadic window-sizing loop.
    """
    n_nodes, k = nbr.shape
    if k < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_nodes, size=min(m, n_nodes))
    return float(_node_coherence(nbr, idx).mean())


def blend_chunglu(core_nbr, k_keep, indeg_dist, N, k, seed, chunk=CHUNK_SIZE):
    """Blend a coherent/structured neighbor matrix with a Chung-Lu hub backbone.

    For each node i, keep its first ``k_keep[i]`` core neighbors and fill the rest of
    its k edges with global degree-weighted random draws: node attractiveness is
    sampled from the real sample in-degree distribution (indeg_dist) and targets are drawn
    proportional to it, so a few nodes accumulate many in-edges — the heavy
    in-degree tail (hubs) the index leans on.  Returns (N, k) int64.

    Note: the result may contain DUPLICATE neighbors and self-loops. This is not a problem because
    `nbr` is a throwaway decoder input. A duplicate just gives that neighbor a little extra
    weight in the order-invariant attention aggregate.
    """
    import cupy as cp

    rng = np.random.default_rng(seed)
    weights = (
        rng.choice(indeg_dist, size=N).astype(np.float64) + 1e-6
    )  # per-node attractiveness
    cdf = cp.cumsum(cp.asarray(weights))  # inverse-CDF over N targets
    cdf /= cdf[-1]  # -> [0, 1]
    del weights

    out = np.empty((N, k), dtype=np.int64)
    ar_k = np.arange(k)
    crng = cp.random.RandomState(seed + 11)
    for s in tqdm(range(0, N, chunk), desc="chung-lu blend", unit="chunk"):
        e = min(s + chunk, N)
        b = e - s
        u = crng.random_sample((b * k,))  # uniform draws
        rand = cp.searchsorted(cdf, u, side="right").reshape(
            b, k
        )  # (b, k) targets ∝ weight
        rand = cp.asnumpy(cp.minimum(rand, N - 1)).astype(np.int64)
        mask = ar_k[None, :] < k_keep[s:e, None]  # first k_keep are core
        out[s:e] = np.where(
            mask, core_nbr[s:e], rand
        )  # dups/self-loops possible; see note
    del cdf
    cp.get_default_memory_pool().free_all_blocks()
    return out


def triadic_coherent_knn(cluster_ids, k, seed, tgt):
    """Coordinate-free coherent graph.

    Within each cluster, lay the nodes in a random order and connect each to k random
    nodes inside a sliding window of width w.  Overlapping windows produce continuous,
    mutually-overlapping neighborhoods (triangles). The window width sets the coherence,
    so w is auto-sized to the sample's measured coherence (tgt) with one proportional
    correction. A smaller window results in a higher coherence.

    Returns (N, k) int64
    """
    import cupy as cp

    N = cluster_ids.shape[0]

    order = np.argsort(
        cluster_ids, kind="stable"
    )  # order[p] = actual global node id at position p
    starts, counts = np.unique(
        cluster_ids[order], return_index=True, return_counts=True
    )[1:]
    bstart = np.repeat(starts, counts).astype(
        np.int64
    )  # (N,) bstart[p] = where p's cluster block starts
    sz_pos = np.repeat(counts, counts).astype(
        np.int64
    )  # (N,) sz_pos[p] = p's cluster size
    loc = (np.arange(N) - bstart).astype(
        np.int64
    )  # (N,) loc[p] = p-bstart[p] = p's local index inside cluster
    order_d = cp.asarray(order.astype(np.int64))  # global ids, on GPU

    def build(w, bseed, chunk=CHUNK_SIZE):
        # Each node connects to k random nodes within +/- w/2 positions inside its
        # cluster block (overlapping windows -> triangles).  Done in node-chunks on
        # the GPU.
        half = max(1, w // 2)
        crng = cp.random.RandomState(bseed)
        nbr = np.empty((N, k), dtype=np.int64)
        for s in tqdm(
            range(0, N, chunk), desc=f"wiring coherence (w={w})", unit="chunk"
        ):
            e = min(s + chunk, N)
            loc_c = cp.asarray(loc[s:e])[:, None]  # (b,1)
            sz_c = cp.asarray(sz_pos[s:e])[:, None]  # (b,1)
            bst_c = cp.asarray(bstart[s:e])[:, None]  # (b,1)
            off = crng.randint(
                -half, half + 1, size=(e - s, k)
            )  # (b,k) window offsets
            tp = (loc_c + off) % sz_c  # local target positions
            tp = cp.where(tp == loc_c, (tp + 1) % sz_c, tp)  # avoid self
            tgt = order_d[bst_c + tp]  # (b,k) local -> global
            nbr[order[s:e]] = cp.asnumpy(tgt)  # scatter to node order
        return nbr

    w = max(k + 1, round(0.7 * k / max(tgt, 1e-3)))
    nbr = build(w, seed + 7)
    C = _est_coherence(nbr, seed=1)
    if abs(C - tgt) > 0.015 and C > 0:  # one proportional correction (C ~ 1/w)
        w = max(k + 1, round(w * C / tgt))
        nbr = build(w, seed + 8)
        C = _est_coherence(nbr, seed=1)
    print(
        f"  coherent seed: windowed in-cluster (w={w}), "
        f"coherence~{C:.3f} (target~{round(tgt, 3)})",
        flush=True,
    )
    return nbr


def generate_graph_knn(stats, N, k, seed, knn_frac=1.0):
    """Build coherent N-node kNN graph, blended with a Chung-Lu hub tail.

    Generate a coherent kNN graph using triadic_coherent_knn. blend_chunglu then
    keeps ~knn_frac*k coherent edges per node and fills the rest with degree-weighted
    random (Chung-Lu) edges. The coherentedges drive recall, the random tail drives
    the hub tail + search difficulty (knn_frac=1.0 = fully coherent). Larger knn_frac
    means an easier data.

    Returns (anchor_ids (N,) int32 — each node's anchor, nbr (N, k) int64).
    """
    import cupy as cp

    rng = np.random.default_rng(seed)
    indeg_dist = stats["indeg_dist"]
    n_anchors = len(indeg_dist)  # #anchors == sample size

    anchor_ids = rng.integers(0, n_anchors, size=N).astype(np.int32)
    knn_nbr = triadic_coherent_knn(
        anchor_ids, k, seed=seed, tgt=stats.get("coherence")
    )
    cp.get_default_memory_pool().free_all_blocks()

    kf_k = knn_frac * k
    if int(np.floor(kf_k)) >= k:
        return anchor_ids, knn_nbr

    # Per-node coherent count so the effective fraction is continuous
    base = int(np.floor(kf_k))
    fracp = kf_k - base
    k_knn_i = np.clip(
        base + (rng.random(N) < fracp).astype(np.int64), 0, k
    )  # (N,)

    # Blend: keep each node's k_knn_i coherent kNN neighbors, fill the rest with
    # the shared Chung-Lu hub backbone.
    nbr = blend_chunglu(knn_nbr, k_knn_i, indeg_dist, N, k, seed + 5)
    print(
        f"  blend: ~{kf_k:.2f} coherent kNN + ~{k - kf_k:.2f} chung-lu random "
        f"per node (knn_frac={knn_frac}, per-node stochastic)",
        flush=True,
    )
    return anchor_ids, nbr
