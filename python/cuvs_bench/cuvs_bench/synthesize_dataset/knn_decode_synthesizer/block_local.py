#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""block_local.py — streaming (block-local) version of generate + decode.

A memory-efficient alternative to (generate_graph_knn + decode_graph), enabled by
generate_data.py's ``--block-local`` flag: nodes are laid out in anchor order and
generated/decoded one block at a time, so the (N,k) graph and O(N) feature arrays
never materialise. The hub tail uses the parametric hub field. See README §5
("Block Decode mode") for the detailed explanation on the design.
"""

from __future__ import annotations

import numpy as np
import torch
from knn_decoder import STRUCT_FEAT_DIM, sample_norms_percentile
from tqdm import tqdm
from upsample import _est_coherence
from utils import normalize_features, sample_residuals, write_fbin_header


def _coherent_edges(loc, sz_of, gstart, k, w, rng):
    """Windowed within-cluster wiring -> (b, k) in-block global neighbor ids."""
    half = max(1, w // 2)
    offd = rng.integers(-half, half + 1, size=(loc.shape[0], k))
    tp = (loc[:, None] + offd) % sz_of[:, None]
    tp = np.where(tp == loc[:, None], (tp + 1) % sz_of[:, None], tp)
    return gstart[:, None] + tp


def _cluster_layout(nc, N, rng):
    """Spread N nodes uniformly over the nc anchors, laid out in anchor order.
    Returns (sizes (nc,), off (nc+1,)) int64 — off[c] = start id of anchor c.
    """
    sizes = rng.multinomial(N, np.full(nc, 1.0 / nc)).astype(np.int64)
    off = np.zeros(nc + 1, dtype=np.int64)
    np.cumsum(sizes, out=off[1:])
    return sizes, off


def _hub_field_setup(indeg_dist, k, knn_frac):
    """Precompute the on-the-fly hub sampler from the sample in-degree histogram.
    Returns (sb_cdf, cl_scale, hub_mni):

    - `sb_cdf`  : (SS,) float64 — inverse-CDF for the size-biased in-degree draw.
    - `cl_scale`: scalar float for transforming weight -> expected Chung-Lu in-degree. Total
                  Chung-Lu edges `e_cl = N·k·(1-knn_frac)` are spread proportional to the weight
                  over all N nodes, so a node with weight w expects
                  `e_cl·w/(N·E[w]) = k·(1-knn_frac)·w/E[w]` edges.
                  `cl_scale = k·(1-knn_frac)/E[w]` (N cancels). Multiplied by the weight
                  to get an in-degree during the decode loop.
    - `hub_mni` : scalar float — a hub's mean-neighbor-in-degree (approx): its k nbrs
                  are ~knn_frac coherent (low) and ~(1-knn_frac) other hubs (high, at
                  the size-biased mean in-degree).
    """
    w = indeg_dist.astype(np.float64) + 1e-9
    mean_w = float(w.mean())
    cl_scale = float(k * (1.0 - knn_frac) / mean_w)
    sb_cdf = np.cumsum(w)
    sb_cdf /= sb_cdf[-1]
    mean_sb_w = float(
        (w * w).sum() / w.sum()
    )  # E[w^2]/E[w] = size-biased mean weight
    mean_sb_indeg = mean_sb_w * cl_scale
    coh_deg = knn_frac * k
    hub_mni = float(knn_frac * coh_deg + (1.0 - knn_frac) * mean_sb_indeg)
    return sb_cdf, cl_scale, hub_mni


@torch.no_grad()
def generate_block_local(
    model,
    anchors,
    mu,
    sd,
    feat_mu,
    feat_sd,
    resid_params,
    norm_q,
    cluster_of_anchor,
    stats,
    N,
    n_queries,
    k,
    knn_frac,
    seed,
    device,
    base_path,
    block_size=1_000_000,
):
    """Cluster-ordered, block-fused generate+decode. Streams base rows to `base_path`
    and returns the held-out queries (n_queries, D). See module docstring.

    anchors : (nc, D) host anchor table (row = anchor = sample point).
    """
    indeg_dist = stats["indeg_dist"]
    nc = len(indeg_dist)  # # anchors (= sample size)
    tgt = stats.get("coherence", 0.3) or 0.3
    D = mu.shape[1]
    w = max(k + 1, round(0.7 * k / max(tgt, 1e-3)))

    rng = np.random.default_rng(seed)
    sizes, off = _cluster_layout(nc, N, rng)

    mu_t = torch.tensor(mu, device=device)
    sd_t = torch.tensor(sd, device=device)

    sb_cdf, cl_scale, hub_mni_p = _hub_field_setup(indeg_dist, k, knn_frac)

    # held-out queries
    q_idx = np.sort(
        np.random.default_rng(seed + 999).choice(
            N, size=n_queries, replace=False
        )
    )
    base_count = N - n_queries
    queries = np.empty((n_queries, D), dtype=np.float32)
    q_off = 0
    fbase = open(base_path, "wb")
    write_fbin_header(fbase, base_count, D)

    # group whole clusters into ~block_size-node blocks
    blocks, c0, acc = [], 0, 0
    for c in range(nc):
        acc += sizes[c]
        if acc >= block_size or c == nc - 1:
            blocks.append((c0, c + 1))
            c0, acc = c + 1, 0

    # one proportional window correction (mirrors triadic) on the first block
    ca0, cb0 = blocks[0]
    lo0, hi0 = int(off[ca0]), int(off[cb0])
    if hi0 - lo0 > k + 1:
        g0 = np.repeat(off[ca0:cb0], sizes[ca0:cb0])
        s0 = np.repeat(sizes[ca0:cb0], sizes[ca0:cb0])
        l0 = np.arange(lo0, hi0) - g0
        c0e = (
            _coherent_edges(l0, s0, g0, k, w, np.random.default_rng(seed + 7))
            - lo0
        )
        C = _est_coherence(c0e.astype(np.int64), seed=1)
        if C > 0 and abs(C - tgt) > 0.015:
            w = max(k + 1, round(w * C / tgt))

    base_k = int(np.floor(knn_frac * k))
    fracp = knn_frac * k - base_k
    ar_k = np.arange(k)
    torch.manual_seed(seed + 1)

    for bi, (ca, cb) in enumerate(
        tqdm(blocks, desc="block-local gen+decode", unit="block")
    ):
        lo, hi = int(off[ca]), int(off[cb])
        b = hi - lo
        if b == 0:
            continue
        brng = np.random.default_rng(seed + 100 + bi)

        # --- per-node cluster / local-position ---
        cl_of = np.repeat(
            np.arange(ca, cb, dtype=np.int64), sizes[ca:cb]
        )  # (b,) anchor id
        gstart = np.repeat(off[ca:cb], sizes[ca:cb])  # (b,) anchor start id
        sz_of = np.repeat(sizes[ca:cb], sizes[ca:cb])  # (b,) anchor size
        loc = np.arange(lo, hi, dtype=np.int64) - gstart  # (b,) local index

        # --- coherent (windowed within-cluster) neighbors ---
        coh_gid = _coherent_edges(
            loc, sz_of, gstart, k, w, brng
        )  # (b, k) in [lo, hi)
        coh_loc = coh_gid - lo  # (b, k) local

        # --- blend mask: k_coh coherent, the rest Chung-Lu ---
        k_coh = np.clip(
            base_k + (brng.random(b) < fracp).astype(np.int64), 0, k
        )
        mask = ar_k[None, :] < k_coh[:, None]  # (b, k) True=coherent

        # --- Chung-Lu targets: sample (anchor, in-degree) per edge ---
        t_anchor = brng.integers(0, nc, size=(b, k)).astype(
            np.int64
        )  # anchor ~ uniform
        t_w = indeg_dist[
            np.searchsorted(sb_cdf, brng.random((b, k)))
        ]  # in-degree ~ size-biased
        t_indeg = t_w.astype(np.float32) * cl_scale  # (b, k)

        # --- in-degree: coherent counted locally + own expected Chung-Lu in-degree ---
        node_indeg = np.bincount(coh_loc[mask], minlength=b).astype(np.float32)
        node_w = indeg_dist[brng.integers(0, nc, size=b)].astype(
            np.float32
        )  # own weight ~ uniform
        node_indeg += node_w * cl_scale

        # neighbor in-degree: coherent -> node_indeg[target]; Chung-Lu -> sampled in-degree
        nbr_indeg = np.where(mask, node_indeg[coh_loc], t_indeg)  # (b, k)
        mni = nbr_indeg.mean(1)  # (b,)

        # --- structural features (node + neighbors), normalized as in training ---
        struct_node = np.stack(
            [np.log1p(node_indeg), np.log1p(mni)], 1
        )  # (b, 2)
        hub_nbr_struct = np.stack(
            [
                np.log1p(t_indeg),
                np.full((b, k), np.log1p(hub_mni_p), np.float32),
            ],
            -1,
        )
        struct_nbr = np.where(
            mask[:, :, None], struct_node[coh_loc], hub_nbr_struct
        )
        struct_node = normalize_features(struct_node, feat_mu, feat_sd)
        struct_nbr = normalize_features(
            struct_nbr.reshape(b * k, STRUCT_FEAT_DIM), feat_mu, feat_sd
        ).reshape(b, k, STRUCT_FEAT_DIM)

        cluster_of_node = cluster_of_anchor[cl_of].astype(np.int64)  # (b,)

        # --- decode: stream this block's anchors, MLP them ---
        block_emb = model.cluster_mlp(
            torch.as_tensor(
                np.ascontiguousarray(anchors[ca:cb]), device=device
            )
        )  # (b_clusters, d_emb)
        node_emb = block_emb[
            torch.tensor(cl_of - ca, device=device)
        ]  # (b, d_emb)

        # coherent nbr shares the node's anchor emb; Chung-Lu nbr -> MLP of its
        # sampled anchor (gathered from the host anchor table, this block only).
        nbr_emb = (
            node_emb[:, None, :].expand(-1, k, -1).contiguous()
        )  # (b, k, d_emb)
        is_cl = ~mask
        if is_cl.any():
            cl_anchor = t_anchor[is_cl]  # (n_cl,)
            cl_rows = torch.as_tensor(
                np.ascontiguousarray(anchors[cl_anchor]), device=device
            )  # (n_cl, D)
            nbr_emb[torch.as_tensor(is_cl, device=device)] = model.cluster_mlp(
                cl_rows
            )

        hi_f = torch.cat(
            [node_emb, torch.as_tensor(struct_node, device=device)], -1
        )
        hj_f = torch.cat(
            [nbr_emb, torch.as_tensor(struct_nbr, device=device)], -1
        )
        pred = model(hi_f, hj_f)
        pred = pred + sample_residuals(
            torch.tensor(cluster_of_node, device=device),
            resid_params,
            D,
            device,
        )
        xb = pred * sd_t + mu_t
        norm = sample_norms_percentile(cluster_of_node, norm_q, seed=seed + bi)
        dirv = xb / (xb.norm(dim=1, keepdim=True) + 1e-12)
        xb = dirv * torch.tensor(norm, device=device).unsqueeze(1)
        xb_np = xb.detach().cpu().numpy()

        # --- split base / queries: this block's query ids ---
        qa, qb = np.searchsorted(q_idx, lo), np.searchsorted(q_idx, hi)
        qloc = q_idx[qa:qb] - lo
        if qloc.size:
            qm = np.zeros(b, dtype=bool)
            qm[qloc] = True
            np.ascontiguousarray(xb_np[~qm]).tofile(fbase)
            queries[q_off : q_off + qloc.size] = xb_np[qloc]
            q_off += qloc.size
        else:
            xb_np.tofile(fbase)

    fbase.close()
    return queries
