#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""kNN decoder: map a graph -> vectors (mean + per-cluster residual)."""

from __future__ import annotations

import threading
import time

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from utils import (
    CHUNK_SIZE,
    batched_mean_std,
    normalize_features,
    sample_residuals,
    ts,
    write_fbin_header,
)

STRUCT_FEAT_DIM = 2
_NORM_QUANTILE_COUNT = (
    256  # per-cluster norm inverse-CDF grid (synthesize_dataset/_fit.py)
)


def mlp(d_in, d_out, hidden, depth=2):
    """A simple GELU MLP: d_in -> [hidden]*depth -> d_out."""
    layers = [nn.Linear(d_in, hidden), nn.GELU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), nn.GELU()]
    layers.append(nn.Linear(hidden, d_out))
    return nn.Sequential(*layers)


def compute_structural_features(nbr, in_deg=None):
    """Per-node topological features from a kNN graph: [log1p(in-degree),
    log1p(mean neighbour in-degree)].  Returns (n_nodes, STRUCT_FEAT_DIM) float32.

    in_deg : optional (n_nodes,) per-node in-degree to reuse.
    """
    n_nodes = nbr.shape[0]
    if in_deg is None:
        in_deg = np.bincount(nbr.reshape(-1), minlength=n_nodes)
    mean_nbr_indeg = in_deg[nbr].mean(axis=1).astype(np.float32)  # (n_nodes,)
    return np.stack(
        [np.log1p(in_deg), np.log1p(mean_nbr_indeg)], axis=1
    ).astype(np.float32)  # (n_nodes, F)


class kNNDecoder(nn.Module):
    """Map local graph structure (anchor + connectivity) -> ambient vector."""

    def __init__(
        self,
        d_out=1024,
        d_emb=64,
        d_struct=STRUCT_FEAT_DIM,
        hidden=1024,
        depth=3,
    ):
        super().__init__()
        self.d_struct = d_struct

        # A node's embedding is a function of its anchor (the raw sample point).
        self.cluster_mlp = mlp(d_out, d_emb, 256, depth=2)  # anchor -> d_emb

        d_node = d_emb + d_struct  # total node feature dimension

        # Score each neighbor relative to the centre node
        self.att = nn.Linear(d_node * 2, 1)

        # Decode [h_i, aggregate_neighbors] -> embedding
        self.out = mlp(d_node * 2, d_out, hidden, depth)

    def node_feat(self, anchor_vecs, struct):
        """Build node feature vectors from anchor rows.

        anchor_vecs : (B, d_out) float32  — the raw anchor rows
        struct      : (B, d_struct) float32  — normalized topological features
        """
        c = self.cluster_mlp(anchor_vecs)  # (B, d_emb)
        return torch.cat([c, struct], dim=-1)  # (B, d_node)

    def forward(self, h_i, h_nbrs):
        """Aggregate neighbors and project to ambient space.

        h_i    : (B, d_node)    — centre node features
        h_nbrs : (B, k, d_node) — neighbor features
        -> (B, d_out)
        """
        B, k, d = h_nbrs.shape
        hi_exp = h_i.unsqueeze(1).expand(B, k, d)  # (B, k, d_node)
        scores = self.att(torch.cat([hi_exp, h_nbrs], -1))  # (B, k, 1)
        w = torch.softmax(scores, dim=1)  # (B, k, 1)
        agg = (w * h_nbrs).sum(1)  # (B, d_node)
        return self.out(torch.cat([h_i, agg], -1))  # (B, d_out)


def train_model(
    X,
    sample_knn,
    in_deg,
    d_emb,
    hidden,
    depth,
    epochs,
    lr,
    batch,
    device,
    seed,
    host_gather=False,
):
    """Train kNNDecoder to map graph structure -> embedding.

    The sample points X are used as the anchors (row i = anchor i) and as the
    regression targets.

    Parameters
    ----------
    X          : (ss, D) float32  real sample embeddings (ss = sample size); used as
                 the anchors and the training targets.
    sample_knn : (ss, k) int64    all-neighbors kNN graph of the sample
    in_deg     : (ss,) per-node in-degree of the sample kNN (the stats' indeg_dist),
                 reused for the structural features.
    host_gather: if False (default), the anchor table / kNN graph / struct features are
                 held GPU-RESIDENT and gathered on-device. This is fast, for samples that
                 fit in GPU memory.  If True, they stay on the HOST and each minibatch's
                 rows are shipped to the GPU per step — slower per step but scales to
                 samples too big for GPU memory.  Same loop either way: only the storage
                 device changes (the per-batch `.to(device)` is a no-op when resident).


    Returns (model, mu, sd, feat_mu, feat_sd, struct_feats):
      mu/sd           per-dim stats of X
      feat_mu/feat_sd stats of the structural features, reused at generate time.
    """
    torch.manual_seed(seed)

    ss, D = X.shape
    k = sample_knn.shape[1]

    mu, sd = batched_mean_std(X)

    # Structural node features from the real kNN graph
    struct_feats = compute_structural_features(sample_knn, in_deg)
    feat_mu, feat_sd = batched_mean_std(struct_feats)
    struct_feats = normalize_features(struct_feats, feat_mu, feat_sd)

    mu_t = torch.as_tensor(mu, device=device)  # (1, D) for per-batch normalize
    sd_t = torch.as_tensor(sd, device=device)

    store = "cpu" if host_gather else device
    Xs = torch.as_tensor(X, device=store)  # (ss, D) anchors == targets
    nbrs = torch.as_tensor(sample_knn, device=store).long()  # (ss, k)
    structs = torch.as_tensor(struct_feats, device=store)  # (ss, F)

    model = kNNDecoder(d_out=D, d_emb=d_emb, hidden=hidden, depth=depth).to(
        device
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n_batches = (ss + batch - 1) // batch
    print(
        f"  [{ts()}] training on {ss:,} nodes, {n_batches:,} batches/epoch, {epochs} "
        f"epochs ({'host-gather' if host_gather else 'GPU-resident'}) ...",
        flush=True,
    )
    t0 = time.perf_counter()

    for ep in range(epochs):
        perm = torch.randperm(ss, device=store)  # on `store`
        total_loss = 0.0

        for bi, i in enumerate(range(0, ss, batch)):
            idx = perm[i : i + batch]  # on `store`
            b = idx.numel()

            xi = (
                Xs[idx].to(device) - mu_t
            ) / sd_t  # (B, D); .to() no-op if resident

            flat = nbrs[idx].reshape(-1)  # (B*k,)
            allids = torch.cat([idx, flat])  # (B + B*k,)
            uniq, inv = torch.unique(allids, return_inverse=True)
            a_uniq = Xs[uniq].to(device)  # (U, D) unique anchor rows
            s_uniq = structs[uniq].to(device)
            ufeat = model.node_feat(a_uniq, s_uniq)  # cluster_mlp once/unique
            inv = inv.to(device)  # gather the device-side ufeat

            hi = ufeat[inv[:b]]  # (B, d_node)
            hj = ufeat[inv[b:]].reshape(b, k, -1)  # (B, k, d_node)

            x_pred = model(hi, hj)  # (B, D)
            loss = ((x_pred - xi) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * b

            if (
                ep == 0
                and n_batches > 200
                and bi > 0
                and bi % (n_batches // 5) == 0
            ):
                print(
                    f"    [{ts()}] epoch 0: batch {bi:,}/{n_batches:,} "
                    f"({100 * bi / n_batches:.0f}%)",
                    flush=True,
                )

        ep_loss = total_loss / ss

        if ep % 10 == 0 or ep == epochs - 1:
            elapsed = time.perf_counter() - t0
            print(
                f"  [{ts()}] ep {ep:4d}/{epochs}  loss={ep_loss:.5f}  "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )

    print(f"  training done in {time.perf_counter() - t0:.0f}s", flush=True)

    return model, mu, sd, feat_mu, feat_sd, struct_feats


@torch.no_grad()
def fit_residuals(
    X,
    sample_knn,
    model,
    mu,
    sd,
    feat_mu,
    feat_sd,
    nc,
    rank,
    device,
    scale=1.0,
    chunk=CHUNK_SIZE,
    struct_feats=None,
    resid_ids=None,
    in_deg=None,
):
    """Fit per-cluster low-rank Gaussian on residuals x - GNN(x).

    Residuals are computed in the model's Z-normalized output space
    ((x - mu) / sd).  Returns a list of nc dicts (or None for tiny clusters),
    each with GPU tensors {mean (D,), comps (r, D), stds (r,), noise_std}.

    Parameters
    ----------
    X            : (ss, D) float32  real sample embeddings (ss = sample size) —
                   the targets whose residual (x - decode(x)) we model.
    sample_knn    : (ss, k) int64    kNN graph, fed to the decoder to produce the
                   per-node mean GNN(x) that the residual is taken against.
    model        : the trained kNNDecoder — run (no-grad) to get the vector embeddings.
    mu, sd       : (1, D) float32  per-dim stats of X; the residual is fit in the
                   z-normalized space (x - mu) / sd (the space the model predicts).
    feat_mu,     : normalization stats for the structural features, reused here so
    feat_sd        the decode pass sees the exact same feature scale as training.
    nc           : int    number of residual groups to fit == number of distinct
                   resid_ids (= # clusters).
    rank         : int    target rank r of the per-cluster low-rank Gaussian
                   (clamped down to the cluster's point count).
    device       : torch device.
    scale        : shrink/scale the sampled residual spread.  1.0 = full real
                   within-cluster spread; 0.0 = GNN mean only (collapsed);
                   >1.0 roughens (the --resid-scale difficulty knob) the manifold.
    chunk        : batch size for the decode pass (caps peak memory).
    struct_feats    : precomputed structural features from train_model,
                   passed in to avoid recomputing them.
    resid_ids    : (ss,) int32  the KMeans cluster label per node.
    """
    ss, D = X.shape
    k = sample_knn.shape[1]

    if (
        struct_feats is None
    ):  # reused from train_model unless we loaded a cached model
        struct_feats = compute_structural_features(sample_knn, in_deg)
        struct_feats = normalize_features(struct_feats, feat_mu, feat_sd)

    # Decode all real nodes -> prediction; residual = target - prediction.
    # For each chunk, gather the unique anchor rows (centers + neighbors) host->GPU
    # and run cluster_mlp once each (Row i is anchor i, so the chunk's center ids are i..end.)
    R = np.empty((ss, D), dtype=np.float32)
    for i in tqdm(
        range(0, ss, chunk), desc="residual: decode sample", unit="chunk"
    ):
        end = min(i + chunk, ss)
        b = end - i
        idx = np.arange(i, end)  # center ids == rows (host)
        flat = sample_knn[idx].reshape(-1)  # (b*k,) neighbor ids (host)
        allids = np.concatenate([idx, flat])
        uniq, inv = np.unique(allids, return_inverse=True)
        a_uniq = torch.as_tensor(
            X[uniq], device=device
        )  # (U, D) anchor rows host->GPU
        s_uniq = torch.as_tensor(struct_feats[uniq], device=device)
        ufeat = model.node_feat(a_uniq, s_uniq)  # MLP once/unique
        inv = torch.as_tensor(inv, device=device)

        hi = ufeat[inv[:b]]  # (b, d_node)
        hj = ufeat[inv[b:]].reshape(b, k, -1)  # (b, k, d_node)
        pred = model(hi, hj)  # normalized space
        xt_chunk = ((X[i:end] - mu) / sd).astype(
            np.float32
        )  # normalize this chunk only
        R[i:end] = xt_chunk - pred.cpu().numpy()

    # Per-cluster PPCA fit
    params = [None] * nc
    for c in tqdm(
        range(nc), desc="residual: per-cluster PPCA", unit="cluster"
    ):
        sel = np.where(resid_ids == c)[0]
        sz = len(sel)
        if sz < 2:
            continue
        Rc = torch.tensor(R[sel], device=device)  # (sz, D)
        mean_c = Rc.mean(0)  # (D,)
        Rc0 = Rc - mean_c

        # SVD: Vh rows are principal directions
        _, S, Vh = torch.linalg.svd(Rc0, full_matrices=False)
        r = int(min(rank, Vh.shape[0]))
        comps = Vh[:r].contiguous()  # (r, D)
        var = (S**2) / max(sz - 1, 1)  # per-direction variance
        stds = torch.sqrt(var[:r]) * scale  # (r,) — anisotropic sheet
        noise_std = torch.zeros((), device=device)
        params[c] = {
            "mean": mean_c * scale,
            "comps": comps,
            "stds": stds,
            "noise_std": noise_std,
        }

    n_fit = sum(1 for p in params if p is not None)
    print(
        f"  fit residual Gaussians for {n_fit}/{nc} clusters "
        f"(rank<={rank}, scale={scale});",
        flush=True,
    )
    return params


def fit_norm_quantiles(X, cluster_ids, nc):
    """Per-cluster norm inverse-CDF grids.

    Mirrors synthesize_dataset/_fit.py.  Returns
    (norm_quantiles: (nc, 256) float32, mean: float, cv: float).
    Empty clusters fall back to the global grid.
    """
    import cupy as cp

    n = len(X)
    norms = cp.empty(n, dtype=cp.float32)  # (ss,) on device
    for s in range(0, n, CHUNK_SIZE):
        e = min(s + CHUNK_SIZE, n)
        chunk_gpu = cp.asarray(X[s:e], dtype=cp.float32)
        norms[s:e] = cp.linalg.norm(chunk_gpu, axis=1)
        del chunk_gpu
    mean = float(norms.mean())
    cv = float(norms.std() / max(mean, 1e-12))
    levels = cp.linspace(0.0, 1.0, _NORM_QUANTILE_COUNT)
    gq = cp.quantile(norms, levels).astype(cp.float32)  # global fallback grid
    cid = cp.asarray(cluster_ids)
    q = cp.empty((nc, _NORM_QUANTILE_COUNT), dtype=cp.float32)
    for c in range(nc):
        m = cid == c
        q[c] = (
            cp.quantile(norms[m], levels).astype(cp.float32)
            if bool(m.any())
            else gq
        )
    cp.get_default_memory_pool().free_all_blocks()
    return cp.asnumpy(q), mean, cv


def sample_norms_percentile(cluster_ids, norm_quantiles, seed=42):
    """Draw each node's target norm from its cluster's inverse-CDF (percentile
    scheme).  Mirrors synthesize_dataset/_generate.py `_rescale_to_scheme`.
    """
    rng = np.random.default_rng(seed + 4)
    nq = np.ascontiguousarray(norm_quantiles, dtype=np.float32)  # (nc, Q)
    Q = nq.shape[1]
    cid = cluster_ids.astype(np.int64)
    u = rng.random(cid.shape[0]).astype(np.float32)  # (N,)
    pos = u * (Q - 1)
    lo = np.minimum(np.floor(pos).astype(np.int64), Q - 2)  # left grid index
    frac = (pos - lo).astype(np.float32)
    q_lo = nq[cid, lo]  # (N,) gather
    q_hi = nq[cid, lo + 1]
    return (q_lo * (1.0 - frac) + q_hi * frac).astype(np.float32)


@torch.no_grad()
def decode_graph(
    anchor_ids,
    nbr,
    model,
    anchors,
    mu,
    sd,
    feat_mu,
    feat_sd,
    resid_params,
    norm_target,
    device,
    seed,
    resid_ids=None,
    base_path=None,
    query_idx=None,
):
    """Decode a kNN graph, streaming base rows to disk -> return held-out queries (N_q, D).
    Runs the trained decoder over the N-node graph to get the vector, adds the
    per-cluster residual, and rescales each vector's radius.  Base rows are written
    straight to base_path and only the query rows are kept and returned.

    Parameters
    ----------
    anchor_ids   : (N,) int32   anchor id per node — indexes the decoder's anchor
                   embedding table.
    nbr          : (N, k) int64  the generated (synthetic) kNN graph to decode.
    model        : the trained kNNDecoder.
    anchors      : (n_anchors, D) float32  the sample rows X (host); nodes index into
                   this table by anchor id to get their anchor embedding.
    mu, sd       : (1, D) float32  per-dim stats of the real sample; used to
                   un-normalize the decoder output back into ambient space.
    feat_mu,     : structural-feature normalization stats from train_model, so the
    feat_sd        synthetic graph's features are scaled exactly as in training.
    resid_params : list of per-cluster residual Gaussians from fit_residuals
    norm_target  : (N,) float32  target L2 norm per node (drawn from the per-cluster
                   norm inverse-CDF)
    device       : torch device for the decode.
    seed         : RNG seed for the residual sampling.
    resid_ids    : (N,) int32  the KMeans cluster label per node
    base_path    : STREAM base rows straight to this .fbin as we decode to cap host RAM usage.
    query_idx    : (N_q,) sorted int64 ids of the held-out query rows
    """
    N, k = nbr.shape
    D = mu.shape[1]
    chunk = CHUNK_SIZE
    if resid_ids is None:
        resid_ids = anchor_ids
    torch.manual_seed(seed + 1)
    mu_t = torch.tensor(mu, device=device)
    sd_t = torch.tensor(sd, device=device)
    struct_all = compute_structural_features(
        nbr
    )  # in-degrees of the synthetic graph
    struct_all = normalize_features(struct_all, feat_mu, feat_sd)

    # Precompute the anchor-embedding table ONCE
    n_anc = anchors.shape[0]
    d_emb = model.cluster_mlp[-1].out_features
    emb_table = torch.empty(n_anc, d_emb, device=device)
    for i in range(0, n_anc, chunk):
        end = min(i + chunk, n_anc)
        emb_table[i:end] = model.cluster_mlp(
            torch.as_tensor(anchors[i:end], device=device)
        )

    # Base rows stream straight to disk; only the held-out queries stay in RAM.
    n_q = len(query_idx)
    queries = np.empty((n_q, D), dtype=np.float32)
    q_off = 0
    fbase = open(base_path, "wb")
    write_fbin_header(fbase, N - n_q, D)  # base row count known up front

    # Overlap disk writes with the GPU decode:
    write_thread = None
    write_exception = None

    def _wait_write():
        nonlocal write_thread, write_exception
        if write_thread is not None:
            write_thread.join()
            write_thread = None
        if write_exception is not None:
            exc = write_exception
            write_exception = None
            raise exc

    def _flush_async(arr):
        nonlocal write_thread

        def _w():
            nonlocal write_exception
            try:
                arr.tofile(fbase)
            except BaseException as e:
                write_exception = e

        write_thread = threading.Thread(target=_w, daemon=True)
        write_thread.start()

    for start in tqdm(range(0, N, chunk), desc="decode", unit="chunk"):
        end = min(start + chunk, N)
        b = end - start

        # --- center-node features: anchor embedding (gathered) + structural features ---
        anchor_i = torch.tensor(
            anchor_ids[start:end].astype(np.int64), device=device
        )
        struct_i = torch.tensor(struct_all[start:end], device=device)
        hi = torch.cat([emb_table[anchor_i], struct_i], dim=-1)  # (b, d_node)

        # --- neighbor features: same, for every node's k neighbours ---
        flat = nbr[start:end].reshape(-1)  # (b*k,) flattened neighbour ids
        anchor_j = torch.tensor(
            anchor_ids[flat].astype(np.int64), device=device
        )
        struct_j = torch.tensor(struct_all[flat], device=device)
        hj = torch.cat([emb_table[anchor_j], struct_j], dim=-1).reshape(
            b, k, -1
        )  # (b, k, d_node)

        # --- decode features into vectors ---
        pred = model(hi, hj)  # (b, D) in z-normalized space

        # --- add the per-cluster residual (restores within-cluster spread) ---
        resid_i = torch.tensor(
            resid_ids[start:end].astype(np.int64), device=device
        )
        pred = pred + sample_residuals(resid_i, resid_params, D, device)

        # --- un-normalize back to ambient space ---
        xb = pred * sd_t + mu_t

        # --- radial rescale: keep only the direction, set the L2 norm to the target ---
        dirv = xb / (xb.norm(dim=1, keepdim=True) + 1e-12)  # unit direction
        r = torch.tensor(norm_target[start:end], device=device).unsqueeze(
            1
        )  # target radius
        xb = dirv * r  # vector = direction * target norm

        xb_np = xb.cpu().numpy()  # (b, D) fresh host tile per chunk
        # which rows in [start, end) are held-out queries — sliced from the sorted
        # id list via searchsorted (no O(N) mask).
        lo, hi = (
            np.searchsorted(query_idx, start),
            np.searchsorted(query_idx, end),
        )
        n_qc = hi - lo
        if n_qc:
            qm = np.zeros(b, dtype=bool)
            qm[query_idx[lo:hi] - start] = True
            queries[q_off : q_off + n_qc] = xb_np[
                qm
            ]  # keep the few query rows in RAM
            q_off += n_qc
            base = np.ascontiguousarray(
                xb_np[~qm]
            )  # base rows the writer thread owns
        else:
            base = xb_np  # no queries here: the whole tile is base

        _wait_write()  # prev flush done -> its buffer is free
        _flush_async(base)  # write this chunk while the next decodes

    _wait_write()
    fbase.close()
    return queries
