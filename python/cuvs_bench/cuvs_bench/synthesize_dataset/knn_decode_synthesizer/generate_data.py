#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""generate_data.py - entry point: fit on a real sample, generate a synthetic benchmark bundle."""

from __future__ import annotations

import argparse
import gc
import os
import time

import numpy as np
import torch
from block_local import generate_block_local
from knn_decoder import (
    decode_graph,
    fit_norm_quantiles,
    fit_residuals,
    kNNDecoder,
    sample_norms_percentile,
    train_model,
)
from upsample import _est_coherence, generate_graph_knn
from utils import (
    PhaseTimer,
    build_all_neighbors,
    build_kmeans,
    holdout_split,
    load_fbin,
    section,
    step,
    write_bundle,
    write_bundle_streamed,
)


def get_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ================================================================== #
    # DATA / RUN — input sample, output, and what to generate
    # ================================================================== #
    g_data = p.add_argument_group(
        "data / run", "input sample, output bundle, and what to generate"
    )
    g_data.add_argument(
        "--sample",
        type=str,
        required=True,
        help="path to real .fbin sample to fit on.",
    )
    g_data.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="dir to write base.fbin / queries.fbin / groundtruth.*",
    )
    g_data.add_argument(
        "--target",
        type=int,
        default=None,
        help="number of synthetic BASE vectors to generate. If this is not given, "
        "we just split the sample data to base and n_queries and return.",
    )
    g_data.add_argument(
        "--n-queries",
        type=int,
        default=10000,
        help="held-out queries, always disjoint from base",
    )
    g_data.add_argument(
        "--gt-k",
        type=int,
        default=100,
        help="exact GT neighbors for each query",
    )
    g_data.add_argument(
        "--seed",
        type=int,
        default=42,
        help="global RNG seed (KMeans, graph gen, decode)",
    )

    # ================================================================== #
    # kNN UPSAMPLING: build the N-node kNN graph from the sample's kNN
    # ================================================================== #
    g_up = p.add_argument_group(
        "kNN upsampling (small real kNN -> large synthetic kNN)"
    )
    g_up.add_argument(
        "--knn",
        type=int,
        default=10,
        help="k for the kNN of the sample training graph AND the generated "
        "graph for decoding. Don't change this unless you know what you are doing.",
    )
    g_up.add_argument(
        "--knn-frac",
        type=float,
        default=0.66,
        help="fraction of each node's edges from the synthesized kNN; the "
        "rest are random. 1.0=fully coherent, lower=harder.",
    )

    # ================================================================== #
    # kNN DECODING: the model that maps the kNN graph -> vectors
    # ================================================================== #
    g_dec = p.add_argument_group(
        "kNN decoding (kNN graph -> vector embeddings)"
    )
    g_dec.add_argument(
        "--d-emb", type=int, default=64, help="cluster-ID embedding dimension"
    )
    g_dec.add_argument(
        "--hidden", type=int, default=1024, help="Model MLP hidden width"
    )
    g_dec.add_argument("--depth", type=int, default=3, help="Model MLP layers")
    g_dec.add_argument(
        "--epochs", type=int, default=400, help="Model training epochs"
    )
    g_dec.add_argument(
        "--lr", type=float, default=5e-3, help="Model learning rate"
    )
    g_dec.add_argument(
        "--batch",
        type=int,
        default=1024,
        help="Model training batch size for SGD",
    )
    g_dec.add_argument(
        "--nc",
        type=int,
        default=None,
        help="# clusters for the per-cluster residual/norm fit — same as "
        "cuvs-bench synthesize_dataset's `nc`. Recommended default: sample_size / 100.",
    )
    g_dec.add_argument(
        "--resid-rank",
        type=int,
        default=64,
        help="rank of the per-cluster residual Gaussian "
        "(within-cluster spread added to the Model output). This is similar to"
        "pca-components in cuvs_bench.",
    )
    g_dec.add_argument(
        "--resid-scale",
        type=float,
        default=1.0,
        help="scale the residual spread (0=Model mean only)",
    )
    g_dec.add_argument(
        "--model-cache",
        type=str,
        default=None,
        help="dir to cache the full fit bundle (model + mu/sd/feat-stats + "
        "resid_params + norm_q + cluster_ids + stats). A cache hit skips "
        "the whole fit and reproduces the dataset: identical bundle + "
        "identical generate args (knn-frac/target/seed) => identical data.",
    )
    g_dec.add_argument(
        "--host-gather",
        action="store_true",
        help="hold the training anchor table / kNN graph / struct on the "
        "HOST and gather each minibatch to the GPU per step. Use only when the sample "
        "is too big to fit in GPU memory (100M+ sample).",
    )

    # ================================================================== #
    # BLOCK-LOCAL: cluster-ordered streaming generate+decode
    # ================================================================== #
    g_bl = p.add_argument_group("block-local (scaling path — see README §5)")
    g_bl.add_argument(
        "--block-local",
        action="store_true",
        help="use the cluster-ordered block-local generator (block_local.py) instead "
        "of the default whole-graph pool+decode. Streams one cluster-ordered block "
        "at a time. This option allows to never materialize the full (N,k) graph or the "
        "O(N) feature arrays. USE WHEN: the target (N, k) graph is too large for the "
        "default path to hold in memory (roughly multi-billion-scale+); the "
        "whole-graph path is simpler and fine for smaller ones (1B and under).",
    )
    g_bl.add_argument(
        "--block-size",
        type=int,
        default=1_000_000,
        help="[--block-local] target # nodes per streamed block — caps peak "
        "GPU memory. Default 1M is okay, but lower it if you hit GPU OOM, raise it to "
        "cut per-block overhead when memory allows.",
    )

    return p.parse_args()


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    timer = PhaseTimer()  # accumulates fit/generate phase times

    # ------------------------------------------------------------------ #
    # CONFIG + LOAD
    # ------------------------------------------------------------------ #
    is_synth = args.target is not None

    section("CONFIG")
    print(f"  sample:  {args.sample}")
    print(f"  out_dir: {args.out_dir}")
    print(f"  run:     {'SYNTH' if is_synth else 'REAL (no target)'}")
    print(f"  target:  {args.target}   n_queries: {args.n_queries}")

    section("LOAD DATA")
    t0 = time.perf_counter()
    X = load_fbin(args.sample)
    D = X.shape[1]
    print(f"  shape: {X.shape}", flush=True)
    timer.lap("fit", "load data", t0)

    # ------------------------------------------------------------------ #
    # REAL reference (no --target): just split the sample and stop
    # ------------------------------------------------------------------ #
    if not is_synth:
        section("REAL — split sample into base + held-out queries")
        base, queries = holdout_split(X, args.n_queries, args.seed)
        write_bundle(args.out_dir, base, queries, args.gt_k)
        section("DONE (just split the sample)")
        return

    # ------------------------------------------------------------------ #
    # SYNTH: FIT — model + residual + norm + clusters + stats.
    # ------------------------------------------------------------------ #
    fit_path = None
    if args.model_cache:
        os.makedirs(args.model_cache, exist_ok=True)
        samp = os.path.splitext(os.path.basename(args.sample))[0][:32]
        req_nc = args.nc if args.nc else max(1, len(X) // 100)
        key = (
            f"fitb_{samp}_ss{len(X)}_emb{args.d_emb}_h{args.hidden}_d{args.depth}"
            f"_knn{args.knn}_ep{args.epochs}_nc{req_nc}_rr{args.resid_rank}"
            f"_rs{args.resid_scale}_s{args.seed}"
        )
        fit_path = os.path.join(args.model_cache, key + ".pt")

    if fit_path and os.path.exists(fit_path):
        section("LOAD FIT BUNDLE (cached — skipping fit)")
        t0 = time.perf_counter()
        step(f"loading cached fit bundle: {fit_path}")
        ck = torch.load(fit_path, map_location=device, weights_only=False)
        model = kNNDecoder(
            d_out=D, d_emb=args.d_emb, hidden=args.hidden, depth=args.depth
        ).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        mu, sd, feat_mu, feat_sd = (
            ck["mu"],
            ck["sd"],
            ck["feat_mu"],
            ck["feat_sd"],
        )
        resid_params = ck["resid_params"]
        norm_q = ck["norm_q"]
        cluster_ids = ck["cluster_ids"]
        stats = ck["stats"]
        nc = ck["nc"]
        print(
            f"  model + resid_params + norm_q + cluster_ids + stats loaded "
            f"(nc={nc}) — fit reproduced from cache",
            flush=True,
        )
        timer.lap("fit", "load fit bundle", t0)
    else:
        section("kNN Graph on sample")
        t0 = time.perf_counter()
        step(f"all-neighbors kNN (k={args.knn}) on {len(X):,} pts ...")
        sample_knn = build_all_neighbors(X, args.knn)
        timer.lap("fit", "sample kNN", t0)

        section("KMeans on sample")
        t0 = time.perf_counter()
        ss = len(X)
        nc = args.nc if args.nc else max(1, ss // 100)
        cluster_ids, _ = build_kmeans(X, nc, seed=args.seed)
        nc = int(cluster_ids.max()) + 1
        timer.lap("fit", "kmeans", t0)

        section("Measuring kNN graph statistics")
        t0 = time.perf_counter()
        stats = {
            "coherence": _est_coherence(
                sample_knn, seed=args.seed
            ),  # float value
            "indeg_dist": np.bincount(
                sample_knn.reshape(-1), minlength=ss
            ).astype(np.float32),  # (ss, )
        }
        print(
            f"  coherence={stats['coherence']:.4f}  "
            f"in-deg max={stats['indeg_dist'].max():.0f}",
            flush=True,
        )
        timer.lap("fit", "measure stats", t0)

        section("TRAIN MODEL")
        t0 = time.perf_counter()
        model, mu, sd, feat_mu, feat_sd, struct_feats = train_model(
            X,
            sample_knn,
            stats["indeg_dist"],
            d_emb=args.d_emb,
            hidden=args.hidden,
            depth=args.depth,
            epochs=args.epochs,
            lr=args.lr,
            batch=args.batch,
            device=device,
            seed=args.seed,
            host_gather=args.host_gather,
        )
        model.eval()
        timer.lap("fit", "train model", t0)

        section("FIT RESIDUAL MODEL")
        t0 = time.perf_counter()
        resid_params = fit_residuals(
            X,
            sample_knn,
            model,
            mu,
            sd,
            feat_mu,
            feat_sd,
            nc=nc,
            rank=args.resid_rank,
            device=device,
            scale=args.resid_scale,
            struct_feats=struct_feats,
            resid_ids=cluster_ids,
            in_deg=stats["indeg_dist"],
        )
        timer.lap("fit", "fit residual", t0)

        section("FIT NORM (radial percentile scheme)")
        t0 = time.perf_counter()
        norm_q, nmean, ncv = fit_norm_quantiles(X, cluster_ids, nc)
        print(
            f"  real norms: mean={nmean:.4f} cv={ncv:.4f} "
            f"-> percentile (per-cluster norm inverse-CDF)",
            flush=True,
        )
        timer.lap("fit", "fit norm", t0)

        if fit_path:
            torch.save(
                {
                    "model": model.state_dict(),
                    "mu": mu,
                    "sd": sd,
                    "feat_mu": feat_mu,
                    "feat_sd": feat_sd,
                    "resid_params": resid_params,
                    "norm_q": norm_q,
                    "cluster_ids": cluster_ids,
                    "stats": stats,
                    "nc": nc,
                },
                fit_path,
            )
            print(f"  cached fit bundle -> {fit_path}", flush=True)

    # ------------------------------------------------------------------ #
    # Build the synthetic kNN graph, decode, split
    # ------------------------------------------------------------------ #
    n_pool = args.target + args.n_queries  # base (=target) + held-out queries
    base_path = os.path.join(args.out_dir, "base.fbin")

    if args.block_local:
        # cluster-ordered streaming generate+decode. Fuses graph gen and decode
        # per block, so the (N,k) graph and O(N) features never materialize.
        section("BLOCK-LOCAL generate+decode  (parametric hub field)")
        t0 = time.perf_counter()
        queries = generate_block_local(
            model,
            X,
            mu,
            sd,
            feat_mu,
            feat_sd,
            resid_params,
            norm_q,
            cluster_ids,
            stats,
            N=n_pool,
            n_queries=args.n_queries,
            k=args.knn,
            knn_frac=args.knn_frac,
            seed=args.seed,
            device=device,
            base_path=base_path,
            block_size=args.block_size,
        )
        print(
            f"  generated {n_pool:,} pts (base streamed to {base_path})",
            flush=True,
        )
        timer.lap("generate", "block-local gen+decode", t0)
    else:
        section("BUILD POOL GRAPH")
        t0 = time.perf_counter()
        anchor_pool, nbr_pool = generate_graph_knn(
            stats, N=n_pool, k=args.knn, seed=args.seed, knn_frac=args.knn_frac
        )
        print(f"  pool graph: {len(anchor_pool):,} nodes", flush=True)
        timer.lap("generate", "pool graph", t0)

        section("DECODE kNN graph — stream base to disk")
        cluster_pool = cluster_ids[
            anchor_pool
        ]  # anchor -> cluster for residual + norm
        norm_target = sample_norms_percentile(
            cluster_pool, norm_q, seed=args.seed
        )
        q_idx = np.sort(
            np.random.default_rng(args.seed).choice(  # held-out query ids
                n_pool, min(args.n_queries, n_pool), replace=False
            )
        )
        t0 = time.perf_counter()
        # Streaming decode: base rows go straight to base.fbin (host RAM ~ one chunk),
        # only the held-out queries come back in memory.
        queries = decode_graph(
            anchor_pool,
            nbr_pool,
            model,
            X,
            mu,
            sd,
            feat_mu,
            feat_sd,
            resid_params,
            norm_target,
            device,
            args.seed,
            resid_ids=cluster_pool,
            base_path=base_path,
            query_idx=q_idx,
        )
        print(
            f"  decoded {n_pool:,} pts (base streamed to {base_path})",
            flush=True,
        )
        timer.lap("generate", "decode", t0)

    model.to("cpu")
    del resid_params
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    section("WRITE queries + exact GT")
    t0 = time.perf_counter()
    write_bundle_streamed(
        args.out_dir, base_path, n_pool - len(queries), queries, args.gt_k, D
    )
    timer.lap("generate", "write + GT", t0)

    timer.summary()
    section("DONE (synth)")


if __name__ == "__main__":
    main()
