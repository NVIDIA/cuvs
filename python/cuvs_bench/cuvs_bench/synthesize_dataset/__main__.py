#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
"""Command-line entry point for the synthesize_dataset module.

Three subcommands:

- ``fit``: extract a cluster fingerprint (.npz) from a real dataset.
- ``generate``: turn a fingerprint into a benchmarkable
  ``base.fbin`` / ``queries.fbin`` / ``groundtruth.{neighbors,distances}.{ibin,fbin}``
  bundle.
- ``verify``: validate the chosen ``nprobes``.

Run ``python -m cuvs_bench.synthesize_dataset --help`` for the full surface.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from ..generate_groundtruth.utils import write_bin
from ._fit import fit_cluster_stats
from ._generate import generate_synthetic_dataset_to_file
from ._ground_truth import (
    compute_groundtruth_nprobe,
    compute_groundtruth_exact,
    generate_queries,
)
from ._io import load_dataset
from ._stats_io import (
    cluster_config_from_stats,
    load_cluster_stats,
    save_cluster_stats,
)
from ._verify import verify_groundtruth


def _add_fit_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "fit",
        help="Extract a dataset fingerprint from a real dataset.",
        description=(
            "Run KMeans + per-cluster PCA on a sample of your real "
            "dataset and save the resulting fingerprint as an NPZ file."
        ),
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="Path to the real dataset (.fbin, .npy, or .pkl).",
    )
    p.add_argument(
        "--output",
        default=None,
        help=(
            "Path to write the fingerprint NPZ. Defaults to "
            "'{dataset_stem}_nc{n_clusters}_ncomp{pca_components}"
            "[_ss{sample_size}]_seed{seed}[_nonorm].npz' in the current "
            "working directory. '_ss{sample_size}' is included only when "
            "--sample_size is passed, and '_nonorm' only when "
            "--no-normalize-for-clustering is passed."
        ),
    )
    p.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help=(
            "Use only the FIRST `sample_size` rows of the input "
            "(default: use entire input). No shuffling is performed, so "
            "make sure your dataset's head-of-file slice is representative "
            "(pre-shuffle the file beforehand if it has any structural "
            "ordering, e.g. by class or time)."
        ),
    )
    p.add_argument(
        "--n_clusters",
        type=int,
        required=True,
        help="Number of KMeans clusters (the `nc` parameter).",
    )
    p.add_argument(
        "--pca_components",
        type=int,
        required=True,
        help="Number of PCA components per cluster (the `ncomp` parameter).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for KMeans init and PCA (default: 42).",
    )
    p.add_argument(
        "--max_iter",
        type=int,
        default=300,
        help="Maximum KMeans iterations (default: 300).",
    )
    p.add_argument(
        "--no-normalize-for-clustering",
        dest="no_normalize_for_clustering",
        action="store_true",
        help=(
            "Skip the on-the-fly L2 normalization "
            "before KMeans. Default behavior (recommended) auto-detects "
            "whether the input is already unit-norm and normalizes it iff "
            "not, so KMeans always clusters on normalized data. Pass this "
            "only for an ablation where you specifically want KMeans to run "
            "on the raw input. Centroids, per-dim variances, and PCA are "
            "always computed in the original space and the "
            "downstream `is_normalized_data` flag is unaffected, so the "
            "benchmark still matches the real data's normalization."
        ),
    )


def _add_generate_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "generate",
        help="Materialize a synthetic dataset + queries + GT from a fingerprint.",
    )
    p.add_argument("--stats", required=True, help="Path to fingerprint NPZ.")
    p.add_argument(
        "--total_rows",
        type=int,
        required=True,
        help="Number of synthetic vectors to generate.",
    )
    p.add_argument(
        "--n_queries",
        type=int,
        default=10_000,
        help="Number of queries to sample (default: 10000).",
    )
    p.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of nearest neighbors in the GT (default: 10).",
    )
    p.add_argument(
        "--nprobes",
        type=int,
        default=10,
        help="Clusters probed per query for nprobe GT (default: 10).",
    )
    p.add_argument(
        "--gt_mode",
        choices=["nprobe", "exact"],
        default="nprobe",
        help=(
            "GT computation mode. `nprobe` is the cheap default"
            "; `exact` does an exact streaming brute force across all "
            "clusters."
        ),
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write base.fbin, queries.fbin, groundtruth.*.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for synthetic data and query sampling (default: 42).",
    )


def _add_verify_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "verify",
        help="Validate that nprobe GT matches exact GT at small scale.",
    )
    p.add_argument("--stats", required=True, help="Path to fingerprint NPZ.")
    p.add_argument(
        "--total_rows",
        type=int,
        default=1_000_000,
        help="Synthetic dataset size to verify against (default: 1M).",
    )
    p.add_argument(
        "--n_queries",
        type=int,
        default=10_000,
        help="Number of queries (default: 10000).",
    )
    p.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of nearest neighbors in the GT (default: 10).",
    )
    p.add_argument(
        "--nprobes",
        type=int,
        default=10,
        help=(
            "Clusters probed per query for the nprobe GT being verified "
            "(default: 10)."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for query sampling (default: 42).",
    )


def _cmd_fit(args: argparse.Namespace) -> int:
    print(f"Loading dataset from {args.dataset}...")
    data = load_dataset(args.dataset, sample_size=args.sample_size)
    print(f"Loaded data shape: {data.shape}")

    stats = fit_cluster_stats(
        data=data,
        n_clusters=args.n_clusters,
        pca_components=args.pca_components,
        seed=args.seed,
        max_iter=args.max_iter,
        normalize_for_clustering=(
            False if args.no_normalize_for_clustering else None
        ),
    )

    output_path = args.output
    if output_path is None:
        stem = os.path.splitext(os.path.basename(args.dataset))[0]
        ss_tag = "" if args.sample_size is None else f"_ss{args.sample_size}"
        nonorm_tag = "_nonorm" if args.no_normalize_for_clustering else ""
        output_path = (
            f"{stem}_nc{args.n_clusters}_ncomp{args.pca_components}"
            f"{ss_tag}_seed{args.seed}{nonorm_tag}.npz"
        )
    save_cluster_stats(output_path, stats)
    print(f"Saved cluster fingerprint to {output_path}")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    print(f"Loading fingerprint from {args.stats}...")
    stats = load_cluster_stats(args.stats)
    config = cluster_config_from_stats(stats, seed=args.seed)
    print(
        f"  {config.nclusters} clusters, {config.ncols} dims, "
        f"ncomp={stats['pca_n_components']}, "
        f"is_normalized_data={config.is_normalized_data} (from fit)"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    base_path = os.path.join(args.output_dir, "base.fbin")
    print(
        f"Streaming {args.total_rows:,} synthetic vectors to {base_path} "
        f"(host RAM stays bounded regardless of total_rows)..."
    )
    generate_synthetic_dataset_to_file(
        config=config,
        total_points=args.total_rows,
        output_path=base_path,
    )

    print(f"Sampling {args.n_queries} queries...")
    queries = generate_queries(
        nqueries=args.n_queries,
        total_rows=args.total_rows,
        config=config,
    )
    queries_path = os.path.join(args.output_dir, "queries.fbin")
    write_bin(queries_path, queries)

    print(f"Computing ground truth in '{args.gt_mode}' mode...")
    if args.gt_mode == "nprobe":
        gt_idx, gt_dist, _ = compute_groundtruth_nprobe(
            queries=queries,
            total_rows=args.total_rows,
            config=config,
            k=args.k,
            nprobes=args.nprobes,
        )
    else:
        gt_idx, gt_dist, _ = compute_groundtruth_exact(
            queries=queries,
            total_rows=args.total_rows,
            config=config,
            k=args.k,
        )

    gt_idx_path = os.path.join(args.output_dir, "groundtruth.neighbors.ibin")
    gt_dist_path = os.path.join(args.output_dir, "groundtruth.distances.fbin")
    write_bin(gt_idx_path, gt_idx.astype(np.uint32))
    write_bin(gt_dist_path, gt_dist.astype(np.float32))

    print(
        f"\nWrote synthetic dataset bundle to {args.output_dir}:\n"
        f"  base.fbin                 ({args.total_rows:,} x {config.ncols})\n"
        f"  queries.fbin              ({args.n_queries} x {config.ncols})\n"
        f"  groundtruth.neighbors.ibin ({args.n_queries} x {args.k})\n"
        f"  groundtruth.distances.fbin ({args.n_queries} x {args.k})"
    )
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    print(f"Loading fingerprint from {args.stats}...")
    stats = load_cluster_stats(args.stats)
    config = cluster_config_from_stats(stats, seed=args.seed)
    print(
        f"  {config.nclusters} clusters, {config.ncols} dims, "
        f"is_normalized_data={config.is_normalized_data} (from fit)"
    )

    result = verify_groundtruth(
        config=config,
        total_rows=args.total_rows,
        nqueries=args.n_queries,
        k=args.k,
        nprobes=args.nprobes,
    )

    print("\n" + "=" * 60)
    print("Verification Result")
    print("=" * 60)
    print(f"  Recall (with ties): {result['recall']:.4f}")
    print(f"  Recall (naive):     {result['simple_recall']:.4f}")
    print(
        f"  Imperfect queries:  {len(result['mismatched_queries'])}"
        f" / {result['nqueries']}"
    )
    if result["recall"] >= 0.999:
        print(
            f"\n  nprobes={args.nprobes} is safe at total_rows={args.total_rows:,}."
        )
    else:
        print(
            f"\n  nprobes={args.nprobes} drops accuracy below 0.999 -- "
            f"consider increasing nprobes."
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cuvs_bench.synthesize_dataset",
        description=(
            "Synthesize an ANN-benchmark dataset of arbitrary size from a "
            "small real-data fingerprint. See README.md for the recommended "
            "workflow."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_fit_parser(subparsers)
    _add_generate_parser(subparsers)
    _add_verify_parser(subparsers)

    args = parser.parse_args(argv)
    if args.command == "fit":
        return _cmd_fit(args)
    if args.command == "generate":
        return _cmd_generate(args)
    if args.command == "verify":
        return _cmd_verify(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
