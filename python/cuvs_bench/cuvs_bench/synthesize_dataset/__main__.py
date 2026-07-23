#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Command-line entry point for the synthesize_dataset module.

- ``fit``: extract a cluster fingerprint (.npz) from a real dataset.
- ``generate``: turn a fingerprint into a ``base.fbin`` / ``queries.fbin`` /
``groundtruth.{neighbors,distances}.{ibin,fbin}`` bundle.
- ``verify``: validate the chosen ``nprobes``.

Run ``python -m cuvs_bench.synthesize_dataset --help`` for help.
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

import numpy as np

from ..generate_groundtruth.utils import (
    groundtruth_neighbors_filename,
    write_bin,
    write_groundtruth_neighbors,
)
from ._fit import fit_fingerprint
from ._generate import (
    generate_queries,
    generate_synthetic_dataset_to_file,
    resolve_norm_scheme,
)
from ._ground_truth import (
    compute_groundtruth_nprobe,
    compute_groundtruth_exact,
)
from ._io import (
    load_dataset,
    load_fingerprint,
    save_fingerprint,
)
from ._verify import verify_groundtruth


def _add_fit_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "fit",
        help="Extract a dataset fingerprint from a real dataset.",
        description=(
            "Run KMeans + per-cluster PCA on a sample of the real "
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
            "[_ss{sample_size}].npz' in the current "
            "working directory. '_ss{sample_size}' is included only when "
            "--sample_size is passed."
        ),
    )
    p.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help=(
            "Use only the FIRST `sample_size` rows of the input "
            "(default: use entire input). No shuffling is performed, so "
            "make sure the dataset's head-of-file slice is representative."
        ),
    )
    p.add_argument(
        "--n_clusters",
        type=int,
        required=True,
        help="Number of KMeans clusters.",
    )
    p.add_argument(
        "--pca_components",
        type=int,
        required=True,
        help="Number of PCA components per cluster.",
    )


def _add_generate_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "generate",
        help="Materialize a synthetic dataset + queries + GT from a fingerprint.",
    )
    p.add_argument(
        "--fingerprint", required=True, help="Path to fingerprint NPZ."
    )
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
        default=None,
        help=(
            "Clusters probed per query for nprobe GT. Must be smaller than "
            "the number of clusters used to fit the fingerprint. "
            "Defaults to 5%% of the fingerprint's cluster count unless specified when "
            "--gt_mode=nprobe."
        ),
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
        help="Validate that nprobe GT matches exact GT.",
        description=(
            "Check whether the chosen nprobes (relative to the number of "
            "clusters in the fingerprint) is sufficient to produce a "
            "high-quality ground truth. Runs nprobe GT and exact GT on a "
            "synthetic dataset and compares recall. Use a small --total_rows "
            "for a quick sanity check before committing to "
            "a full-scale generate run using ``generate''."
        ),
    )
    p.add_argument(
        "--fingerprint", required=True, help="Path to fingerprint NPZ."
    )
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
        default=None,
        help=(
            "Clusters probed per query for the nprobe GT being verified. "
            "Must be smaller than the number of clusters in the fingerprint. "
            "Defaults to 5%% of the fingerprint's cluster count."
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

    fingerprint = fit_fingerprint(
        data=data,
        n_clusters=args.n_clusters,
        pca_components=args.pca_components,
    )

    output_path = args.output
    if output_path is None:
        stem = os.path.splitext(os.path.basename(args.dataset))[0]
        ss_tag = "" if args.sample_size is None else f"_ss{args.sample_size}"
        output_path = (
            f"{stem}_nc{args.n_clusters}_ncomp{args.pca_components}"
            f"{ss_tag}.npz"
        )
    save_fingerprint(output_path, fingerprint)
    print(f"Saved fingerprint to {output_path}")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    print(f"Loading fingerprint from {args.fingerprint}...")
    config = load_fingerprint(args.fingerprint, seed=args.seed)
    if args.nprobes is not None and args.nprobes > config.nclusters:
        sys.exit(
            f"--nprobes ({args.nprobes}) must be <= the fingerprint's cluster "
            f"count ({config.nclusters})."
        )
    print(
        f"  {config.nclusters} clusters, {config.ncols} dims, "
        f"ncomp={config.pca_n_components}, "
        f"norm_scheme={resolve_norm_scheme(config)} "
    )
    os.makedirs(args.output_dir, exist_ok=True)

    base_path = os.path.join(args.output_dir, "base.fbin")
    generate_synthetic_dataset_to_file(
        config=config,
        total_points=args.total_rows,
        output_path=base_path,
    )

    queries = generate_queries(
        nqueries=args.n_queries,
        total_rows=args.total_rows,
        config=config,
    )
    queries_path = os.path.join(args.output_dir, "queries.fbin")
    write_bin(queries_path, queries)

    nprobes = args.nprobes
    if args.gt_mode == "nprobe" and nprobes is None:
        nprobes = max(1, round(config.nclusters * 0.05))
        print(
            f"  nprobes not set; using 5% of {config.nclusters} clusters = {nprobes}."
        )

    if args.gt_mode == "nprobe":
        gt_idx, gt_dist, _ = compute_groundtruth_nprobe(
            queries=queries,
            total_rows=args.total_rows,
            config=config,
            k=args.k,
            nprobes=nprobes,
        )
    else:
        gt_idx, gt_dist, _ = compute_groundtruth_exact(
            queries=queries,
            total_rows=args.total_rows,
            config=config,
            k=args.k,
        )

    gt_neighbors_fname = groundtruth_neighbors_filename(args.total_rows)
    gt_idx_path = os.path.join(args.output_dir, gt_neighbors_fname)
    gt_dist_path = os.path.join(args.output_dir, "groundtruth.distances.fbin")
    write_groundtruth_neighbors(gt_idx_path, gt_idx, args.total_rows)
    write_bin(gt_dist_path, gt_dist.astype(np.float32))

    dataset_name = os.path.basename(os.path.abspath(args.output_dir))
    yaml_entry = [
        {
            "name": dataset_name,
            "dims": config.ncols,
            "distance": "euclidean",
            "base_file": f"{dataset_name}/base.fbin",
            "query_file": f"{dataset_name}/queries.fbin",
            "groundtruth_neighbors_file": f"{dataset_name}/{gt_neighbors_fname}",
            "groundtruth_distances_file": f"{dataset_name}/groundtruth.distances.fbin",
        }
    ]
    yaml_path = os.path.join(args.output_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_entry, f, default_flow_style=False, sort_keys=False)

    print(
        f"\nWrote synthetic dataset bundle to {args.output_dir}:\n"
        f"  base.fbin                  ({args.total_rows:,} x {config.ncols})\n"
        f"  queries.fbin               ({args.n_queries} x {config.ncols})\n"
        f"  {gt_neighbors_fname} ({args.n_queries} x {args.k})\n"
        f"  groundtruth.distances.fbin ({args.n_queries} x {args.k})\n"
        f"  dataset.yaml               (pass to --dataset-configuration)"
    )
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    print(f"Loading fingerprint from {args.fingerprint}...")
    config = load_fingerprint(args.fingerprint, seed=args.seed)
    if args.nprobes is not None and args.nprobes > config.nclusters:
        sys.exit(
            f"--nprobes ({args.nprobes}) must be <= the fingerprint's cluster "
            f"count ({config.nclusters})."
        )
    nprobes = args.nprobes
    if nprobes is None:
        nprobes = max(1, round(config.nclusters * 0.05))
        print(
            f"  nprobes not set; using 5% of {config.nclusters} clusters = {nprobes}."
        )
    print(
        f"  {config.nclusters} clusters, {config.ncols} dims, "
        f"norm_scheme={resolve_norm_scheme(config)}"
    )

    result = verify_groundtruth(
        config=config,
        total_rows=args.total_rows,
        nqueries=args.n_queries,
        k=args.k,
        nprobes=nprobes,
    )

    print("\n" + "=" * 60)
    print("Verification Result")
    print("=" * 60)
    print(f"  Recall: {result['recall']:.4f}")
    if result["recall"] >= 0.999:
        print(
            f"\n  nprobes={nprobes} is safe at total_rows={args.total_rows:,}."
        )
    else:
        print(
            f"\n  nprobes={nprobes} drops recall below 0.999 -- "
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
        if args.sample_size is not None and args.sample_size <= 0:
            parser.error(
                f"--sample_size must be > 0 when provided "
                f"(got {args.sample_size})."
            )
        if args.n_clusters <= 0:
            parser.error(f"--n_clusters must be > 0 (got {args.n_clusters}).")
        if args.pca_components <= 0:
            parser.error(
                f"--pca_components must be > 0 (got {args.pca_components})."
            )
    elif args.command in ("generate", "verify"):
        if args.total_rows <= 0:
            parser.error(f"--total_rows must be > 0 (got {args.total_rows}).")
        if args.n_queries <= 0:
            parser.error(f"--n_queries must be > 0 (got {args.n_queries}).")
        if args.n_queries > args.total_rows:
            parser.error(
                f"--n_queries ({args.n_queries}) must be <= --total_rows "
                f"({args.total_rows}) for sampling without replacement."
            )
        if args.k <= 0:
            parser.error(f"--k must be > 0 (got {args.k}).")
        if args.nprobes is not None and args.nprobes <= 0:
            parser.error(f"--nprobes must be > 0 (got {args.nprobes}).")

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
