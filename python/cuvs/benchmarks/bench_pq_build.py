#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
End-to-end benchmark for cuVS Product Quantizer build.

Mirrors the C++ setup:
  - 256k train vectors (sampled from SIFT1M .fbin)
  - pq_bits=8 (256 centers), pq_dim=128 subspaces, use_vq=False
  - classic kmeans with max_iter=12 (default init = k-means|| / scalable++)

Example:
  python bench_pq_build.py \\
      --dataset /path/to/sift-128-euclidean/base.fbin \\
      --num-train 262144 --repeats 3
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np

from cuvs.common import Resources
from cuvs.preprocessing.quantize import pq


def read_fbin(path: Path) -> np.ndarray:
    """Load a .fbin file (int32 n, int32 dim, then n*dim float32)."""
    with open(path, "rb") as f:
        n, dim = np.fromfile(f, dtype=np.int32, count=2)
        data = np.fromfile(f, dtype=np.float32, count=int(n) * int(dim))
    return np.ascontiguousarray(data.reshape(int(n), int(dim)))


def load_train(dataset: str, num_train: int, dim: int, seed: int) -> np.ndarray:
    path = Path(dataset)
    if not path.exists():
        raise FileNotFoundError(path)

    data = read_fbin(path)
    if data.ndim != 2:
        raise ValueError(f"expected 2D dataset, got shape {data.shape}")
    if data.shape[1] != dim:
        raise ValueError(f"dataset dim={data.shape[1]} does not match --dim={dim}")
    if data.shape[0] < num_train:
        raise ValueError(
            f"dataset has {data.shape[0]} rows, need at least --num-train={num_train}"
        )

    if data.shape[0] == num_train:
        train = data
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(data.shape[0], size=num_train, replace=False)
        train = np.ascontiguousarray(data[idx])

    print(f"Loaded {path}: using {train.shape[0]} / {data.shape[0]} rows, dim={train.shape[1]}")
    return train


def time_pq_build(
    train: np.ndarray,
    *,
    pq_bits: int,
    pq_dim: int,
    kmeans_n_iters: int,
    pq_kmeans_type: str,
    use_subspaces: bool,
    warmup: int,
    repeats: int,
) -> list[float]:
    """Return wall times (seconds) for end-to-end pq.build calls."""
    resources = Resources()
    params = pq.QuantizerParams(
        pq_bits=pq_bits,
        pq_dim=pq_dim,
        use_subspaces=use_subspaces,
        use_vq=False,
        vq_n_centers=0,
        kmeans_n_iters=kmeans_n_iters,
        pq_kmeans_type=pq_kmeans_type,
        # Use the full trainset (matches C++ max_train_points_per_pq_code=num_train).
        max_train_points_per_pq_code=train.shape[0],
        max_train_points_per_vq_cluster=train.shape[0],
    )

    print(
        "QuantizerParams("
        f"pq_bits={params.pq_bits}, pq_dim={params.pq_dim}, "
        f"use_subspaces={params.use_subspaces}, use_vq={params.use_vq}, "
        f"kmeans_n_iters={params.kmeans_n_iters}, pq_kmeans_type={pq_kmeans_type}, "
        f"max_train_points_per_pq_code={params.max_train_points_per_pq_code})"
    )

    def once() -> float:
        resources.sync()
        t0 = time.perf_counter()
        quantizer = pq.build(params, train, resources=resources)
        resources.sync()
        elapsed = time.perf_counter() - t0
        _ = quantizer.pq_bits
        return elapsed

    for i in range(warmup):
        t = once()
        print(f"  warmup[{i}]: {t:.3f}s")

    times: list[float] = []
    for i in range(repeats):
        t = once()
        times.append(t)
        print(f"  repeat[{i}]: {t:.3f}s")
    return times


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to SIFT (or other) base.fbin",
    )
    p.add_argument("--num-train", type=int, default=262144, help="Train vectors (default 256k)")
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--pq-bits", type=int, default=8, help="Implies n_centers = 2**pq_bits")
    p.add_argument("--pq-dim", type=int, default=128, help="Number of PQ subspaces / chunks")
    p.add_argument("--kmeans-n-iters", type=int, default=12)
    p.add_argument(
        "--pq-kmeans-type",
        choices=("kmeans", "kmeans_balanced"),
        default="kmeans",
        help="'kmeans' uses classic kmeans (default init = scalable k-means++)",
    )
    p.add_argument("--no-subspaces", action="store_true", help="Disable per-subspace codebooks")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    train = load_train(args.dataset, args.num_train, args.dim, args.seed)
    n_centers = 1 << args.pq_bits
    print(
        f"Benchmark: n_train={train.shape[0]}, dim={train.shape[1]}, "
        f"pq_dim={args.pq_dim}, n_centers={n_centers}, max_iter={args.kmeans_n_iters}"
    )

    times = time_pq_build(
        train,
        pq_bits=args.pq_bits,
        pq_dim=args.pq_dim,
        kmeans_n_iters=args.kmeans_n_iters,
        pq_kmeans_type=args.pq_kmeans_type,
        use_subspaces=not args.no_subspaces,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    mean = statistics.mean(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0.0
    print("-" * 60)
    print(
        f"pq.build: mean={mean:.3f}s  stdev={stdev:.3f}s  "
        f"min={min(times):.3f}s  max={max(times):.3f}s  (n={len(times)})"
    )


if __name__ == "__main__":
    main()
