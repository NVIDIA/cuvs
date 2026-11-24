# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate a synthetic clustered dataset as a big-ann style .fbin file.

The cluster-assignment and fit-tuning benchmarks read every dataset, synthetic or real, from an
.fbin file (uint32 n_rows, uint32 dim, then row-major float32). Use this script when no real
dataset is at hand: it draws points from a Gaussian mixture, so unlike uniform random data the
points form clusters of uneven size and spread, which is closer to real embeddings.

Each point is its component's center plus Gaussian noise:
  - centers are drawn from N(0, I)
  - component weights are log-normal, so cluster sizes are uneven
  - each component has its own noise std, drawn uniformly from [--min-std, --max-std]

Usage:
  python generate_clustered_dataset.py --n-rows 10000000 --dim 128 --out synthetic_10M_128.fbin
"""

import argparse
import struct

import numpy as np

CHUNK_ROWS = 1 << 20


def generate(
    out_path: str,
    n_rows: int,
    dim: int,
    n_components: int,
    min_std: float,
    max_std: float,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_components, dim), dtype=np.float32)
    stds = rng.uniform(min_std, max_std, n_components).astype(np.float32)
    weights = rng.lognormal(mean=0.0, sigma=1.0, size=n_components)
    weights /= weights.sum()

    with open(out_path, "wb") as f:
        f.write(struct.pack("<II", n_rows, dim))
        for first in range(0, n_rows, CHUNK_ROWS):
            count = min(CHUNK_ROWS, n_rows - first)
            comp = rng.choice(n_components, size=count, p=weights)
            noise = rng.standard_normal((count, dim), dtype=np.float32)
            points = centers[comp] + noise * stds[comp, None]
            f.write(np.ascontiguousarray(points, dtype=np.float32).tobytes())
    print(
        f"wrote {n_rows:,} x {dim} points from {n_components} components "
        f"-> {out_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", required=True, help="output .fbin path")
    parser.add_argument("--n-rows", type=int, required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument(
        "--n-components",
        type=int,
        default=2000,
        help="number of mixture components (true clusters in the data)",
    )
    parser.add_argument("--min-std", type=float, default=0.2)
    parser.add_argument("--max-std", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    generate(
        args.out,
        args.n_rows,
        args.dim,
        args.n_components,
        args.min_std,
        args.max_std,
        args.seed,
    )


if __name__ == "__main__":
    main()
