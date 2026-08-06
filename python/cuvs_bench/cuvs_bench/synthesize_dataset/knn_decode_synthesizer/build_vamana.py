#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a cuVS Vamana index on a subset of an fbin dataset.

Reads an .fbin file, takes the first N vectors,
moves them to the GPU and runs cuVS Vamana with the given parameters while
timing the build.
"""

import argparse
import os
import time
from contextlib import contextmanager

import numpy as np


@contextmanager
def timer(label):
    """Context manager that prints the wall-clock time spent in the block."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[timer] {label}: {elapsed:.3f} s")


def read_fbin_subset(path, num_vectors, header_dtype=np.int32):
    """Read the first ``num_vectors`` vectors from an .fbin file.

    The .fbin layout is a 2-element header ``[n_vectors, dim]`` followed by
    ``n_vectors * dim`` float32 values. Standard .fbin uses int32 headers;
    pass ``np.int64`` for .fbin64-style files.
    """
    header_bytes = header_dtype().itemsize * 2
    with open(path, "rb") as f:
        header = np.fromfile(f, count=2, dtype=header_dtype)
        total_vectors, dim = int(header[0]), int(header[1])

        vectors_to_read = min(num_vectors, total_vectors)
        print(
            f"File: {total_vectors:,} vectors, dim={dim}. "
            f"Reading first {vectors_to_read:,} vectors."
        )

        # Header is already consumed; read only the needed float32 block.
        f.seek(header_bytes)
        data = np.fromfile(f, count=vectors_to_read * dim, dtype=np.float32)

    actual = data.size // dim
    if actual < vectors_to_read:
        print(
            f"Warning: only read {actual:,} vectors instead of {vectors_to_read:,}"
        )
        vectors_to_read = actual

    data = data[: vectors_to_read * dim].reshape(vectors_to_read, dim)
    return data, dim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fbin", help="Path to the input .fbin file")
    parser.add_argument(
        "--num-vectors",
        type=int,
        default=10_000_000,
        help="Number of vectors from the start of the file to use (default: 10M)",
    )
    parser.add_argument("--max-fraction", type=float, default=0.06)
    parser.add_argument("--visited-size", type=int, default=256)
    parser.add_argument("--graph-degree", type=int, default=64)
    parser.add_argument("--vamana-iters", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1.2)
    parser.add_argument(
        "--metric",
        default="sqeuclidean",
        help="Distance metric (default: sqeuclidean)",
    )
    parser.add_argument(
        "--header-dtype",
        choices=["int32", "int64"],
        default="int32",
        help="dtype of the 2-element fbin header (default: int32)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the built index as vamana.index in the input .fbin's directory",
    )
    args = parser.parse_args()

    # Imported here so the script can at least show --help without cuVS/cupy.
    import cupy as cp
    from cuvs.neighbors import vamana

    header_dtype = np.int32 if args.header_dtype == "int32" else np.int64

    with timer("read fbin subset (host)"):
        host_data, dim = read_fbin_subset(
            args.fbin, args.num_vectors, header_dtype
        )

    n = host_data.shape[0]
    gb = host_data.nbytes / 1e9
    print(f"Loaded {n:,} x {dim} float32 ({gb:.2f} GB) on host.")

    with timer("copy dataset host -> device"):
        device_data = cp.asarray(host_data)
        cp.cuda.Stream.null.synchronize()

    index_params = vamana.IndexParams(
        metric=args.metric,
        graph_degree=args.graph_degree,
        visited_size=args.visited_size,
        vamana_iters=args.vamana_iters,
        alpha=args.alpha,
        max_fraction=args.max_fraction,
    )

    print(
        "Vamana params: "
        f"metric={args.metric}, graph_degree={args.graph_degree}, "
        f"visited_size={args.visited_size}, vamana_iters={args.vamana_iters}, "
        f"alpha={args.alpha}, max_fraction={args.max_fraction}"
    )

    with timer(f"cuVS Vamana build ({n:,} vectors)"):
        index = vamana.build(index_params, device_data)
        cp.cuda.Stream.null.synchronize()

    print("Build complete.")

    if args.save:
        save_path = os.path.join(
            os.path.dirname(os.path.abspath(args.fbin)), "vamana.index"
        )
        with timer("save index"):
            vamana.save(save_path, index)
        print(f"Saved index to {save_path}")


if __name__ == "__main__":
    main()
