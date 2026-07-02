#!/usr/bin/env python
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import argparse
import importlib
import os
import sys
import warnings

from .utils import (
    groundtruth_neighbors_filename,
    memmap_bin_file,
    offset_neighbor_indices,
    suffix_from_dtype,
    write_bin,
    write_groundtruth_neighbors,
)


def import_with_fallback(primary_lib, secondary_lib=None, alias=None):
    """
    Attempt to import a primary library, with an optional fallback to a
    secondary library.
    Optionally assigns the imported module to a global alias.

    Parameters
    ----------
    primary_lib : str
        Name of the primary library to import.
    secondary_lib : str, optional
        Name of the secondary library to use as a fallback. If `None`,
        no fallback is attempted.
    alias : str, optional
        Alias to assign the imported module globally.

    Returns
    -------
    module or None
        The imported module if successful; otherwise, `None`.

    Examples
    --------
    >>> xp = import_with_fallback('cupy', 'numpy')
    >>> mod = import_with_fallback('nonexistent_lib')
    >>> if mod is None:
    ...     print("Library not found.")
    """
    try:
        module = importlib.import_module(primary_lib)
    except ImportError:
        if secondary_lib is not None:
            try:
                module = importlib.import_module(secondary_lib)
            except ImportError:
                module = None
        else:
            module = None
    if alias and module is not None:
        globals()[alias] = module
    return module


xp = import_with_fallback("cupy", "numpy")
rmm = import_with_fallback("rmm")
gpu_system = False


def force_fallback_to_numpy():
    global xp, gpu_system
    xp = import_with_fallback("numpy")
    gpu_system = False
    warnings.warn(
        "Consider using a GPU-based system to greatly accelerate "
        " generating groundtruths using cuVS."
    )


if rmm is not None:
    gpu_system = True
    try:
        from rmm.allocators.cupy import rmm_cupy_allocator

        from cuvs.common import Resources
        from cuvs.neighbors import filters
        from cuvs.neighbors.brute_force import build, search
    except ImportError:
        # RMM is available, cupy is available, but cuVS is not
        force_fallback_to_numpy()
else:
    # No RMM, no cuVS, but cupy is available
    force_fallback_to_numpy()


def generate_random_queries(n_queries, n_features, dtype=xp.float32):
    print("Generating random queries")
    if xp.issubdtype(dtype, xp.integer):
        queries = xp.random.randint(
            0, 255, size=(n_queries, n_features), dtype=dtype
        )
    else:
        queries = xp.random.uniform(size=(n_queries, n_features)).astype(dtype)
    return queries


def choose_random_queries(dataset, n_queries):
    print("Choosing random vector from dataset as query vectors")
    query_idx = xp.random.choice(
        dataset.shape[0], size=(n_queries,), replace=False
    )
    return dataset[query_idx, :]


def create_bitset_filter(n_samples, filter_reject_rate):
    """
    Creates a packed uint32 bitset where bit i is set iff vector i passes the
    filter.  Uses a modulo-1000 bucket scheme: vector i passes when
    ``i % 1000 >= round(filter_reject_rate * 1000)``, giving a reject rate
    within 0.1% of the requested value.

    Parameters
    ----------
    n_samples : int
        Number of vectors in the dataset.
    filter_reject_rate : float
        Fraction of vectors to reject, in [0.0, 1.0).

    Returns
    -------
    numpy.ndarray
        Packed uint32 array of shape ``(ceil(n_samples / 32),)``.
    """
    import numpy as np

    fail_buckets = round(filter_reject_rate * 1000)
    n_padded = ((n_samples + 31) // 32) * 32
    bool_mask = np.zeros(n_padded, dtype=bool)
    bool_mask[:n_samples] = (np.arange(n_samples) % 1000) >= fail_buckets
    # Pack with little-endian bit order: bit j maps to bit (j%32) of uint32
    # word (j//32), LSB first — matching cuVS bitset layout.
    return np.packbits(bool_mask, bitorder="little").view(np.uint32)


def cpu_search(dataset, queries, k, metric="squeclidean", accept_mask=None):
    """
    Find the k nearest neighbors for each query point in the dataset using the
    specified metric.

    Parameters
    ----------
    dataset : numpy.ndarray
        An array of shape (n_samples, n_features) representing the dataset.
    queries : numpy.ndarray
        An array of shape (n_queries, n_features) representing the query
        points.
    k : int
        The number of nearest neighbors to find.
    metric : str, optional
        The distance metric to use. Can be 'squeclidean' or 'inner_product'.
        Default is 'squeclidean'.
    accept_mask : numpy.ndarray, optional
        Boolean array of shape (n_samples,). Where False, the corresponding
        dataset vector is excluded from results.

    Returns
    -------
    distances : numpy.ndarray
        An array of shape (n_queries, k) containing the distances
        (for 'squeclidean') or similarities
        (for 'inner_product') to the k nearest neighbors for each query.
    indices : numpy.ndarray
        An array of shape (n_queries, k) containing the indices of the
        k nearest neighbors in the dataset for each query.

    """
    if metric == "squeclidean":
        diff = queries[:, xp.newaxis, :] - dataset[xp.newaxis, :, :]
        dist_sq = xp.sum(diff**2, axis=2)  # Shape: (n_queries, n_samples)

        if accept_mask is not None:
            dist_sq[:, ~accept_mask] = xp.inf

        indices = xp.argpartition(dist_sq, kth=k - 1, axis=1)[:, :k]
        distances = xp.take_along_axis(dist_sq, indices, axis=1)

        sorted_idx = xp.argsort(distances, axis=1)
        distances = xp.take_along_axis(distances, sorted_idx, axis=1)
        indices = xp.take_along_axis(indices, sorted_idx, axis=1)

    elif metric == "inner_product":
        similarities = xp.dot(
            queries, dataset.T
        )  # Shape: (n_queries, n_samples)

        if accept_mask is not None:
            similarities[:, ~accept_mask] = -xp.inf

        neg_similarities = -similarities
        indices = xp.argpartition(neg_similarities, kth=k - 1, axis=1)[:, :k]
        distances = xp.take_along_axis(similarities, indices, axis=1)

        sorted_idx = xp.argsort(-distances, axis=1)

    else:
        raise ValueError(
            "Unsupported metric in cuvs-bench-cpu. "
            "Use 'squeclidean' or 'inner_product' or use the GPU package"
            "to use any distance supported by cuVS."
        )

    distances = xp.take_along_axis(distances, sorted_idx, axis=1)
    indices = xp.take_along_axis(indices, sorted_idx, axis=1)

    return distances, indices


def calc_truth(dataset, queries, k, metric="sqeuclidean", bitset=None):
    """
    Calculate exact nearest neighbors, optionally with a prefilter.

    Parameters
    ----------
    dataset : array-like
        Dataset of shape (n_samples, n_features).
    queries : array-like
        Queries of shape (n_queries, n_features).
    k : int
        Number of neighbors.
    metric : str
        Distance metric.
    bitset : numpy.ndarray, optional
        Packed uint32 array of shape (ceil(n_samples / 32),) as returned by
        :func:`create_bitset_filter`.  Bit i set means vector i passes the
        filter.  When None, all vectors are considered.
    """
    import numpy as np

    n_samples = dataset.shape[0]
    n = 500000  # batch size for processing neighbors
    i = 0
    indices = None
    distances = None
    queries = xp.asarray(queries, dtype=xp.float32)

    if gpu_system:
        resources = Resources()

    while i < n_samples:
        print("Step {0}/{1}:".format(i // n, n_samples // n))
        n_batch = n if i + n <= n_samples else n_samples - i

        X = xp.asarray(dataset[i : i + n_batch, :], xp.float32)

        if gpu_system:
            index = build(X, metric=metric, resources=resources)
            prefilter = None
            if bitset is not None:
                word_start = i // 32
                word_end = (i + n_batch + 31) // 32
                batch_words = xp.asarray(bitset[word_start:word_end])
                prefilter = filters.from_bitset(batch_words)
            D, Ind = search(
                index, queries, k, resources=resources, prefilter=prefilter
            )
            resources.sync()
        else:
            accept_mask = None
            if bitset is not None:
                word_start = i // 32
                word_end = (i + n_batch + 31) // 32
                batch_bytes = bitset[word_start:word_end].view(np.uint8)
                accept_mask = np.unpackbits(batch_bytes, bitorder="little")[
                    :n_batch
                ].astype(bool)
            D, Ind = cpu_search(
                X, queries, k, metric=metric, accept_mask=accept_mask
            )

        D, Ind = xp.asarray(D), xp.asarray(Ind)
        Ind = offset_neighbor_indices(Ind, i, n_samples)

        if distances is None:
            distances = D
            indices = Ind
        else:
            distances = xp.concatenate([distances, D], axis=1)
            indices = xp.concatenate([indices, Ind], axis=1)
            sort_keys = -distances if metric == "inner_product" else distances
            idx = xp.argsort(sort_keys, axis=1)[:, :k]
            distances = xp.take_along_axis(distances, idx, axis=1)
            indices = xp.take_along_axis(indices, idx, axis=1)

        i += n_batch

    return distances, indices


def main():
    if gpu_system and xp.__name__ == "cupy":
        pool = rmm.mr.PoolMemoryResource(
            rmm.mr.CudaMemoryResource(), initial_pool_size=2**30
        )
        rmm.mr.set_current_device_resource(pool)
        xp.cuda.set_allocator(rmm_cupy_allocator)
    else:
        # RMM is available, but cupy is not
        force_fallback_to_numpy()

    parser = argparse.ArgumentParser(
        prog="generate_groundtruth",
        description="Generate true neighbors using exact NN search. "
        "The input and output files are in big-ann-benchmark's binary format.",
        epilog="""Example usage
    # With existing query file
    python -m cuvs_bench.generate_groundtruth --dataset /dataset/base.\
fbin --output=groundtruth_dir --queries=/dataset/query.public.10K.fbin

    # With randomly generated queries
    python -m cuvs_bench.generate_groundtruth --dataset /dataset/base.\
fbin --output=groundtruth_dir --queries=random --n_queries=10000

    # Using only a subset of the dataset. Define queries by randomly
    # selecting vectors from the (subset of the) dataset.
    python -m cuvs_bench.generate_groundtruth --dataset /dataset/base.\
fbin --nrows=2000000 --cols=128 --output=groundtruth_dir \
--queries=random-choice --n_queries=10000

    # Prefiltered ground truth using a saved bitset file
    python -m cuvs_bench.generate_groundtruth --dataset /dataset/base.\
fbin --output=groundtruth_dir --queries=/dataset/query.fbin \
--bitset=/dataset/filter.npy

    # Prefiltered ground truth generated on-the-fly from a reject rate
    python -m cuvs_bench.generate_groundtruth --dataset /dataset/base.\
fbin --output=groundtruth_dir --queries=/dataset/query.fbin \
--filter_reject_rate=0.1
    """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("dataset", type=str, help="input dataset file name")
    parser.add_argument(
        "--queries",
        type=str,
        default="random",
        help="Queries file name, or one of 'random-choice' or 'random' "
        "(default). 'random-choice': select n_queries vectors from the input "
        "dataset. 'random': generate n_queries as uniform random numbers.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="output directory name (default current dir)",
    )

    parser.add_argument(
        "--n_queries",
        type=int,
        default=10000,
        help="Number of queries to generate (if no query file is given). "
        "Default: 10000.",
    )

    parser.add_argument(
        "-N",
        "--rows",
        default=None,
        type=int,
        help="use only first N rows from dataset, by default the whole "
        "dataset is used",
    )
    parser.add_argument(
        "-D",
        "--cols",
        default=None,
        type=int,
        help="number of features (dataset columns). "
        "Default: read from dataset file.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        help="Dataset dtype. When not specified, then derived from extension."
        " Supported types: 'float32', 'float16', 'uint8', 'int8'",
    )

    parser.add_argument(
        "-k",
        type=int,
        default=100,
        help="Number of neighbors (per query) to calculate",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="sqeuclidean",
        help="Metric to use while calculating distances. Valid metrics are "
        "those that are accepted by cuvs.neighbors.brute_force.knn. Most"
        " commonly used with cuVS are 'sqeuclidean' and 'inner_product'",
    )

    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument(
        "--bitset",
        type=str,
        default=None,
        help="Path to a .npy file containing a packed uint32 prefilter "
        "bitset of shape (ceil(n_samples / 32),). Bit i set means vector i "
        "passes the filter. Mutually exclusive with --filter_reject_rate.",
    )
    filter_group.add_argument(
        "--filter_reject_rate",
        type=float,
        default=None,
        help="Fraction of vectors to reject, in [0.0, 1.0). Generates a "
        "bitset using a modulo-1000 bucket scheme (vector i passes when "
        "i %% 1000 >= round(filter_reject_rate * 1000)). Mutually exclusive "
        "with --bitset.",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    if args.rows is not None:
        print("Reading subset of the data, nrows=", args.rows)
    else:
        print("Reading whole dataset")

    # Load input data
    dataset = memmap_bin_file(
        args.dataset, args.dtype, shape=(args.rows, args.cols)
    )
    n_features = dataset.shape[1]
    n_samples = dataset.shape[0]
    dtype = dataset.dtype

    print(
        "Dataset size {:6.1f} GB, shape {}, dtype {}".format(
            dataset.size * dataset.dtype.itemsize / 1e9,
            dataset.shape,
            xp.dtype(dtype),
        )
    )

    if len(args.output) > 0:
        os.makedirs(args.output, exist_ok=True)

    if args.queries == "random" or args.queries == "random-choice":
        if args.n_queries is None:
            raise RuntimeError(
                "n_queries must be given to generate random queries"
            )
        if args.queries == "random":
            queries = generate_random_queries(
                args.n_queries, n_features, dtype
            )
        elif args.queries == "random-choice":
            queries = choose_random_queries(dataset, args.n_queries)

        queries_filename = os.path.join(
            args.output, "queries" + suffix_from_dtype(dtype)
        )
        print("Writing queries file", queries_filename)
        write_bin(queries_filename, queries)
    else:
        print("Reading queries from file", args.queries)
        queries = memmap_bin_file(args.queries, dtype)

    # Resolve prefilter bitset.
    bitset = None
    if args.bitset is not None:
        import numpy as np

        print("Loading prefilter bitset from", args.bitset)
        bitset = np.load(args.bitset)
    elif args.filter_reject_rate is not None:
        print(
            f"Generating prefilter bitset for filter_reject_rate="
            f"{args.filter_reject_rate}"
        )
        bitset = create_bitset_filter(n_samples, args.filter_reject_rate)

    print("Calculating true nearest neighbors")
    distances, indices = calc_truth(
        dataset, queries, args.k, args.metric, bitset=bitset
    )

    n_base = dataset.shape[0]
    write_groundtruth_neighbors(
        os.path.join(args.output, groundtruth_neighbors_filename(n_base)),
        indices,
        n_base,
    )
    write_bin(
        os.path.join(args.output, "groundtruth.distances.fbin"),
        distances.astype(xp.float32),
    )


if __name__ == "__main__":
    main()
