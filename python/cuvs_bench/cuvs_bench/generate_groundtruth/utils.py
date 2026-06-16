#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import os

import numpy as np

from cuvs_bench._bin_format import read_bin_header, write_bin_header


def is_l2_normalized(
    data,
    sample_size: int = 10_000,
    tol: float = 1e-2,
    seed: int = 0,
) -> bool:
    """Cheaply check whether ``data`` rows are L2-unit-norm.

    Samples up to ``sample_size`` rows uniformly at random and returns ``True``
    iff every sampled row has ``|‖x‖ - 1| < tol``.
    """
    n = len(data)
    if n == 0:
        return False
    rng = np.random.default_rng(seed)
    take = min(sample_size, n)
    idx = rng.choice(n, size=take, replace=False)
    norms = np.linalg.norm(data[idx].astype(np.float32), axis=1)
    return bool(np.all(np.abs(norms - 1.0) < tol))


def add_jitter(
    queries: np.ndarray,
    rng: np.random.Generator,
    normalize: bool,
) -> np.ndarray:
    """Add Gaussian jitter to query vectors and optionally re-normalize."""
    noise_scale = float(np.std(queries)) * 0.1
    queries = queries + rng.normal(0, noise_scale, queries.shape).astype(
        np.float32
    )
    if normalize:
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        queries = queries / np.maximum(norms, 1e-8)
    return queries.astype(np.float32)


def dtype_from_filename(filename):
    ext = os.path.splitext(filename)[1]
    if ext == ".fbin":
        return np.float32
    if ext == ".hbin":
        return np.float16
    elif ext == ".ibin":
        return np.int32
    elif ext == ".u64bin":
        return np.uint64
    elif ext == ".u8bin":
        return np.ubyte
    elif ext == ".i8bin":
        return np.byte
    else:
        raise RuntimeError("Not supported file extension" + ext)


def suffix_from_dtype(dtype):
    if dtype == np.float32:
        return ".fbin"
    if dtype == np.float16:
        return ".hbin"
    elif dtype == np.int32:
        return ".ibin"
    elif dtype == np.uint64:
        return ".u64bin"
    elif dtype == np.ubyte:
        return ".u8bin"
    elif dtype == np.byte:
        return ".i8bin"
    else:
        raise RuntimeError("Not supported dtype extension" + dtype)


def neighbor_index_dtype(n_base: int) -> np.dtype:
    """Return the dtype used to store neighbor row IDs for a base set size."""
    if n_base > np.iinfo(np.int32).max:
        return np.uint64
    return np.int32


def neighbor_index_accumulator_dtype(n_base: int) -> np.dtype:
    """Return the in-memory dtype for neighbor IDs during GT computation.

    cuVS brute-force search returns ``int64`` neighbors. Use ``int64`` for
    large bases so batch offsets up to multi-billion row counts do not
    overflow; cast to :func:`neighbor_index_dtype` only when writing files.
    """
    if n_base > np.iinfo(np.int32).max:
        return np.int64
    return np.int32


def groundtruth_neighbors_filename(n_base: int) -> str:
    """Return the ground-truth neighbors filename for a base set size."""
    if n_base > np.iinfo(np.int32).max:
        return "groundtruth.neighbors.u64bin"
    return "groundtruth.neighbors.ibin"


def offset_neighbor_indices(indices, batch_offset: int, n_base: int):
    """Shift local neighbor IDs by a batch offset without integer overflow."""
    dtype = neighbor_index_accumulator_dtype(n_base)
    return indices.astype(dtype) + batch_offset


def _to_host_numpy(array, dtype):
    """Copy *array* to a host NumPy array, including from CuPy if needed."""
    get = getattr(array, "get", None)
    if callable(get):
        array = get()
    return np.asarray(array, dtype=dtype)


def write_groundtruth_neighbors(path, indices, n_base: int):
    """Write a ground-truth neighbor matrix using the correct on-disk dtype."""
    storage_dtype = neighbor_index_dtype(n_base)
    data = _to_host_numpy(indices, storage_dtype)
    write_bin(path, data)


def memmap_bin_file(
    bin_file, dtype, shape=None, mode="r", size_dtype=np.uint32
):
    """Memory-map a cuvs-bench binary file.

    Supports both the legacy 8-byte ``[uint32 n_rows, uint32 n_cols]`` and
    the extended 16-byte ``[uint64 n_rows, uint64 n_cols]`` headers. In read
    mode the layout is auto-detected from the file size; in write mode the
    legacy layout is used unless ``size_dtype=np.uint64`` or one of the shape
    dimensions exceeds ``UINT32_MAX``.

    Parameters
    ----------
    bin_file : str or None
        Path to the binary file. ``None`` short-circuits and returns ``None``
        (preserves the historical "skip optional file" behavior).
    dtype : numpy dtype or None
        Element dtype. If ``None``, inferred from the file extension via
        :func:`dtype_from_filename`.
    shape : tuple or None
        Read mode: optionally override ``(n_rows, n_cols)`` from the header;
        any ``None`` entries are filled in from the header value. Write mode:
        required ``(n_rows, n_cols)`` of the file to create.
    mode : str
        Standard ``np.memmap`` mode string (``"r"``, ``"r+"``, ``"w+"``).
    size_dtype : numpy dtype
        Write mode only: ``np.uint32`` for the legacy 8-byte header (default),
        or ``np.uint64`` to force the extended 16-byte header. Ignored in read
        mode (auto-detected).
    """
    if bin_file is None:
        return None
    if dtype is None:
        dtype = dtype_from_filename(bin_file)
    itemsize = np.dtype(dtype).itemsize

    if shape is not None and len(shape) != 2:
        raise ValueError(
            f"shape must have exactly 2 dimensions (n_rows, n_cols), got {shape!r}"
        )

    if mode[0] == "r":
        n_rows, n_cols, header_bytes = read_bin_header(bin_file, itemsize)
        if shape is None:
            final_shape = (n_rows, n_cols)
        else:
            header_dims = (n_rows, n_cols)
            final_shape = tuple(
                aval if sval is None else sval
                for aval, sval in zip(header_dims, shape)
            )
        return np.memmap(
            bin_file,
            mode=mode,
            dtype=dtype,
            offset=header_bytes,
            shape=final_shape,
        )
    elif mode[0] == "w":
        if shape is None:
            raise ValueError("Need to specify shape to map file in write mode")

        print("creating file", bin_file)
        dirname = os.path.dirname(bin_file)
        if len(dirname) > 0:
            os.makedirs(dirname, exist_ok=True)
        with open(bin_file, "wb") as f:
            header_bytes = write_bin_header(
                f, shape[0], shape[1], size_dtype=size_dtype
            )
        return np.memmap(
            bin_file,
            mode="r+",
            dtype=dtype,
            offset=header_bytes,
            shape=shape,
        )


def write_bin(fname, data):
    """Write a 2-D numpy array to a cuvs-bench binary file."""
    print("writing", fname, data.shape, data.dtype, "...")
    with open(fname, "wb") as f:
        write_bin_header(f, data.shape[0], data.shape[1])
        data.tofile(f)
