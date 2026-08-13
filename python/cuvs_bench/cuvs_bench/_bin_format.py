#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

"""
On-disk header helpers for the cuvs-bench binary file format.

cuvs-bench inherits the big-ann-benchmarks binary layout: a small header
listing ``n_rows`` and ``n_cols`` followed by a dense ``n_rows * n_cols``
array of the dtype implied by the file extension. Two layouts are supported:

- **Legacy**:  ``[uint32 n_rows, uint32 n_cols, data ...]``  (8-byte header).
  This is what every existing ``.fbin`` / ``.ibin`` / ``.u8bin`` / ``.i8bin``
  / ``.f16bin`` / ``.hbin`` / ``.u64bin`` file on disk uses today.

- **Extended**: ``[uint64 n_rows, uint64 n_cols, data ...]``  (16-byte header).
  For datasets whose ``n_rows`` or ``n_cols`` exceeds ``UINT32_MAX`` (~4.29B).

Detection is **size-based**: a well-formed cuvs-bench binary is exactly
``header_bytes + n_rows * n_cols * itemsize`` bytes long. :func:`read_bin_header` reads the first 16 bytes
of the file and:

1. Tries the legacy layout (first 8 bytes as two ``uint32``s, 8-byte
   header). The layout is accepted if ``8 + n_rows * n_cols * itemsize``
   matches the on-disk file size.
2. Otherwise tries the extended layout (first 16 bytes as two
   ``uint64``s, 16-byte header). Accepted if
   ``16 + n_rows * n_cols * itemsize`` matches the file size instead.
3. If neither layout matches, raises ``ValueError`` -- the file is
   truncated, padded, or has a mismatched dtype extension.
"""

import os
import struct
from typing import BinaryIO, Tuple

import numpy as np

UINT32_MAX = (1 << 32) - 1

LEGACY_HEADER_BYTES = 8
EXTENDED_HEADER_BYTES = 16


def read_bin_header(path: str, itemsize: int) -> Tuple[int, int, int]:
    """Read the header of a cuvs-bench binary file.

    Auto-detects the on-disk layout from the file size by checking which
    of the two layouts (legacy 8-byte uint32 header, extended 16-byte uint64
    header) makes ``file_size == header_bytes + n_rows * n_cols * itemsize``
    balance.

    Parameters
    ----------
    path : str
        Path to the binary file.
    itemsize : int
        Per-element size in bytes (e.g. ``4`` for ``float32``, ``1`` for
        ``int8``) used for the size-equation check.

    Returns
    -------
    (n_rows, n_cols, header_bytes) : Tuple[int, int, int]
        Row count, column count, and the number of bytes the header
        occupies on disk (``8`` for legacy, ``16`` for extended).

    Raises
    ------
    ValueError
        If neither the legacy nor the extended interpretation matches.
    FileNotFoundError
        If ``path`` does not exist.
    """
    if itemsize < 1:
        raise ValueError(
            f"itemsize must be a positive integer, got {itemsize!r}"
        )
    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(EXTENDED_HEADER_BYTES)

    if len(head) < LEGACY_HEADER_BYTES:
        raise ValueError(
            f"File too small to contain a valid header (expected at least "
            f"{LEGACY_HEADER_BYTES} bytes, got {len(head)}): {path}"
        )

    n_rows_32, n_cols_32 = struct.unpack("<II", head[:LEGACY_HEADER_BYTES])
    if file_size == LEGACY_HEADER_BYTES + n_rows_32 * n_cols_32 * itemsize:
        return int(n_rows_32), int(n_cols_32), LEGACY_HEADER_BYTES

    if len(head) == EXTENDED_HEADER_BYTES:
        n_rows_64, n_cols_64 = struct.unpack("<QQ", head)
        if (
            file_size
            == EXTENDED_HEADER_BYTES + n_rows_64 * n_cols_64 * itemsize
        ):
            return int(n_rows_64), int(n_cols_64), EXTENDED_HEADER_BYTES

    raise ValueError(
        f"File size {file_size:,} bytes does not match either the legacy "
        f"(8-byte uint32) or extended (16-byte uint64) header layout for "
        f"itemsize={itemsize}: {path}. The file may be truncated, padded, "
        f"or have a mismatched dtype extension."
    )


def write_bin_header(
    f: BinaryIO,
    n_rows: int,
    n_cols: int,
    *,
    size_dtype=np.uint32,
) -> int:
    """Write the canonical cuvs-bench binary header at the current position.

    The legacy 8-byte uint32 layout is used whenever both ``n_rows`` and
    ``n_cols`` fit in a ``uint32``. The 16-byte uint64 layout is used
    otherwise, or when explicitly requested via ``size_dtype=np.uint64``.

    Parameters
    ----------
    f : BinaryIO
        Open binary file handle, positioned where the header should go.
    n_rows, n_cols : int
        Header values to write. Must be non-negative.
    size_dtype : numpy dtype
        ``np.uint32`` for the legacy 8-byte header (default), or
        ``np.uint64`` to force the extended 16-byte header.

    Returns
    -------
    int
        Number of bytes written (``8`` for legacy, ``16`` for extended).
    """
    if n_rows < 0 or n_cols < 0:
        raise ValueError(
            f"n_rows and n_cols must be non-negative, got ({n_rows}, {n_cols})"
        )
    use_uint64 = (
        np.dtype(size_dtype) == np.uint64
        or n_rows > UINT32_MAX
        or n_cols > UINT32_MAX
    )
    if use_uint64:
        f.write(struct.pack("<QQ", int(n_rows), int(n_cols)))
        return EXTENDED_HEADER_BYTES
    f.write(struct.pack("<II", int(n_rows), int(n_cols)))
    return LEGACY_HEADER_BYTES
