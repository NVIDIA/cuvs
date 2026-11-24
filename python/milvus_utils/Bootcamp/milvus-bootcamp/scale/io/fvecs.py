# fvecs.py
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .task import WorkUnit, VectorReader, VectorWriter


def read_fvecs_header(path: Path) -> tuple[int, int]:
    """Return (dim, n) for an FVECs file.

    FVECs has no global header; `dim` is taken from the first record prefix and `n`
    is derived from file size.
    """
    dim, n = read_fvecs_dim_and_n(Path(path))
    return int(dim), int(n)


def _bytes_per_vec(dim: int) -> int:
    """Return bytes per FVECs record: [int32 dim][float32 * dim]."""
    return 4 * (1 + int(dim))


def read_fvecs_dim_and_n(path: Path) -> tuple[int, int]:
    """Read FVECs dimensionality and vector count from file metadata."""
    path = Path(path)
    size = path.stat().st_size
    if size < 4:
        raise ValueError("FVECs file too small to read dim")

    with path.open("rb") as f:
        dim = int(np.frombuffer(f.read(4), dtype=np.int32)[0])

    bpv = _bytes_per_vec(dim)
    if size % bpv != 0:
        raise ValueError("FVECs file size is not aligned to record size")

    n = size // bpv
    return int(dim), int(n)


def _pack_fvecs_batch(dim: int, x: np.ndarray) -> bytes:
    """Pack a (n, dim) float32 matrix into FVECs record bytes."""
    x = np.asarray(x, dtype=np.float32, order="C")
    if x.ndim != 2 or x.shape[1] != int(dim):
        raise ValueError(f"Expected (n, {dim}), got {x.shape}")

    n = int(x.shape[0])
    bpv = _bytes_per_vec(int(dim))
    total = n * bpv

    buf = bytearray(total)
    mv = memoryview(buf)

    dim_i32 = np.int32(int(dim)).tobytes()
    for i in range(n):
        off = i * bpv
        mv[off : off + 4] = dim_i32
        mv[off + 4 : off + bpv] = x[i].tobytes(order="C")

    return bytes(buf)


def precreate_fvecs_file(output_path: Path, dim: int, total_n: int) -> None:
    """Preallocate the full FVECs output file to its final size."""
    output_path = Path(output_path)
    bpv = _bytes_per_vec(int(dim))
    total_bytes = int(total_n) * int(bpv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        f.truncate(total_bytes)




@dataclass
class FvecsReader(VectorReader):
    """FVECs reader (per-thread). Uses `os.pread()` and strips per-record dim prefix."""

    input_path: Path
    dim: int

    _opened: bool = field(default=False, init=False, repr=False)
    _bpv: int = field(default=0, init=False, repr=False)
    _in_fd: Optional[int] = field(default=None, init=False, repr=False)

    def open(self) -> None:
        """Open the input FVECs file for reading (idempotent).

        Computes bytes-per-vector (record size) and opens an FD for `os.pread()`.
        """
        if self._opened:
            return
        self._bpv = _bytes_per_vec(int(self.dim))
        self._in_fd = os.open(os.fspath(self.input_path), os.O_RDONLY)
        self._opened = True

    def close(self) -> None:
        """Close the input FD (best-effort, idempotent)."""
        if self._in_fd is not None:
            try:
                os.close(self._in_fd)
            except Exception:
                pass
            self._in_fd = None
        self._opened = False

    def read_range(self, wu: WorkUnit) -> np.ndarray:
        """Read `wu.count` FVECs records starting at `wu.in_start`.

        Uses `os.pread()` and returns a float32 array of shape (count, dim),
        dropping the per-record int32 dim prefix.

        Raises
        ------
        RuntimeError
            If called before `open()`.
        ValueError
            If the read is truncated.
        """
        count = int(wu.count)
        if count <= 0:
            return np.empty((0, int(self.dim)), dtype=np.float32)

        if self._in_fd is None:
            raise RuntimeError("FvecsReader is not open()")

        byte_off = int(wu.in_start) * int(self._bpv)
        nbytes = count * int(self._bpv)

        buf = os.pread(self._in_fd, nbytes, byte_off)
        if len(buf) != nbytes:
            raise ValueError("Truncated FVECs read")

        return np.frombuffer(buf, dtype=np.float32).reshape(-1, int(self.dim) + 1)[:, 1:].copy()


@dataclass
class FvecsWriter(VectorWriter):
    """FVECs writer (per-thread). Uses `os.pwrite()` to avoid shared file cursors."""

    output_path: Path
    dim: int

    _opened: bool = field(default=False, init=False, repr=False)
    _bpv: int = field(default=0, init=False, repr=False)
    _out_fd: Optional[int] = field(default=None, init=False, repr=False)

    def open(self) -> None:
        """Open the output FVECs file for writing (idempotent).

        Computes bytes-per-vector (record size) and opens an FD for `os.pwrite()`.
        """
        if self._opened:
            return
        self._bpv = _bytes_per_vec(int(self.dim))
        self._out_fd = os.open(os.fspath(self.output_path), os.O_RDWR)
        self._opened = True

    def close(self) -> None:
        """Close the output FD (best-effort, idempotent)."""
        if self._out_fd is not None:
            try:
                os.close(self._out_fd)
            except Exception:
                pass
            self._out_fd = None
        self._opened = False

    def write_range(self, wu: WorkUnit, x: np.ndarray) -> None:
        """Write matrix `x` as FVECs records at `wu.out_global_start`.

        Encodes each row as: [int32 dim][float32 * dim] and writes with `os.pwrite()`
        using an absolute byte offset (no shared file cursor).

        Raises
        ------
        RuntimeError
            If called before `open()`.
        ValueError
            If `x` shape is not (n, dim).
        """
        if x.size == 0:
            return
        if x.ndim != 2 or x.shape[1] != int(self.dim):
            raise ValueError(f"Expected (n, {self.dim}), got {x.shape}")

        if self._out_fd is None:
            raise RuntimeError("FvecsWriter is not open()")

        byte_off = int(wu.out_global_start) * int(self._bpv)
        payload = _pack_fvecs_batch(int(self.dim), x)
        os.pwrite(self._out_fd, payload, byte_off)

