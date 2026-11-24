# fbin.py
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import Config
from .task import WorkUnit, VectorReader, VectorWriter


_INT32_MAX = np.iinfo(np.int32).max


def read_fbin_header(path: Path) -> tuple[int, int]:
    """Read FBIN header and return (dim, base_n).

    FBIN file format:
        [int32 count][int32 dim]  (8-byte header)
        followed by count * dim float32 payload.

    Parameters
    ----------
    path:
        Input FBIN path.

    Returns
    -------
    (dim, base_n):
        `dim` is vector dimensionality.
        `base_n` is the number of vectors in the input file.

    Notes
    -----
    `WorkManager.__init__` expects `(dim, base_n)` ordering.
    """
    path = Path(path)
    with path.open("rb") as f:
        hdr = np.frombuffer(f.read(8), dtype=np.int32)
    if hdr.size != 2:
        raise ValueError("Invalid FBIN header")
    return int(hdr[1]), int(hdr[0])


def compute_fbin_shard_size(cfg: Config) -> int:
    """Compute the effective FBIN shard size in vectors.

    If `cfg.shard_size` is None, returns the maximum allowed by the FBIN int32 header
    limit. Otherwise clamps the provided shard size to int32 max.

    Returns
    -------
    int
        Vectors per shard (always in [1, int32_max]).
    """
    effective_shard_size = _INT32_MAX if cfg.shard_size is None else int(cfg.shard_size)
    effective_shard_size = min(int(effective_shard_size), _INT32_MAX)
    return int(effective_shard_size)


def fbin_shard_path(output_path: Path, *, shard_id: int, shard_size: int) -> Path:
    """Return the on-disk path for a specific FBIN output shard."""
    output_path = Path(output_path)
    shard_id = int(shard_id)
    shard_size = int(shard_size)

    if shard_id == 0 and shard_size >= _INT32_MAX:
        return output_path

    stem = output_path.stem
    suf = output_path.suffix
    return output_path.with_name(f"{stem}_part{shard_id:03d}{suf}")


def precreate_fbin_shards(output_path: Path, dim: int, shard_size: int, total_n: int) -> None:
    """Create all FBIN shard files with placeholder headers (no payload preallocation)."""
    shard_size = min(int(shard_size), _INT32_MAX)
    output_path = Path(output_path)
    shard_size = int(shard_size)
    total_n = int(total_n)

    n_shards = (total_n + shard_size - 1) // shard_size if total_n > 0 else 1
    hdr = np.array([0, int(dim)], dtype=np.int32).tobytes()

    for shard_id in range(n_shards):
        p = fbin_shard_path(output_path, shard_id=shard_id, shard_size=shard_size)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            f.write(hdr)


def backpatch_fbin_headers(output_path: Path, dim: int, shard_size: int, total_n: int) -> list[int]:
    """Finalize FBIN shard headers after all payload writes complete."""
    shard_size = min(int(shard_size), _INT32_MAX)
    output_path = Path(output_path)
    shard_size = int(shard_size)
    total_n = int(total_n)

    n_shards = (total_n + shard_size - 1) // shard_size if total_n > 0 else 1

    counts: list[int] = []
    for shard_id in range(n_shards):
        start = shard_id * shard_size
        end = min(total_n, (shard_id + 1) * shard_size)
        count_in_shard = max(0, end - start)
        counts.append(int(count_in_shard))

        p = fbin_shard_path(output_path, shard_id=shard_id, shard_size=shard_size)
        hdr = np.array([int(count_in_shard), int(dim)], dtype=np.int32).tobytes()
        with p.open("r+b") as f:
            f.seek(0)
            f.write(hdr)

    return counts



def shard_paths_for_total_n(output_path: Path, shard_size: int, total_n: int) -> list[Path]:
    """Return the expected FBIN shard paths for a given (shard_size, total_n) layout.

    This is a pure layout helper: it computes
        n_shards = ceil(total_n / shard_size)
    (and returns at least one shard when total_n == 0),
    then maps shard ids 0..n_shards-1 through `fbin_shard_path()`.

    Parameters
    ----------
    output_path:
        Base output path / shard naming stem.
    shard_size:
        Maximum number of vectors per shard.
    total_n:
        Total vectors across all shards.

    Returns
    -------
    list[Path]
        Shard paths in shard_id order.
    """
    output_path = Path(output_path)
    shard_size = int(shard_size)
    total_n = int(total_n)
    n_shards = (total_n + shard_size - 1) // shard_size if total_n > 0 else 1
    return [fbin_shard_path(output_path, shard_id=i, shard_size=shard_size) for i in range(n_shards)]





@dataclass
class FbinReader(VectorReader):
    """FBIN reader (per-thread). Uses `os.pread()` and returns float32 (count, dim)."""

    input_path: Path
    dim: int

    _opened: bool = field(default=False, init=False, repr=False)
    _in_fd: Optional[int] = field(default=None, init=False, repr=False)

    def open(self) -> None:
        """Open the input FBIN file for reading (idempotent).

        Uses `os.open(..., O_RDONLY)` and stores the resulting FD for `os.pread()` calls.
        """
        if self._opened:
            return
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
        """Read `wu.count` vectors starting at `wu.in_start` from the FBIN payload.

        Uses `os.pread()` with an absolute offset (no shared file cursor).
        Returns a new contiguous float32 array of shape (count, dim).

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
            raise RuntimeError("FbinReader is not open()")

        byte_off = 8 + int(wu.in_start) * int(self.dim) * 4
        nbytes = count * int(self.dim) * 4

        buf = os.pread(self._in_fd, nbytes, byte_off)
        if len(buf) != nbytes:
            raise ValueError("Truncated FBIN read")

        return np.frombuffer(buf, dtype=np.float32).reshape(count, int(self.dim)).copy()


@dataclass
class FbinWriter(VectorWriter):
    """FBIN writer (per-thread). Uses shard files + `os.pwrite()` (payload only)."""

    output_path: Path
    dim: int
    shard_size: int

    _opened: bool = field(default=False, init=False, repr=False)
    _out_fds: dict[int, int] = field(default_factory=dict, init=False, repr=False)

    def open(self) -> None:
        """Mark the writer as ready (idempotent).
        Shard files are opened lazily on first write per shard via `_get_out_fd()`.
        """
        self._opened = True

    def close(self) -> None:
        """Close all cached shard FDs (best-effort, idempotent)."""
        for fd in list(self._out_fds.values()):
            try:
                os.close(fd)
            except Exception:
                pass
        self._out_fds.clear()
        self._opened = False

    def _get_out_fd(self, shard_id: int) -> int:
        """Return an open FD for the given shard file, opening and caching it if needed."""
        fd = self._out_fds.get(int(shard_id))
        if fd is not None:
            return fd
        p = fbin_shard_path(Path(self.output_path), shard_id=int(shard_id), shard_size=int(self.shard_size))
        fd = os.open(os.fspath(p), os.O_RDWR)
        self._out_fds[int(shard_id)] = fd
        return fd

    def write_range(self, wu: WorkUnit, x: np.ndarray) -> None:
        """Write matrix `x` into the FBIN output at `wu.out_global_start`.

        Writes payload only (no header). Automatically splits the write across shard
        boundaries and uses `os.pwrite()` with absolute offsets.

        Parameters
        ----------
        wu:
            Work unit describing the global output start index.
        x:
            Float32 matrix of shape (n, dim). `n` may be <= `wu.count` (caller-controlled).

        Raises
        ------
        ValueError
            If `x` shape is not (n, dim).
        """
        if x.size == 0:
            return
        if x.ndim != 2 or x.shape[1] != int(self.dim):
            raise ValueError(f"Expected (n, {self.dim}), got {x.shape}")

        x = np.asarray(x, dtype=np.float32, order="C")
        n = int(x.shape[0])
        off = 0
        while off < n:
            gidx = int(wu.out_global_start) + off
            shard_id = gidx // int(self.shard_size)
            idx_in_shard = gidx % int(self.shard_size)
            take = min(n - off, int(self.shard_size) - idx_in_shard)

            out_fd = self._get_out_fd(shard_id)
            byte_off = 8 + idx_in_shard * int(self.dim) * 4
            chunk = x[off : off + take]
            os.pwrite(out_fd, chunk.tobytes(order="C"), byte_off)
            off += take

