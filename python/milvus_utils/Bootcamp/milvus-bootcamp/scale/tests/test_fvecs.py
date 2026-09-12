# tests/test_fvecs.py
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from scale.io.fvecs import (
    FvecsReader,
    FvecsWriter,
    _bytes_per_vec,
    _pack_fvecs_batch,
    precreate_fvecs_file,
    read_fvecs_dim_and_n,
    read_fvecs_header,
)
from scale.io.task import WorkUnit


def _write_fvecs(path: Path, x: np.ndarray) -> None:
    """Write a minimal FVECs file for tests: [int32 dim][float32*dim] per row."""
    x = np.asarray(x, dtype=np.float32, order="C")
    assert x.ndim == 2
    dim = int(x.shape[1])

    with path.open("wb") as f:
        for i in range(int(x.shape[0])):
            f.write(np.int32(dim).tobytes())
            f.write(x[i].tobytes(order="C"))


def test__bytes_per_vec() -> None:
    assert _bytes_per_vec(1) == 8
    assert _bytes_per_vec(3) == 16
    assert _bytes_per_vec(10) == 44


def test_read_fvecs_dim_and_n_happy(tmp_path: Path) -> None:
    p = tmp_path / "a.fvecs"
    x = np.arange(12, dtype=np.float32).reshape(4, 3)
    _write_fvecs(p, x)

    dim, n = read_fvecs_dim_and_n(p)
    assert dim == 3
    assert n == 4

    dim2, n2 = read_fvecs_header(p)
    assert (dim2, n2) == (3, 4)


def test_read_fvecs_dim_and_n_too_small(tmp_path: Path) -> None:
    p = tmp_path / "tiny.fvecs"
    p.write_bytes(b"\x00\x01\x02")  # < 4 bytes
    with pytest.raises(ValueError, match="too small"):
        read_fvecs_dim_and_n(p)


def test_read_fvecs_dim_and_n_misaligned_size(tmp_path: Path) -> None:
    p = tmp_path / "bad.fvecs"
    # dim=3 => bpv=16. Make size not divisible by 16.
    p.write_bytes(np.int32(3).tobytes() + b"\x00" * 1)
    with pytest.raises(ValueError, match="not aligned"):
        read_fvecs_dim_and_n(p)


def test_pack_fvecs_batch_happy() -> None:
    dim = 3
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    b = _pack_fvecs_batch(dim, x)
    assert isinstance(b, (bytes, bytearray))

    bpv = _bytes_per_vec(dim)
    assert len(b) == 2 * bpv

    # Validate record 0 prefix + payload
    rec0 = b[0:bpv]
    assert np.frombuffer(rec0[0:4], dtype=np.int32)[0] == dim
    assert np.allclose(np.frombuffer(rec0[4:], dtype=np.float32), x[0])

    # Validate record 1
    rec1 = b[bpv : 2 * bpv]
    assert np.frombuffer(rec1[0:4], dtype=np.int32)[0] == dim
    assert np.allclose(np.frombuffer(rec1[4:], dtype=np.float32), x[1])


def test_pack_fvecs_batch_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="Expected"):
        _pack_fvecs_batch(3, np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="Expected"):
        _pack_fvecs_batch(3, np.zeros((3,), dtype=np.float32))


def test_precreate_fvecs_file_allocates_size(tmp_path: Path) -> None:
    out = tmp_path / "out.fvecs"
    precreate_fvecs_file(out, dim=4, total_n=10)
    assert out.exists()
    assert out.stat().st_size == 10 * _bytes_per_vec(4)


def test_fvecs_reader_open_close_idempotent(tmp_path: Path) -> None:
    p = tmp_path / "in.fvecs"
    x = np.arange(8, dtype=np.float32).reshape(2, 4)
    _write_fvecs(p, x)

    r = FvecsReader(input_path=p, dim=4)
    r.open()
    r.open()  # idempotent
    r.close()
    r.close()  # idempotent


def test_fvecs_reader_read_range_happy(tmp_path: Path) -> None:
    p = tmp_path / "in.fvecs"
    x = np.arange(30, dtype=np.float32).reshape(5, 6)
    _write_fvecs(p, x)

    r = FvecsReader(input_path=p, dim=6)
    r.open()
    try:
        wu = WorkUnit(batch_id=0, replica_id=0, in_start=1, count=3, out_global_start=0)
        got = r.read_range(wu)
        assert got.shape == (3, 6)
        assert got.dtype == np.float32
        assert np.allclose(got, x[1:4])
        assert got.flags["OWNDATA"]  # .copy() in implementation
    finally:
        r.close()


def test_fvecs_reader_read_range_empty_count(tmp_path: Path) -> None:
    p = tmp_path / "in.fvecs"
    _write_fvecs(p, np.zeros((2, 3), dtype=np.float32))

    r = FvecsReader(input_path=p, dim=3)
    r.open()
    try:
        wu = WorkUnit(batch_id=0, replica_id=0, in_start=0, count=0, out_global_start=0)
        got = r.read_range(wu)
        assert got.shape == (0, 3)
        assert got.dtype == np.float32
    finally:
        r.close()


def test_fvecs_reader_requires_open(tmp_path: Path) -> None:
    p = tmp_path / "in.fvecs"
    _write_fvecs(p, np.zeros((2, 3), dtype=np.float32))

    r = FvecsReader(input_path=p, dim=3)
    wu = WorkUnit(batch_id=0, replica_id=0, in_start=0, count=1, out_global_start=0)
    with pytest.raises(RuntimeError, match="not open"):
        r.read_range(wu)


def test_fvecs_reader_truncated_read(tmp_path: Path) -> None:
    p = tmp_path / "in.fvecs"
    x = np.arange(12, dtype=np.float32).reshape(2, 6)
    _write_fvecs(p, x)

    # Truncate file to cause a short pread.
    os.truncate(p, p.stat().st_size - 1)

    r = FvecsReader(input_path=p, dim=6)
    r.open()
    try:
        wu = WorkUnit(batch_id=0, replica_id=0, in_start=0, count=2, out_global_start=0)
        with pytest.raises(ValueError, match="Truncated"):
            r.read_range(wu)
    finally:
        r.close()


def test_fvecs_writer_open_close_idempotent(tmp_path: Path) -> None:
    out = tmp_path / "out.fvecs"
    precreate_fvecs_file(out, dim=3, total_n=5)

    w = FvecsWriter(output_path=out, dim=3)
    w.open()
    w.open()  # idempotent
    w.close()
    w.close()  # idempotent


def test_fvecs_writer_requires_open(tmp_path: Path) -> None:
    out = tmp_path / "out.fvecs"
    precreate_fvecs_file(out, dim=3, total_n=5)

    w = FvecsWriter(output_path=out, dim=3)
    wu = WorkUnit(batch_id=0, replica_id=0, in_start=0, count=2, out_global_start=0)
    with pytest.raises(RuntimeError, match="not open"):
        w.write_range(wu, np.zeros((2, 3), dtype=np.float32))


def test_fvecs_writer_rejects_wrong_shape(tmp_path: Path) -> None:
    out = tmp_path / "out.fvecs"
    precreate_fvecs_file(out, dim=3, total_n=5)

    w = FvecsWriter(output_path=out, dim=3)
    w.open()
    try:
        wu = WorkUnit(batch_id=0, replica_id=0, in_start=0, count=2, out_global_start=0)
        with pytest.raises(ValueError, match="Expected"):
            w.write_range(wu, np.zeros((2, 2), dtype=np.float32))
    finally:
        w.close()


def test_fvecs_writer_write_and_read_back(tmp_path: Path) -> None:
    out = tmp_path / "out.fvecs"
    dim = 4
    total_n = 6
    precreate_fvecs_file(out, dim=dim, total_n=total_n)

    w = FvecsWriter(output_path=out, dim=dim)
    w.open()
    try:
        # Write 2 vectors at global start 0
        x0 = np.array([[1, 2, 3, 4], [10, 20, 30, 40]], dtype=np.float32)
        wu0 = WorkUnit(batch_id=0, replica_id=0, in_start=0, count=2, out_global_start=0)
        w.write_range(wu0, x0)

        # Write 3 vectors at global start 2
        x1 = np.array([[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.float32)
        wu1 = WorkUnit(batch_id=1, replica_id=0, in_start=0, count=3, out_global_start=2)
        w.write_range(wu1, x1)

        # Write empty should be a no-op
        wu_empty = WorkUnit(batch_id=2, replica_id=0, in_start=0, count=0, out_global_start=5)
        w.write_range(wu_empty, np.empty((0, dim), dtype=np.float32))
    finally:
        w.close()

    # Read back with the real reader
    r = FvecsReader(input_path=out, dim=dim)
    r.open()
    try:
        got0 = r.read_range(WorkUnit(batch_id=0, replica_id=0, in_start=0, count=2, out_global_start=0))
        got1 = r.read_range(WorkUnit(batch_id=0, replica_id=0, in_start=2, count=3, out_global_start=0))
        assert np.allclose(got0, x0)
        assert np.allclose(got1, x1)
    finally:
        r.close()


def test_close_swallow_os_close_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "in.fvecs"
    _write_fvecs(p, np.zeros((1, 2), dtype=np.float32))

    r = FvecsReader(input_path=p, dim=2)
    r.open()

    # Force os.close to raise; close() should swallow and reset state anyway.
    def _boom(_fd: int) -> None:
        raise OSError("boom")

    monkeypatch.setattr(os, "close", _boom)
    r.close()  # should not raise