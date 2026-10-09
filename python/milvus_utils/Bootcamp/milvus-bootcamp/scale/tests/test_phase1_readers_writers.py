# test_phase1_readers_writers.py
from __future__ import annotations

from pathlib import Path

import numpy as np

from scale.io.fbin import (
    FbinReader,
    FbinWriter,
    precreate_fbin_shards,
    fbin_shard_path,
)
from scale.io.fvecs import (
    FvecsReader,
    FvecsWriter,
    precreate_fvecs_file,
    _bytes_per_vec,
)
from scale.io.task import WorkUnit


def _write_fbin(path: Path, x: np.ndarray) -> None:
    x = np.asarray(x, dtype=np.float32, order="C")
    n, dim = x.shape
    hdr = np.array([n, dim], dtype=np.int32).tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(hdr)
        f.write(x.tobytes(order="C"))


def _write_fvecs(path: Path, x: np.ndarray) -> None:
    x = np.asarray(x, dtype=np.float32, order="C")
    n, dim = x.shape
    bpv = _bytes_per_vec(dim)
    buf = bytearray(n * bpv)
    mv = memoryview(buf)
    dim_i32 = np.int32(dim).tobytes()
    for i in range(n):
        off = i * bpv
        mv[off : off + 4] = dim_i32
        mv[off + 4 : off + bpv] = x[i].tobytes(order="C")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(bytes(buf))


def test_fbin_reader_reads_exact_slice(tmp_path: Path) -> None:
    dim = 4
    base = np.arange(10 * dim, dtype=np.float32).reshape(10, dim)
    in_path = tmp_path / "in.fbin"
    _write_fbin(in_path, base)

    r = FbinReader(in_path, dim=dim)
    r.open()
    try:
        wu = WorkUnit(batch_id=0, replica_id=0, in_start=3, count=4, out_global_start=0)
        got = r.read_range(wu)
        assert got.shape == (4, dim)
        np.testing.assert_allclose(got, base[3:7])
    finally:
        r.close()


def test_fbin_writer_splits_across_shards(tmp_path: Path) -> None:
    dim = 3
    shard_size = 4

    out_base = tmp_path / "out.fbin"
    total_n = 7  # requires 2 shards if shard_size=4
    precreate_fbin_shards(out_base, dim=dim, shard_size=shard_size, total_n=total_n)

    w = FbinWriter(out_base, dim=dim, shard_size=shard_size)
    w.open()
    try:
        x = np.arange(5 * dim, dtype=np.float32).reshape(5, dim)
        # Start at global index 2, write 5 vectors -> indices [2..6], crosses shard boundary at 4.
        wu = WorkUnit(batch_id=0, replica_id=0, in_start=0, count=5, out_global_start=2)
        w.write_range(wu, x)
    finally:
        w.close()

    # Read back payload from both shards and validate placement.
    shard0 = fbin_shard_path(out_base, shard_id=0, shard_size=shard_size)
    shard1 = fbin_shard_path(out_base, shard_id=1, shard_size=shard_size)

    # shard0 has indices 0..3, shard1 has indices 4..7 (payload only after 8-byte header)
    with shard0.open("rb") as f:
        f.read(8)
        payload0 = np.frombuffer(f.read(), dtype=np.float32).reshape(-1, dim)
    with shard1.open("rb") as f:
        f.read(8)
        payload1 = np.frombuffer(f.read(), dtype=np.float32).reshape(-1, dim)

    # We wrote x to global indices 2..6:
    # shard0 indices 2..3 => x[0..1]
    # shard1 indices 0..2 => x[2..4]
    np.testing.assert_allclose(payload0[2:4], x[0:2])
    np.testing.assert_allclose(payload1[0:3], x[2:5])


def test_fvecs_reader_writer_roundtrip_at_offset(tmp_path: Path) -> None:
    dim = 5
    base = (np.arange(9 * dim, dtype=np.float32).reshape(9, dim) + 0.5)

    in_path = tmp_path / "in.fvecs"
    out_path = tmp_path / "out.fvecs"

    _write_fvecs(in_path, base)
    precreate_fvecs_file(out_path, dim=dim, total_n=20)

    r = FvecsReader(in_path, dim=dim)
    w = FvecsWriter(out_path, dim=dim)

    r.open()
    w.open()
    try:
        wu_r = WorkUnit(batch_id=0, replica_id=0, in_start=2, count=4, out_global_start=0)
        x = r.read_range(wu_r)

        wu_w = WorkUnit(batch_id=0, replica_id=0, in_start=0, count=4, out_global_start=7)
        w.write_range(wu_w, x)
    finally:
        r.close()
        w.close()

    # Read back written records from out_path at offset 7
    bpv = _bytes_per_vec(dim)
    byte_off = 7 * bpv
    nbytes = 4 * bpv
    raw = out_path.read_bytes()[byte_off : byte_off + nbytes]
    got = np.frombuffer(raw, dtype=np.float32).reshape(-1, dim + 1)[:, 1:]
    np.testing.assert_allclose(got, base[2:6])
