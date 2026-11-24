import os
from pathlib import Path

import numpy as np
import pytest

from scale.config import Config, NoiseScheme
from scale.io.fbin import (
    FbinReader,
    FbinWriter,
    backpatch_fbin_headers,
    compute_fbin_shard_size,
    fbin_shard_path,
    precreate_fbin_shards,
    read_fbin_header,
    shard_paths_for_total_n,
)
from scale.io.task import WorkUnit


def _write_fbin(path: Path, *, dim: int, x: np.ndarray) -> None:
    """
    Write a minimal FBIN file:
      [int32 count][int32 dim] + payload float32 row-major.
    """
    x = np.asarray(x, dtype=np.float32, order="C")
    assert x.ndim == 2 and x.shape[1] == dim
    count = int(x.shape[0])

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(np.array([count, dim], dtype=np.int32).tobytes())
        f.write(x.tobytes(order="C"))


def _read_fbin_payload(path: Path, *, dim: int, count: int) -> np.ndarray:
    with path.open("rb") as f:
        f.seek(8)
        buf = f.read(count * dim * 4)
    return np.frombuffer(buf, dtype=np.float32).reshape(count, dim).copy()


def test_read_fbin_header_ok(tmp_path: Path) -> None:
    p = tmp_path / "in.fbin"
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    _write_fbin(p, dim=4, x=x)

    dim, n = read_fbin_header(p)
    assert dim == 4
    assert n == 3


def test_read_fbin_header_invalid_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.fbin"
    p.write_bytes(b"\x00\x00\x00")  # < 8 bytes
    with pytest.raises(ValueError, match="buffer size"):
        read_fbin_header(p)


def test_compute_fbin_shard_size_default_and_clamp(tmp_path: Path) -> None:
    cfg_none = Config(
        input_path=tmp_path / "in.fbin",
        output_path=tmp_path / "out.fbin",
        scale=1,
        batch_size=1,
        noise_amplitude=0.0,
        noise_scheme=NoiseScheme.ALL,
        normalize=False,
        tolerance=1e-3,
        shard_size=None,
        seed=None,
        json_metadata=None,
        workers=1,
    )
    s = compute_fbin_shard_size(cfg_none)
    assert isinstance(s, int)
    assert s > 0

    cfg_huge = cfg_none.__class__(**{**cfg_none.__dict__, "shard_size": 2**40})
    s2 = compute_fbin_shard_size(cfg_huge)
    # Implementation clamps to int32 max.
    assert s2 == np.iinfo(np.int32).max


def test_fbin_shard_path_naming_rules(tmp_path: Path) -> None:
    out = tmp_path / "out.fbin"

    # "Single shard" naming rule: shard 0 + shard_size >= int32 max returns base path.
    p0 = fbin_shard_path(out, shard_id=0, shard_size=np.iinfo(np.int32).max)
    assert p0 == out

    # Otherwise suffix uses _partXXX
    p1 = fbin_shard_path(out, shard_id=1, shard_size=100)
    assert p1.name == "out_part001.fbin"

    p12 = fbin_shard_path(out, shard_id=12, shard_size=100)
    assert p12.name == "out_part012.fbin"


def test_shard_paths_for_total_n_layout(tmp_path: Path) -> None:
    out = tmp_path / "out.fbin"

    # total_n == 0 still yields at least one shard
    paths0 = shard_paths_for_total_n(out, shard_size=10, total_n=0)
    assert len(paths0) == 1

    paths = shard_paths_for_total_n(out, shard_size=10, total_n=21)
    assert len(paths) == 3
    assert paths[0].name == "out_part000.fbin" or paths[0] == out  # depends on shard rule; here shard_size small => part000


def test_precreate_and_backpatch_headers(tmp_path: Path) -> None:
    out = tmp_path / "out.fbin"
    dim = 4
    shard_size = 5
    total_n = 12  # -> 3 shards: 5,5,2

    precreate_fbin_shards(out, dim=dim, shard_size=shard_size, total_n=total_n)

    paths = shard_paths_for_total_n(out, shard_size=shard_size, total_n=total_n)
    assert len(paths) == 3

    # Placeholder header count is 0, dim is set.
    for p in paths:
        with p.open("rb") as f:
            hdr = np.frombuffer(f.read(8), dtype=np.int32)
        assert hdr.tolist() == [0, dim]

    counts = backpatch_fbin_headers(out, dim=dim, shard_size=shard_size, total_n=total_n)
    assert counts == [5, 5, 2]

    # Verify headers updated with correct per-shard counts.
    for p, c in zip(paths, counts):
        with p.open("rb") as f:
            hdr = np.frombuffer(f.read(8), dtype=np.int32)
        assert hdr.tolist() == [c, dim]


def test_fbin_reader_requires_open_and_handles_empty(tmp_path: Path) -> None:
    p = tmp_path / "in.fbin"
    x = np.arange(20, dtype=np.float32).reshape(5, 4)
    _write_fbin(p, dim=4, x=x)

    r = FbinReader(p, dim=4)

    # Empty read is OK even before open (implementation checks count first)
    wu0 = WorkUnit(batch_id=0, replica_id=0, in_start=0, count=0, out_global_start=0)
    got0 = r.read_range(wu0)
    assert got0.shape == (0, 4)

    # Non-empty requires open
    wu = WorkUnit(batch_id=0, replica_id=0, in_start=1, count=2, out_global_start=0)
    with pytest.raises(RuntimeError, match="not open"):
        r.read_range(wu)

    r.open()
    r.open()  # idempotent
    got = r.read_range(wu)
    assert got.shape == (2, 4)
    np.testing.assert_allclose(got, x[1:3])

    r.close()
    r.close()  # idempotent


def test_fbin_reader_truncated_read_raises(tmp_path: Path) -> None:
    p = tmp_path / "trunc.fbin"
    # Write header claiming count=3, dim=4 but only 1 vector payload.
    with p.open("wb") as f:
        f.write(np.array([3, 4], dtype=np.int32).tobytes())
        f.write(np.zeros((1, 4), dtype=np.float32).tobytes(order="C"))

    r = FbinReader(p, dim=4)
    r.open()
    wu = WorkUnit(batch_id=0, replica_id=0, in_start=0, count=3, out_global_start=0)
    with pytest.raises(ValueError, match="Truncated FBIN read"):
        r.read_range(wu)
    r.close()


def test_fbin_writer_validates_shape_and_noop_on_empty(tmp_path: Path) -> None:
    out = tmp_path / "out.fbin"
    dim = 4
    shard_size = 3
    total_n = 7  # -> 3 shards (3,3,1)

    precreate_fbin_shards(out, dim=dim, shard_size=shard_size, total_n=total_n)

    w = FbinWriter(out, dim=dim, shard_size=shard_size)
    w.open()

    wu = WorkUnit(batch_id=0, replica_id=0, in_start=0, count=2, out_global_start=0)

    # empty is noop
    w.write_range(wu, np.empty((0, dim), dtype=np.float32))

    # bad shape raises
    with pytest.raises(ValueError, match="Expected"):
        w.write_range(wu, np.zeros((2, dim + 1), dtype=np.float32))

    w.close()
    w.close()


def test_fbin_writer_splits_across_shards_and_writes_payload(tmp_path: Path) -> None:
    out = tmp_path / "out.fbin"
    dim = 4
    shard_size = 3
    total_n = 7  # shards: [3,3,1]

    precreate_fbin_shards(out, dim=dim, shard_size=shard_size, total_n=total_n)
    paths = shard_paths_for_total_n(out, shard_size=shard_size, total_n=total_n)
    assert len(paths) == 3

    w = FbinWriter(out, dim=dim, shard_size=shard_size)
    w.open()

    # Write 5 vectors starting at global index 2:
    # shard0: indices 2 (1 vec)
    # shard1: indices 3,4,5 (3 vecs)
    # shard2: index 6 (1 vec)
    x = (np.arange(5 * dim, dtype=np.float32).reshape(5, dim) + 1000.0).astype(np.float32)
    wu = WorkUnit(batch_id=0, replica_id=0, in_start=0, count=5, out_global_start=2)
    w.write_range(wu, x)

    # Validate payload bytes at the right positions per shard.
    # shard0 has one vector written at idx_in_shard=2
    shard0_payload = _read_fbin_payload(paths[0], dim=dim, count=3)
    np.testing.assert_allclose(shard0_payload[2], x[0])

    # shard1 has three vectors written at idx 0..2
    shard1_payload = _read_fbin_payload(paths[1], dim=dim, count=3)
    np.testing.assert_allclose(shard1_payload[0], x[1])
    np.testing.assert_allclose(shard1_payload[1], x[2])
    np.testing.assert_allclose(shard1_payload[2], x[3])

    # shard2 has one vector written at idx 0
    shard2_payload = _read_fbin_payload(paths[2], dim=dim, count=1)
    np.testing.assert_allclose(shard2_payload[0], x[4])

    # Also exercise FD caching: after writes across shards, cache should have 3 fds.
    assert len(w._out_fds) == 3  # implementation detail; fine for unit coverage here

    w.close()

    # After close, all fds are gone.
    assert w._out_fds == {}


def test_fbin_writer_opens_shards_lazily(tmp_path: Path) -> None:
    out = tmp_path / "out.fbin"
    dim = 4
    shard_size = 3
    total_n = 7

    precreate_fbin_shards(out, dim=dim, shard_size=shard_size, total_n=total_n)

    w = FbinWriter(out, dim=dim, shard_size=shard_size)
    w.open()
    assert w._out_fds == {}

    # Write entirely into shard 0 only => should open exactly one shard FD.
    x = np.ones((2, dim), dtype=np.float32)
    wu = WorkUnit(batch_id=0, replica_id=0, in_start=0, count=2, out_global_start=0)
    w.write_range(wu, x)

    assert len(w._out_fds) == 1
    w.close()