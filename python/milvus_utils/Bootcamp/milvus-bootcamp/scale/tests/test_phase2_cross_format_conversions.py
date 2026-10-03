from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scale.work_manager import WorkManager
from scale.io.fbin import read_fbin_header
from scale.io.fvecs import read_fvecs_dim_and_n


@dataclass
class _Cfg:
    # Minimal config shim for WorkManager runtime access.
    input_path: Path
    output_path: Path
    scale: int = 1
    shard_size: int | None = 4  # small to exercise sharding in FBIN output


def _write_fbin(path: Path, x: np.ndarray) -> None:
    x = np.asarray(x, dtype=np.float32, order="C")
    n, dim = x.shape
    hdr = np.array([n, dim], dtype=np.int32).tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(hdr)
        f.write(x.tobytes(order="C"))


def _read_fbin_payload(path: Path) -> np.ndarray:
    dim, n = read_fbin_header(path)
    with Path(path).open("rb") as f:
        f.seek(8)
        buf = f.read(n * dim * 4)
    return np.frombuffer(buf, dtype=np.float32).reshape(n, dim).copy()


def _write_fvecs(path: Path, x: np.ndarray) -> None:
    x = np.asarray(x, dtype=np.float32, order="C")
    n, dim = x.shape
    bpv = 4 * (1 + dim)
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


def _read_fvecs_payload(path: Path) -> np.ndarray:
    dim, n = read_fvecs_dim_and_n(path)
    bpv = 4 * (1 + dim)
    raw = Path(path).read_bytes()
    arr = np.frombuffer(raw, dtype=np.float32).reshape(n, dim + 1)[:, 1:]
    return arr.copy()


def _run_copy_pipeline(wm: WorkManager, batch_size: int = 4) -> None:
    work = wm.plan_work(base_n=int(wm.base_n), batch_size=batch_size, scale=int(wm.conf.scale))
    task = wm.get_task_factory()()
    task.open()
    try:
        for wu in work:
            x = task.read_range(wu)
            # Phase-2 test: no transforms/noise here. Just copy into output location.
            task.write_range(wu, x)
    finally:
        task.close()
    wm.finalize()


def test_fbin_to_fvecs_scale1_copy_values_equal(tmp_path: Path) -> None:
    dim = 5
    x = (np.arange(13 * dim, dtype=np.float32).reshape(13, dim) + 0.25)

    in_path = tmp_path / "in.fbin"
    out_path = tmp_path / "out.fvecs"
    _write_fbin(in_path, x)

    cfg = _Cfg(input_path=in_path, output_path=out_path, scale=1, shard_size=4)
    wm = WorkManager(cfg)  # type: ignore[arg-type]
    _run_copy_pipeline(wm, batch_size=4)

    got = _read_fvecs_payload(out_path)
    np.testing.assert_allclose(got, x, rtol=0, atol=0)


def test_fvecs_to_fbin_scale1_copy_values_equal(tmp_path: Path) -> None:
    dim = 4
    x = (np.arange(9 * dim, dtype=np.float32).reshape(9, dim) * 0.1 - 1.0)

    in_path = tmp_path / "in.fvecs"
    out_path = tmp_path / "out.fbin"
    _write_fvecs(in_path, x)

    cfg = _Cfg(input_path=in_path, output_path=out_path, scale=1, shard_size=4)
    wm = WorkManager(cfg)  # type: ignore[arg-type]
    _run_copy_pipeline(wm, batch_size=3)

    # FBIN output may be sharded if shard_size is small.
    # WorkManager.finalize() returns paths + counts, but for value check we read only the first shard
    # when shard_size >= n. Here shard_size=4 and n=9, so sharded output is expected.
    paths, counts = wm.finalize()
    assert sum(counts) == x.shape[0]

    # Reconstruct output from shards in order.
    parts = []
    for p, c in zip(paths, counts):
        part = _read_fbin_payload(p)
        assert part.shape[0] == c
        parts.append(part)
    got = np.vstack(parts) if parts else np.empty((0, dim), dtype=np.float32)

    np.testing.assert_allclose(got, x, rtol=0, atol=0)
