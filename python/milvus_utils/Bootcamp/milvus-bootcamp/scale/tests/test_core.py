from __future__ import annotations

import importlib
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


# ----------------------------
# Helpers: locate package under test dynamically
# ----------------------------



@pytest.fixture(scope="session")
def core_mod():
    pkg = "scale"
    # Ensure repo root is importable
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module(f"{pkg}.core")


@pytest.fixture(scope="session")
def config_mod():
    pkg = "scale"
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module(f"{pkg}.config")


@pytest.fixture(scope="session")
def task_mod():
    pkg = "scale"
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module(f"{pkg}.io.task")


# ----------------------------
# Dummy I/O task for worker/provider tests
# ----------------------------

class DummyIOTask:
    def __init__(self, read_returns: np.ndarray | None = None, *, fail_on_open: bool = False):
        self.open_calls = 0
        self.close_calls = 0
        self.read_calls = 0
        self.write_calls = 0
        self.fail_on_open = fail_on_open
        self.read_returns = read_returns if read_returns is not None else np.zeros((0, 0), dtype=np.float32)
        self.writes: list[tuple[Any, np.ndarray]] = []

    def open(self) -> None:
        self.open_calls += 1
        if self.fail_on_open:
            raise RuntimeError("open failed")

    def close(self) -> None:
        self.close_calls += 1

    def read_range(self, wu) -> np.ndarray:
        self.read_calls += 1
        return self.read_returns

    def write_range(self, wu, x: np.ndarray) -> None:
        self.write_calls += 1
        self.writes.append((wu, np.asarray(x)))


# ----------------------------
# ThreadLocalTaskProvider tests
# ----------------------------

def test_thread_local_provider_single_thread_reuses_task(core_mod):
    created: list[DummyIOTask] = []

    def factory():
        t = DummyIOTask()
        created.append(t)
        return t

    p = core_mod.ThreadLocalTaskProvider(factory)
    t1 = p.get()
    t2 = p.get()

    assert t1 is t2
    assert len(created) == 1
    assert created[0].open_calls == 1

    p.close_all()
    assert created[0].close_calls == 1


def test_thread_local_provider_creates_one_per_thread(core_mod):
    created: list[DummyIOTask] = []

    def factory():
        t = DummyIOTask()
        created.append(t)
        return t

    p = core_mod.ThreadLocalTaskProvider(factory)

    got: list[DummyIOTask] = []
    lock = threading.Lock()

    def worker():
        t = p.get()
        with lock:
            got.append(t)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert len(created) == 2
    assert got[0] is not got[1]
    assert created[0].open_calls == 1
    assert created[1].open_calls == 1

    p.close_all()
    assert created[0].close_calls == 1
    assert created[1].close_calls == 1


def test_thread_local_provider_close_all_swallows_close_errors(core_mod):
    class BadCloseTask(DummyIOTask):
        def close(self) -> None:
            super().close()
            raise RuntimeError("close failed")

    created: list[BadCloseTask] = []

    def factory():
        t = BadCloseTask()
        created.append(t)
        return t

    p = core_mod.ThreadLocalTaskProvider(factory)
    _ = p.get()

    # should not raise
    p.close_all()
    assert created[0].close_calls == 1


# ----------------------------
# DatasetScaleWorker.process_wu tests
# ----------------------------

def test_worker_process_wu_no_noise_on_replica0(core_mod, config_mod, task_mod):
    cfg = config_mod.Config(
        input_path=Path("in.fbin"),
        output_path=Path("out.fbin"),
        scale=2,
        batch_size=10,
        noise_amplitude=0.5,
        noise_scheme=config_mod.NoiseScheme.ALL,
        normalize=False,
        tolerance=1e-3,
        shard_size=None,
        seed=123,
        json_metadata=None,
        workers=1,
    )

    x_in = np.ones((3, 4), dtype=np.float32)
    io = DummyIOTask(read_returns=x_in)
    provider = SimpleNamespace(get=lambda: io)

    worker = core_mod.DatasetScaleWorker(cfg=cfg, provider=provider)
    wu = task_mod.WorkUnit(batch_id=0, replica_id=0, in_start=0, count=3, out_global_start=0)

    written = worker.process_wu(wu)

    assert written == 3
    assert io.read_calls == 1
    assert io.write_calls == 1
    _, x_written = io.writes[0]
    assert np.allclose(x_written, x_in)


def test_worker_process_wu_applies_noise_and_renorm(core_mod, config_mod, task_mod, monkeypatch):
    cfg = config_mod.Config(
        input_path=Path("in.fbin"),
        output_path=Path("out.fbin"),
        scale=2,
        batch_size=10,
        noise_amplitude=0.5,
        noise_scheme=config_mod.NoiseScheme.ALL,
        normalize=True,
        tolerance=1e-3,
        shard_size=None,
        seed=7,
        json_metadata=None,
        workers=1,
    )

    x_in = np.ones((2, 3), dtype=np.float32)
    io = DummyIOTask(read_returns=x_in)
    provider = SimpleNamespace(get=lambda: io)
    worker = core_mod.DatasetScaleWorker(cfg=cfg, provider=provider)
    wu = task_mod.WorkUnit(batch_id=5, replica_id=1, in_start=0, count=2, out_global_start=0)

    calls = {"make_work_seed": 0, "make_rng": 0, "make_noise_strategy": 0, "apply": 0, "renorm": 0}

    def fake_make_work_seed(base_seed, replica_id, batch_id):
        calls["make_work_seed"] += 1
        assert base_seed == 7
        assert replica_id == 1
        assert batch_id == 5
        return 12345

    def fake_make_rng(seed):
        calls["make_rng"] += 1
        assert seed == 12345
        return object()

    class FakeNoise:
        def apply(self, x, rng):
            calls["apply"] += 1
            assert rng is not None
            return x + np.float32(2.0)

    def fake_make_noise_strategy(scheme, amp):
        calls["make_noise_strategy"] += 1
        assert amp == 0.5
        return FakeNoise()

    def fake_renormalize(x, tolerance):
        calls["renorm"] += 1
        return x * np.float32(0.5)

    monkeypatch.setattr(core_mod, "make_work_seed", fake_make_work_seed)
    monkeypatch.setattr(core_mod, "make_rng", fake_make_rng)
    monkeypatch.setattr(core_mod, "make_noise_strategy", fake_make_noise_strategy)
    monkeypatch.setattr(core_mod, "renormalize", fake_renormalize)

    written = worker.process_wu(wu)

    assert written == 2
    assert calls["make_work_seed"] == 1
    assert calls["make_rng"] == 1
    assert calls["make_noise_strategy"] == 1
    assert calls["apply"] == 1
    assert calls["renorm"] == 1

    _, x_written = io.writes[0]
    assert np.allclose(x_written, np.full((2, 3), 1.5, dtype=np.float32))


def test_worker_process_wu_noise_not_applied_when_amplitude_zero(core_mod, config_mod, task_mod, monkeypatch):
    cfg = config_mod.Config(
        input_path=Path("in.fbin"),
        output_path=Path("out.fbin"),
        scale=2,
        batch_size=10,
        noise_amplitude=0.0,
        noise_scheme=config_mod.NoiseScheme.ALL,
        normalize=False,
        tolerance=1e-3,
        shard_size=None,
        seed=7,
        json_metadata=None,
        workers=1,
    )

    x_in = np.ones((1, 2), dtype=np.float32)
    io = DummyIOTask(read_returns=x_in)
    provider = SimpleNamespace(get=lambda: io)
    worker = core_mod.DatasetScaleWorker(cfg=cfg, provider=provider)
    wu = task_mod.WorkUnit(batch_id=0, replica_id=1, in_start=0, count=1, out_global_start=0)

    monkeypatch.setattr(
        core_mod,
        "make_noise_strategy",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("noise called")),
    )

    worker.process_wu(wu)
    _, x_written = io.writes[0]
    assert np.allclose(x_written, x_in)


def test_worker_process_wu_propagates_read_failure(core_mod, config_mod, task_mod):
    cfg = config_mod.Config(
        input_path=Path("in.fbin"),
        output_path=Path("out.fbin"),
        scale=1,
        batch_size=10,
        noise_amplitude=0.0,
        noise_scheme=config_mod.NoiseScheme.ALL,
        normalize=False,
        tolerance=1e-3,
        shard_size=None,
        seed=None,
        json_metadata=None,
        workers=1,
    )

    class FailRead(DummyIOTask):
        def read_range(self, wu):
            raise RuntimeError("read failed")

    io = FailRead()
    provider = SimpleNamespace(get=lambda: io)
    worker = core_mod.DatasetScaleWorker(cfg=cfg, provider=provider)
    wu = task_mod.WorkUnit(batch_id=0, replica_id=0, in_start=0, count=1, out_global_start=0)

    with pytest.raises(RuntimeError, match="read failed"):
        worker.process_wu(wu)


# ----------------------------
# _execute_mt tests
# ----------------------------

def test_execute_mt_sums_results_and_handles_no_tqdm(core_mod, config_mod, task_mod, monkeypatch):
    cfg = config_mod.Config(
        input_path=Path("in.fbin"),
        output_path=Path("out.fbin"),
        scale=1,
        batch_size=10,
        noise_amplitude=0.0,
        noise_scheme=config_mod.NoiseScheme.ALL,
        normalize=False,
        tolerance=1e-3,
        shard_size=None,
        seed=None,
        json_metadata=None,
        workers=2,
    )

    # Force tqdm import to fail so pbar=None branch is covered.
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "tqdm":
            raise ImportError("no tqdm")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(core_mod, "__builtins__", dict(getattr(core_mod, "__builtins__")))
    core_mod.__builtins__["__import__"] = fake_import  # type: ignore[index]

    class FakeWorker:
        def process_wu(self, wu):
            return int(wu.count)

    work = [
        task_mod.WorkUnit(batch_id=0, replica_id=0, in_start=0, count=3, out_global_start=0),
        task_mod.WorkUnit(batch_id=1, replica_id=0, in_start=3, count=2, out_global_start=3),
    ]

    total = core_mod._execute_mt(cfg, FakeWorker(), work)
    assert total == 5


def test_execute_mt_propagates_worker_exception(core_mod, config_mod, task_mod):
    cfg = config_mod.Config(
        input_path=Path("in.fbin"),
        output_path=Path("out.fbin"),
        scale=1,
        batch_size=10,
        noise_amplitude=0.0,
        noise_scheme=config_mod.NoiseScheme.ALL,
        normalize=False,
        tolerance=1e-3,
        shard_size=None,
        seed=None,
        json_metadata=None,
        workers=2,
    )

    class BadWorker:
        def process_wu(self, wu):
            raise RuntimeError("boom")

    work = [task_mod.WorkUnit(batch_id=0, replica_id=0, in_start=0, count=1, out_global_start=0)]

    with pytest.raises(RuntimeError, match="boom"):
        core_mod._execute_mt(cfg, BadWorker(), work)


# ----------------------------
# write_report tests
# ----------------------------

def test_write_report_noop_when_path_none(core_mod):
    core_mod.write_report(None, {"a": 1})  # should not raise


def test_write_report_writes_json_and_adds_timestamp(core_mod, tmp_path):
    out = tmp_path / "reports" / "r.json"
    core_mod.write_report(out, {"hello": "world"})
    data = out.read_text(encoding="utf-8")
    assert '"hello": "world"' in data
    assert "created_at_unix" in data


# ----------------------------
# run() tests (with heavy mocking to avoid I/O)
# ----------------------------

@dataclass
class _DummyWM:
    cfg: Any
    in_kind: str = "fbin"
    out_kind: str = "fbin"
    dim: int = 4
    base_n: int = 10
    total_n: int = 20

    @staticmethod
    def plan_work(*, base_n: int, batch_size: int, scale: int):
        return []

    def get_task_factory(self):
        return lambda: DummyIOTask(read_returns=np.zeros((0, self.dim), dtype=np.float32))

    def finalize(self):
        return [Path("out_part000.fbin")], [self.total_n]


def test_run_rejects_shard_size_for_fvecs_output(core_mod, config_mod):
    cfg = config_mod.Config(
        input_path=Path("in.fbin"),
        output_path=Path("out.fvecs"),
        scale=1,
        batch_size=10,
        noise_amplitude=0.0,
        noise_scheme=config_mod.NoiseScheme.ALL,
        normalize=False,
        tolerance=1e-3,
        shard_size=123,  # invalid with FVECs output
        seed=None,
        json_metadata=None,
        workers=1,
    )

    # make WorkManager say output is fvecs so the validation triggers
    class WM(_DummyWM):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.out_kind = "fvecs"

    core_mod.WorkManager = WM  # type: ignore[assignment]

    with pytest.raises(ValueError, match="shard_size is not supported for FVECs output"):
        core_mod.run(cfg)


def test_run_calls_close_all_even_when_execute_fails(core_mod, config_mod, monkeypatch):
    cfg = config_mod.Config(
        input_path=Path("in.fbin"),
        output_path=Path("out.fbin"),
        scale=1,
        batch_size=10,
        noise_amplitude=0.0,
        noise_scheme=config_mod.NoiseScheme.ALL,
        normalize=False,
        tolerance=1e-3,
        shard_size=None,
        seed=None,
        json_metadata=None,
        workers=1,
    )

    monkeypatch.setattr(core_mod, "WorkManager", _DummyWM)

    closed = {"called": 0}

    class FakeProvider:
        def __init__(self, factory):
            self.factory = factory

        def close_all(self):
            closed["called"] += 1

        def get(self):
            raise AssertionError("not expected")

    monkeypatch.setattr(core_mod, "ThreadLocalTaskProvider", FakeProvider)

    def boom_execute(*args, **kwargs):
        raise RuntimeError("execute failed")

    monkeypatch.setattr(core_mod, "_execute_mt", boom_execute)

    with pytest.raises(RuntimeError, match="execute failed"):
        core_mod.run(cfg)

    assert closed["called"] == 1


def test_run_success_summary_fields_and_report_called(core_mod, config_mod, monkeypatch):
    cfg = config_mod.Config(
        input_path=Path("in.fbin"),
        output_path=Path("out.fbin"),
        scale=2,
        batch_size=4,
        noise_amplitude=0.1,
        noise_scheme=config_mod.NoiseScheme.ALL,
        normalize=True,
        tolerance=1e-3,
        shard_size=None,
        seed=11,
        json_metadata=Path("meta.json"),
        workers=3,
    )

    monkeypatch.setattr(core_mod, "WorkManager", _DummyWM)
    monkeypatch.setattr(core_mod, "_execute_mt", lambda cfg, worker, work: 123)

    def fin(self):
        return [Path("out_part000.fbin"), Path("out_part001.fbin")], [10, 10]

    monkeypatch.setattr(_DummyWM, "finalize", fin, raising=True)

    reported: dict[str, Any] = {}

    def fake_write_report(path, payload):
        reported["path"] = path
        reported["payload"] = payload

    monkeypatch.setattr(core_mod, "write_report", fake_write_report)

    summary = core_mod.run(cfg)

    assert summary["input_format"] == "fbin"
    assert summary["output_format"] == "fbin"
    assert summary["total_written"] == 123
    assert summary["dim"] == 4
    assert summary["base_n"] == 10
    assert summary["scale"] == 2
    assert summary["workers"] == 3
    assert isinstance(summary["timings_s"]["total"], float)
    assert summary["shards"] == [
        {"path": "out_part000.fbin", "count": 10},
        {"path": "out_part001.fbin", "count": 10},
    ]

    assert reported["path"] == cfg.json_metadata
    assert reported["payload"]["total_written"] == 123