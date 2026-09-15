import pytest
from pathlib import Path

from scale.config import Config, NoiseScheme
from scale.work_manager import WorkManager, _format_kind


def _cfg(tmp_path: Path, *, in_name: str, out_name: str, scale: int = 3, shard_size=None) -> Config:
    in_path = tmp_path / in_name
    out_path = tmp_path / out_name
    in_path.write_bytes(b"")  # file exists; headers are monkeypatched anyway

    return Config(
        input_path=in_path,
        output_path=out_path,
        scale=scale,
        batch_size=4,
        noise_amplitude=0.02,
        noise_scheme=NoiseScheme.PARTIAL,
        normalize=False,
        tolerance=1e-3,
        shard_size=shard_size,
        seed=123,
        json_metadata=None,
        workers=1,
    )


# -----------------------------
# _format_kind
# -----------------------------

def test_format_kind_fbin_and_fvecs():
    assert _format_kind(Path("a.fbin")) == "fbin"
    assert _format_kind(Path("a.fvecs")) == "fvecs"
    assert _format_kind(Path("a.fvec")) == "fvecs"


def test_format_kind_unsupported_suffix_raises():
    with pytest.raises(ValueError, match="Unsupported dataset suffix"):
        _format_kind(Path("a.npy"))


# -----------------------------
# WorkManager.__init__
# -----------------------------

def test_init_fbin_to_fbin_precreates_shards_and_sets_metadata(tmp_path, monkeypatch):
    calls = {}

    def fake_read_fbin_header(path):
        calls["read_fbin_header"] = Path(path)
        return (7, 10)  # (dim, base_n)

    def fake_compute_fbin_shard_size(cfg):
        calls["compute_fbin_shard_size"] = True
        return 6

    def fake_precreate_fbin_shards(output_path, dim, shard_size, total_n):
        calls["precreate_fbin_shards"] = {
            "output_path": Path(output_path),
            "dim": dim,
            "shard_size": shard_size,
            "total_n": total_n,
        }

    # Patch the names as imported inside work_manager.py
    import scale.work_manager as wm

    monkeypatch.setattr(wm, "read_fbin_header", fake_read_fbin_header)
    monkeypatch.setattr(wm, "compute_fbin_shard_size", fake_compute_fbin_shard_size)
    monkeypatch.setattr(wm, "precreate_fbin_shards", fake_precreate_fbin_shards)

    cfg = _cfg(tmp_path, in_name="in.fbin", out_name="out.fbin", scale=3)
    manager = WorkManager(cfg)

    assert manager.in_kind == "fbin"
    assert manager.out_kind == "fbin"
    assert manager.dim == 7
    assert manager.base_n == 10
    assert manager.total_n == 30
    assert manager._effective_shard_size == 6

    assert calls["read_fbin_header"] == cfg.input_path
    assert calls["compute_fbin_shard_size"] is True
    assert calls["precreate_fbin_shards"] == {
        "output_path": cfg.output_path,
        "dim": 7,
        "shard_size": 6,
        "total_n": 30,
    }


def test_init_fvecs_to_fvecs_precreates_file_and_no_shard_size(tmp_path, monkeypatch):
    calls = {}

    def fake_read_fvecs_header(path):
        calls["read_fvecs_header"] = Path(path)
        return (5, 11)

    def fake_precreate_fvecs_file(output_path, dim, total_n):
        calls["precreate_fvecs_file"] = {
            "output_path": Path(output_path),
            "dim": dim,
            "total_n": total_n,
        }

    import scale.work_manager as wm

    monkeypatch.setattr(wm, "read_fvecs_header", fake_read_fvecs_header)
    monkeypatch.setattr(wm, "precreate_fvecs_file", fake_precreate_fvecs_file)

    cfg = _cfg(tmp_path, in_name="in.fvecs", out_name="out.fvecs", scale=2)
    manager = WorkManager(cfg)

    assert manager.in_kind == "fvecs"
    assert manager.out_kind == "fvecs"
    assert manager.dim == 5
    assert manager.base_n == 11
    assert manager.total_n == 22
    assert manager._effective_shard_size is None

    assert calls["read_fvecs_header"] == cfg.input_path
    assert calls["precreate_fvecs_file"] == {
        "output_path": cfg.output_path,
        "dim": 5,
        "total_n": 22,
    }


def test_init_unsupported_input_suffix_fails(tmp_path):
    cfg = _cfg(tmp_path, in_name="in.npy", out_name="out.fbin", scale=2)
    with pytest.raises(ValueError, match="Unsupported dataset suffix"):
        WorkManager(cfg)


def test_init_unsupported_output_suffix_fails(tmp_path):
    cfg = _cfg(tmp_path, in_name="in.fbin", out_name="out.npy", scale=2)
    with pytest.raises(ValueError, match="Unsupported dataset suffix"):
        WorkManager(cfg)


# -----------------------------
# get_task_factory / _make_task_factory
# -----------------------------

def test_task_factory_builds_read_write_task_with_expected_types(tmp_path, monkeypatch):
    """
    Verifies the task factory returns ReadWriteTask(reader=..., writer=...)
    with reader/writer chosen by in_kind/out_kind, and writer receives shard_size for FBIN.
    """
    import scale.work_manager as wm

    # Header + shard sizing
    monkeypatch.setattr(wm, "read_fbin_header", lambda p: (3, 8))
    monkeypatch.setattr(wm, "compute_fbin_shard_size", lambda cfg: 10)
    monkeypatch.setattr(wm, "precreate_fbin_shards", lambda *a, **k: None)

    # Replace concrete IO classes with tiny fakes so we can assert constructor args.
    created = {}

    class FakeReader:
        def __init__(self, path, *, dim):
            created["reader"] = {"path": Path(path), "dim": dim}

    class FakeWriter:
        def __init__(self, path, *, dim, shard_size):
            created["writer"] = {"path": Path(path), "dim": dim, "shard_size": shard_size}

    monkeypatch.setattr(wm, "FbinReader", FakeReader)
    monkeypatch.setattr(wm, "FbinWriter", FakeWriter)

    cfg = _cfg(tmp_path, in_name="in.fbin", out_name="out.fbin", scale=2)
    manager = WorkManager(cfg)

    factory = manager.get_task_factory()
    task = factory()

    # ReadWriteTask surface: attributes must exist
    assert hasattr(task, "reader")
    assert hasattr(task, "writer")

    assert created["reader"] == {"path": cfg.input_path, "dim": 3}
    assert created["writer"] == {"path": cfg.output_path, "dim": 3, "shard_size": 10}


# -----------------------------
# finalize (idempotent + both formats)
# -----------------------------

def test_finalize_fbin_calls_backpatch_and_returns_shard_paths_then_idempotent(tmp_path, monkeypatch):
    import scale.work_manager as wm

    # init path
    monkeypatch.setattr(wm, "read_fbin_header", lambda p: (4, 5))
    monkeypatch.setattr(wm, "compute_fbin_shard_size", lambda cfg: 3)
    monkeypatch.setattr(wm, "precreate_fbin_shards", lambda *a, **k: None)

    calls = {"backpatch": 0, "paths": 0}

    def fake_backpatch(output_path, *, dim, shard_size, total_n):
        calls["backpatch"] += 1
        assert Path(output_path) == Path(tmp_path / "out.fbin")
        assert dim == 4
        assert shard_size == 3
        assert total_n == 10  # base_n=5 * scale=2
        return [3, 3, 3, 1]

    def fake_paths(output_path, *, shard_size, total_n):
        calls["paths"] += 1
        assert shard_size == 3
        assert total_n == 10
        return [Path(f"p{i}") for i in range(4)]

    monkeypatch.setattr(wm, "backpatch_fbin_headers", fake_backpatch)
    monkeypatch.setattr(wm, "shard_paths_for_total_n", fake_paths)

    cfg = _cfg(tmp_path, in_name="in.fbin", out_name="out.fbin", scale=2)
    manager = WorkManager(cfg)

    p1, c1 = manager.finalize()
    assert [Path(p) for p in p1] == [Path("p0"), Path("p1"), Path("p2"), Path("p3")]
    assert list(c1) == [3, 3, 3, 1]
    assert calls["backpatch"] == 1
    assert calls["paths"] == 1

    # idempotent: should not call helpers again
    p2, c2 = manager.finalize()
    assert [Path(p) for p in p2] == [Path("p0"), Path("p1"), Path("p2"), Path("p3")]
    assert list(c2) == [3, 3, 3, 1]
    assert calls["backpatch"] == 1
    assert calls["paths"] == 1


def test_finalize_fvecs_returns_single_path_and_total_count_then_idempotent(tmp_path, monkeypatch):
    import scale.work_manager as wm

    monkeypatch.setattr(wm, "read_fvecs_header", lambda p: (2, 9))
    monkeypatch.setattr(wm, "precreate_fvecs_file", lambda *a, **k: None)

    cfg = _cfg(tmp_path, in_name="in.fvecs", out_name="out.fvecs", scale=4)
    manager = WorkManager(cfg)

    p1, c1 = manager.finalize()
    assert p1 == [Path(cfg.output_path)]
    assert c1 == [36]

    p2, c2 = manager.finalize()
    assert p2 == [Path(cfg.output_path)]
    assert c2 == [36]


# -----------------------------
# plan_work
# -----------------------------

def test_plan_work_shapes_and_contiguous_output():
    # base_n=10, batch_size=4 => batches: [4,4,2] => 3 batches, scale=3 => 9 work units
    work = WorkManager.plan_work(base_n=10, batch_size=4, scale=3)
    assert len(work) == 9

    # Batch 0: count=4, out starts: 0,4,8
    b0 = [w for w in work if w.batch_id == 0]
    assert [w.replica_id for w in b0] == [0, 1, 2]
    assert [w.in_start for w in b0] == [0, 0, 0]
    assert [w.count for w in b0] == [4, 4, 4]
    assert [w.out_global_start for w in b0] == [0, 4, 8]

    # Batch 1: count=4, batch_base advanced by 4*3=12, out starts: 12,16,20
    b1 = [w for w in work if w.batch_id == 1]
    assert [w.in_start for w in b1] == [4, 4, 4]
    assert [w.out_global_start for w in b1] == [12, 16, 20]

    # Batch 2: count=2, batch_base advanced by 4*3=12 again => 24, out starts: 24,26,28
    b2 = [w for w in work if w.batch_id == 2]
    assert [w.in_start for w in b2] == [8, 8, 8]
    assert [w.count for w in b2] == [2, 2, 2]
    assert [w.out_global_start for w in b2] == [24, 26, 28]


def test_plan_work_rejects_bad_inputs_via_python_errors():
    # plan_work currently does not validate explicitly; it will fail naturally (division by zero).
    with pytest.raises(ZeroDivisionError):
        WorkManager.plan_work(base_n=10, batch_size=0, scale=2)
