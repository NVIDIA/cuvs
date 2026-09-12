from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

from .config import Config
from .io.task import IOTask, WorkUnit
from .rng import make_rng, make_work_seed
from .transforms import make_noise_strategy, renormalize
from .work_manager import WorkManager


# ============================================================
# MT-only execution: thread-local tasks
# ============================================================


class ThreadLocalTaskProvider:
    """
    Thread-local IOTask factory/registry (MT-only).

    Contract
    --------
    - Each worker thread lazily creates exactly one `IOTask` instance.
    - Each task owns its own file descriptors (FDs), so it can do I/O without locks.
    - The provider tracks all created tasks so the caller can clean them up via
      `close_all()` (best-effort).

    Notes
    -----
    `threading.local()` ensures each thread gets its own `proc` attribute.
    `_all` is only for cleanup; it is not used for I/O synchronization.
    """
    def __init__(self, factory):
        """
        Parameters
        ----------
        factory:
            Callable that returns a new, unopened `IOTask` instance.
            The provider will call `open()` on it exactly once per thread.
        """
        self._factory = factory
        self._local = threading.local()
        self._all: list[IOTask] = []
        self._all_lock = threading.Lock()

    def get(self) -> IOTask:
        """
        Get the current thread's `IOTask`, creating it on first use.

        Returns
        -------
        IOTask
            A thread-owned task instance that is guaranteed to be `open()`.

        Notes
        -----
        - The first call in a thread creates a task via `factory()`, calls `open()`,
          stores it in thread-local storage, and registers it for later cleanup.
        - Subsequent calls in the same thread reuse the same instance.
        """
        proc = getattr(self._local, "proc", None)
        if proc is not None:
            return proc

        proc = self._factory()
        proc.open()
        self._local.proc = proc

        with self._all_lock:
            self._all.append(proc)

        return proc

    def close_all(self) -> None:
        """
        Best-effort close of every `IOTask` created by this provider.

        Notes
        -----
        - Intended to be called once the ThreadPoolExecutor is done.
        - Exceptions from individual task closures are swallowed to maximize cleanup.
        """
        with self._all_lock:
            procs = list(self._all)
            self._all.clear()

        for p in procs:
            try:
                p.close()
            except Exception:
                pass


@dataclass
class DatasetScaleWorker:
    """Work-unit runner: read → transform → write (threaded)."""

    cfg: Config
    provider: ThreadLocalTaskProvider

    def process_wu(self, wu: WorkUnit) -> int:
        """
        Execute a single WorkUnit.

        Workflow
        --------
        1) Read the input slice described by `wu` (format-specific).
        2) Optionally apply noise (only for replica_id > 0, and only if amplitude > 0).
        3) Optionally renormalize vectors to unit length (within tolerance).
        4) Write the output slice described by `wu` (format-specific).

        Parameters
        ----------
        wu:
            The planned unit of work (batch-major). Contains input start/count and
            output global start for this replica/batch.

        Returns
        -------
        int
            Number of vectors written (equals `wu.count`).

        Notes
        -----
        - Noise seeding uses (cfg.seed, wu.replica_id, wu.batch_id) via `make_work_seed`.
        - I/O safety relies on planning: work units must not overlap in output offsets.
        """
        proc = self.provider.get()
        x = proc.read_range(wu)

        if wu.replica_id > 0 and self.cfg.noise_amplitude > 0.0:
            seed_wu = make_work_seed(self.cfg.seed, wu.replica_id, wu.batch_id)
            rng = make_rng(seed_wu)
            noise = make_noise_strategy(self.cfg.noise_scheme, float(self.cfg.noise_amplitude))
            x = noise.apply(x, rng)

        if self.cfg.normalize:
            x = renormalize(x, tolerance=float(self.cfg.tolerance))

        proc.write_range(wu, x)
        return int(wu.count)


def _execute_mt(cfg: Config, worker: DatasetScaleWorker, work: list[WorkUnit]) -> int:
    """
    Execute all work units using a ThreadPoolExecutor.

    Parameters
    ----------
    cfg:
        Run configuration (notably `workers`).
    worker:
        Work-unit executor (read/transform/write).
    work:
        Planned work units.

    Returns
    -------
    int
        Total number of vectors written across all completed work units.
    """
    try:
        from tqdm import tqdm  # type: ignore
        pbar = tqdm(total=len(work), desc="Work units", unit="wu")
    except Exception:
        pbar = None

    total_written = 0
    with ThreadPoolExecutor(max_workers=int(cfg.workers)) as ex:
        futs = [ex.submit(worker.process_wu, wu) for wu in work]
        for fut in as_completed(futs):
            total_written += int(fut.result())
            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    return int(total_written)


# ============================================================
# Reporting
# ============================================================

def write_report(path: Optional[Path], payload: dict[str, Any]) -> None:
    """
    Write a JSON report to disk (if enabled).

    Parameters
    ----------
    path:
        Output path for the JSON metadata. If None, this is a no-op.
    payload:
        JSON-serializable dictionary to write. A `created_at_unix` field is added.
    """
    if path is None:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["created_at_unix"] = time.time()
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


# ============================================================
# Main entry
# ============================================================

def run(cfg: Config) -> dict:
    """Run the dataset scaler (multi-threaded, thread-local I/O).

    High-level flow:
    1) Validate config and read input metadata (dim, base_n).
    2) Precreate output (FBIN shards or a full FVECs file).
    3) Plan batch-major WorkUnits (replicas contiguous per input batch).
    4) Execute work in a ThreadPool: read → optional noise → optional renorm → write.
    5) Finalize output (FBIN header backpatching) and emit an optional JSON report.

    Returns
    -------
    dict
        A JSON-serializable summary including formats, sizes, shard layout, timings,
        and the effective config.
    """
    cfg.validate()

    start_wall = time.perf_counter()
    work_manager = WorkManager(cfg=cfg)
    in_kind = work_manager.in_kind
    out_kind = work_manager.out_kind

    # Validation: shard_size is only meaningful for FBIN output.
    if out_kind == "fvecs" and cfg.shard_size is not None:
        raise ValueError("shard_size is not supported for FVECs output (use --shard-size only with .fbin output)")

    work = WorkManager.plan_work(base_n=work_manager.base_n, batch_size=cfg.batch_size, scale=cfg.scale)
    provider = ThreadLocalTaskProvider(work_manager.get_task_factory())
    worker = DatasetScaleWorker(cfg=cfg, provider=provider)

    try:
        total_written = _execute_mt(cfg, worker, work)
    finally:
        provider.close_all()

    shard_paths, shard_counts = work_manager.finalize()

    elapsed = time.perf_counter() - start_wall

    summary = {
        "input_path": str(cfg.input_path),
        "output_path": str(cfg.output_path),
        # Backward-compat: "format" used to mean the output format.
        "format": out_kind,
        "input_format": in_kind,
        "output_format": out_kind,
        "dim": int(work_manager.dim),
        "base_n": int(work_manager.base_n),
        "scale": int(cfg.scale),
        "normalize": bool(cfg.normalize),
        "tolerance": float(cfg.tolerance),
        "noise_scheme": str(cfg.noise_scheme.value),
        "noise_amplitude": float(cfg.noise_amplitude),
        "seed": cfg.seed,
        "workers": int(cfg.workers),
        "total_written": int(total_written),
        "shards": [{"path": str(p), "count": int(c)} for p, c in zip(shard_paths, shard_counts)],
        "timings_s": {"total": float(elapsed)},
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
    }

    write_report(cfg.json_metadata, summary)
    return summary
