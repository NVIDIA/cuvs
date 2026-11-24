from __future__ import annotations

from pathlib import Path

from .config import Config
from .io import (
    FbinReader,
    FbinWriter,
    FvecsReader,
    FvecsWriter,
    ReadWriteTask,
    WorkUnit,
    backpatch_fbin_headers,
    compute_fbin_shard_size,
    precreate_fbin_shards,
    precreate_fvecs_file,
    read_fbin_header,
    read_fvecs_header,
    shard_paths_for_total_n,
)


def _format_kind(path: Path) -> str:
    """Return normalized format kind name for a dataset path."""
    suf = Path(path).suffix.lower()
    if suf == ".fbin":
        return "fbin"
    if suf in (".fvecs", ".fvec"):
        return "fvecs"
    raise ValueError(f"Unsupported dataset suffix: {suf}")


class WorkManager:
    def __init__(self, cfg: Config):
        """Manage input/output metadata and per-thread I/O tasks."""
        self.conf = cfg

        self.in_kind = _format_kind(cfg.input_path)
        self.out_kind = _format_kind(cfg.output_path)

        # Input metadata
        if self.in_kind == "fbin":
            self.dim, self.base_n = read_fbin_header(cfg.input_path)
        else:
            self.dim, self.base_n = read_fvecs_header(cfg.input_path)

        self.total_n = int(self.base_n) * int(cfg.scale)

        # Output precreate
        self._effective_shard_size: int | None
        if self.out_kind == "fbin":
            self._effective_shard_size = int(compute_fbin_shard_size(cfg))
            precreate_fbin_shards(
                cfg.output_path,
                dim=int(self.dim),
                shard_size=int(self._effective_shard_size),
                total_n=int(self.total_n),
            )
        else:
            self._effective_shard_size = None
            precreate_fvecs_file(cfg.output_path, dim=int(self.dim), total_n=int(self.total_n))

        self._task_factory = self._make_task_factory()

        # finalize() caching (idempotent)
        self._finalized: bool = False
        self._finalize_paths: list[Path] | None = None
        self._finalize_counts: list[int] | None = None

    def _make_task_factory(self):
        """Build a per-thread task factory (reader+writer) for the configured formats.

        Returns
        -------
        Callable[[], ReadWriteTask]
            A factory that creates a new per-thread reader and writer instance.
        """
        cfg = self.conf
        dim = int(self.dim)

        def make_reader():
            if self.in_kind == "fbin":
                return FbinReader(cfg.input_path, dim=dim)
            return FvecsReader(cfg.input_path, dim=dim)

        def make_writer():
            if self.out_kind == "fbin":
                assert self._effective_shard_size is not None
                return FbinWriter(cfg.output_path, dim=dim, shard_size=int(self._effective_shard_size))
            return FvecsWriter(cfg.output_path, dim=dim)

        return lambda: ReadWriteTask(reader=make_reader(), writer=make_writer())

    def get_task_factory(self):
        """Return the cached per-thread task factory created at initialization."""
        return self._task_factory

    def finalize(self):
        """Finalize output after all work units have completed (idempotent)."""
        if self._finalized:
            return list(self._finalize_paths or []), list(self._finalize_counts or [])

        if self.out_kind == "fbin":
            assert self._effective_shard_size is not None
            shard_size = int(self._effective_shard_size)
            counts = backpatch_fbin_headers(
                Path(self.conf.output_path),
                dim=int(self.dim),
                shard_size=shard_size,
                total_n=int(self.total_n),
            )
            paths = shard_paths_for_total_n(
                Path(self.conf.output_path),
                shard_size=shard_size,
                total_n=int(self.total_n),
            )
        else:
            paths = [Path(self.conf.output_path)]
            counts = [int(self.total_n)]

        self._finalized = True
        self._finalize_paths = list(paths)
        self._finalize_counts = list(counts)
        return paths, counts

    @staticmethod
    def plan_work(*, base_n: int, batch_size: int, scale: int) -> list[WorkUnit]:
        """Plan batch-major work units with contiguous output placement."""
        base_n = int(base_n)
        batch_size = int(batch_size)
        scale = int(scale)

        B = (base_n + batch_size - 1) // batch_size
        work: list[WorkUnit] = []
        batch_base = 0

        for batch_id in range(B):
            in_start = batch_id * batch_size
            count = min(batch_size, base_n - in_start)

            for replica_id in range(scale):
                out_global_start = batch_base + replica_id * count
                work.append(
                    WorkUnit(
                        batch_id=batch_id,
                        replica_id=replica_id,
                        in_start=in_start,
                        count=count,
                        out_global_start=out_global_start,
                    )
                )
            batch_base += count * scale

        return work
