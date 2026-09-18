# task.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class WorkUnit:
    """
    A single planned piece of scaling work.

    Each WorkUnit describes:

    - Which input slice to read:
        - `in_start` – starting vector index in the base dataset
        - `count` – number of vectors to read

    - Which replica to produce:
        - `replica_id` – replica number for this batch
        - `batch_id` – batch number in the input dataset

    - Where to write the result:
        - `out_global_start` – starting index in the final output dataset

    Work units are generated in batch-major order:
    for each input batch, all replicas are placed contiguously
    in the output.

    The total number of WorkUnit instances equals:
        number_of_batches * scale

    WorkUnit intentionally does NOT contain:
        - input file path
        - output file path
        - dataset dimension
        - total number of vectors
        - scale factor

    There is no reason to duplicate such global properties in every
    WorkUnit instance. They belong to:
        - `Config` (paths, scale, global parameters)
        - `WorkManager` (dim, base_n, total_n)

    The entity responsible for mapping the slice described by WorkUnit
    into the correct position in the output file is the format-specific
    I/O implementation (e.g., FBIN/FVECs reader+writer).
    """

    batch_id: int
    replica_id: int
    in_start: int
    count: int
    out_global_start: int


@runtime_checkable
class IOTask(Protocol):
    """Format-specific I/O interface used by the threaded execution engine."""

    def open(self) -> None:
        """Open underlying resources (file descriptors, mmaps, etc.). Idempotent."""

    def close(self) -> None:
        """Close underlying resources. Safe to call multiple times."""

    def read_range(self, wu: WorkUnit) -> np.ndarray:
        """Read a contiguous range of vectors described by `wu`."""

    def write_range(self, wu: WorkUnit, x: np.ndarray) -> None:
        """Write vectors `x` at the output location described by `wu`."""


@runtime_checkable
class VectorReader(Protocol):
    """Read interface for one vector file format (per-thread instance)."""

    def open(self) -> None:
        """Open underlying resources for reading. Idempotent."""

    def close(self) -> None:
        """Close underlying resources. Safe to call multiple times."""

    def read_range(self, wu: WorkUnit) -> np.ndarray:
        """Read a contiguous range of vectors described by `wu`."""


@runtime_checkable
class VectorWriter(Protocol):
    """Write interface for one vector file format (per-thread instance)."""

    def open(self) -> None:
        """Open underlying resources for writing. Idempotent."""

    def close(self) -> None:
        """Close underlying resources. Safe to call multiple times."""

    def write_range(self, wu: WorkUnit, x: np.ndarray) -> None:
        """Write vectors `x` at the output location described by `wu`."""


@dataclass
class ReadWriteTask:
    """Glue task: compose a reader and a writer into the existing `IOTask` surface."""

    reader: VectorReader
    writer: VectorWriter

    def open(self) -> None:
        self.reader.open()
        self.writer.open()

    def close(self) -> None:
        # Best-effort close in both directions.
        try:
            self.reader.close()
        finally:
            self.writer.close()

    def read_range(self, wu: WorkUnit) -> np.ndarray:
        return self.reader.read_range(wu)

    def write_range(self, wu: WorkUnit, x: np.ndarray) -> None:
        self.writer.write_range(wu, x)
