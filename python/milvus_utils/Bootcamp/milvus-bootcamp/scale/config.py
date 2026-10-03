
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


class NoiseScheme(str, Enum):
    ALL = "ALL"
    PARTIAL = "PARTIAL"


@dataclass(frozen=True)
class Config:
    input_path: Path
    output_path: Path
    scale: int
    batch_size: int
    noise_amplitude: float
    noise_scheme: NoiseScheme
    normalize: bool
    tolerance: float
    shard_size: Optional[int]
    seed: Optional[int]
    json_metadata: Optional[Path]
    # Thread count for MT-only engine.
    workers: int = 1

    def validate(self) -> None:
        """Validate config values and raise ValueError on invalid combinations/limits."""
        if self.scale < 1:
            raise ValueError("scale must be >= 1")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.noise_amplitude < 0:
            raise ValueError("noise_amplitude must be >= 0")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be > 0")
        if self.workers <= 0:
            raise ValueError("workers must be >= 1")
        if self.shard_size is not None and int(self.shard_size) < 1:
            raise ValueError("shard_size must be >= 1")
