from __future__ import annotations

from typing import Optional
import numpy as np


def make_rng(seed: Optional[int]) -> np.random.Generator:
    """
    Create a numpy RNG.
    If seed is provided, results become deterministic across runs.
    """
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(int(seed))

def make_work_seed(base_seed: Optional[int], replica_id: int, batch_id: int) -> Optional[int]:
    """
    Derive a deterministic per-work-unit seed from a base seed and work identifiers.

    Parameters
    ----------
    base_seed:
        The global/user-provided seed. If None, returns None (meaning "unseeded").
    replica_id:
        Which replica is being produced for the given batch.
    batch_id:
        Which batch within the input this work unit corresponds to.

    Returns
    -------
    Optional[int]
        A 64-bit-masked deterministic integer seed unique for (replica_id, batch_id),
        or None if `base_seed` is None.

    Notes
    -----
    - This function is intentionally pure and stable: same inputs => same output.
    - The mixing uses fixed odd constants and XOR to avoid collisions in common cases.
    - This seed is used for noise generation so output is reproducible for a given
      (seed, batch_size, scale, ...).
    """
    if base_seed is None:
        return None
    x = int(base_seed) & 0xFFFFFFFFFFFFFFFF
    x ^= (int(replica_id) * 0x9E3779B185EBCA87) & 0xFFFFFFFFFFFFFFFF
    x ^= (int(batch_id) * 0xC2B2AE3D27D4EB4F) & 0xFFFFFFFFFFFFFFFF
    return x
