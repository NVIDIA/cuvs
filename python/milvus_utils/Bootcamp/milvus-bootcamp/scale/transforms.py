from __future__ import annotations

from dataclasses import dataclass
from .config import NoiseScheme
from typing import Protocol

import numpy as np


def renormalize(x: np.ndarray, tolerance: float) -> np.ndarray:
    """Return a (mostly) unit-normalized version of `x` within a tolerance.

    Computes L2 norms row-wise and renormalizes only rows whose norm differs from 1.0
    by more than `tolerance`.

    Parameters
    ----------
    x:
        Input vectors of shape (n, d). Expected float32/float64, but any float dtype
        works as long as `np.linalg.norm` supports it.
    tolerance:
        Rows with |norm - 1.0| <= tolerance are left unchanged.

    Returns
    -------
    np.ndarray
        If no rows need renormalization, returns `x` as is.
        Otherwise, renormalizes the selected rows **in-place on the provided `x`**
        and returns `x` (same object).

    """
    norms = np.linalg.norm(x, axis=1)
    mask = np.abs(norms - 1.0) > tolerance

    if not np.any(mask):
        return x
    safe_norms = np.maximum(norms[mask], 1e-12)
    x[mask, :] /= safe_norms[:, None]

    return x

class NoiseStrategy(Protocol):
    """Strategy interface for applying noise to a batch of vectors."""
    def apply(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray: ...


@dataclass(frozen=True)
class AllDimsSignNoise:
    """
    Apply ±noise_amplitude to all dimensions.
    """
    noise_amplitude: float

    def apply(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if x.size == 0 or self.noise_amplitude == 0.0:
            return x
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=x.shape, replace=True)
        return (x + signs * np.float32(self.noise_amplitude)).astype(x.dtype, copy=False)


@dataclass(frozen=True)
class PartialDimsSignNoise:
    """
    For each vector, perturb exactly floor(d/2) dimensions by ±noise_amplitude.
    The perturbed dimensions are chosen uniformly at random per vector.
    """
    noise_amplitude: float

    def apply(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if x.size == 0 or self.noise_amplitude == 0.0:
            return x
        n, d = x.shape
        if d < 2:
            raise ValueError("PARTIAL noise scheme requires dim >= 2")

        k = d // 2
        out = x.copy()

        # Generate per-row indices of perturbed dimensions.
        # We generate random scores and take argpartition to get k smallest indices.
        scores = rng.random((n, d), dtype=np.float64)
        idx = np.argpartition(scores, kth=k - 1, axis=1)[:, :k]  # shape (n, k)

        # Random signs for each perturbed element.
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(n, k), replace=True)

        row_ids = np.arange(n)[:, None]
        out[row_ids, idx] = out[row_ids, idx] + signs * np.float32(self.noise_amplitude)
        return out.astype(x.dtype, copy=False)


def make_noise_strategy(scheme: NoiseScheme | str, noise_amplitude: float) -> NoiseStrategy:
    """Construct a noise strategy implementation from a scheme selector.

    Parameters
    ----------
    scheme:
        Either a `NoiseScheme` enum value or a string (case-insensitive) that maps
        to a `NoiseScheme` member name (e.g. "ALL", "PARTIAL").
    noise_amplitude:
        Magnitude of the additive ± noise applied by the selected strategy.

    Returns
    -------
    NoiseStrategy
        A concrete strategy instance:
        - `AllDimsSignNoise` for NoiseScheme.ALL
        - `PartialDimsSignNoise` for NoiseScheme.PARTIAL

    Raises
    ------
    ValueError
        If `scheme` cannot be converted to `NoiseScheme`, or if the scheme is not
        supported.

    Notes
    -----
    The returned strategy’s `apply()` is expected to be deterministic given a
    deterministic `rng` state.
    """
    if isinstance(scheme, NoiseScheme):
        scheme_enum = scheme
    else:
        scheme_enum = NoiseScheme(str(scheme).upper())

    if scheme_enum == NoiseScheme.ALL:
        return AllDimsSignNoise(noise_amplitude=float(noise_amplitude))
    if scheme_enum == NoiseScheme.PARTIAL:
        return PartialDimsSignNoise(noise_amplitude=float(noise_amplitude))
    raise ValueError(f"Unsupported noise scheme: {scheme_enum}")
