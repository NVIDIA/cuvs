import numpy as np
import pytest

from scale.config import NoiseScheme
from scale.transforms import (
    AllDimsSignNoise,
    PartialDimsSignNoise,
    make_noise_strategy,
    renormalize,
)


def test_renormalize_no_rows_need_change_returns_same_object():
    x = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    x_id = id(x)

    out = renormalize(x, tolerance=1e-6)

    assert out is x
    assert id(out) == x_id
    np.testing.assert_allclose(out, np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))


def test_renormalize_renormalizes_only_masked_rows_in_place():
    # Row0 norm=2 -> should be renormalized to unit.
    # Row1 norm=1 -> should remain unchanged.
    x = np.array([[2.0, 0.0], [0.6, 0.8]], dtype=np.float32)
    x_id = id(x)

    out = renormalize(x, tolerance=1e-6)

    assert out is x
    assert id(out) == x_id
    np.testing.assert_allclose(out[0], np.array([1.0, 0.0], dtype=np.float32), rtol=0, atol=1e-6)
    np.testing.assert_allclose(out[1], np.array([0.6, 0.8], dtype=np.float32), rtol=0, atol=1e-6)


def test_renormalize_handles_zero_norm_without_infs_or_nans():
    x = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    out = renormalize(x, tolerance=1e-6)

    assert np.isfinite(out).all()
    np.testing.assert_allclose(out[0], np.array([0.0, 0.0], dtype=np.float32), rtol=0, atol=0)
    np.testing.assert_allclose(out[1], np.array([0.6, 0.8], dtype=np.float32), rtol=0, atol=1e-6)


def test_all_dims_noise_noop_on_empty_input_returns_same_object():
    rng = np.random.default_rng(0)
    x = np.empty((0, 4), dtype=np.float32)

    out = AllDimsSignNoise(noise_amplitude=0.02).apply(x, rng)

    assert out is x
    assert out.shape == (0, 4)


def test_all_dims_noise_noop_on_zero_amplitude_returns_same_object():
    rng = np.random.default_rng(0)
    x = np.zeros((3, 5), dtype=np.float32)
    x_id = id(x)

    out = AllDimsSignNoise(noise_amplitude=0.0).apply(x, rng)

    assert out is x
    assert id(out) == x_id
    np.testing.assert_allclose(out, 0.0)


def test_all_dims_noise_applies_abs_amplitude_everywhere_and_preserves_dtype():
    rng = np.random.default_rng(123)
    amp = 0.25
    x = np.zeros((4, 6), dtype=np.float32)

    out = AllDimsSignNoise(noise_amplitude=amp).apply(x, rng)

    assert out.shape == x.shape
    assert out.dtype == x.dtype
    np.testing.assert_allclose(np.abs(out), np.float32(amp))


def test_partial_dims_noise_noop_on_empty_input_returns_same_object():
    rng = np.random.default_rng(0)
    x = np.empty((0, 4), dtype=np.float32)

    out = PartialDimsSignNoise(noise_amplitude=0.02).apply(x, rng)

    assert out is x
    assert out.shape == (0, 4)


def test_partial_dims_noise_noop_on_zero_amplitude_returns_same_object():
    rng = np.random.default_rng(0)
    x = np.zeros((3, 6), dtype=np.float32)
    x_id = id(x)

    out = PartialDimsSignNoise(noise_amplitude=0.0).apply(x, rng)

    assert out is x
    assert id(out) == x_id
    np.testing.assert_allclose(out, 0.0)


def test_partial_dims_noise_raises_for_dim_lt_2():
    rng = np.random.default_rng(0)
    x = np.zeros((3, 1), dtype=np.float32)

    with pytest.raises(ValueError, match=r"requires dim >= 2"):
        PartialDimsSignNoise(noise_amplitude=0.1).apply(x, rng)


def test_partial_dims_noise_changes_exactly_floor_half_dimensions_per_row_and_does_not_mutate_input():
    rng = np.random.default_rng(7)
    amp = 0.5
    n, d = 5, 7
    k = d // 2  # floor(d/2)
    x = np.zeros((n, d), dtype=np.float32)

    out = PartialDimsSignNoise(noise_amplitude=amp).apply(x, rng)

    # Input must remain unchanged (implementation uses copy()).
    np.testing.assert_allclose(x, 0.0)

    assert out.shape == (n, d)
    assert out.dtype == x.dtype

    # Exactly k dims per row should be changed to +/- amp.
    changed = out != 0.0
    assert changed.sum(axis=1).tolist() == [k] * n
    np.testing.assert_allclose(np.abs(out[changed]), np.float32(amp))


def test_make_noise_strategy_accepts_enum_and_string_case_insensitive():
    s1 = make_noise_strategy(NoiseScheme.ALL, 0.1)
    assert isinstance(s1, AllDimsSignNoise)
    assert s1.noise_amplitude == pytest.approx(0.1)

    s2 = make_noise_strategy("partial", 0.2)
    assert isinstance(s2, PartialDimsSignNoise)
    assert s2.noise_amplitude == pytest.approx(0.2)


def test_make_noise_strategy_rejects_invalid_scheme_string():
    with pytest.raises(ValueError):
        make_noise_strategy("nope", 0.1)