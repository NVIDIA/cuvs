# tests/test_rng.py
import numpy as np
import pytest

from scale.rng import make_rng, make_work_seed


def test_make_rng_none_seed_returns_generator() -> None:
    rng = make_rng(None)
    assert isinstance(rng, np.random.Generator)
    # Smoke: can generate values
    x = rng.random(3)
    assert x.shape == (3,)


def test_make_rng_deterministic_matches_numpy_default_rng() -> None:
    seed = 12345
    rng1 = make_rng(seed)
    rng2 = np.random.default_rng(seed)

    a = rng1.random(10)
    b = rng2.random(10)
    assert np.allclose(a, b)


def test_make_rng_accepts_int_like_seed() -> None:
    rng1 = make_rng(7)
    rng2 = make_rng(int("7"))
    assert np.allclose(rng1.random(5), rng2.random(5))


def test_make_rng_rejects_non_int_like_seed() -> None:
    with pytest.raises((TypeError, ValueError)):
        make_rng("nope")  # type: ignore[arg-type]


def test_make_work_seed_none_base_seed_returns_none() -> None:
    assert make_work_seed(None, replica_id=0, batch_id=0) is None


def test_make_work_seed_is_pure_and_stable() -> None:
    s1 = make_work_seed(123, replica_id=5, batch_id=9)
    s2 = make_work_seed(123, replica_id=5, batch_id=9)
    assert s1 == s2
    assert isinstance(s1, int)


def test_make_work_seed_changes_with_replica_or_batch() -> None:
    base = 123
    a = make_work_seed(base, replica_id=1, batch_id=2)
    b = make_work_seed(base, replica_id=2, batch_id=2)
    c = make_work_seed(base, replica_id=1, batch_id=3)
    assert a != b
    assert a != c


def test_make_work_seed_matches_manual_mixing_and_is_u64_masked() -> None:
    base_seed = 123
    replica_id = 7
    batch_id = 11

    got = make_work_seed(base_seed, replica_id=replica_id, batch_id=batch_id)

    mask = 0xFFFFFFFFFFFFFFFF
    x = int(base_seed) & mask
    x ^= (int(replica_id) * 0x9E3779B185EBCA87) & mask
    x ^= (int(batch_id) * 0xC2B2AE3D27D4EB4F) & mask
    expected = x

    assert got == expected
    assert 0 <= got <= mask  # type: ignore[operator]


def test_make_work_seed_rejects_non_int_like_inputs() -> None:
    with pytest.raises((TypeError, ValueError)):
        make_work_seed("x", replica_id=1, batch_id=1)  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError)):
        make_work_seed(1, replica_id="y", batch_id=1)  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError)):
        make_work_seed(1, replica_id=1, batch_id="z")  # type: ignore[arg-type]