from pathlib import Path
import dataclasses
import pytest

from scale.config import Config, NoiseScheme


def _cfg(**overrides) -> Config:
    """Factory with known-good defaults; override only what a test needs."""
    base = dict(
        input_path=Path("in.fbin"),
        output_path=Path("out.fbin"),
        scale=1,
        batch_size=1,
        noise_amplitude=0.0,
        noise_scheme=NoiseScheme.ALL,
        normalize=False,
        tolerance=1e-6,
        shard_size=None,
        seed=None,
        json_metadata=None,
        workers=1,
    )
    base.update(overrides)
    return Config(**base)


def test_noise_scheme_values():
    assert NoiseScheme.ALL.value == "ALL"
    assert NoiseScheme.PARTIAL.value == "PARTIAL"
    assert NoiseScheme("ALL") is NoiseScheme.ALL
    assert NoiseScheme("PARTIAL") is NoiseScheme.PARTIAL


def test_validate_ok_defaults():
    cfg = _cfg()
    cfg.validate()  # should not raise


@pytest.mark.parametrize(
    "overrides, msg",
    [
        ({"scale": 0}, "scale must be >= 1"),
        ({"scale": -1}, "scale must be >= 1"),
        ({"batch_size": 0}, "batch_size must be > 0"),
        ({"batch_size": -5}, "batch_size must be > 0"),
        ({"noise_amplitude": -0.01}, "noise_amplitude must be >= 0"),
        ({"tolerance": 0.0}, "tolerance must be > 0"),
        ({"tolerance": -1.0}, "tolerance must be > 0"),
        ({"workers": 0}, "workers must be >= 1"),
        ({"workers": -2}, "workers must be >= 1"),
        ({"shard_size": 0}, "shard_size must be >= 1"),
        ({"shard_size": -10}, "shard_size must be >= 1"),
    ],
)
def test_validate_rejects_invalid_values(overrides, msg):
    cfg = _cfg(**overrides)
    with pytest.raises(ValueError, match=msg):
        cfg.validate()


def test_validate_allows_none_shard_size():
    cfg = _cfg(shard_size=None)
    cfg.validate()


def test_validate_allows_non_negative_noise_amplitude():
    cfg = _cfg(noise_amplitude=0.0)
    cfg.validate()
    cfg = _cfg(noise_amplitude=1.234)
    cfg.validate()


def test_config_is_frozen():
    cfg = _cfg()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.scale = 2  # type: ignore[attr-defined]