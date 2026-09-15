import pytest


def test_build_parser_workers_default_is_cpu_count(monkeypatch):
    # Import inside test so monkeypatching affects the module used.
    import scale.cli as cli

    monkeypatch.setattr(cli.multiprocessing, "cpu_count", lambda: 7)
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "-i",
            "in.fbin",
            "-o",
            "out.fbin",
            "-s",
            "1",
        ]
    )

    assert args.workers == 7


def test_main_happy_path_builds_config_and_prints_summary(tmp_path, monkeypatch, capsys):
    import scale.cli as cli
    from scale.config import NoiseScheme

    in_path = tmp_path / "in.fbin"
    out_path = tmp_path / "out.fbin"
    in_path.write_bytes(b"")

    captured_cfg = {}

    def fake_run(cfg):
        # Capture a few representative config fields to ensure wiring.
        captured_cfg.update(
            {
                "input_path": cfg.input_path,
                "output_path": cfg.output_path,
                "scale": cfg.scale,
                "batch_size": cfg.batch_size,
                "noise_amplitude": cfg.noise_amplitude,
                "noise_scheme": cfg.noise_scheme,
                "normalize": cfg.normalize,
                "tolerance": cfg.tolerance,
                "shard_size": cfg.shard_size,
                "seed": cfg.seed,
                "json_metadata": cfg.json_metadata,
                "workers": cfg.workers,
            }
        )
        return {"total_written": 123}

    monkeypatch.setattr(cli, "run", fake_run)

    rc = cli.main(
        [
            "-i",
            str(in_path),
            "-o",
            str(out_path),
            "-s",
            "3",
            "-b",
            "10",
            "-a",
            "0.5",
            "-m",
            "ALL",
            "-n",
            "-t",
            "0.001",
            "-k",
            "100",
            "-x",
            "42",
            "-j",
            str(tmp_path / "report.json"),
            "-w",
            "2",
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "Done. Total vectors written: 123" in out

    assert captured_cfg["input_path"] == in_path
    assert captured_cfg["output_path"] == out_path
    assert captured_cfg["scale"] == 3
    assert captured_cfg["batch_size"] == 10
    assert captured_cfg["noise_amplitude"] == 0.5
    assert captured_cfg["noise_scheme"] == NoiseScheme.ALL
    assert captured_cfg["normalize"] is True
    assert captured_cfg["tolerance"] == 0.001
    assert captured_cfg["shard_size"] == 100
    assert captured_cfg["seed"] == 42
    assert str(captured_cfg["json_metadata"]).endswith("report.json")
    assert captured_cfg["workers"] == 2


def test_main_rejects_invalid_noise_scheme(tmp_path):
    import scale.cli as cli

    in_path = tmp_path / "in.fbin"
    out_path = tmp_path / "out.fbin"
    in_path.write_bytes(b"")

    with pytest.raises(SystemExit) as e:
        cli.main(
            [
                "-i",
                str(in_path),
                "-o",
                str(out_path),
                "-s",
                "1",
                "-m",
                "BOGUS",
            ]
        )
    # argparse uses exit code 2 for parsing errors
    assert e.value.code == 2


def test_main_propagates_run_exception(tmp_path, monkeypatch):
    import scale.cli as cli

    in_path = tmp_path / "in.fbin"
    out_path = tmp_path / "out.fbin"
    in_path.write_bytes(b"")

    def boom(_cfg):
        raise RuntimeError("run failed")

    monkeypatch.setattr(cli, "run", boom)

    with pytest.raises(RuntimeError, match="run failed"):
        cli.main(
            [
                "-i",
                str(in_path),
                "-o",
                str(out_path),
                "-s",
                "1",
            ]
        )
