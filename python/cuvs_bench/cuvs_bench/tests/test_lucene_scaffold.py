# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the opt-in Lucene backend scaffold."""

import os
import subprocess
import tomllib
from importlib import resources
from pathlib import Path
from unittest.mock import patch

import yaml
from click.testing import CliRunner

from cuvs_bench.run.__main__ import main as run_main


_PROJECT_ROOT = Path(__file__).parents[2]


def _invoke_run(tmp_path, *extra_args, input_text=""):
    captured = {}

    class RecordingOrchestrator:
        def __init__(self, backend_type):
            captured["backend_type"] = backend_type

        def run_benchmark(self, **kwargs):
            captured["run_kwargs"] = kwargs
            return []

    args = [
        "--dataset",
        "test-data",
        "--dataset-path",
        str(tmp_path),
        "--batch-size",
        "10",
        "-k",
        "10",
        "--groups",
        "test",
        "-m",
        "latency",
        "--dry-run",
        *extra_args,
    ]
    with patch(
        "cuvs_bench.run.__main__.BenchmarkOrchestrator",
        RecordingOrchestrator,
    ):
        result = CliRunner().invoke(run_main, args, input=input_text)
    return result, captured


def test_ordinary_cli_keeps_cpp_backend_and_cagra_default(tmp_path):
    result, captured = _invoke_run(tmp_path, input_text="\n")

    assert result.exit_code == 0, result.output
    assert captured["backend_type"] == "cpp_gbench"
    assert captured["run_kwargs"]["algorithms"] == "cuvs_cagra"


def test_lucene_backend_selects_cagra_default(tmp_path):
    result, captured = _invoke_run(
        tmp_path, "--backend", "lucene", input_text="\n"
    )

    assert result.exit_code == 0, result.output
    assert captured["backend_type"] == "lucene"
    assert captured["run_kwargs"]["algorithms"] == "lucene_cuvs_cagra"


def test_lucene_backend_preserves_explicit_cpu_algorithm(tmp_path):
    result, captured = _invoke_run(
        tmp_path,
        "--backend",
        "lucene",
        "--algorithms",
        "lucene_cpu_hnsw",
    )

    assert result.exit_code == 0, result.output
    assert captured["backend_type"] == "lucene"
    assert captured["run_kwargs"]["algorithms"] == "lucene_cpu_hnsw"


def test_lucene_backend_does_not_rewrite_explicit_algorithm(tmp_path):
    result, captured = _invoke_run(
        tmp_path,
        "--backend",
        "lucene",
        "--algorithms",
        "cuvs_cagra",
    )

    assert result.exit_code == 0, result.output
    assert captured["run_kwargs"]["algorithms"] == "cuvs_cagra"


def test_lucene_backend_preserves_an_algorithm_typed_at_the_prompt(tmp_path):
    result, captured = _invoke_run(
        tmp_path,
        "--backend",
        "lucene",
        input_text="cuvs_cagra\n",
    )

    assert result.exit_code == 0, result.output
    assert captured["run_kwargs"]["algorithms"] == "cuvs_cagra"


def test_lucene_backend_config_sets_the_prompt_default(tmp_path):
    backend_config = tmp_path / "backend.yaml"
    backend_config.write_text("backend: lucene\n", encoding="utf-8")

    result, captured = _invoke_run(
        tmp_path,
        "--backend-config",
        str(backend_config),
        input_text="\n",
    )

    assert result.exit_code == 0, result.output
    assert captured["backend_type"] == "lucene"
    assert captured["run_kwargs"]["algorithms"] == "lucene_cuvs_cagra"


def test_backend_config_keeps_selecting_backend_without_new_option(tmp_path):
    backend_config = tmp_path / "backend.yaml"
    backend_config.write_text("backend: opensearch\nhost: search.example\n")

    result, captured = _invoke_run(
        tmp_path,
        "--backend-config",
        str(backend_config),
        "--algorithms",
        "opensearch_faiss_hnsw",
    )

    assert result.exit_code == 0, result.output
    assert captured["backend_type"] == "opensearch"
    assert captured["run_kwargs"]["host"] == "search.example"


def test_explicit_backend_must_match_backend_config(tmp_path):
    backend_config = tmp_path / "backend.yaml"
    backend_config.write_text("backend: cpp_gbench\n")

    result, captured = _invoke_run(
        tmp_path,
        "--backend",
        "lucene",
        "--backend-config",
        str(backend_config),
        "--algorithms",
        "lucene_cuvs_cagra",
    )

    assert result.exit_code != 0
    assert "must match" in str(result.exception)
    assert captured == {}


def test_lucene_plugin_uses_lazy_entry_points():
    pyproject = tomllib.loads((_PROJECT_ROOT / "pyproject.toml").read_text())
    entry_points = pyproject["project"]["entry-points"]

    target = "cuvs_bench.backends.lucene:register"
    assert entry_points["cuvs_bench.backends"]["lucene"] == target
    assert entry_points["cuvs_bench.config_loaders"]["lucene"] == target


def test_lucene_algorithm_configs_are_packaged_resources():
    expected_codecs = {
        "lucene_accelerated_hnsw": "Lucene101AcceleratedHNSWCodec",
        "lucene_cpu_hnsw": "Lucene101",
        "lucene_cuvs_cagra": "CuVS2510GPUSearchCodec",
    }
    algorithm_resources = resources.files("cuvs_bench.config.algos")

    for algorithm, codec in expected_codecs.items():
        config = yaml.safe_load(
            algorithm_resources.joinpath(f"{algorithm}.yaml").read_text()
        )
        assert config["name"] == algorithm
        assert set(config["groups"]) == {"base", "test"}
        for group in config["groups"].values():
            assert group == {
                "build": {"codec": [codec]},
                "search": {},
            }


def test_pylucene_source_builder_is_self_describing() -> None:
    tool_directory = _PROJECT_ROOT / "tools" / "pylucene"
    helper = tool_directory / "build_pylucene_10_2.sh"
    compatibility_patch = tool_directory / "pylucene-10.2.0.patch"

    completed = subprocess.run(
        ["bash", str(helper), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert os.access(helper, os.X_OK)
    assert compatibility_patch.is_file()
    assert completed.returncode == 0, completed.stderr
    assert "Build an isolated PyLucene 10.2.0 environment" in completed.stdout


def test_pylucene_source_builder_rejects_unsafe_build_roots(
    tmp_path: Path,
) -> None:
    helper = _PROJECT_ROOT / "tools" / "pylucene" / "build_pylucene_10_2.sh"

    for unsafe_character in (" ", ":", ";", "$", "`", "&", "#", "|"):
        build_root = tmp_path / f"unsafe{unsafe_character}root"
        completed = subprocess.run(
            [
                "bash",
                str(helper),
                "--build-root",
                str(build_root),
                "--prepare-only",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode != 0
        assert "--build-root may contain only" in completed.stderr
        assert not build_root.exists()


def test_pylucene_source_builder_rejects_unsafe_resolved_build_root(
    tmp_path: Path,
) -> None:
    helper = _PROJECT_ROOT / "tools" / "pylucene" / "build_pylucene_10_2.sh"
    unsafe_target = tmp_path / "unsafe;target"
    unsafe_target.mkdir()
    build_root = tmp_path / "safe-link"
    build_root.symlink_to(unsafe_target, target_is_directory=True)

    completed = subprocess.run(
        [
            "bash",
            str(helper),
            "--build-root",
            str(build_root),
            "--prepare-only",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "--build-root may contain only" in completed.stderr
    assert not (unsafe_target / ".build.lock").exists()
