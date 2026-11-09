#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI contracts shared by pluggable cuVS Bench backends."""

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from cuvs_bench.backends.base import BenchmarkBackend, BuildResult
from cuvs_bench.backends.cpp_gbench import CppGoogleBenchmarkBackend
from cuvs_bench.orchestrator import BenchmarkOrchestrator
from cuvs_bench.run.__main__ import main as run_main


class _PluginBackend:
    default_algorithm = "plugin_default"

    @classmethod
    def result_failure_message(cls, results):
        failures = [result for result in results if not result.success]
        if not failures:
            return None
        return "; ".join(result.error_message for result in failures)


def _write_backend_config(
    tmp_path: Path, backend: str, *settings: str
) -> Path:
    backend_config = tmp_path / "backend.yaml"
    lines = [f"backend: {backend}", *settings]
    backend_config.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return backend_config


def _invoke_run(
    tmp_path: Path,
    *extra_args: str,
    input_text: str = "",
    run_results=None,
    dry_run: bool = True,
    plugin_backend=_PluginBackend,
):
    captured = {}
    backend_classes = {
        "cpp_gbench": CppGoogleBenchmarkBackend,
        "opensearch": plugin_backend,
        "third_party": plugin_backend,
    }

    class RecordingOrchestrator:
        def __init__(self, backend_type):
            captured["backend_type"] = backend_type
            self.backend_class = backend_classes[backend_type]

        def run_benchmark(self, **kwargs):
            captured["run_kwargs"] = kwargs
            return [] if run_results is None else run_results

        def result_failure_message(self, results):
            return self.backend_class.result_failure_message(results)

    arguments = [
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
        *extra_args,
    ]
    if dry_run:
        arguments.append("--dry-run")

    with (
        patch(
            "cuvs_bench.run.__main__.get_backend_class",
            side_effect=backend_classes.__getitem__,
        ),
        patch(
            "cuvs_bench.run.__main__.BenchmarkOrchestrator",
            RecordingOrchestrator,
        ),
        patch("cuvs_bench.run.__main__.write_results_to_csv") as write_results,
    ):
        result = CliRunner().invoke(run_main, arguments, input=input_text)

    return result, captured, write_results


def test_cli_keeps_the_cpp_backend_and_algorithm_defaults(tmp_path: Path):
    result, captured, _write_results = _invoke_run(tmp_path, input_text="\n")

    assert result.exit_code == 0, result.output
    assert captured["backend_type"] == "cpp_gbench"
    assert captured["run_kwargs"]["algorithms"] == "cuvs_cagra"


def test_registered_backend_controls_its_prompt_default(tmp_path: Path):
    backend_config = _write_backend_config(tmp_path, "third_party")

    result, captured, _write_results = _invoke_run(
        tmp_path,
        "--backend-config",
        str(backend_config),
        input_text="\n",
    )

    assert result.exit_code == 0, result.output
    assert captured["backend_type"] == "third_party"
    assert captured["run_kwargs"]["algorithms"] == "plugin_default"


def test_backend_default_and_dispatch_share_one_config_snapshot(
    tmp_path: Path,
):
    backend_config = _write_backend_config(tmp_path, "third_party")
    changing_snapshots = (
        {"backend": "third_party"},
        {"backend": "opensearch"},
        {"backend": "cpp_gbench"},
    )

    with patch(
        "cuvs_bench.run.__main__._read_backend_config",
        side_effect=changing_snapshots,
    ) as read_backend_config:
        result, captured, _write_results = _invoke_run(
            tmp_path,
            "--backend-config",
            str(backend_config),
            input_text="\n",
        )

    assert result.exit_code == 0, result.output
    assert read_backend_config.call_count == 1
    assert captured["backend_type"] == "third_party"
    assert captured["run_kwargs"]["algorithms"] == "plugin_default"


def test_backend_without_a_default_keeps_the_cli_default(tmp_path: Path):
    class BackendWithoutDefault(_PluginBackend):
        default_algorithm = None

    backend_config = _write_backend_config(tmp_path, "third_party")

    result, captured, _write_results = _invoke_run(
        tmp_path,
        "--backend-config",
        str(backend_config),
        input_text="\n",
        plugin_backend=BackendWithoutDefault,
    )

    assert result.exit_code == 0, result.output
    assert captured["run_kwargs"]["algorithms"] == "cuvs_cagra"


def test_cli_preserves_an_explicit_algorithm(tmp_path: Path):
    backend_config = _write_backend_config(tmp_path, "third_party")

    result, captured, _write_results = _invoke_run(
        tmp_path,
        "--backend-config",
        str(backend_config),
        "--algorithms",
        "user_selected_algorithm",
    )

    assert result.exit_code == 0, result.output
    assert captured["run_kwargs"]["algorithms"] == "user_selected_algorithm"


def test_backend_config_keeps_selecting_the_backend(tmp_path: Path):
    backend_config = _write_backend_config(
        tmp_path, "opensearch", "host: search.example"
    )

    result, captured, _write_results = _invoke_run(
        tmp_path,
        "--backend-config",
        str(backend_config),
        "--algorithms",
        "plugin_algorithm",
    )

    assert result.exit_code == 0, result.output
    assert captured["backend_type"] == "opensearch"
    assert captured["run_kwargs"]["host"] == "search.example"


def test_cli_invokes_export_before_reporting_backend_failure(tmp_path: Path):
    reported_failure = "i-am-fake-backend-failure"
    failed = BuildResult(
        index_path="",
        build_time_seconds=0.0,
        index_size_bytes=0,
        algorithm="plugin_algorithm",
        build_params={},
        success=False,
        error_message=reported_failure,
    )

    backend_config = _write_backend_config(tmp_path, "third_party")

    result, _captured, write_results = _invoke_run(
        tmp_path,
        "--backend-config",
        str(backend_config),
        "--algorithms",
        "plugin_algorithm",
        run_results=[failed],
        dry_run=False,
    )

    assert result.exit_code == 1
    assert reported_failure in result.output
    write_results.assert_called_once()


def test_backend_failure_policy_is_opt_in():
    failed = BuildResult(
        index_path="",
        build_time_seconds=0.0,
        index_size_bytes=0,
        algorithm="plugin_algorithm",
        build_params={},
        success=False,
        error_message="backend operation failed",
    )

    assert BenchmarkBackend.result_failure_message([failed]) is None


def test_orchestrator_delegates_failure_policy_to_selected_backend():
    reported_failure = "i-am-fake-backend-failure"
    orchestrator = BenchmarkOrchestrator.__new__(BenchmarkOrchestrator)
    orchestrator.backend_class = _PluginBackend
    failed = BuildResult(
        index_path="",
        build_time_seconds=0.0,
        index_size_bytes=0,
        algorithm="plugin_algorithm",
        build_params={},
        success=False,
        error_message=reported_failure,
    )

    assert orchestrator.result_failure_message([failed]) == reported_failure
