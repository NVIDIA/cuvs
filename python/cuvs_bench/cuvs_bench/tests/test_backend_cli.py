#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI contracts shared by pluggable cuVS Bench backends."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from cuvs_bench.backends import registry
from cuvs_bench.backends.base import BenchmarkBackend, BuildResult
from cuvs_bench.backends.cpp_gbench import CppGoogleBenchmarkBackend
from cuvs_bench.orchestrator import BenchmarkOrchestrator
from cuvs_bench.orchestrator import orchestrator as orchestrator_module
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


def _run_arguments(tmp_path: Path, *extra_args: str) -> list[str]:
    return [
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

    arguments = _run_arguments(tmp_path, *extra_args)
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
        {"backend": "third_party", "host": "first-snapshot.example"},
        {"backend": "opensearch", "host": "second-snapshot.example"},
        {"backend": "cpp_gbench", "host": "third-snapshot.example"},
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
    assert captured["run_kwargs"]["host"] == "first-snapshot.example"


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


@pytest.mark.parametrize(
    "algorithm_args",
    [(), ("--algorithms", "cuvs_cagra")],
    ids=["prompted-algorithm", "explicit-algorithm"],
)
@pytest.mark.parametrize(
    "config_text, expected_reason",
    [
        pytest.param("[]", "must parse to a mapping", id="nonmapping"),
        pytest.param(
            "host: localhost",
            "must include a 'backend' field",
            id="missing-backend",
        ),
        pytest.param(
            "backend: i-am-not-a-registered-backend",
            "Backend 'i-am-not-a-registered-backend' not found",
            id="unknown-backend",
        ),
        pytest.param("backend: [", "Cannot read", id="malformed-yaml"),
        pytest.param(None, "No such file or directory", id="missing-file"),
    ],
)
def test_invalid_backend_config_reports_a_cli_usage_error(
    tmp_path,
    monkeypatch,
    algorithm_args,
    config_text,
    expected_reason,
):
    backend_config = tmp_path / "backend.yaml"
    if config_text is not None:
        backend_config.write_text(config_text, encoding="utf-8")
    # Keep the real registry's missing-name behavior, without installed plugins.
    monkeypatch.setattr(
        registry.importlib.metadata,
        "entry_points",
        lambda **_filters: registry.importlib.metadata.EntryPoints(),
    )
    with patch(
        "cuvs_bench.run.__main__.BenchmarkOrchestrator"
    ) as orchestrator:
        result = CliRunner().invoke(
            run_main,
            _run_arguments(
                tmp_path,
                "--backend-config",
                str(backend_config),
                *algorithm_args,
            ),
            input="\n",
        )

    assert result.exit_code == 2, result.output
    assert "Invalid value for --backend-config" in result.output
    assert expected_reason in result.output
    orchestrator.assert_not_called()


@pytest.mark.parametrize(
    "algorithm_args",
    [(), ("--algorithms", "cuvs_cagra")],
    ids=["prompted-algorithm", "explicit-algorithm"],
)
def test_plugin_programming_error_is_not_reported_as_invalid_config(
    tmp_path,
    monkeypatch,
    algorithm_args,
):
    backend_config = _write_backend_config(tmp_path, "i-am-a-fake-plugin")
    plugin_error = ValueError("i-am-a-fake-plugin-registration-failure")
    monkeypatch.setattr(
        registry,
        "_try_load_plugin",
        Mock(side_effect=plugin_error),
    )

    result = CliRunner().invoke(
        run_main,
        _run_arguments(
            tmp_path,
            "--backend-config",
            str(backend_config),
            *algorithm_args,
        ),
        input="\n",
    )

    assert result.exception is plugin_error
    assert "Invalid value for --backend-config" not in result.output


@pytest.mark.parametrize(
    "algorithm_args",
    [(), ("--algorithms", "cuvs_cagra")],
    ids=["prompted-algorithm", "explicit-algorithm"],
)
def test_loader_programming_error_is_not_reported_as_invalid_config(
    tmp_path,
    monkeypatch,
    algorithm_args,
):
    backend_config = _write_backend_config(tmp_path, "cpp_gbench")
    loader_error = ValueError("i-am-a-fake-loader-construction-failure")
    fake_loader_constructor = Mock(side_effect=loader_error)
    monkeypatch.setattr(
        orchestrator_module,
        "get_config_loader",
        lambda _name: fake_loader_constructor,
    )

    result = CliRunner().invoke(
        run_main,
        _run_arguments(
            tmp_path,
            "--backend-config",
            str(backend_config),
            *algorithm_args,
        ),
        input="\n",
    )

    assert result.exception is loader_error
    assert "Invalid value for --backend-config" not in result.output


@pytest.mark.parametrize(
    "algorithm_args",
    [(), ("--algorithms", "cuvs_cagra")],
    ids=["prompted-algorithm", "explicit-algorithm"],
)
def test_benchmark_programming_error_is_not_reported_as_invalid_config(
    tmp_path,
    monkeypatch,
    algorithm_args,
):
    backend_config = _write_backend_config(tmp_path, "cpp_gbench")
    benchmark_error = ValueError("i-am-a-fake-benchmark-failure")
    monkeypatch.setattr(
        BenchmarkOrchestrator,
        "run_benchmark",
        Mock(side_effect=benchmark_error),
    )

    result = CliRunner().invoke(
        run_main,
        _run_arguments(
            tmp_path,
            "--backend-config",
            str(backend_config),
            *algorithm_args,
        ),
        input="\n",
    )

    assert result.exception is benchmark_error
    assert "Invalid value for --backend-config" not in result.output


def test_missing_builtin_algorithm_default_remains_an_internal_error(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(CppGoogleBenchmarkBackend, "default_algorithm", None)

    result, _captured, _write_results = _invoke_run(tmp_path, input_text="\n")

    assert isinstance(result.exception, RuntimeError)
    assert (
        str(result.exception)
        == "The default backend must define a default algorithm"
    )


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
