#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Exercise real orchestration/export with explicitly fabricated measurements."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from cuvs_bench.backends import registry, search_spaces
from cuvs_bench.backends.base import (
    BenchmarkBackend,
    BuildResult,
    SearchResult,
)
from cuvs_bench.orchestrator import BenchmarkOrchestrator
from cuvs_bench.orchestrator.config_loaders import (
    BenchmarkConfig,
    DatasetConfig,
    IndexConfig,
)
from cuvs_bench.run.__main__ import main
from cuvs_bench.run.data_export import write_results_to_csv


def _fake_config(name="fake-index", search_params=None):
    return BenchmarkConfig(
        indexes=[
            IndexConfig(
                name=name,
                algo="fake-algorithm",
                build_param={},
                search_params=[{}] if search_params is None else search_params,
                file=name,
            )
        ],
        backend_config={"name": name, "group": "base"},
    )


def _fake_result(name, search_params, recall=0.75):
    return SearchResult(
        neighbors=np.empty((0, 2), dtype=np.int64),
        distances=np.empty((0, 2), dtype=np.float32),
        search_time_ms=10.0,
        queries_per_second=100.0,
        recall=recall,
        algorithm="fake-algorithm",
        search_params=search_params,
        metadata={"group": "base", "index_name": name},
    )


@pytest.fixture
def fake_backend(monkeypatch):
    """Replace the plugin boundary, not orchestration or serialization."""
    state = SimpleNamespace(
        configurations=[[_fake_config()]],
        returned_batches=[[_fake_result("fake-index", [{}])]],
        events=[],
        search_error=None,
    )

    class FakeLoader:
        def load(self, **kwargs):
            return DatasetConfig(name="fake-data"), state.configurations.pop(0)

    class FakeBackend(BenchmarkBackend):
        @property
        def algo(self):
            return "fake-algorithm"

        def initialize(self):
            state.events.append(("initialize", self.name))

        def cleanup(self):
            state.events.append(("cleanup", self.name))

        def build(self, dataset, indexes, **kwargs):
            state.events.append(("build", self.name))
            return BuildResult(
                index_path=self.name,
                build_time_seconds=1.0,
                index_size_bytes=0,
                algorithm=self.algo,
                build_params={},
                metadata={"group": "base", "index_name": self.name},
            )

        def search(self, dataset, indexes, **kwargs):
            state.events.append(("search", self.name))
            if state.search_error is not None:
                raise state.search_error
            return state.returned_batches.pop(0)

    monkeypatch.setitem(
        registry.get_registry()._backends, "fake_backend", FakeBackend
    )
    monkeypatch.setitem(
        registry._CONFIG_LOADER_REGISTRY, "fake_backend", FakeLoader
    )
    monkeypatch.setitem(
        search_spaces.ALGORITHM_SEARCH_SPACES,
        "fake-algorithm",
        {"build": {}, "search": {}},
    )
    return state


@pytest.mark.parametrize("layout", ["later-config", "later-index"])
@pytest.mark.parametrize(
    "invalid_value",
    [
        pytest.param(object(), id="fake-object"),
        pytest.param(np.timedelta64(1, "ns"), id="numpy-time"),
        pytest.param({"nested": {1, 2}}, id="nested-set"),
    ],
)
def test_invalid_search_plan_stops_sweep_before_any_backend_work(
    fake_backend, layout, invalid_value
):
    first = _fake_config("first-index")
    invalid = _fake_config("invalid-index", [{"value": invalid_value}])
    configs = [first, invalid]
    if layout == "later-index":
        first.indexes.extend(invalid.indexes)
        configs = [first]
    fake_backend.configurations = [configs]
    # The unfixed orchestrator must be able to finish, so the negative control
    # fails for missing validation rather than exhausting fake result batches.
    fake_backend.returned_batches = [
        [_fake_result(config.index_name, [{}])] for config in configs
    ]

    with pytest.raises(TypeError) as failure:
        BenchmarkOrchestrator("fake_backend").run_benchmark()

    assert "fake_backend" in str(failure.value)
    assert "invalid-index" in str(failure.value)
    assert "configured search_params" in str(failure.value)
    assert "Unsupported search parameter type" in str(failure.value)
    assert fake_backend.events == []


def _invoke_cli(tmp_path, *extra, backend_type="fake_backend"):
    config = tmp_path / "fake-backend.yaml"
    config.write_text(f"backend: {backend_type}\n", encoding="utf-8")
    return CliRunner().invoke(
        main,
        [
            "--backend-config",
            str(config),
            "--dataset",
            "fake-data",
            "--dataset-path",
            str(tmp_path),
            "--algorithms",
            "fake-algorithm",
            "--groups",
            "base",
            "--batch-size",
            "2",
            "-k",
            "2",
            "--search-mode",
            "latency",
            *extra,
        ],
    )


def _raw_search_results(tmp_path):
    return pd.read_csv(
        tmp_path / "fake-data/result/search/fake-algorithm,base,k2,bs2,raw.csv"
    )


def _write_historical_tuning_measurements(tmp_path):
    """Seed both CSVs so an aborted tuning run must retain existing evidence."""
    previous_build = BuildResult(
        index_path="historical-only",
        build_time_seconds=2.0,
        index_size_bytes=0,
        algorithm="fake-algorithm",
        build_params={},
        metadata={"group": "base", "index_name": "historical-only"},
    )
    previous_search = _fake_result("historical-only", [{}], recall=0.5)
    write_results_to_csv(
        [previous_build, previous_search], "fake-data", str(tmp_path), 2, 2
    )


def _invoke_tune_cli(
    tmp_path, *extra, n_trials=2, backend_type="fake_backend"
):
    return _invoke_cli(
        tmp_path,
        "--mode",
        "tune",
        "--n-trials",
        str(n_trials),
        "--constraints",
        '{"recall": "maximize"}',
        *extra,
        backend_type=backend_type,
    )


def test_invalid_backend_plan_keeps_good_measurements_and_fails_cli(
    fake_backend, tmp_path
):
    historical = [
        _fake_result("invalid-index", [{"ef_search": 100}], recall=0.5),
        _fake_result("historical-only", [{}], recall=0.6),
    ]
    write_results_to_csv(historical, "fake-data", str(tmp_path), 2, 2)
    fake_backend.returned_batches = [
        [
            _fake_result("first-good", [{"ef_search": 50}]),
            _fake_result("invalid-index", [{"value": object()}]),
            _fake_result("last-good", [{"ef_search": 200}]),
        ]
    ]

    outcome = _invoke_cli(tmp_path)

    assert _raw_search_results(tmp_path).set_index("index_name")[
        "recall"
    ].to_dict() == {
        "first-good": 0.75,
        "last-good": 0.75,
        "invalid-index": 0.5,
        "historical-only": 0.6,
    }
    assert outcome.exit_code == 1, outcome.output
    assert "returned search_params" in outcome.output
    assert "invalid-index" in outcome.output
    assert "object" in outcome.output
    assert fake_backend.events[-1] == ("cleanup", "fake-index")
    for metric in ("latency", "throughput"):
        frontier = pd.read_csv(
            tmp_path
            / f"fake-data/result/search/fake-algorithm,base,k2,bs2,{metric}.csv"
        )
        assert not frontier.empty
        assert "_search_params_identity" not in frontier


@pytest.mark.parametrize(
    ("invalid_source", "expected_started_indexes"),
    [
        pytest.param("loaded-plan", ["first-trial"], id="reject-before-build"),
        pytest.param(
            "returned-plan",
            ["first-trial", "second-trial"],
            id="reject-after-search",
        ),
    ],
)
@pytest.mark.parametrize(
    "group_source", ["backend-config", "result-metadata-only"]
)
def test_invalid_second_tuning_trial_preserves_previous_exports_and_fails_cli(
    fake_backend,
    tmp_path,
    invalid_source,
    expected_started_indexes,
    group_source,
):
    pytest.importorskip("optuna", reason="tune mode requires optional Optuna")
    # Optuna is real; its parameter space is empty so trial ordering and the
    # fabricated results do not depend on sampling or measured wall time.
    first_trial = _fake_config("first-trial")
    second_trial = _fake_config("second-trial")
    if invalid_source == "loaded-plan":
        second_trial.indexes[0].search_params = [{"value": object()}]
    if group_source == "result-metadata-only":
        del first_trial.backend_config["group"]
        del second_trial.backend_config["group"]
    fake_backend.configurations = [[first_trial], [second_trial]]
    fake_backend.returned_batches = [
        [_fake_result("first-trial", [{}])],
        [_fake_result("second-trial", [{"value": object()}])],
    ]
    _write_historical_tuning_measurements(tmp_path)

    outcome = _invoke_tune_cli(tmp_path)

    exported_searches = _raw_search_results(tmp_path)
    assert exported_searches.set_index("index_name")["recall"].to_dict() == {
        "historical-only": 0.5,
        "first-trial": 0.75,
    }
    initialized_indexes = [
        name for event, name in fake_backend.events if event == "initialize"
    ]
    assert initialized_indexes == expected_started_indexes
    exported_builds = pd.read_csv(
        tmp_path / "fake-data/result/build/fake-algorithm,base.csv"
    )
    assert exported_builds["index_name"].tolist() == [
        "historical-only",
        *expected_started_indexes,
    ]
    assert outcome.exit_code == 1, outcome.output
    assert "Unsupported search parameter type" in outcome.output


@pytest.mark.parametrize("backend_type", ["fake_backend", "cpp_gbench"])
@pytest.mark.parametrize("dry_run", [False, True], ids=["normal", "dry-run"])
def test_invalid_tune_plan_is_fatal_on_every_cli_route(
    fake_backend, tmp_path, monkeypatch, backend_type, dry_run
):
    pytest.importorskip("optuna", reason="tune mode requires optional Optuna")
    # Select the actual C++ CLI route with fake plugin boundaries: the invalid
    # loader must prevent backend initialization, not launch a native process.
    monkeypatch.setitem(
        registry.get_registry()._backends,
        "cpp_gbench",
        registry.get_registry()._backends["fake_backend"],
    )
    monkeypatch.setitem(
        registry._CONFIG_LOADER_REGISTRY,
        "cpp_gbench",
        registry._CONFIG_LOADER_REGISTRY["fake_backend"],
    )
    fake_backend.configurations = [
        [_fake_config("invalid-trial", [{"value": object()}])]
    ]

    outcome = _invoke_tune_cli(
        tmp_path,
        *(["--dry-run"] if dry_run else []),
        n_trials=1,
        backend_type=backend_type,
    )

    assert outcome.exit_code == 1, outcome.output
    assert "configured search_params" in outcome.output
    assert "invalid-trial" in outcome.output
    assert "Consider relaxing constraints" not in outcome.output
    assert fake_backend.events == []
    assert not list(tmp_path.rglob("*.csv"))


@pytest.mark.parametrize("outcome", ["failed", "skipped", "dry_run"])
def test_unexported_result_does_not_require_serializable_parameters(
    fake_backend, outcome
):
    result = _fake_result("unmeasured", [{"value": object()}])
    if outcome == "failed":
        result.success = False
    else:
        result.metadata[outcome] = True
    fake_backend.returned_batches = [[result]]
    orchestrator = BenchmarkOrchestrator("fake_backend")

    results = orchestrator.run_benchmark(build=False, search=True)

    assert results == [result]
    assert result.error_message is None
    assert orchestrator.result_failure_message(results) is None


@pytest.mark.parametrize("plans", [[], [{"ef": 50}, {"ef": 100}]])
def test_empty_and_aggregate_result_plans_remain_supported(
    fake_backend, plans
):
    result = _fake_result("aggregate", plans)
    fake_backend.returned_batches = [[result]]

    results = BenchmarkOrchestrator("fake_backend").run_benchmark(
        build=False, search=True
    )

    assert results[0].success
    assert results[0].search_params == plans


def test_unrelated_backend_error_still_propagates_and_cleans_up(fake_backend):
    fake_backend.search_error = ValueError("fake backend programming error")

    with pytest.raises(ValueError, match="fake backend programming error"):
        BenchmarkOrchestrator("fake_backend").run_benchmark()

    assert fake_backend.events[-1] == ("cleanup", "fake-index")
