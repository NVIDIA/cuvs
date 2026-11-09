#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pandas as pd
import pytest

from cuvs_bench.backends.base import (
    BenchmarkBackend,
    BuildResult,
    SearchResult,
)
from cuvs_bench.orchestrator.config_loaders import (
    BenchmarkConfig,
    DatasetConfig,
    IndexConfig,
)
from cuvs_bench.orchestrator.orchestrator import BenchmarkOrchestrator
from cuvs_bench.plot.__main__ import load_all_results
from cuvs_bench.run.data_export import (
    _SEARCH_PARAMS_IDENTITY_COLUMN,
    write_results_to_csv,
)


def _search_result(
    algorithm,
    index_name,
    search_params,
    recall,
    *,
    outcome="completed",
):
    metadata = {"group": "base", "index_name": index_name}
    if outcome == "skipped":
        metadata["skipped"] = True
    return SearchResult(
        neighbors=np.empty((0, 2), dtype=np.int64),
        distances=np.empty((0, 2), dtype=np.float32),
        search_time_ms=10.0,
        queries_per_second=100.0,
        recall=recall,
        algorithm=algorithm,
        search_params=[search_params],
        metadata=metadata,
        success=outcome != "failed",
        error_message="search failed" if outcome == "failed" else None,
    )


@pytest.mark.parametrize("index_name", ["test-index", "001", "NA"])
def test_python_backend_csv_is_plot_compatible(tmp_path, index_name):
    dataset = "test-dataset"
    algorithm = "opensearch_faiss_hnsw"
    results = [
        BuildResult(
            index_path="temporary-build-path",
            build_time_seconds=1.5,
            index_size_bytes=1024,
            algorithm=algorithm,
            build_params={"m": 16},
            metadata={"group": "base", "index_name": index_name},
        ),
        SearchResult(
            neighbors=np.empty((0, 2), dtype=np.int64),
            distances=np.empty((0, 2), dtype=np.float32),
            search_time_ms=20.0,
            queries_per_second=100.0,
            recall=0.5,
            algorithm=algorithm,
            search_params=[{"ef_search": 50}],
            metadata={
                "group": "base",
                "index_name": index_name,
                "latency_seconds": 0.01,
            },
        ),
        SearchResult(
            neighbors=np.empty((0, 2), dtype=np.int64),
            distances=np.empty((0, 2), dtype=np.float32),
            search_time_ms=10.0,
            queries_per_second=200.0,
            recall=1.0,
            algorithm=algorithm,
            search_params=[{"ef_search": 100}],
            metadata={
                "group": "base",
                "index_name": index_name,
                "latency_seconds": 0.005,
            },
        ),
    ]

    write_results_to_csv(
        results, dataset, str(tmp_path), count=2, batch_size=2
    )

    result_path = tmp_path / dataset / "result"
    raw_file = result_path / "search" / f"{algorithm},base,k2,bs2,raw.csv"
    raw = pd.read_csv(
        raw_file, dtype={"index_name": str}, keep_default_na=False
    )
    assert raw.columns[:5].tolist() == [
        "algo_name",
        "index_name",
        "recall",
        "throughput",
        "latency",
    ]
    assert raw["ef_search"].tolist() == [50, 100]
    assert raw["index_name"].tolist() == [index_name, index_name]
    assert raw["build time"].tolist() == [1.5, 1.5]
    assert not list(result_path.rglob("*.json"))

    write_results_to_csv(
        [
            BuildResult(
                index_path=index_name,
                build_time_seconds=0.0,
                index_size_bytes=0,
                algorithm=algorithm,
                build_params={"m": 16},
                metadata={"group": "base", "skipped": True},
            )
        ],
        dataset,
        str(tmp_path),
        count=2,
        batch_size=2,
    )
    build = pd.read_csv(result_path / "build" / f"{algorithm},base.csv")
    assert build["time"].tolist() == [1.5]

    for mode in ("throughput", "latency"):
        plotted = load_all_results(
            str(result_path.parent),
            algorithms=[algorithm],
            groups=["base"],
            algo_groups=[],
            k=2,
            batch_size=2,
            method="search",
            index_key="algo",
            raw=False,
            mode=mode,
            time_unit="s",
        )
        assert algorithm in plotted
        assert plotted[algorithm]


def test_failed_and_dry_results_preserve_existing_measurements(tmp_path):
    dataset = "test-dataset"
    algorithm = "test-backend-algorithm"
    index_name = "test-index"
    build = BuildResult(
        index_path=index_name,
        build_time_seconds=1.5,
        index_size_bytes=1024,
        algorithm=algorithm,
        build_params={},
        metadata={"group": "base"},
    )
    search = SearchResult(
        neighbors=np.empty((0, 2), dtype=np.int64),
        distances=np.empty((0, 2), dtype=np.float32),
        search_time_ms=10.0,
        queries_per_second=100.0,
        recall=1.0,
        algorithm=algorithm,
        search_params=[{}],
        metadata={
            "group": "base",
            "index_name": index_name,
            "latency_seconds": 0.01,
        },
    )
    write_results_to_csv(
        [build, search], dataset, str(tmp_path), count=2, batch_size=2
    )

    result_root = tmp_path / dataset / "result"
    exported_files = tuple(result_root.rglob("*.csv"))
    original_contents = {path: path.read_bytes() for path in exported_files}
    dry_build = BuildResult(
        index_path=index_name,
        build_time_seconds=0.0,
        index_size_bytes=0,
        algorithm=algorithm,
        build_params={},
        metadata={"group": "base", "dry_run": True},
    )
    failed_build = BuildResult(
        index_path=index_name,
        build_time_seconds=0.0,
        index_size_bytes=0,
        algorithm=algorithm,
        build_params={},
        metadata={"group": "base"},
        success=False,
        error_message="build failed",
    )
    dry_search = SearchResult(
        neighbors=np.empty((0, 2), dtype=np.int64),
        distances=np.empty((0, 2), dtype=np.float32),
        search_time_ms=0.0,
        queries_per_second=0.0,
        recall=0.0,
        algorithm=algorithm,
        search_params=[{}],
        metadata={
            "group": "base",
            "index_name": index_name,
            "dry_run": True,
        },
    )
    failed_search = SearchResult(
        neighbors=np.empty((0, 2), dtype=np.int64),
        distances=np.empty((0, 2), dtype=np.float32),
        search_time_ms=0.0,
        queries_per_second=0.0,
        recall=0.0,
        algorithm=algorithm,
        search_params=[{}],
        metadata={"group": "base", "index_name": index_name},
        success=False,
        error_message="search failed",
    )
    skipped_search = SearchResult(
        neighbors=np.empty((0, 2), dtype=np.int64),
        distances=np.empty((0, 2), dtype=np.float32),
        search_time_ms=0.0,
        queries_per_second=0.0,
        recall=0.0,
        algorithm=algorithm,
        search_params=[{}],
        metadata={
            "group": "base",
            "index_name": index_name,
            "skipped": True,
        },
    )

    write_results_to_csv(
        [
            dry_build,
            failed_build,
            dry_search,
            failed_search,
            skipped_search,
        ],
        dataset,
        str(tmp_path),
        count=2,
        batch_size=2,
    )

    assert {path: path.read_bytes() for path in exported_files} == (
        original_contents
    )


@pytest.mark.parametrize(
    "incomplete_outcome",
    ["failed", "skipped"],
)
@pytest.mark.parametrize(
    ("index_name", "retained_index_name"),
    [
        pytest.param("test-index", "retained-index", id="text"),
        pytest.param("123", "456", id="numeric-text"),
        pytest.param("001", "002", id="leading-zeros"),
        pytest.param("NA", "null", id="na-like-text"),
    ],
)
def test_partial_search_run_replaces_only_completed_measurements(
    tmp_path, incomplete_outcome, index_name, retained_index_name
):
    dataset = "test-dataset"
    algorithm = "test-backend-algorithm"

    write_results_to_csv(
        [
            _search_result(
                algorithm,
                index_name,
                {"ef_search": 50, "search_width": 16},
                0.5,
            ),
            _search_result(
                algorithm,
                index_name,
                {"ef_search": 50, "search_width": 32},
                1.0,
            ),
            _search_result(
                algorithm,
                retained_index_name,
                {"ef_search": 50, "search_width": 16},
                0.6,
            ),
        ],
        dataset,
        str(tmp_path),
        count=2,
        batch_size=2,
    )
    write_results_to_csv(
        [
            _search_result(
                algorithm,
                index_name,
                {"search_width": 16, "ef_search": 50},
                0.75,
            ),
            _search_result(
                algorithm,
                index_name,
                {"search_width": 32, "ef_search": 50},
                0.0,
                outcome=incomplete_outcome,
            ),
        ],
        dataset,
        str(tmp_path),
        count=2,
        batch_size=2,
    )

    raw_file = (
        tmp_path
        / dataset
        / "result"
        / "search"
        / f"{algorithm},base,k2,bs2,raw.csv"
    )
    measurements = pd.read_csv(
        raw_file, dtype={"index_name": str}, keep_default_na=False
    )
    assert len(measurements) == 3
    identity_columns = ["index_name", "ef_search", "search_width"]
    assert not measurements.duplicated(subset=identity_columns).any()
    measurements_by_identity = measurements.set_index(identity_columns)[
        "recall"
    ].to_dict()
    assert measurements_by_identity == {
        (index_name, 50, 16): 0.75,
        (index_name, 50, 32): 1.0,
        (retained_index_name, 50, 16): 0.6,
    }


def test_failed_search_with_new_parameter_preserves_existing_results(
    tmp_path,
):
    dataset = "test-dataset"
    algorithm = "test-backend-algorithm"
    index_name = "test-index"

    write_results_to_csv(
        [_search_result(algorithm, index_name, {}, 0.5)],
        dataset,
        str(tmp_path),
        count=2,
        batch_size=2,
    )
    write_results_to_csv(
        [
            _search_result(algorithm, index_name, {}, 0.75),
            _search_result(
                algorithm,
                index_name,
                {"ef_search": 100},
                0.0,
                outcome="failed",
            ),
        ],
        dataset,
        str(tmp_path),
        count=2,
        batch_size=2,
    )

    raw_file = (
        tmp_path
        / dataset
        / "result"
        / "search"
        / f"{algorithm},base,k2,bs2,raw.csv"
    )
    measurements = pd.read_csv(raw_file)
    assert measurements["recall"].tolist() == [0.75]
    assert "ef_search" not in measurements


@pytest.mark.parametrize(
    "legacy_csv", [False, True], ids=["current", "legacy"]
)
@pytest.mark.parametrize(
    "completed_index_name",
    ["historical-index", "new-index"],
    ids=["same-index", "different-index"],
)
def test_partial_search_run_preserves_unmatched_parameter_plans(
    tmp_path, legacy_csv, completed_index_name
):
    dataset = "test-dataset"
    algorithm = "test-backend-algorithm"
    historical_index_name = "historical-index"

    write_results_to_csv(
        [
            _search_result(
                algorithm,
                historical_index_name,
                {"ef_search": 50},
                0.5,
            ),
            _search_result(
                algorithm,
                historical_index_name,
                {"ef_search": 100},
                1.0,
            ),
        ],
        dataset,
        str(tmp_path),
        count=2,
        batch_size=2,
    )
    raw_file = (
        tmp_path
        / dataset
        / "result"
        / "search"
        / f"{algorithm},base,k2,bs2,raw.csv"
    )
    if legacy_csv:
        legacy_measurements = pd.read_csv(raw_file).drop(
            columns=[_SEARCH_PARAMS_IDENTITY_COLUMN]
        )
        legacy_measurements.to_csv(raw_file, index=False)

    write_results_to_csv(
        [
            _search_result(algorithm, completed_index_name, {}, 0.75),
            _search_result(
                algorithm,
                "failed-index",
                {},
                0.0,
                outcome="failed",
            ),
        ],
        dataset,
        str(tmp_path),
        count=2,
        batch_size=2,
    )

    measurements = pd.read_csv(raw_file)
    historical_measurements = measurements.loc[
        measurements["ef_search"].notna()
    ]
    assert historical_measurements.set_index("ef_search")[
        "recall"
    ].to_dict() == {50: 0.5, 100: 1.0}
    completed_measurements = measurements.loc[
        (measurements["index_name"] == completed_index_name)
        & measurements["ef_search"].isna()
    ]
    assert completed_measurements["recall"].tolist() == [0.75]
    assert len(measurements) == 3

    latency_frontier = raw_file.with_name(
        raw_file.name.replace(",raw.csv", ",latency.csv")
    )
    assert (
        _SEARCH_PARAMS_IDENTITY_COLUMN
        not in pd.read_csv(latency_frontier).columns
    )


@pytest.mark.parametrize(
    ("numpy_parameters", "native_parameters"),
    [
        pytest.param({"ef_search": np.int64(50)}, {"ef_search": 50}, id="int"),
        pytest.param(
            {"seed": np.uint64(2**64 - 1)},
            {"seed": 2**64 - 1},
            id="uint64-max",
        ),
        pytest.param({"ratio": np.float32(0.5)}, {"ratio": 0.5}, id="float"),
        pytest.param(
            {"enabled": np.bool_(True)}, {"enabled": True}, id="bool"
        ),
        pytest.param(
            {"options": {"weights": [np.float32(0.5)], "limit": np.int64(50)}},
            {"options": {"weights": [0.5], "limit": 50}},
            id="nested",
        ),
    ],
)
@pytest.mark.parametrize(
    "numpy_first", [True, False], ids=["numpy-first", "native-first"]
)
def test_numpy_and_python_parameters_replace_the_same_measurement(
    tmp_path, numpy_parameters, native_parameters, numpy_first
):
    earlier, replacement = (
        (numpy_parameters, native_parameters)
        if numpy_first
        else (native_parameters, numpy_parameters)
    )
    algorithm = "test-backend-algorithm"
    write_results_to_csv(
        [_search_result(algorithm, "test-index", earlier, 1.0)],
        "test-dataset",
        str(tmp_path),
        2,
        2,
    )
    write_results_to_csv(
        [
            _search_result(algorithm, "test-index", replacement, 0.75),
            _search_result(
                algorithm, "failed-index", {}, 0.0, outcome="failed"
            ),
        ],
        "test-dataset",
        str(tmp_path),
        2,
        2,
    )

    search_directory = tmp_path / "test-dataset/result/search"
    for suffix in ("raw", "latency", "throughput"):
        measurements = pd.read_csv(
            search_directory / f"{algorithm},base,k2,bs2,{suffix}.csv"
        )
        assert measurements["recall"].tolist() == [0.75]


@pytest.mark.parametrize(
    "unsupported_value",
    [
        pytest.param(np.longdouble(1), id="extended-float"),
        pytest.param(np.timedelta64(1, "ns"), id="time-delta"),
        pytest.param(object(), id="object"),
    ],
)
def test_unsupported_parameter_types_are_not_silently_coerced(
    tmp_path, unsupported_value
):
    result = _search_result(
        "test-algorithm", "test-index", {"value": unsupported_value}, 0.5
    )

    with pytest.raises(TypeError, match="Unsupported search parameter type"):
        write_results_to_csv([result], "test-dataset", str(tmp_path), 2, 2)


def _run_sweep_with_second_build_outcome(algorithm, second_build_outcome):
    """Use the real orchestrator to decide whether the second index is searched."""
    index_names = ("completed-index", "second-index")

    class SweepLoader:
        def load(self, **kwargs):
            return DatasetConfig(name="test-dataset"), [
                BenchmarkConfig(
                    indexes=[IndexConfig(name, algorithm, {}, [{}], name)],
                    backend_config={"name": name},
                )
                for name in index_names
            ]

    class BuildOutcomeBackend(BenchmarkBackend):
        @property
        def algo(self):
            return algorithm

        def build(self, dataset, indexes, force=False, dry_run=False):
            outcome = (
                second_build_outcome
                if self.name == "second-index"
                else "completed"
            )
            return BuildResult(
                index_path=self.name,
                build_time_seconds=1.5,
                index_size_bytes=1024,
                algorithm=self.algo,
                build_params={},
                metadata={
                    "group": "base",
                    "index_name": self.name,
                    "skipped": outcome == "cached",
                },
                success=outcome != "failed",
                error_message="build failed" if outcome == "failed" else None,
            )

        def search(self, dataset, indexes, k, **kwargs):
            return [_search_result(self.algo, self.name, {}, 0.75)]

    orchestrator = BenchmarkOrchestrator("cpp_gbench")
    orchestrator.config_loader = SweepLoader()
    orchestrator.backend_class = BuildOutcomeBackend
    return orchestrator.run_benchmark(
        build=True, search=True, count=2, batch_size=2
    )


@pytest.mark.parametrize(
    "second_build_outcome", ["failed", "cached", "completed"]
)
def test_build_outcome_controls_preservation_of_previous_searches(
    tmp_path, second_build_outcome
):
    algorithm = "test-backend-algorithm"
    old_recalls = {
        "completed-index": 0.5,
        "second-index": 1.0,
        "historical-only": 0.6,
    }
    write_results_to_csv(
        [
            _search_result(algorithm, name, {}, recall)
            for name, recall in old_recalls.items()
        ],
        "test-dataset",
        str(tmp_path),
        2,
        2,
    )

    results = _run_sweep_with_second_build_outcome(
        algorithm, second_build_outcome
    )
    searched_names = [
        result.metadata["index_name"]
        for result in results
        if isinstance(result, SearchResult)
    ]
    expected_recalls = {"completed-index": 0.75, "second-index": 0.75}
    if second_build_outcome == "failed":
        assert searched_names == ["completed-index"]
        expected_recalls = {**old_recalls, "completed-index": 0.75}
    else:
        assert searched_names == ["completed-index", "second-index"]

    write_results_to_csv(results, "test-dataset", str(tmp_path), 2, 2)

    raw_file = (
        tmp_path
        / "test-dataset/result/search"
        / f"{algorithm},base,k2,bs2,raw.csv"
    )
    measurements = pd.read_csv(raw_file)
    assert (
        measurements.set_index("index_name")["recall"].to_dict()
        == expected_recalls
    )
    assert len(measurements) == len(expected_recalls)


def test_tune_trial_retains_build_and_search_results():
    algorithm = "opensearch_faiss_hnsw"
    index = IndexConfig(
        name="test-index",
        algo=algorithm,
        build_param={"m": 16},
        search_params=[{"ef_search": 100}],
        file="test-index",
    )
    dataset = DatasetConfig(name="test-dataset")

    class FakeLoader:
        def load(self, **kwargs):
            return dataset, [
                BenchmarkConfig(
                    indexes=[index],
                    backend_config={"name": index.name, "group": "base"},
                )
            ]

    class FakeBackend:
        def __init__(self, config):
            pass

        def initialize(self):
            pass

        def cleanup(self):
            pass

        def build(self, dataset, indexes, force, dry_run):
            return BuildResult(
                index_path=index.name,
                build_time_seconds=1.5,
                index_size_bytes=1024,
                algorithm=algorithm,
                build_params=index.build_param,
                metadata={"group": "base"},
            )

        def search(self, dataset, indexes, k, **kwargs):
            return [
                SearchResult(
                    neighbors=np.array([[0, 1]], dtype=np.int64),
                    distances=np.zeros((1, k), dtype=np.float32),
                    search_time_ms=10.0,
                    queries_per_second=100.0,
                    recall=0.0,
                    algorithm=algorithm,
                    search_params=index.search_params,
                    metadata={
                        "group": "base",
                        "index_name": index.name,
                    },
                )
            ]

    orchestrator = BenchmarkOrchestrator(backend_type="opensearch")
    orchestrator.config_loader = FakeLoader()
    orchestrator.backend_class = FakeBackend
    orchestrator._create_dataset = lambda config: type(
        "Dataset",
        (),
        {"groundtruth_neighbors": np.array([[0, 1]], dtype=np.int64)},
    )()

    results = orchestrator._run_trial(
        algorithm=algorithm,
        build_params=index.build_param,
        search_params=index.search_params[0],
        build=True,
        search=True,
        force=False,
        dry_run=False,
        count=2,
        batch_size=1,
        search_mode="latency",
        search_threads=None,
    )

    assert [type(result) for result in results] == [BuildResult, SearchResult]
    assert results[1].recall == 1.0
