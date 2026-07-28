#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pandas as pd
from click.testing import CliRunner

from cuvs_bench.backends.base import BuildResult, SearchResult
from cuvs_bench.orchestrator.config_loaders import (
    BenchmarkConfig,
    DatasetConfig,
    IndexConfig,
)
from cuvs_bench.orchestrator.orchestrator import BenchmarkOrchestrator
from cuvs_bench.plot.__main__ import load_all_results
from cuvs_bench.run.__main__ import main
from cuvs_bench.run.data_export import write_results_to_csv


def test_python_backend_csv_is_plot_compatible(tmp_path):
    dataset = "test-dataset"
    algorithm = "opensearch_faiss_hnsw"
    index_name = "test-index"
    results = [
        BuildResult(
            index_path=index_name,
            build_time_seconds=1.5,
            index_size_bytes=1024,
            algorithm=algorithm,
            build_params={"m": 16},
            metadata={"group": "base"},
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
    raw = pd.read_csv(raw_file)
    assert raw.columns[:5].tolist() == [
        "algo_name",
        "index_name",
        "recall",
        "throughput",
        "latency",
    ]
    assert raw["ef_search"].tolist() == [50, 100]
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


def test_run_command_always_writes_python_backend_csv(tmp_path, monkeypatch):
    algorithm = "opensearch_faiss_hnsw"
    result = SearchResult(
        neighbors=np.empty((0, 2), dtype=np.int64),
        distances=np.empty((0, 2), dtype=np.float32),
        search_time_ms=10.0,
        queries_per_second=200.0,
        recall=1.0,
        algorithm=algorithm,
        search_params=[{"ef_search": 100}],
        metadata={
            "group": "base",
            "index_name": "test-index",
            "latency_seconds": 0.01,
        },
    )

    class FakeOrchestrator:
        def __init__(self, backend_type):
            assert backend_type == "opensearch"

        def run_benchmark(self, **kwargs):
            return [result]

    monkeypatch.setattr(
        "cuvs_bench.run.__main__.BenchmarkOrchestrator", FakeOrchestrator
    )
    backend_config = tmp_path / "backend.yaml"
    backend_config.write_text("backend: opensearch\n")

    cli_result = CliRunner().invoke(
        main,
        [
            "--dataset",
            "test-dataset",
            "--dataset-path",
            str(tmp_path),
            "--algorithms",
            algorithm,
            "--groups",
            "base",
            "--count",
            "2",
            "--batch-size",
            "2",
            "--search-mode",
            "latency",
            "--search",
            "--backend-config",
            str(backend_config),
        ],
    )

    assert cli_result.exit_code == 0, cli_result.output
    assert (
        tmp_path
        / "test-dataset"
        / "result"
        / "search"
        / f"{algorithm},base,k2,bs2,raw.csv"
    ).exists()
    help_output = CliRunner().invoke(main, ["--help"]).output
    assert "--data-export" in help_output
    assert "Deprecated" in help_output


def test_data_export_option_is_deprecated(tmp_path, monkeypatch):
    converted = []
    monkeypatch.setattr(
        "cuvs_bench.run.__main__.convert_json_to_csv_build",
        lambda dataset, dataset_path: converted.append("build"),
    )
    monkeypatch.setattr(
        "cuvs_bench.run.__main__.convert_json_to_csv_search",
        lambda dataset, dataset_path: converted.append("search"),
    )

    cli_result = CliRunner().invoke(
        main,
        [
            "--dataset",
            "test-dataset",
            "--dataset-path",
            str(tmp_path),
            "--algorithms",
            "cuvs_cagra",
            "--groups",
            "base",
            "--count",
            "2",
            "--batch-size",
            "2",
            "--search-mode",
            "latency",
            "--data-export",
        ],
    )

    assert cli_result.exit_code == 0, cli_result.output
    assert "--data-export is deprecated" in cli_result.output
    assert converted == ["build", "search"]


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

    orchestrator = object.__new__(BenchmarkOrchestrator)
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
