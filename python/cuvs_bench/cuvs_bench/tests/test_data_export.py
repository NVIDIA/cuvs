#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pandas as pd
import pytest

from cuvs_bench.backends.base import BuildResult, SearchResult
from cuvs_bench.orchestrator.config_loaders import (
    BenchmarkConfig,
    DatasetConfig,
    IndexConfig,
)
from cuvs_bench.orchestrator.orchestrator import BenchmarkOrchestrator
from cuvs_bench.plot.__main__ import load_all_results
from cuvs_bench.run.data_export import (
    validate_dataset_name,
    write_results_to_csv,
)


def test_python_backend_csv_is_plot_compatible(tmp_path):
    dataset = "test-dataset"
    algorithm = "opensearch_faiss_hnsw"
    index_name = "test-index"
    results = [
        BuildResult(
            index_path="/tmp/physical-index-path",
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
            latency_percentiles={"p50": 8.0, "p95": 12.0, "p99": 15.0},
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
    assert raw.loc[0, ["p50", "p95", "p99"]].tolist() == [8.0, 12.0, 15.0]
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


def test_subset_scopes_use_independent_result_files(tmp_path):
    dataset = "test-dataset"
    algorithm = "pylucene_cuvs_hnsw"

    for scope in ("subset100", "subset200"):
        index_name = f"test-index-{scope}"
        write_results_to_csv(
            [
                BuildResult(
                    index_path=f"/tmp/{index_name}",
                    build_time_seconds=1.0,
                    index_size_bytes=100,
                    algorithm=algorithm,
                    build_params={},
                    metadata={
                        "group": "base",
                        "index_name": index_name,
                        "result_scope": scope,
                    },
                ),
                SearchResult(
                    neighbors=np.empty((0, 1), dtype=np.int64),
                    distances=np.empty((0, 1), dtype=np.float32),
                    search_time_ms=1.0,
                    queries_per_second=10.0,
                    recall=1.0,
                    algorithm=algorithm,
                    search_params=[{}],
                    metadata={
                        "group": "base",
                        "index_name": index_name,
                        "result_scope": scope,
                    },
                ),
            ],
            dataset,
            str(tmp_path),
            count=1,
            batch_size=1,
        )

    result_path = tmp_path / dataset / "result"
    for scope in ("subset100", "subset200"):
        assert (
            result_path / "build" / f"{algorithm},base,{scope}.csv"
        ).is_file()
        assert (
            result_path / "search" / f"{algorithm},base,{scope},k1,bs1,raw.csv"
        ).is_file()

    plotted = load_all_results(
        str(tmp_path / dataset),
        algorithms=[algorithm],
        groups=["base"],
        algo_groups=[],
        k=1,
        batch_size=1,
        method="search",
        index_key="algo",
        raw=True,
        mode="latency",
        time_unit="s",
    )
    assert set(plotted) == {
        f"{algorithm}[scope=subset100]",
        f"{algorithm}[scope=subset200]",
    }


def test_algorithm_and_group_names_remain_distinct_in_plot_series(tmp_path):
    dataset = "test-dataset"
    identities = (
        ("a_b", "c", "first-index"),
        ("a", "b_c", "second-index"),
    )

    for algorithm, group, index_name in identities:
        write_results_to_csv(
            [
                SearchResult(
                    neighbors=np.empty((0, 1), dtype=np.int64),
                    distances=np.empty((0, 1), dtype=np.float32),
                    search_time_ms=1.0,
                    queries_per_second=10.0,
                    recall=1.0,
                    algorithm=algorithm,
                    search_params=[{}],
                    metadata={"group": group, "index_name": index_name},
                )
            ],
            dataset,
            str(tmp_path),
            count=1,
            batch_size=1,
        )

    plotted = load_all_results(
        str(tmp_path / dataset),
        algorithms=["a_b", "a"],
        groups=["c", "b_c"],
        algo_groups=[],
        k=1,
        batch_size=1,
        method="search",
        index_key="algo",
        raw=True,
        mode="latency",
        time_unit="s",
    )

    assert set(plotted) == {"a_b[group=c]", "a[group=b_c]"}


def test_failed_build_removes_only_matching_stale_search_files(tmp_path):
    dataset = "test-dataset"
    algorithm = "pylucene_cuvs_hnsw"
    search_dir = tmp_path / dataset / "result" / "search"
    search_dir.mkdir(parents=True)

    matching_stem = f"{algorithm},base,subset100,k1,bs1"
    other_stem = f"{algorithm},base,subset200,k1,bs1"
    suffixes = (",raw.csv", ",throughput.csv", ",latency.csv")
    for stem in (matching_stem, other_stem):
        for suffix in suffixes:
            (search_dir / f"{stem}{suffix}").write_text("stale\n")

    write_results_to_csv(
        [
            BuildResult(
                index_path="/tmp/failed-index",
                build_time_seconds=0.0,
                index_size_bytes=0,
                algorithm=algorithm,
                build_params={},
                metadata={
                    "group": "base",
                    "index_name": "failed-index",
                    "result_scope": "subset100",
                },
                success=False,
                error_message="build failed",
            )
        ],
        dataset,
        str(tmp_path),
        count=1,
        batch_size=1,
        search_requested=True,
    )

    assert not list(search_dir.glob(f"{matching_stem}*"))
    assert len(list(search_dir.glob(f"{other_stem}*"))) == 3


@pytest.mark.parametrize("dataset", ["", ".", "..", "nested/dataset"])
def test_dataset_name_must_be_a_safe_path_component(dataset):
    with pytest.raises(ValueError, match="Invalid benchmark dataset name"):
        validate_dataset_name(dataset)


@pytest.mark.parametrize(
    ("algorithm", "metadata", "match"),
    [
        ("../pylucene", {"group": "base"}, "benchmark algorithm name"),
        ("pylucene,other", {"group": "base"}, "benchmark algorithm name"),
        ("pylucene[other]", {"group": "base"}, "benchmark algorithm name"),
        ("pylucene", {"group": "../base"}, "benchmark group name"),
        ("pylucene", {"group": "base\nother"}, "benchmark group name"),
        (
            "pylucene",
            {"group": "base", "result_scope": "../subset100"},
            "benchmark result scope",
        ),
    ],
)
def test_result_identity_must_use_safe_path_components(
    tmp_path, algorithm, metadata, match
):
    result = BuildResult(
        index_path="index",
        build_time_seconds=1.0,
        index_size_bytes=1,
        algorithm=algorithm,
        build_params={},
        metadata=metadata,
    )

    with pytest.raises(ValueError, match=match):
        write_results_to_csv(
            [result], "test-dataset", str(tmp_path), count=1, batch_size=1
        )
