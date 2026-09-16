#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Result, parameter, and configuration contracts for the Lucene backend."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from _lucene_test_support import (
    ALGORITHM_CASES,
    RecordingRuntime,
    _backend_and_index,
    _dataset,
)
from cuvs_bench.backends._lucene_runtime import CAGRA_CODEC, CPU_HNSW_CODEC
from cuvs_bench.backends.lucene import (
    CAGRA_ALGORITHM,
    CPU_HNSW_ALGORITHM,
    LuceneBackend,
    LuceneConfigLoader,
    _codec_for,
    _score_to_squared_euclidean,
    _search_parameters,
)
from cuvs_bench.run.data_export import write_results_to_csv


@pytest.mark.parametrize(("algorithm", "codec", "_path"), ALGORITHM_CASES)
def test_backend_accepts_only_the_codec_owned_by_each_algorithm(
    tmp_path: Path, algorithm: str, codec: str, _path: str
) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(tmp_path, algorithm, runtime)

    assert backend.algorithm == algorithm
    assert backend.codec == codec
    assert _codec_for(index.algo, index.build_param) == codec


@pytest.mark.parametrize(
    ("algorithm", "codec"),
    (
        pytest.param(CPU_HNSW_ALGORITHM, CAGRA_CODEC, id="cpu-with-cagra"),
        pytest.param(CAGRA_ALGORITHM, CPU_HNSW_CODEC, id="cagra-with-cpu"),
    ),
)
def test_backend_rejects_mismatched_algorithm_and_codec(
    tmp_path: Path, algorithm: str, codec: str
) -> None:
    with pytest.raises(ValueError, match="Invalid Lucene algorithm/codec pair"):
        LuceneBackend(
            {
                "name": "invalid",
                "algo": algorithm,
                "codec": codec,
                "index_root": str(tmp_path),
            }
        )


@pytest.mark.parametrize(("algorithm", "codec", "_path"), ALGORITHM_CASES)
def test_build_parameters_accept_only_the_fixed_codec(
    algorithm: str, codec: str, _path: str
) -> None:
    assert _codec_for(algorithm, {}) == codec
    assert _codec_for(algorithm, {"codec": codec}) == codec

    with pytest.raises(ValueError, match="Unsupported Lucene build parameters"):
        _codec_for(algorithm, {"codec": codec, "graph_degree": 32})


@pytest.mark.parametrize(
    ("parameters", "k", "expected"),
    (
        pytest.param({}, 3, {"num_candidates": 3}, id="default-to-k"),
        pytest.param(
            {"num_candidates": 7},
            3,
            {"num_candidates": 7},
            id="explicit-candidates",
        ),
    ),
)
def test_cpu_search_parameter_validation_accepts_candidate_budgets(
    parameters: dict[str, int], k: int, expected: dict[str, int]
) -> None:
    assert _search_parameters(CPU_HNSW_ALGORITHM, parameters, k) == expected


@pytest.mark.parametrize(
    "parameters",
    (
        pytest.param({"num_candidates": 2}, id="below-k"),
        pytest.param({"num_candidates": True}, id="boolean"),
        pytest.param({"search_width": 16}, id="unsupported-key"),
    ),
)
def test_cpu_search_parameter_validation_rejects_invalid_budgets(
    parameters: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        _search_parameters(CPU_HNSW_ALGORITHM, parameters, 3)


def test_cagra_search_accepts_only_fixed_parameters_within_its_route_limit() -> None:
    assert _search_parameters(CAGRA_ALGORITHM, {}, 1024) == {"num_candidates": 1024}

    with pytest.raises(ValueError, match="accepts no search parameters"):
        _search_parameters(CAGRA_ALGORITHM, {"search_width": 16}, 10)
    with pytest.raises(ValueError, match=r"supports k <= 1024"):
        _search_parameters(CAGRA_ALGORITHM, {}, 1025)


@pytest.mark.parametrize(
    ("score", "expected_distance"),
    (
        pytest.param(1.0, 0.0, id="zero-distance"),
        pytest.param(0.5, 1.0, id="unit-distance"),
        pytest.param(0.2, 4.0, id="distance-four"),
    ),
)
def test_lucene_scores_are_inverted_to_squared_euclidean_distance(
    score: float, expected_distance: float
) -> None:
    assert _score_to_squared_euclidean(score) == pytest.approx(expected_distance)


@pytest.mark.parametrize(
    "score",
    (
        0.0,
        -1.0,
        1.0 + 2.0 * float(np.spacing(np.float32(1.0))),
        np.nan,
        np.inf,
    ),
)
def test_score_inversion_rejects_values_outside_lucenes_score_domain(
    score: float,
) -> None:
    with pytest.raises(RuntimeError, match="invalid Euclidean score"):
        _score_to_squared_euclidean(score)


def test_score_inversion_tolerates_float32_roundoff_above_one() -> None:
    score = float(np.nextafter(np.float32(1.0), np.float32(2.0)))

    assert _score_to_squared_euclidean(score) == 0.0


def test_result_metadata_is_exported_instead_of_silently_dropped(
    tmp_path: Path,
) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(tmp_path, CPU_HNSW_ALGORITHM, runtime)
    dataset = _dataset()
    build = backend.build(dataset, [index])
    search = backend.search(dataset, [index], k=2, batch_size=2)[0]

    write_results_to_csv(
        [build, search], dataset.name, str(tmp_path), count=2, batch_size=2
    )

    result_root = tmp_path / dataset.name / "result"
    build_csv = result_root / "build" / "lucene_cpu_hnsw,test.csv"
    search_csv = result_root / "search" / "lucene_cpu_hnsw,test,k2,bs2,raw.csv"
    assert build_csv.is_file()
    assert search_csv.is_file()
    with search_csv.open(newline="", encoding="utf-8") as stream:
        [row] = csv.DictReader(stream)
    assert row["index_name"] == CPU_HNSW_ALGORITHM
    assert float(row["build time"]) == pytest.approx(build.build_time_seconds)


def test_dry_and_failed_exports_preserve_existing_measurements(
    tmp_path: Path,
) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(tmp_path, CPU_HNSW_ALGORITHM, runtime)
    dataset = _dataset()
    build = backend.build(dataset, [index])
    search = backend.search(dataset, [index], k=2, batch_size=2)[0]
    assert build.success and search.success
    write_results_to_csv(
        [build, search], dataset.name, str(tmp_path), count=2, batch_size=2
    )

    result_root = tmp_path / dataset.name / "result"
    exported_files = (
        result_root / "build" / "lucene_cpu_hnsw,test.csv",
        result_root / "search" / "lucene_cpu_hnsw,test,k2,bs2,raw.csv",
        result_root / "search" / "lucene_cpu_hnsw,test,k2,bs2,throughput.csv",
        result_root / "search" / "lucene_cpu_hnsw,test,k2,bs2,latency.csv",
    )
    original_contents = {path: path.read_bytes() for path in exported_files}

    dry_build = backend.build(dataset, [index], dry_run=True)
    dry_search = backend.search(dataset, [index], k=2, dry_run=True)[0]
    runtime.build_error = RuntimeError("replacement build failed")
    failed_build = backend.build(dataset, [index], force=True)
    runtime.search_error = RuntimeError("replacement search failed")
    failed_search = backend.search(dataset, [index], k=2)[0]
    assert not failed_build.success
    assert not failed_search.success

    write_results_to_csv(
        [dry_build, dry_search, failed_build, failed_search],
        dataset.name,
        str(tmp_path),
        count=2,
        batch_size=2,
    )

    assert {path: path.read_bytes() for path in exported_files} == (original_contents)


def test_config_loader_maps_each_algorithm_to_its_codec_and_requirement(
    tmp_path: Path,
) -> None:
    dataset_configuration = tmp_path / "datasets.yaml"
    dataset_configuration.write_text(
        "- name: tiny-l2\n  distance: euclidean\n  dims: 2\n",
        encoding="utf-8",
    )
    loader = LuceneConfigLoader()

    _dataset_config, configurations = loader.load(
        dataset="tiny-l2",
        dataset_path=str(tmp_path),
        dataset_configuration=str(dataset_configuration),
        algorithms=f"{CPU_HNSW_ALGORITHM},{CAGRA_ALGORITHM}",
        groups="test",
    )
    by_algorithm = {
        configuration.indexes[0].algo: configuration for configuration in configurations
    }

    assert set(by_algorithm) == {CPU_HNSW_ALGORITHM, CAGRA_ALGORITHM}
    assert by_algorithm[CPU_HNSW_ALGORITHM].indexes[0].build_param == {
        "codec": CPU_HNSW_CODEC
    }
    assert by_algorithm[CAGRA_ALGORITHM].indexes[0].build_param == {
        "codec": CAGRA_CODEC
    }
    assert by_algorithm[CPU_HNSW_ALGORITHM].backend_config["requires_cuvs"] is False
    assert by_algorithm[CAGRA_ALGORITHM].backend_config["requires_cuvs"] is True
    assert by_algorithm[CPU_HNSW_ALGORITHM].backend_config["include_cuvs"] is True
    assert by_algorithm[CAGRA_ALGORITHM].backend_config["include_cuvs"] is True
    assert by_algorithm[CPU_HNSW_ALGORITHM].backend_config["group"] == "test"
    assert by_algorithm[CAGRA_ALGORITHM].backend_config["group"] == "test"


def test_config_loader_rejects_dataset_names_that_can_escape_the_root(
    tmp_path: Path,
) -> None:
    dataset_configuration = tmp_path / "datasets.yaml"
    dataset_configuration.write_text(
        "- name: ../escape\n  distance: euclidean\n  dims: 2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsafe Lucene dataset"):
        LuceneConfigLoader().load(
            dataset="../escape",
            dataset_path=str(tmp_path),
            dataset_configuration=str(dataset_configuration),
            algorithms=CPU_HNSW_ALGORITHM,
            groups="test",
        )


def test_cpu_only_config_does_not_include_optional_cuvs_artifacts(
    tmp_path: Path,
) -> None:
    dataset_configuration = tmp_path / "datasets.yaml"
    dataset_configuration.write_text(
        "- name: tiny-l2\n  distance: euclidean\n  dims: 2\n",
        encoding="utf-8",
    )

    _dataset_config, [configuration] = LuceneConfigLoader().load(
        dataset="tiny-l2",
        dataset_path=str(tmp_path),
        dataset_configuration=str(dataset_configuration),
        algorithms=CPU_HNSW_ALGORITHM,
        groups="test",
    )

    assert configuration.backend_config["requires_cuvs"] is False
    assert configuration.backend_config["include_cuvs"] is False
