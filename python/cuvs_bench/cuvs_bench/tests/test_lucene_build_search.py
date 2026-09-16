#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Build and search behavior for the opt-in Lucene benchmark backend."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from _lucene_test_support import (
    ALGORITHM_CASES,
    _FAKE_ARTIFACT_PROVENANCE,
    RecordingRuntime,
    RecordingRuntimeFactory,
    _backend_and_index,
    _dataset,
    _dataset_with_dimensions,
    _write_fbin,
)
from cuvs_bench.backends._lucene_runtime import RuntimeSearchResult, SearchHit
from cuvs_bench.backends.base import Dataset
from cuvs_bench.backends.lucene import (
    CAGRA_ALGORITHM,
    CPU_HNSW_ALGORITHM,
    MAX_CAGRA_DIMENSIONS,
    MAX_CPU_HNSW_DIMENSIONS,
    LuceneBackend,
    _file_backed_dataset_identity,
)
from cuvs_bench.orchestrator.config_loaders import IndexConfig


@pytest.mark.parametrize(("algorithm", "codec", "path"), ALGORITHM_CASES)
def test_successful_build_reports_the_verified_persisted_index_kind(
    tmp_path: Path, algorithm: str, codec: str, path: str
) -> None:
    runtime = RecordingRuntime()
    backend, index, factory = _backend_and_index(tmp_path, algorithm, runtime)

    result = backend.build(_dataset(), [index])

    assert result.success, result.error_message
    assert result.algorithm == algorithm
    assert result.build_params == {"codec": codec}
    assert result.metadata["codec"] == codec
    assert result.metadata["persisted_index_kind"] == (
        "gpu_cagra_only" if algorithm == CAGRA_ALGORITHM else path
    )
    assert result.metadata["segment_count"] == 1
    assert result.metadata["pylucene_version"] == "10.2.0"
    assert result.index_size_bytes > 0
    assert len(factory.calls) == 1
    assert [call[2] for call in runtime.build_calls] == [codec]
    if algorithm == CAGRA_ALGORITHM:
        [(verified_path, vector_count, dimensions)] = runtime.cagra_verifier.calls
        destination = Path(index.file).resolve()
        assert verified_path.parent == destination.parent
        assert verified_path.name.startswith(f".{destination.name}.build-")
        assert not verified_path.exists()
        assert (vector_count, dimensions) == (4, 2)
    else:
        assert runtime.cagra_verifier.calls == []


@pytest.mark.parametrize(
    ("algorithm", "dimensions", "maximum_dimensions", "expected_success"),
    (
        pytest.param(
            CPU_HNSW_ALGORITHM,
            1024,
            MAX_CPU_HNSW_DIMENSIONS,
            True,
            id="cpu-maximum-dimensions",
        ),
        pytest.param(
            CPU_HNSW_ALGORITHM,
            1025,
            MAX_CPU_HNSW_DIMENSIONS,
            False,
            id="cpu-above-maximum-dimensions",
        ),
        pytest.param(
            CAGRA_ALGORITHM,
            1025,
            MAX_CAGRA_DIMENSIONS,
            True,
            id="cagra-above-cpu-maximum",
        ),
        pytest.param(
            CAGRA_ALGORITHM,
            4096,
            MAX_CAGRA_DIMENSIONS,
            True,
            id="cagra-maximum-dimensions",
        ),
        pytest.param(
            CAGRA_ALGORITHM,
            4097,
            MAX_CAGRA_DIMENSIONS,
            False,
            id="cagra-above-maximum-dimensions",
        ),
    ),
)
def test_build_enforces_each_algorithms_dimension_limit(
    tmp_path: Path,
    algorithm: str,
    dimensions: int,
    maximum_dimensions: int,
    expected_success: bool,
) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(tmp_path, algorithm, runtime)

    result = backend.build(_dataset_with_dimensions(dimensions), [index])

    assert result.success is expected_success
    if expected_success:
        assert runtime.dimensions == dimensions
    else:
        assert result.error_message == (
            "ValueError: training vectors dimensions must not exceed "
            f"{maximum_dimensions}"
        )
        assert runtime.build_calls == []


def test_file_backed_identity_rejects_cpu_dimensions_before_hashing(
    tmp_path: Path,
) -> None:
    vectors = np.zeros((2, MAX_CPU_HNSW_DIMENSIONS + 1), dtype=np.float32)
    source = tmp_path / "base.fbin"
    _write_fbin(source, vectors)
    dataset = _dataset_with_dimensions(2)
    dataset.base_file = str(source)

    with pytest.raises(ValueError, match="must not exceed 1024"):
        _file_backed_dataset_identity(
            dataset,
            {},
            maximum_dimensions=MAX_CPU_HNSW_DIMENSIONS,
        )


def test_numpy_subset_size_is_normalized_before_manifest_serialization(
    tmp_path: Path,
) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(tmp_path, CPU_HNSW_ALGORITHM, runtime)
    vectors = _dataset().training_vectors
    base_file = tmp_path / "base.fbin"
    _write_fbin(base_file, vectors)
    dataset = Dataset(
        name="tiny-l2-subset",
        query_vectors=vectors[:2].copy(),
        distance_metric="euclidean",
        base_file=str(base_file),
        metadata={"subset_size": np.int64(3)},
    )

    first_result = backend.build(dataset, [index])
    second_result = backend.build(dataset, [index])

    assert first_result.success, first_result.error_message
    assert second_result.success, second_result.error_message
    assert second_result.metadata["skipped"] is True
    assert runtime.document_count == 3
    assert len(runtime.build_calls) == 1
    assert type(dataset.metadata["subset_size"]) is int
    manifest = json.loads(
        (Path(index.file) / ".cuvs-bench-lucene.json").read_text(encoding="utf-8")
    )
    assert manifest["dataset"]["subset_size"] == 3


def test_boolean_subset_size_is_rejected_before_runtime_initialization(
    tmp_path: Path,
) -> None:
    runtime = RecordingRuntime()
    backend, index, factory = _backend_and_index(tmp_path, CPU_HNSW_ALGORITHM, runtime)
    dataset = _dataset()
    dataset.metadata = {"subset_size": True}

    result = backend.build(dataset, [index])

    assert not result.success
    assert result.error_message == (
        "ValueError: subset_size must be a positive integer, got True"
    )
    assert factory.calls == []
    assert runtime.build_calls == []
    assert not Path(index.file).exists()


@pytest.mark.parametrize(("algorithm", "codec", "path"), ALGORITHM_CASES)
def test_successful_search_reports_path_parameters_and_distances(
    tmp_path: Path, algorithm: str, codec: str, path: str
) -> None:
    runtime = RecordingRuntime()
    search_params = [{"num_candidates": 4}] if algorithm == CPU_HNSW_ALGORITHM else [{}]
    backend, index, _factory = _backend_and_index(
        tmp_path,
        algorithm,
        runtime,
        search_params=search_params,
    )
    build_result = backend.build(_dataset(), [index])
    assert build_result.success, build_result.error_message

    result = backend.search(_dataset(), [index], k=2, batch_size=1)[0]

    assert result.success, result.error_message
    np.testing.assert_array_equal(result.neighbors, [[0, 1], [0, 1]])
    np.testing.assert_allclose(result.distances, [[0.0, 1.0], [0.0, 1.0]])
    assert result.metadata == {
        "codec": codec,
        "pylucene_version": "10.2.0",
        "latency_seconds": 0.002,
        "batch_count": 2,
        "batch_size": 1,
        "mode": "latency",
        "group": "test",
        "index_name": algorithm,
        **(
            {
                f"{role}_{key}": value
                for role in ("build_runtime", "search_runtime")
                for key, value in _FAKE_ARTIFACT_PROVENANCE.items()
            }
            if algorithm == CAGRA_ALGORITHM
            else {}
        ),
        "expected_search_route": path,
        "persisted_index_kind": (
            "gpu_cagra_only" if algorithm == CAGRA_ALGORITHM else path
        ),
        "segment_count": 1,
        "field_count": 1,
        "vector_count": 4,
        "dimensions": 2,
    }
    assert result.search_params == (
        [{"num_candidates": 4}] if algorithm == CPU_HNSW_ALGORITHM else [{}]
    )
    assert runtime.search_calls[0]["num_candidates"] == (
        4 if algorithm == CPU_HNSW_ALGORITHM else 2
    )


def test_search_parameter_sweep_verifies_artifacts_once(
    tmp_path: Path,
) -> None:
    runtime = RecordingRuntime()
    search_params = [{"num_candidates": 2}, {"num_candidates": 4}]
    backend, index, _factory = _backend_and_index(
        tmp_path,
        CPU_HNSW_ALGORITHM,
        runtime,
        search_params=search_params,
    )
    assert backend.build(_dataset(), [index]).success
    verifications_after_build = runtime.artifact_verification_count

    results = backend.search(_dataset(), [index], k=2, batch_size=2)

    assert all(result.success for result in results)
    assert len(results) == len(search_params)
    assert runtime.artifact_verification_count == verifications_after_build + 1
    assert [search_call["num_candidates"] for search_call in runtime.search_calls] == [
        2,
        4,
    ]


def test_search_latency_reports_complete_batches(
    tmp_path: Path,
) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(tmp_path, CPU_HNSW_ALGORITHM, runtime)
    dataset = _dataset()
    assert backend.build(dataset, [index]).success
    runtime.search_result = RuntimeSearchResult(
        hits=[
            [SearchHit(0, 1.0), SearchHit(1, 0.5)],
            [SearchHit(0, 1.0), SearchHit(1, 0.5)],
        ],
        batch_latencies_ms=[8.0],
        document_count=4,
        dimensions=2,
    )

    result = backend.search(dataset, [index], k=2, batch_size=2)[0]

    assert result.success, result.error_message
    assert result.latency_percentiles == {"p50": 8.0, "p95": 8.0, "p99": 8.0}
    assert result.metadata["latency_seconds"] == 0.008
    assert result.metadata["batch_count"] == 1


@pytest.mark.parametrize(
    ("hits", "message"),
    (
        pytest.param(
            [[SearchHit(0, 1.0), SearchHit(0, 0.5)]] * 2,
            "duplicate IDs",
            id="duplicate",
        ),
        pytest.param(
            [[SearchHit(0, 1.0), SearchHit(4, 0.5)]] * 2,
            "an invalid ID",
            id="out-of-range",
        ),
    ),
)
def test_search_rejects_invalid_hit_identifiers(
    tmp_path: Path, hits: list[list[SearchHit]], message: str
) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(tmp_path, CPU_HNSW_ALGORITHM, runtime)
    assert backend.build(_dataset(), [index]).success
    runtime.search_result = RuntimeSearchResult(
        hits=hits,
        batch_latencies_ms=[2.0],
        document_count=4,
        dimensions=2,
    )

    result = backend.search(_dataset(), [index], k=2)[0]

    assert not result.success
    assert f"RuntimeError: Lucene query 0 returned {message}" in result.error_message


def test_search_rejects_a_runtime_result_that_drops_a_query(
    tmp_path: Path,
) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(tmp_path, CPU_HNSW_ALGORITHM, runtime)
    dataset = _dataset()
    assert backend.build(dataset, [index]).success
    runtime.search_result = RuntimeSearchResult(
        hits=[[SearchHit(0, 1.0), SearchHit(1, 0.5)]],
        batch_latencies_ms=[2.0],
        document_count=4,
        dimensions=2,
    )

    result = backend.search(dataset, [index], k=2)[0]

    assert not result.success
    assert result.error_message == (
        "RuntimeError: Lucene returned results for 1 queries, expected 2"
    )


def test_cagra_search_above_1024_fails_before_issuing_a_query(
    tmp_path: Path,
) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(tmp_path, CAGRA_ALGORITHM, runtime)
    assert backend.build(_dataset(), [index]).success

    result = backend.search(_dataset(), [index], k=1025)[0]

    assert not result.success
    assert "CAGRA search supports k <= 1024" in result.error_message
    assert runtime.search_calls == []


def test_invalid_top_k_returns_the_validation_error_without_a_secondary_failure(
    tmp_path: Path,
) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(tmp_path, CPU_HNSW_ALGORITHM, runtime)

    result = backend.search(_dataset(), [index], k=-1)[0]

    assert not result.success
    assert result.error_message == "ValueError: k must be a positive integer"
    assert result.neighbors.shape == (0, 0)
    assert result.distances.shape == (0, 0)


@pytest.mark.parametrize("batch_size", (0, 2.5, True, np.bool_(True)))
def test_invalid_batch_size_fails_before_issuing_a_query(
    tmp_path: Path, batch_size: object
) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(tmp_path, CPU_HNSW_ALGORITHM, runtime)

    result = backend.search(_dataset(), [index], k=2, batch_size=batch_size)[0]

    assert not result.success
    assert result.error_message == ("ValueError: batch_size must be a positive integer")
    assert runtime.search_calls == []


def test_numpy_integer_batch_size_is_normalized_before_search(
    tmp_path: Path,
) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(tmp_path, CPU_HNSW_ALGORITHM, runtime)
    assert backend.build(_dataset(), [index]).success

    result = backend.search(_dataset(), [index], k=2, batch_size=np.int64(2))[0]

    assert result.success, result.error_message
    assert runtime.search_calls[0]["batch_size"] == 2


def test_build_failure_is_actionable_and_removes_partial_index(
    tmp_path: Path,
) -> None:
    runtime = RecordingRuntime()
    runtime.build_error = RuntimeError("GPU device is unavailable")
    backend, index, _factory = _backend_and_index(tmp_path, CAGRA_ALGORITHM, runtime)

    result = backend.build(_dataset(), [index])

    assert not result.success
    assert result.error_message == "RuntimeError: GPU device is unavailable"
    assert not Path(index.file).exists()


def test_build_timing_excludes_validation_and_index_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(tmp_path, CPU_HNSW_ALGORITHM, runtime)
    events = []
    original_verify_artifacts = runtime.verify_artifacts
    original_build_index = runtime.build_index

    def verify_artifacts() -> None:
        events.append("artifact verification")
        original_verify_artifacts()

    def build_index(index_path: Path, vectors: np.ndarray, codec_name: str) -> int:
        events.append("build")
        return original_build_index(index_path, vectors, codec_name)

    runtime.verify_artifacts = verify_artifacts
    runtime.build_index = build_index
    clock = iter((10.0, 12.0, 20.0, 25.0, 30.0, 37.0))

    def read_clock() -> float:
        events.append("clock")
        return next(clock)

    monkeypatch.setattr("cuvs_bench.backends.lucene.time.perf_counter", read_clock)

    result = backend.build(_dataset(), [index])

    assert result.success, result.error_message
    assert result.build_time_seconds == 2.0
    assert result.metadata["validation_time_seconds"] == 5.0
    assert result.metadata["install_time_seconds"] == 7.0
    assert events[:3] == ["artifact verification", "clock", "build"]
    assert runtime.artifact_verification_count == 1


def test_search_failure_is_returned_with_context(
    tmp_path: Path,
) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(tmp_path, CPU_HNSW_ALGORITHM, runtime)
    assert backend.build(_dataset(), [index]).success
    runtime.search_error = RuntimeError("Lucene reader could not open the index")

    result = backend.search(_dataset(), [index], k=2)[0]

    assert not result.success
    assert result.error_message == (
        "RuntimeError: Lucene reader could not open the index"
    )
    assert result.neighbors.shape == (0, 2)
    assert result.distances.shape == (0, 2)


def test_failure_diagnostics_include_exception_notes(tmp_path: Path) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(tmp_path, CPU_HNSW_ALGORITHM, runtime)
    error = RuntimeError("writer failed")
    error.add_note("Rollback also failed: disk unavailable")
    runtime.build_error = error

    result = backend.build(_dataset(), [index])

    assert not result.success
    assert result.error_message == (
        "RuntimeError: writer failed\nRollback also failed: disk unavailable"
    )


def test_throughput_mode_fails_instead_of_reporting_serial_search_as_throughput(
    tmp_path: Path,
) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(tmp_path, CPU_HNSW_ALGORITHM, runtime)

    result = backend.search(_dataset(), [index], k=2, mode="throughput")[0]

    assert not result.success
    assert "supports only latency mode" in result.error_message
    assert runtime.search_calls == []


@pytest.mark.parametrize(("algorithm", "codec", "_path"), ALGORITHM_CASES)
def test_dry_runs_do_not_resolve_or_start_the_runtime(
    tmp_path: Path, algorithm: str, codec: str, _path: str
) -> None:
    runtime = RecordingRuntime()
    factory = RecordingRuntimeFactory(runtime)
    index_root = tmp_path / "indexes"
    backend = LuceneBackend(
        {
            "name": algorithm,
            "algo": algorithm,
            "codec": codec,
            "group": "test",
            "index_root": str(index_root),
            "requires_cuvs": algorithm == CAGRA_ALGORITHM,
        },
        runtime_factory=factory,
    )
    index = IndexConfig(
        name=algorithm,
        algo=algorithm,
        build_param={"codec": codec},
        search_params=[{}],
        file=str(index_root / algorithm),
    )

    build_result = backend.build(_dataset(), [index], dry_run=True)
    search_result = backend.search(_dataset(), [index], k=2, dry_run=True)[0]

    assert build_result.success
    assert build_result.metadata == {
        "dry_run": True,
        "codec": codec,
        "group": "test",
        "index_name": algorithm,
    }
    assert search_result.success
    assert search_result.metadata == {
        "dry_run": True,
        "codec": codec,
        "group": "test",
        "index_name": algorithm,
    }
    assert factory.calls == []
    assert not index_root.exists()


def test_dry_runs_do_not_materialize_lazy_vectors(tmp_path: Path) -> None:
    class UnreadableVectors(Dataset):
        @property
        def training_vectors(self) -> np.ndarray:
            raise AssertionError("dry-run loaded training vectors")

        @property
        def query_vectors(self) -> np.ndarray:
            raise AssertionError("dry-run loaded query vectors")

    runtime = RecordingRuntime()
    backend, index, factory = _backend_and_index(tmp_path, CPU_HNSW_ALGORITHM, runtime)
    dataset = UnreadableVectors(name="tiny-l2", distance_metric="euclidean")

    assert backend.build(dataset, [index], dry_run=True).success
    assert backend.search(dataset, [index], k=2, dry_run=True)[0].success
    assert factory.calls == []
