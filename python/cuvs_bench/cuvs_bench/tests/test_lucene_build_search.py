#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Build and search behavior for the opt-in Lucene benchmark backend."""

from __future__ import annotations

import json
from dataclasses import replace
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
    _runtime_search_result,
    _write_fbin,
)
from cuvs_bench.backends._lucene_runtime import (
    RuntimeBuildResult,
    RuntimeBuildTiming,
    RuntimeSearchResult,
    SearchHit,
)
from cuvs_bench.backends.base import Dataset
from cuvs_bench.backends.lucene import (
    ACCELERATED_HNSW_ALGORITHM,
    CAGRA_ALGORITHM,
    CPU_HNSW_ALGORITHM,
    MAX_CAGRA_DIMENSIONS,
    MAX_CPU_HNSW_DIMENSIONS,
    LuceneBackend,
    _file_backed_dataset_identity,
    _runtime_build_timing_metadata,
    _runtime_search_timing_metadata,
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
    expected_persisted_kind = {
        CPU_HNSW_ALGORITHM: "cpu_hnsw",
        ACCELERATED_HNSW_ALGORITHM: "hnsw",
        CAGRA_ALGORITHM: "gpu_cagra_only",
    }[algorithm]
    assert result.metadata["persisted_index_kind"] == expected_persisted_kind
    assert (
        result.metadata["build_route_policy"]
        == {
            CPU_HNSW_ALGORITHM: "cpu_hnsw",
            ACCELERATED_HNSW_ALGORITHM: "gpu_cagra_or_cpu_hnsw_fallback",
            CAGRA_ALGORITHM: "gpu_cagra",
        }[algorithm]
    )
    assert result.metadata["segment_count"] == 1
    assert result.metadata["pylucene_version"] == "10.2.0"
    assert result.index_size_bytes > 0
    assert len(factory.calls) == 1
    assert [call[2] for call in runtime.build_calls] == [codec]
    if algorithm == CAGRA_ALGORITHM:
        [(verified_path, vector_count, dimensions)] = (
            runtime.cagra_verifier.calls
        )
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
            ACCELERATED_HNSW_ALGORITHM,
            4096,
            MAX_CAGRA_DIMENSIONS,
            True,
            id="accelerated-hnsw-maximum-dimensions",
        ),
        pytest.param(
            ACCELERATED_HNSW_ALGORITHM,
            4097,
            MAX_CAGRA_DIMENSIONS,
            False,
            id="accelerated-hnsw-above-maximum-dimensions",
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
    backend, index, _factory = _backend_and_index(
        tmp_path, CPU_HNSW_ALGORITHM, runtime
    )
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
        (Path(index.file) / ".cuvs-bench-lucene.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["dataset"]["subset_size"] == 3


def test_boolean_subset_size_is_rejected_before_runtime_initialization(
    tmp_path: Path,
) -> None:
    runtime = RecordingRuntime()
    backend, index, factory = _backend_and_index(
        tmp_path, CPU_HNSW_ALGORITHM, runtime
    )
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
    search_params = (
        [{"num_candidates": 4}] if algorithm != CAGRA_ALGORITHM else [{}]
    )
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
    assert result.metadata["codec"] == codec
    assert result.metadata["pylucene_version"] == "10.2.0"
    assert result.metadata["latency_seconds"] == 0.0025
    assert result.metadata["timing_contract_version"] == 1
    assert result.metadata["latency_scope"] == "client_query"
    assert result.metadata["throughput_scope"] == "query_corpus_wall"
    assert result.metadata["sample_unit"] == "query"
    assert result.metadata["execution_model"] == "serial_single_query"
    assert result.metadata["requested_batch_size"] == 1
    assert result.metadata["effective_search_batch_size"] == 1
    assert result.metadata["query_count"] == 2
    assert result.metadata["pylucene_search_dispatch_count"] == 2
    assert result.metadata["search_dispatch_kind"] == (
        "thin_jar_timing_bridge"
        if algorithm != CPU_HNSW_ALGORITHM
        else "direct_pylucene"
    )
    assert result.metadata["batch_size"] == 1
    assert result.metadata["mode"] == "latency"
    assert result.metadata["group"] == "test"
    assert result.metadata["index_name"] == algorithm
    assert result.metadata["expected_search_route"] == path
    assert (
        result.metadata["persisted_index_kind"]
        == {
            CPU_HNSW_ALGORITHM: "cpu_hnsw",
            ACCELERATED_HNSW_ALGORITHM: "hnsw",
            CAGRA_ALGORITHM: "gpu_cagra_only",
        }[algorithm]
    )
    assert result.metadata["segment_count"] == 1
    assert result.metadata["field_count"] == 1
    assert result.metadata["vector_count"] == 4
    assert result.metadata["dimensions"] == 2
    assert result.metadata["index_prewarm_bytes"] > 0
    assert result.metadata["index_prewarm_file_count"] >= 2
    assert result.metadata["java_timing_available"] is (
        algorithm != CPU_HNSW_ALGORITHM
    )
    if algorithm != CPU_HNSW_ALGORITHM:
        for role in ("build_runtime", "search_runtime"):
            for key, value in _FAKE_ARTIFACT_PROVENANCE.items():
                assert result.metadata[f"{role}_{key}"] == value
    assert result.search_params == (
        [{"num_candidates": 4}] if algorithm != CAGRA_ALGORITHM else [{}]
    )
    assert runtime.search_calls[0]["num_candidates"] == (
        4 if algorithm != CAGRA_ALGORITHM else 2
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
    assert [
        search_call["num_candidates"] for search_call in runtime.search_calls
    ] == [
        2,
        4,
    ]
    invocation_totals = {
        result.metadata["backend_search_invocation_total_ms"]
        for result in results
    }
    assert len(invocation_totals) == 1
    assert next(iter(invocation_totals)) >= max(
        result.metadata["search_plan_total_ms"] for result in results
    )
    assert all(
        "backend_search_total_ms" not in result.metadata for result in results
    )


def test_search_timing_uses_one_sample_per_serial_query(
    tmp_path: Path,
) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(
        tmp_path, CPU_HNSW_ALGORITHM, runtime
    )
    dataset = _dataset()
    assert backend.build(dataset, [index]).success
    runtime.search_result = _runtime_search_result(
        [
            [SearchHit(0, 1.0), SearchHit(1, 0.5)],
            [SearchHit(0, 1.0), SearchHit(1, 0.5)],
        ],
        search_dispatch_ns=(2_000_000, 8_000_000),
        query_corpus_wall_ns=12_000_000,
    )

    result = backend.search(dataset, [index], k=2, batch_size=2)[0]

    assert result.success, result.error_message
    assert result.latency_percentiles == {
        "p50": 5.5,
        "p95": pytest.approx(8.2),
        "p99": pytest.approx(8.44),
    }
    assert result.search_time_ms == 11.0
    assert result.queries_per_second == pytest.approx(2.0 / 0.012)
    assert result.metadata["latency_seconds"] == 0.0055
    assert result.metadata["pylucene_search_dispatch_count"] == 2
    assert result.metadata["first_query_pylucene_search_dispatch_ms"] == 2.0
    assert result.metadata["first_query_client_query_ms"] == 2.5
    assert result.metadata["subsequent_query_count"] == 1
    assert (
        result.metadata["subsequent_pylucene_search_dispatch_mean_ms"] == 8.0
    )
    assert result.metadata["subsequent_client_query_mean_ms"] == 8.5
    assert result.metadata["requested_batch_size"] == 2
    assert result.metadata["effective_search_batch_size"] == 1


def test_exact_java_timing_reports_first_and_subsequent_queries_only_when_available() -> (
    None
):
    hits = [
        [SearchHit(0, 1.0), SearchHit(1, 0.5)],
        [SearchHit(0, 1.0), SearchHit(1, 0.5)],
    ]
    bridged = _runtime_search_result(
        hits,
        java_search_ns=(1_000_000, 4_000_000),
    )
    direct = _runtime_search_result(hits)

    bridged_metadata, _ = _runtime_search_timing_metadata(bridged, 2)
    direct_metadata, _ = _runtime_search_timing_metadata(direct, 2)

    assert bridged_metadata["first_query_java_index_searcher_search_ms"] == 1.0
    assert (
        bridged_metadata["subsequent_java_index_searcher_search_mean_ms"]
        == 4.0
    )
    assert "first_query_java_index_searcher_search_ms" not in direct_metadata
    assert (
        "subsequent_java_index_searcher_search_mean_ms" not in direct_metadata
    )


def test_requested_batch_size_never_groups_lucene_query_samples(
    tmp_path: Path,
) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(
        tmp_path, CPU_HNSW_ALGORITHM, runtime
    )
    dataset = _dataset()
    dataset.query_vectors = dataset.training_vectors[:3].copy()
    assert backend.build(dataset, [index]).success

    result = backend.search(dataset, [index], k=2, batch_size=2)[0]

    assert result.success, result.error_message
    assert result.metadata["query_count"] == 3
    assert result.metadata["client_query_count"] == 3
    assert result.metadata["pylucene_search_dispatch_count"] == 3
    assert result.metadata["requested_batch_size"] == 2
    assert result.metadata["effective_search_batch_size"] == 1
    assert "batch_count" not in result.metadata


def test_search_rejects_missing_per_query_timing_data(tmp_path: Path) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(
        tmp_path, CPU_HNSW_ALGORITHM, runtime
    )
    dataset = _dataset()
    assert backend.build(dataset, [index]).success
    single_query_result = _runtime_search_result(
        [[SearchHit(0, 1.0), SearchHit(1, 0.5)]]
    )
    runtime.search_result = RuntimeSearchResult(
        hits=[
            [SearchHit(0, 1.0), SearchHit(1, 0.5)],
            [SearchHit(0, 1.0), SearchHit(1, 0.5)],
        ],
        timing=single_query_result.timing,
        document_count=4,
        dimensions=2,
    )

    result = backend.search(dataset, [index], k=2)[0]

    assert not result.success
    assert result.error_message == (
        "RuntimeError: Lucene returned timing data for 1 queries, expected 2"
    )


def test_build_timing_rejects_phases_longer_than_the_enclosing_wall() -> None:
    timing = RuntimeBuildTiming(
        directory_open_ns=1,
        writer_setup_ns=2,
        document_ingest_ns=3,
        writer_commit_close_ns=4,
        post_build_reader_ns=5,
        directory_close_ns=6,
        runtime_build_wall_ns=20,
    )

    with pytest.raises(RuntimeError, match="impossible runtime_build_wall"):
        _runtime_build_timing_metadata(
            RuntimeBuildResult(segment_count=1, timing=timing)
        )


def test_search_timing_rejects_query_phases_outside_client_wall() -> None:
    result = _runtime_search_result([[SearchHit(0, 1.0), SearchHit(1, 0.5)]])
    query = replace(result.timing.queries[0], client_query_ns=2_000_000)
    contradictory = replace(
        result,
        timing=replace(result.timing, queries=(query,)),
    )

    with pytest.raises(RuntimeError, match=r"impossible client_query\[0\]"):
        _runtime_search_timing_metadata(contradictory, 1)


def test_search_timing_rejects_client_total_longer_than_corpus_wall() -> None:
    result = _runtime_search_result([[SearchHit(0, 1.0), SearchHit(1, 0.5)]])
    contradictory = replace(
        result,
        timing=replace(result.timing, query_corpus_wall_ns=2_499_999),
    )

    with pytest.raises(RuntimeError, match="impossible query_corpus_wall"):
        _runtime_search_timing_metadata(contradictory, 1)


def test_search_timing_rejects_plan_phases_longer_than_plan_wall() -> None:
    result = _runtime_search_result([[SearchHit(0, 1.0), SearchHit(1, 0.5)]])
    contradictory = replace(
        result,
        timing=replace(result.timing, runtime_plan_wall_ns=3_999_999),
    )

    with pytest.raises(RuntimeError, match="impossible runtime_plan_wall"):
        _runtime_search_timing_metadata(contradictory, 1)


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
    backend, index, _factory = _backend_and_index(
        tmp_path, CPU_HNSW_ALGORITHM, runtime
    )
    assert backend.build(_dataset(), [index]).success
    runtime.search_result = _runtime_search_result(hits)

    result = backend.search(_dataset(), [index], k=2)[0]

    assert not result.success
    assert (
        f"RuntimeError: Lucene query 0 returned {message}"
        in result.error_message
    )


def test_search_rejects_a_runtime_result_that_drops_a_query(
    tmp_path: Path,
) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(
        tmp_path, CPU_HNSW_ALGORITHM, runtime
    )
    dataset = _dataset()
    assert backend.build(dataset, [index]).success
    runtime.search_result = _runtime_search_result(
        [[SearchHit(0, 1.0), SearchHit(1, 0.5)]]
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
    backend, index, _factory = _backend_and_index(
        tmp_path, CAGRA_ALGORITHM, runtime
    )
    assert backend.build(_dataset(), [index]).success

    result = backend.search(_dataset(), [index], k=1025)[0]

    assert not result.success
    assert "CAGRA search supports k <= 1024" in result.error_message
    assert runtime.search_calls == []


def test_invalid_top_k_returns_the_validation_error_without_a_secondary_failure(
    tmp_path: Path,
) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(
        tmp_path, CPU_HNSW_ALGORITHM, runtime
    )

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
    backend, index, _factory = _backend_and_index(
        tmp_path, CPU_HNSW_ALGORITHM, runtime
    )

    result = backend.search(_dataset(), [index], k=2, batch_size=batch_size)[0]

    assert not result.success
    assert result.error_message == (
        "ValueError: batch_size must be a positive integer"
    )
    assert runtime.search_calls == []


def test_numpy_integer_batch_size_is_reported_without_changing_execution(
    tmp_path: Path,
) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(
        tmp_path, CPU_HNSW_ALGORITHM, runtime
    )
    assert backend.build(_dataset(), [index]).success

    result = backend.search(_dataset(), [index], k=2, batch_size=np.int64(2))[
        0
    ]

    assert result.success, result.error_message
    assert "batch_size" not in runtime.search_calls[0]
    assert result.metadata["requested_batch_size"] == 2
    assert result.metadata["effective_search_batch_size"] == 1


def test_build_failure_is_actionable_and_removes_partial_index(
    tmp_path: Path,
) -> None:
    runtime = RecordingRuntime()
    runtime.build_error = RuntimeError("GPU device is unavailable")
    backend, index, _factory = _backend_and_index(
        tmp_path, CAGRA_ALGORITHM, runtime
    )

    result = backend.build(_dataset(), [index])

    assert not result.success
    assert result.error_message == "RuntimeError: GPU device is unavailable"
    assert not Path(index.file).exists()


def test_build_timing_separates_build_validation_and_index_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(
        tmp_path, CPU_HNSW_ALGORITHM, runtime
    )
    clock = iter(
        int(seconds * 1_000_000_000)
        for seconds in (
            0,
            10,
            11,
            12,
            13,
            14,
            15,
            20,
            22,
            30,
            35,
            40,
            47,
            50,
            51,
            60,
        )
    )

    def read_clock() -> int:
        return next(clock)

    monkeypatch.setattr(
        "cuvs_bench.backends.lucene.time.perf_counter_ns", read_clock
    )

    result = backend.build(_dataset(), [index])

    assert result.success, result.error_message
    assert result.build_time_seconds == 2.0
    assert result.metadata["validation_time_seconds"] == 5.0
    assert result.metadata["install_time_seconds"] == 7.0
    assert result.metadata["dataset_load_validate_seconds"] == 1.0
    assert result.metadata["runtime_setup_seconds"] == 1.0
    assert result.metadata["artifact_validation_seconds"] == 1.0
    assert result.metadata["index_build_call_seconds"] == 2.0
    assert result.metadata["index_validation_manifest_seconds"] == 5.0
    assert result.metadata["index_install_seconds"] == 7.0
    assert result.metadata["index_size_measurement_seconds"] == 1.0
    assert result.metadata["backend_build_total_seconds"] == 60.0
    assert result.metadata["runtime_document_ingest_seconds"] == 0.0003
    assert runtime.artifact_verification_count == 1


def test_search_failure_is_returned_with_context(
    tmp_path: Path,
) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(
        tmp_path, CPU_HNSW_ALGORITHM, runtime
    )
    assert backend.build(_dataset(), [index]).success
    runtime.search_error = RuntimeError(
        "Lucene reader could not open the index"
    )

    result = backend.search(_dataset(), [index], k=2)[0]

    assert not result.success
    assert result.error_message == (
        "RuntimeError: Lucene reader could not open the index"
    )
    assert result.neighbors.shape == (0, 2)
    assert result.distances.shape == (0, 2)


def test_failure_diagnostics_include_exception_notes(tmp_path: Path) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(
        tmp_path, CPU_HNSW_ALGORITHM, runtime
    )
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
    backend, index, _factory = _backend_and_index(
        tmp_path, CPU_HNSW_ALGORITHM, runtime
    )

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
            "requires_cuvs": algorithm != CPU_HNSW_ALGORITHM,
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
    backend, index, factory = _backend_and_index(
        tmp_path, CPU_HNSW_ALGORITHM, runtime
    )
    dataset = UnreadableVectors(name="tiny-l2", distance_metric="euclidean")

    assert backend.build(dataset, [index], dry_run=True).success
    assert backend.search(dataset, [index], k=2, dry_run=True)[0].success
    assert factory.calls == []
