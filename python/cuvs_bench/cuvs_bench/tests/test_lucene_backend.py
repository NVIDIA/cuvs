#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for the opt-in Lucene benchmark backend."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

from cuvs_bench.backends._lucene_runtime import (
    CAGRA_CODEC,
    CPU_HNSW_CODEC,
    CagraVerification,
    LuceneIndexVerification,
    RuntimeSearchResult,
    SearchHit,
)
from cuvs_bench.backends.base import Dataset
from cuvs_bench.backends.lucene import (
    CAGRA_ALGORITHM,
    CPU_HNSW_ALGORITHM,
    MAX_CAGRA_DIMENSIONS,
    MAX_CPU_HNSW_DIMENSIONS,
    LuceneBackend,
    LuceneConfigLoader,
    _codec_for,
    _file_backed_dataset_identity,
    _score_to_squared_euclidean,
    _search_parameters,
    _source_identity,
)
from cuvs_bench.backends._lucene_runtime_config import maven_artifact_version
from cuvs_bench.orchestrator.config_loaders import IndexConfig
from cuvs_bench.run.data_export import write_results_to_csv


ALGORITHM_CASES = (
    pytest.param(
        CPU_HNSW_ALGORITHM, CPU_HNSW_CODEC, "cpu_hnsw", id="cpu-hnsw"
    ),
    pytest.param(CAGRA_ALGORITHM, CAGRA_CODEC, "gpu_cagra", id="cagra"),
)
_ARTIFACT_VERSION = maven_artifact_version()
_FAKE_ARTIFACT_PROVENANCE = {
    "cuvs_java_coordinates": (
        f"com.nvidia.cuvs:cuvs-java:{_ARTIFACT_VERSION}"
    ),
    "cuvs_java_jar_path": "/artifacts/cuvs-java.jar",
    "cuvs_java_jar_sha256": "a" * 64,
    "cuvs_lucene_coordinates": (
        f"com.nvidia.cuvs.lucene:cuvs-lucene:{_ARTIFACT_VERSION}"
    ),
    "cuvs_lucene_jar_path": "/artifacts/cuvs-lucene.jar",
    "cuvs_lucene_jar_sha256": "b" * 64,
}


class RecordingIndexVerifier:
    """Return the physical index facts recorded by the runtime substitute."""

    def __init__(self, runtime: "RecordingRuntime") -> None:
        self.runtime = runtime
        self.calls: list[tuple[Path, str, int, int]] = []

    def verify(
        self,
        index_path: Path,
        *,
        expected_codec: str,
        expected_vector_count: int,
        expected_dimensions: int,
    ) -> LuceneIndexVerification:
        self.calls.append(
            (
                index_path,
                expected_codec,
                expected_vector_count,
                expected_dimensions,
            )
        )
        if self.runtime.verification_error is not None:
            raise self.runtime.verification_error
        return LuceneIndexVerification(
            codec=expected_codec,
            segment_count=self.runtime.segment_count,
            field_count=1,
            vector_count=expected_vector_count,
            dimensions=expected_dimensions,
        )


class RecordingCagraVerifier:
    """Return deterministic persisted-path evidence and retain each request."""

    def __init__(self) -> None:
        self.calls: list[tuple[Path, int, int]] = []

    def verify(
        self,
        index_path: Path,
        *,
        expected_vector_count: int,
        expected_dimensions: int,
    ) -> CagraVerification:
        self.calls.append(
            (index_path, expected_vector_count, expected_dimensions)
        )
        return CagraVerification(
            segment_count=1,
            field_count=1,
            vector_count=expected_vector_count,
            dimensions=expected_dimensions,
        )


class RecordingRuntime:
    """Small in-process substitute for the JVM boundary."""

    pylucene_version = "10.2.0"

    def __init__(self) -> None:
        self.artifact_provenance: dict[str, str] = {}
        self.index_verifier = RecordingIndexVerifier(self)
        self.cagra_verifier = RecordingCagraVerifier()
        self.build_calls: list[tuple[Path, np.ndarray, str]] = []
        self.search_calls: list[dict[str, Any]] = []
        self.build_error: Exception | None = None
        self.search_error: Exception | None = None
        self.verification_error: Exception | None = None
        self.search_result: RuntimeSearchResult | None = None
        self.document_count = 0
        self.dimensions = 0
        self.segment_count = 1
        self.artifact_verification_count = 0

    def verify_artifacts(self) -> None:
        self.artifact_verification_count += 1

    def build_index(
        self, index_path: Path, vectors: np.ndarray, codec_name: str
    ) -> int:
        self.build_calls.append((index_path, vectors.copy(), codec_name))
        if self.build_error is not None:
            raise self.build_error
        self.document_count, self.dimensions = vectors.shape
        (index_path / "segments.fake").write_text(codec_name, encoding="utf-8")
        return 1

    def search_index(
        self,
        index_path: Path,
        queries: np.ndarray,
        *,
        k: int,
        batch_size: int,
        num_candidates: int,
    ) -> RuntimeSearchResult:
        self.search_calls.append(
            {
                "index_path": index_path,
                "queries": queries.copy(),
                "k": k,
                "batch_size": batch_size,
                "num_candidates": num_candidates,
            }
        )
        if self.search_error is not None:
            raise self.search_error
        if self.search_result is not None:
            return self.search_result
        hits = [
            [
                SearchHit(
                    document_id=rank,
                    score=1.0 / (1.0 + float(rank)),
                )
                for rank in range(k)
            ]
            for _query in queries
        ]
        batch_count = (len(queries) + batch_size - 1) // batch_size
        return RuntimeSearchResult(
            hits=hits,
            batch_latencies_ms=[2.0] * batch_count,
            document_count=self.document_count,
            dimensions=self.dimensions,
        )


class RecordingRuntimeFactory:
    """Inject one runtime without importing or initializing PyLucene."""

    def __init__(self, runtime: RecordingRuntime) -> None:
        self.runtime = runtime
        self.calls: list[dict[str, Any]] = []

    def __call__(self, config: Mapping[str, Any]) -> RecordingRuntime:
        self.calls.append(dict(config))
        return self.runtime


def _dataset(*, offset: float = 0.0) -> Dataset:
    training_vectors = np.asarray(
        [
            [0.0 + offset, 0.0],
            [1.0, 0.0],
            [0.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float32,
    )
    return Dataset(
        name="tiny-l2",
        training_vectors=training_vectors,
        query_vectors=training_vectors[:2].copy(),
        distance_metric="euclidean",
    )


def _dataset_with_dimensions(dimensions: int) -> Dataset:
    vectors = np.zeros((2, dimensions), dtype=np.float32)
    return Dataset(
        name=f"dimensions-{dimensions}",
        training_vectors=vectors,
        query_vectors=vectors[:1].copy(),
        distance_metric="euclidean",
    )


def _write_fbin(path: Path, vectors: np.ndarray) -> None:
    rows, dimensions = vectors.shape
    path.write_bytes(
        np.asarray([rows, dimensions], dtype=np.uint32).tobytes()
        + np.ascontiguousarray(vectors, dtype=np.float32).tobytes()
    )


def _artifact_pair(tmp_path: Path) -> tuple[Path, Path]:
    java_jar = tmp_path / "cuvs-java.jar"
    lucene_jar = tmp_path / "cuvs-lucene.jar"
    java_jar.touch()
    lucene_jar.touch()
    return java_jar, lucene_jar


def _backend_and_index(
    tmp_path: Path,
    algorithm: str,
    runtime: RecordingRuntime,
    *,
    search_params: list[dict[str, Any]] | None = None,
) -> tuple[LuceneBackend, IndexConfig, RecordingRuntimeFactory]:
    codec = {
        CPU_HNSW_ALGORITHM: CPU_HNSW_CODEC,
        CAGRA_ALGORITHM: CAGRA_CODEC,
    }[algorithm]
    index_root = tmp_path / "indexes"
    config: dict[str, Any] = {
        "name": algorithm,
        "algo": algorithm,
        "codec": codec,
        "group": "test",
        "index_root": str(index_root),
        "requires_cuvs": algorithm == CAGRA_ALGORITHM,
    }
    if algorithm == CAGRA_ALGORITHM:
        runtime.artifact_provenance = dict(_FAKE_ARTIFACT_PROVENANCE)
        java_jar, lucene_jar = _artifact_pair(tmp_path)
        (tmp_path / "libcuvs_c.so").touch()
        config.update(
            {
                "cuvs_java_jar": str(java_jar),
                "cuvs_lucene_jar": str(lucene_jar),
                "java_library_path": str(tmp_path),
            }
        )
    factory = RecordingRuntimeFactory(runtime)
    backend = LuceneBackend(config, runtime_factory=factory)
    index = IndexConfig(
        name=algorithm,
        algo=algorithm,
        build_param={"codec": codec},
        search_params=search_params if search_params is not None else [{}],
        file=str(index_root / algorithm),
    )
    return backend, index, factory


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
    with pytest.raises(
        ValueError, match="Invalid Lucene algorithm/codec pair"
    ):
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

    with pytest.raises(
        ValueError, match="Unsupported Lucene build parameters"
    ):
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


def test_cagra_search_accepts_only_fixed_parameters_within_its_route_limit() -> (
    None
):
    assert _search_parameters(CAGRA_ALGORITHM, {}, 1024) == {
        "num_candidates": 1024
    }

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
    assert _score_to_squared_euclidean(score) == pytest.approx(
        expected_distance
    )


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
        [{"num_candidates": 4}] if algorithm == CPU_HNSW_ALGORITHM else [{}]
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
    assert [
        search_call["num_candidates"] for search_call in runtime.search_calls
    ] == [2, 4]


def test_search_latency_reports_complete_batches(
    tmp_path: Path,
) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(
        tmp_path, CPU_HNSW_ALGORITHM, runtime
    )
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
    backend, index, _factory = _backend_and_index(
        tmp_path, CPU_HNSW_ALGORITHM, runtime
    )
    assert backend.build(_dataset(), [index]).success
    runtime.search_result = RuntimeSearchResult(
        hits=hits,
        batch_latencies_ms=[2.0],
        document_count=4,
        dimensions=2,
    )

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


def test_numpy_integer_batch_size_is_normalized_before_search(
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
    assert runtime.search_calls[0]["batch_size"] == 2


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


def test_build_timing_excludes_validation_and_index_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(
        tmp_path, CPU_HNSW_ALGORITHM, runtime
    )
    events = []
    original_verify_artifacts = runtime.verify_artifacts
    original_build_index = runtime.build_index

    def verify_artifacts() -> None:
        events.append("artifact verification")
        original_verify_artifacts()

    def build_index(
        index_path: Path, vectors: np.ndarray, codec_name: str
    ) -> int:
        events.append("build")
        return original_build_index(index_path, vectors, codec_name)

    runtime.verify_artifacts = verify_artifacts
    runtime.build_index = build_index
    clock = iter((10.0, 12.0, 20.0, 25.0, 30.0, 37.0))

    def read_clock() -> float:
        events.append("clock")
        return next(clock)

    monkeypatch.setattr(
        "cuvs_bench.backends.lucene.time.perf_counter", read_clock
    )

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
    backend, index, factory = _backend_and_index(
        tmp_path, CPU_HNSW_ALGORITHM, runtime
    )
    dataset = UnreadableVectors(name="tiny-l2", distance_metric="euclidean")

    assert backend.build(dataset, [index], dry_run=True).success
    assert backend.search(dataset, [index], k=2, dry_run=True)[0].success
    assert factory.calls == []


def test_force_rebuild_never_removes_a_path_outside_the_configured_root(
    tmp_path: Path,
) -> None:
    runtime = RecordingRuntime()
    backend, _index, factory = _backend_and_index(
        tmp_path, CPU_HNSW_ALGORITHM, runtime
    )
    outside = tmp_path / "outside" / "index"
    outside.mkdir(parents=True)
    sentinel = outside / "keep-me"
    sentinel.write_text("preserve", encoding="utf-8")
    index = IndexConfig(
        name="outside",
        algo=CPU_HNSW_ALGORITHM,
        build_param={"codec": CPU_HNSW_CODEC},
        search_params=[{}],
        file=str(outside),
    )

    result = backend.build(_dataset(), [index], force=True)

    assert not result.success
    assert "outside its configured root" in result.error_message
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert factory.calls == []


def test_force_rebuild_rejects_a_symlink_to_a_sibling_index(
    tmp_path: Path,
) -> None:
    runtime = RecordingRuntime()
    backend, index, factory = _backend_and_index(
        tmp_path, CPU_HNSW_ALGORITHM, runtime
    )
    root = Path(backend.config["index_root"])
    root.mkdir(parents=True)
    sibling = root / "existing-index"
    sibling.mkdir()
    sentinel = sibling / "keep-me"
    sentinel.write_text("preserve", encoding="utf-8")
    Path(index.file).symlink_to(sibling, target_is_directory=True)

    result = backend.build(_dataset(), [index], force=True)

    assert not result.success
    assert "must not be a symlink" in result.error_message
    assert Path(index.file).is_symlink()
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert factory.calls == []


def test_force_rebuild_removes_only_the_valid_index_directory(
    tmp_path: Path,
) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(
        tmp_path, CPU_HNSW_ALGORITHM, runtime
    )
    path = Path(index.file)
    path.mkdir(parents=True)
    stale_file = path / "stale"
    stale_file.write_text("old", encoding="utf-8")
    sibling = path.parent / "sibling"
    sibling.mkdir()
    sibling_file = sibling / "keep-me"
    sibling_file.write_text("preserve", encoding="utf-8")

    result = backend.build(_dataset(), [index], force=True)

    assert result.success, result.error_message
    assert not stale_file.exists()
    assert sibling_file.read_text(encoding="utf-8") == "preserve"


def test_failed_force_rebuild_preserves_the_previous_valid_index(
    tmp_path: Path,
) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(
        tmp_path, CPU_HNSW_ALGORITHM, runtime
    )
    first_result = backend.build(_dataset(), [index])
    assert first_result.success, first_result.error_message
    path = Path(index.file)
    original_manifest = (path / ".cuvs-bench-lucene.json").read_bytes()
    original_payload = (path / "segments.fake").read_bytes()
    runtime.build_error = RuntimeError("replacement build failed")

    replacement = backend.build(_dataset(offset=0.25), [index], force=True)

    assert not replacement.success
    assert (
        replacement.error_message == "RuntimeError: replacement build failed"
    )
    assert (path / ".cuvs-bench-lucene.json").read_bytes() == original_manifest
    assert (path / "segments.fake").read_bytes() == original_payload
    assert list(path.parent.glob(f".{path.name}.build-*")) == []


def test_backup_cleanup_failure_does_not_report_a_published_index_as_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(
        tmp_path, CPU_HNSW_ALGORITHM, runtime
    )
    assert backend.build(_dataset(), [index]).success

    def fail_cleanup(_path: Path) -> None:
        raise OSError("cleanup denied")

    monkeypatch.setattr(
        "cuvs_bench.backends.lucene.shutil.rmtree", fail_cleanup
    )
    replacement = backend.build(_dataset(offset=0.25), [index], force=True)

    assert replacement.success, replacement.error_message
    assert "Published the new index" in replacement.metadata["cleanup_warning"]
    assert "cleanup denied" in replacement.metadata["cleanup_warning"]
    assert Path(index.file).is_dir()


def test_failed_index_install_restores_the_previous_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "index"
    staged = tmp_path / "staged"
    destination.mkdir()
    staged.mkdir()
    (destination / "old").write_text("preserve", encoding="utf-8")
    original_rename = Path.rename

    def fail_install(path: Path, target: Path) -> Path:
        if path == staged:
            raise OSError("install denied")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_install)

    with pytest.raises(OSError, match="install denied"):
        LuceneBackend._install_staged_index(staged, destination)

    assert (destination / "old").read_text(encoding="utf-8") == "preserve"


def test_failed_index_restore_identifies_the_recoverable_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "index"
    staged = tmp_path / "staged"
    destination.mkdir()
    staged.mkdir()
    original_rename = Path.rename

    def fail_install_and_restore(path: Path, target: Path) -> Path:
        if path == staged:
            raise OSError("install denied")
        if path.name.startswith(f".{destination.name}.backup-"):
            raise OSError("restore denied")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_install_and_restore)

    with pytest.raises(OSError, match="install denied") as failure:
        LuceneBackend._install_staged_index(staged, destination)

    [note] = failure.value.__notes__
    assert "Failed to restore the previous index" in note
    assert "restore denied" in note
    assert "recoverable backup" in note


@pytest.mark.parametrize("control_error", (KeyboardInterrupt, SystemExit))
def test_restore_process_control_takes_precedence_over_install_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control_error: type[BaseException],
) -> None:
    destination = tmp_path / "index"
    staged = tmp_path / "staged"
    destination.mkdir()
    staged.mkdir()
    original_rename = Path.rename

    def fail_install_then_interrupt_restore(path: Path, target: Path) -> Path:
        if path == staged:
            raise OSError("install denied")
        if path.name.startswith(f".{destination.name}.backup-"):
            raise control_error("restore interrupted")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_install_then_interrupt_restore)

    with pytest.raises(control_error) as failure:
        LuceneBackend._install_staged_index(staged, destination)

    [note] = failure.value.__notes__
    assert "Index installation first failed: OSError: install denied" in note
    assert "The previous index remains in" in note
    assert list(tmp_path.glob(".index.backup-*"))


def test_backup_cleanup_does_not_swallow_process_control_exceptions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "index"
    staged = tmp_path / "staged"
    destination.mkdir()
    staged.mkdir()

    def interrupt_cleanup(_path: Path) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(
        "cuvs_bench.backends.lucene.shutil.rmtree", interrupt_cleanup
    )

    with pytest.raises(KeyboardInterrupt):
        LuceneBackend._install_staged_index(staged, destination)


def test_reusing_an_index_rejects_a_different_dataset_fingerprint(
    tmp_path: Path,
) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(
        tmp_path, CPU_HNSW_ALGORITHM, runtime
    )
    first_result = backend.build(_dataset(), [index])
    assert first_result.success, first_result.error_message

    reuse_result = backend.build(_dataset(offset=0.25), [index])

    assert not reuse_result.success
    assert (
        "does not match this dataset and configuration"
        in reuse_result.error_message
    )
    assert "rerun with --force" in reuse_result.error_message
    assert len(runtime.build_calls) == 1


def test_reusing_a_file_backed_index_does_not_materialize_training_vectors(
    tmp_path: Path,
) -> None:
    class ReuseOnlyDataset(Dataset):
        @property
        def training_vectors(self) -> np.ndarray:
            raise AssertionError("index reuse materialized the base dataset")

    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(
        tmp_path, CPU_HNSW_ALGORITHM, runtime
    )
    vectors = _dataset().training_vectors
    base_file = tmp_path / "base.fbin"
    _write_fbin(base_file, vectors)
    build_dataset = Dataset(
        name="tiny-l2",
        training_vectors=vectors,
        query_vectors=vectors[:2].copy(),
        distance_metric="euclidean",
        base_file=str(base_file),
    )
    assert backend.build(build_dataset, [index]).success
    reuse_dataset = ReuseOnlyDataset(
        name="tiny-l2",
        query_vectors=vectors[:2].copy(),
        distance_metric="euclidean",
        base_file=str(base_file),
    )

    result = backend.build(reuse_dataset, [index])

    assert result.success, result.error_message
    assert result.metadata["skipped"] is True
    assert len(runtime.build_calls) == 1


def test_search_rejects_changed_vectors_with_the_same_dataset_name(
    tmp_path: Path,
) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(
        tmp_path, CPU_HNSW_ALGORITHM, runtime
    )
    assert backend.build(_dataset(), [index]).success

    result = backend.search(_dataset(offset=0.25), [index], k=2)[0]

    assert not result.success
    assert (
        "does not match this dataset and configuration" in result.error_message
    )
    assert runtime.search_calls == []


def test_file_backed_search_does_not_materialize_training_vectors(
    tmp_path: Path,
) -> None:
    class SearchOnlyDataset(Dataset):
        @property
        def training_vectors(self) -> np.ndarray:
            raise AssertionError("search materialized the base dataset")

    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(
        tmp_path, CPU_HNSW_ALGORITHM, runtime
    )
    vectors = _dataset().training_vectors
    base_file = tmp_path / "base.fbin"
    _write_fbin(base_file, vectors)
    build_dataset = Dataset(
        name="tiny-l2",
        training_vectors=vectors,
        query_vectors=vectors[:2].copy(),
        distance_metric="euclidean",
        base_file=str(base_file),
    )
    assert backend.build(build_dataset, [index]).success
    search_dataset = SearchOnlyDataset(
        name="tiny-l2",
        query_vectors=vectors[:2].copy(),
        distance_metric="euclidean",
        base_file=str(base_file),
    )

    result = backend.search(search_dataset, [index], k=2)[0]

    assert result.success, result.error_message


def test_file_backed_search_rejects_changed_content_when_tokens_collide(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(
        tmp_path, CPU_HNSW_ALGORITHM, runtime
    )
    vectors = _dataset().training_vectors
    base_file = tmp_path / "base.fbin"
    _write_fbin(base_file, vectors)
    dataset = Dataset(
        name="tiny-l2",
        training_vectors=vectors,
        query_vectors=vectors[:2].copy(),
        distance_metric="euclidean",
        base_file=str(base_file),
    )
    assert backend.build(dataset, [index]).success
    original_source = _source_identity(dataset)
    changed_vectors = vectors.copy()
    changed_vectors[0, 0] = 42.0
    _write_fbin(base_file, changed_vectors)
    monkeypatch.setattr(
        "cuvs_bench.backends.lucene._source_identity",
        lambda _dataset: original_source,
    )
    search_dataset = Dataset(
        name="tiny-l2",
        query_vectors=vectors[:2].copy(),
        distance_metric="euclidean",
        base_file=str(base_file),
    )

    result = backend.search(search_dataset, [index], k=2)[0]

    assert not result.success
    assert (
        "does not match this dataset and configuration" in result.error_message
    )
    assert runtime.search_calls == []


def test_search_rejects_a_file_that_was_not_the_explicit_build_array(
    tmp_path: Path,
) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(
        tmp_path, CPU_HNSW_ALGORITHM, runtime
    )
    indexed_vectors = _dataset().training_vectors
    file_vectors = indexed_vectors.copy()
    file_vectors[0, 0] = 42.0
    base_file = tmp_path / "base.fbin"
    _write_fbin(base_file, file_vectors)
    build_dataset = Dataset(
        name="tiny-l2",
        training_vectors=indexed_vectors,
        query_vectors=indexed_vectors[:2].copy(),
        distance_metric="euclidean",
        base_file=str(base_file),
    )
    assert backend.build(build_dataset, [index]).success
    search_dataset = Dataset(
        name="tiny-l2",
        query_vectors=indexed_vectors[:2].copy(),
        distance_metric="euclidean",
        base_file=str(base_file),
    )

    result = backend.search(search_dataset, [index], k=2)[0]

    assert not result.success
    assert (
        "does not match this dataset and configuration" in result.error_message
    )
    assert runtime.search_calls == []


def test_search_rejects_a_physical_index_that_fails_verification(
    tmp_path: Path,
) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(
        tmp_path, CPU_HNSW_ALGORITHM, runtime
    )
    dataset = _dataset()
    assert backend.build(dataset, [index]).success
    runtime.verification_error = RuntimeError(
        "Segment '_0' uses CuVS2510GPUSearchCodec, not Lucene101"
    )

    result = backend.search(dataset, [index], k=2)[0]

    assert not result.success
    assert "uses CuVS2510GPUSearchCodec, not Lucene101" in result.error_message
    assert runtime.search_calls == []


def test_search_rejects_malformed_build_runtime_provenance(
    tmp_path: Path,
) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(
        tmp_path, CAGRA_ALGORITHM, runtime
    )
    dataset = _dataset()
    assert backend.build(dataset, [index]).success
    manifest_path = Path(index.file) / ".cuvs-bench-lucene.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["build_runtime_artifacts"]["cuvs_lucene_jar_sha256"] = "invalid"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = backend.search(dataset, [index], k=2)[0]

    assert not result.success
    assert "invalid cuvs_lucene_jar_sha256" in result.error_message
    assert runtime.search_calls == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        pytest.param(
            "schema_version",
            True,
            "Unsupported Lucene index manifest",
            id="boolean-schema-version",
        ),
        pytest.param(
            "segment_count",
            1.0,
            "invalid segment count",
            id="floating-segment-count",
        ),
        pytest.param(
            "segment_count",
            0,
            "invalid segment count",
            id="empty-index-segment-count",
        ),
    ),
)
def test_search_rejects_invalid_manifest_integer_fields(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(
        tmp_path, CPU_HNSW_ALGORITHM, runtime
    )
    dataset = _dataset()
    assert backend.build(dataset, [index]).success
    manifest_path = Path(index.file) / ".cuvs-bench-lucene.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = backend.search(dataset, [index], k=2)[0]

    assert not result.success
    assert message in result.error_message
    assert runtime.search_calls == []


def test_search_rejects_non_integer_manifest_dataset_dimensions(
    tmp_path: Path,
) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(
        tmp_path, CPU_HNSW_ALGORITHM, runtime
    )
    dataset = _dataset()
    assert backend.build(dataset, [index]).success
    manifest_path = Path(index.file) / ".cuvs-bench-lucene.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset"]["dimensions"] = 2.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = backend.search(dataset, [index], k=2)[0]

    assert not result.success
    assert "invalid dataset dimensions" in result.error_message
    assert runtime.search_calls == []


def test_search_rejects_manifest_segment_count_that_disagrees_with_index(
    tmp_path: Path,
) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(
        tmp_path, CPU_HNSW_ALGORITHM, runtime
    )
    dataset = _dataset()
    assert backend.build(dataset, [index]).success
    runtime.segment_count = 2

    result = backend.search(dataset, [index], k=2)[0]

    assert not result.success
    assert "segment count does not match the physical index: 1 != 2" in (
        result.error_message
    )
    assert runtime.search_calls == []


def test_result_metadata_is_exported_instead_of_silently_dropped(
    tmp_path: Path,
) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(
        tmp_path, CPU_HNSW_ALGORITHM, runtime
    )
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
    backend, index, _factory = _backend_and_index(
        tmp_path, CPU_HNSW_ALGORITHM, runtime
    )
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

    assert {path: path.read_bytes() for path in exported_files} == (
        original_contents
    )


def test_reusing_the_same_index_reports_a_skipped_build(
    tmp_path: Path,
) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(
        tmp_path, CPU_HNSW_ALGORITHM, runtime
    )
    assert backend.build(_dataset(), [index]).success
    assert runtime.artifact_verification_count == 1

    reuse_result = backend.build(_dataset(), [index])

    assert reuse_result.success, reuse_result.error_message
    assert reuse_result.metadata == {
        "skipped": True,
        "codec": CPU_HNSW_CODEC,
        "group": "test",
        "index_name": CPU_HNSW_ALGORITHM,
        "persisted_index_kind": "cpu_hnsw",
        "segment_count": 1,
        "field_count": 1,
        "vector_count": 4,
        "dimensions": 2,
    }
    assert len(runtime.build_calls) == 1
    assert runtime.artifact_verification_count == 2


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
        configuration.indexes[0].algo: configuration
        for configuration in configurations
    }

    assert set(by_algorithm) == {CPU_HNSW_ALGORITHM, CAGRA_ALGORITHM}
    assert by_algorithm[CPU_HNSW_ALGORITHM].indexes[0].build_param == {
        "codec": CPU_HNSW_CODEC
    }
    assert by_algorithm[CAGRA_ALGORITHM].indexes[0].build_param == {
        "codec": CAGRA_CODEC
    }
    assert (
        by_algorithm[CPU_HNSW_ALGORITHM].backend_config["requires_cuvs"]
        is False
    )
    assert (
        by_algorithm[CAGRA_ALGORITHM].backend_config["requires_cuvs"] is True
    )
    assert (
        by_algorithm[CPU_HNSW_ALGORITHM].backend_config["include_cuvs"] is True
    )
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
