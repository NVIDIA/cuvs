#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Test doubles and builders shared by the Lucene backend unit tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

from cuvs_bench.backends._lucene_runtime import (
    ACCELERATED_HNSW_CODEC,
    CAGRA_CODEC,
    CPU_HNSW_CODEC,
    CagraVerification,
    DIRECT_PYLUCENE_DISPATCH,
    LuceneIndexVerification,
    QueryTiming,
    RuntimeBuildResult,
    RuntimeBuildTiming,
    RuntimeSearchResult,
    RuntimeSearchTiming,
    SearchHit,
    TIMED_BRIDGE_PYLUCENE_DISPATCH,
)
from cuvs_bench.backends._lucene_runtime_config import maven_artifact_version
from cuvs_bench.backends.base import Dataset
from cuvs_bench.backends.lucene import (
    ACCELERATED_HNSW_ALGORITHM,
    CAGRA_ALGORITHM,
    CPU_HNSW_ALGORITHM,
    LuceneBackend,
)
from cuvs_bench.orchestrator.config_loaders import IndexConfig


ALGORITHM_CASES = (
    pytest.param(
        CPU_HNSW_ALGORITHM, CPU_HNSW_CODEC, "cpu_hnsw", id="cpu-hnsw"
    ),
    pytest.param(
        ACCELERATED_HNSW_ALGORITHM,
        ACCELERATED_HNSW_CODEC,
        "cpu_hnsw",
        id="accelerated-hnsw",
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
    ) -> RuntimeBuildResult:
        self.build_calls.append((index_path, vectors.copy(), codec_name))
        if self.build_error is not None:
            raise self.build_error
        self.document_count, self.dimensions = vectors.shape
        (index_path / "segments.fake").write_text(codec_name, encoding="utf-8")
        return RuntimeBuildResult(
            segment_count=1,
            timing=RuntimeBuildTiming(
                directory_open_ns=100_000,
                writer_setup_ns=200_000,
                document_ingest_ns=300_000,
                writer_commit_close_ns=400_000,
                post_build_reader_ns=500_000,
                directory_close_ns=600_000,
                runtime_build_wall_ns=2_100_000,
            ),
        )

    def search_index(
        self,
        index_path: Path,
        queries: np.ndarray,
        *,
        k: int,
        num_candidates: int,
    ) -> RuntimeSearchResult:
        self.search_calls.append(
            {
                "index_path": index_path,
                "queries": queries.copy(),
                "k": k,
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
        query_timings = tuple(
            QueryTiming(
                query_prepare_ns=100_000,
                pylucene_search_dispatch_ns=2_000_000,
                java_index_searcher_search_ns=(
                    1_500_000 if self.artifact_provenance else None
                ),
                result_materialization_ns=200_000,
                client_query_ns=2_500_000,
            )
            for _query in queries
        )
        return RuntimeSearchResult(
            hits=hits,
            timing=RuntimeSearchTiming(
                directory_open_ns=250_000,
                reader_searcher_setup_ns=750_000,
                query_corpus_wall_ns=3_000_000 * len(queries),
                reader_close_ns=250_000,
                directory_close_ns=250_000,
                runtime_plan_wall_ns=3_000_000 * len(queries) + 1_500_000,
                search_dispatch_kind=(
                    TIMED_BRIDGE_PYLUCENE_DISPATCH
                    if self.artifact_provenance
                    else DIRECT_PYLUCENE_DISPATCH
                ),
                queries=query_timings,
            ),
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


def _runtime_search_result(
    hits: list[list[SearchHit]],
    *,
    search_dispatch_ns: tuple[int, ...] | None = None,
    java_search_ns: tuple[int, ...] | None = None,
    document_count: int = 4,
    dimensions: int = 2,
    query_corpus_wall_ns: int | None = None,
) -> RuntimeSearchResult:
    """Create an explicit per-query timing result for backend tests."""
    query_count = len(hits)
    query_prepare_ns = 100_000
    result_materialization_ns = 200_000
    unclassified_client_overhead_ns = 200_000
    directory_open_ns = 250_000
    reader_searcher_setup_ns = 750_000
    reader_close_ns = 250_000
    directory_close_ns = 250_000
    dispatch_samples = (
        search_dispatch_ns
        if search_dispatch_ns is not None
        else (2_000_000,) * query_count
    )
    if len(dispatch_samples) != query_count:
        raise ValueError("search_dispatch_ns must contain one value per query")
    if java_search_ns is not None and len(java_search_ns) != query_count:
        raise ValueError("java_search_ns must contain one value per query")
    query_timings = tuple(
        QueryTiming(
            query_prepare_ns=query_prepare_ns,
            pylucene_search_dispatch_ns=dispatch_samples[index],
            java_index_searcher_search_ns=(
                java_search_ns[index] if java_search_ns is not None else None
            ),
            result_materialization_ns=result_materialization_ns,
            client_query_ns=(
                query_prepare_ns
                + dispatch_samples[index]
                + result_materialization_ns
                + unclassified_client_overhead_ns
            ),
        )
        for index in range(query_count)
    )
    corpus_ns = (
        query_corpus_wall_ns
        if query_corpus_wall_ns is not None
        else sum(item.client_query_ns for item in query_timings)
    )
    return RuntimeSearchResult(
        hits=hits,
        timing=RuntimeSearchTiming(
            directory_open_ns=directory_open_ns,
            reader_searcher_setup_ns=reader_searcher_setup_ns,
            query_corpus_wall_ns=corpus_ns,
            reader_close_ns=reader_close_ns,
            directory_close_ns=directory_close_ns,
            runtime_plan_wall_ns=(
                directory_open_ns
                + reader_searcher_setup_ns
                + corpus_ns
                + reader_close_ns
                + directory_close_ns
            ),
            search_dispatch_kind=(
                TIMED_BRIDGE_PYLUCENE_DISPATCH
                if java_search_ns is not None
                else DIRECT_PYLUCENE_DISPATCH
            ),
            queries=query_timings,
        ),
        document_count=document_count,
        dimensions=dimensions,
    )


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
        ACCELERATED_HNSW_ALGORITHM: ACCELERATED_HNSW_CODEC,
        CAGRA_ALGORITHM: CAGRA_CODEC,
    }[algorithm]
    index_root = tmp_path / "indexes"
    config: dict[str, Any] = {
        "name": algorithm,
        "algo": algorithm,
        "codec": codec,
        "group": "test",
        "index_root": str(index_root),
        "requires_cuvs": algorithm != CPU_HNSW_ALGORITHM,
    }
    if algorithm != CPU_HNSW_ALGORITHM:
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
