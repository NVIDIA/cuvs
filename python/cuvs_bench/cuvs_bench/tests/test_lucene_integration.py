#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Opt-in end-to-end tests for the Lucene CPU and cuVS-backed routes."""

import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from _lucene_log_capture import (
    case_used_cpu_hnsw_fallback,
    lucene_case_output,
    lucene_log_case,
)
from cuvs_bench.backends.base import Dataset
from cuvs_bench.backends._lucene_runtime import (
    ACCELERATED_HNSW_CODEC,
    DIRECT_PYLUCENE_DISPATCH,
    TIMED_BRIDGE_PYLUCENE_DISPATCH,
)
from cuvs_bench.backends.lucene import (
    ACCELERATED_HNSW_ALGORITHM,
    CAGRA_ALGORITHM,
    CAGRA_CODEC,
    CPU_HNSW_ALGORITHM,
    CPU_HNSW_CODEC,
    LuceneBackend,
)
from cuvs_bench.backends._lucene_runtime_config import maven_artifact_version
from cuvs_bench.orchestrator.config_loaders import IndexConfig

pytestmark = [
    pytest.mark.lucene_e2e,
    pytest.mark.filterwarnings(
        "ignore:builtin type .* has no __module__ attribute:DeprecationWarning"
    ),
]

_GRAPH_CLAMP_WARNINGS = (
    "Intermediate graph degree cannot be larger",
    "cannot be larger than intermediate graph degree",
    "for nn-descent needs to match cagra intermediate graph degree",
)
_FRESH_PROCESS_TIMEOUT_SECONDS = 600
_MINIMUM_RECALL = 0.75
_ARTIFACT_VERSION = maven_artifact_version()
_GPU_HIDDEN_FALLBACK_CASE = (
    "test_lucene_integration.py::"
    "test_accelerated_hnsw_logs_cpu_fallback_when_gpu_is_hidden"
)


def _case(algorithm: str, dimensions: int) -> tuple[Dataset, np.ndarray]:
    document_count = 1024 if algorithm != CPU_HNSW_ALGORITHM else 512
    rng = np.random.default_rng(174 + dimensions)
    vectors = rng.standard_normal((document_count, dimensions)).astype(
        np.float32
    )
    query_ids = np.asarray([0, 127, document_count - 1], dtype=np.int64)
    queries = vectors[query_ids].copy()
    squared_distances = np.sum(
        (queries[:, np.newaxis, :] - vectors[np.newaxis, :, :]) ** 2,
        axis=2,
    )
    ground_truth = np.argsort(squared_distances, axis=1)[:, :10]
    return (
        Dataset(
            name=f"lucene-e2e-{algorithm}-{dimensions}",
            training_vectors=vectors,
            query_vectors=queries,
            groundtruth_neighbors=ground_truth.astype(np.int32),
            groundtruth_distances=np.take_along_axis(
                squared_distances, ground_truth, axis=1
            ).astype(np.float32),
            distance_metric="euclidean",
        ),
        query_ids,
    )


def _backend_and_index(
    tmp_path: Path,
    algorithm: str,
    codec: str,
    search_params: list[dict],
    *,
    include_cuvs: bool = True,
) -> tuple[LuceneBackend, IndexConfig]:
    root = tmp_path / algorithm
    return (
        LuceneBackend(
            {
                "name": algorithm,
                "algo": algorithm,
                "codec": codec,
                "group": "test",
                "index_root": str(root),
                "requires_cuvs": algorithm != CPU_HNSW_ALGORITHM,
                "include_cuvs": include_cuvs,
            }
        ),
        IndexConfig(
            name=algorithm,
            algo=algorithm,
            build_param={"codec": codec},
            search_params=search_params,
            file=str(root / "index"),
        ),
    )


def _recall(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(
        np.mean(
            [
                len(set(row).intersection(truth)) / expected.shape[1]
                for row, truth in zip(actual, expected)
            ]
        )
    )


def _write_bin(path: Path, values: np.ndarray) -> None:
    rows, columns = values.shape
    path.write_bytes(
        np.asarray([rows, columns], dtype=np.uint32).tobytes()
        + np.ascontiguousarray(values).tobytes()
    )


def _write_segmented_cpu_index(
    runtime,
    index_path: Path,
    vectors: np.ndarray,
    external_ids: np.ndarray,
    *,
    force_merge: bool,
) -> int:
    """Write three commits while keeping dataset IDs distinct from doc IDs."""
    runtime.attach_current_thread()
    index_path.mkdir(parents=True)
    directory = runtime.FSDirectory.open(runtime.Paths.get(str(index_path)))
    try:
        config = runtime.IndexWriterConfig()
        config.setOpenMode(runtime.IndexWriterConfig.OpenMode.CREATE)
        config.setCodec(runtime.resolve_codec(CPU_HNSW_CODEC))
        if not force_merge:
            from org.apache.lucene.index import NoMergePolicy

            config.setMergePolicy(NoMergePolicy.INSTANCE)
        writer = runtime.IndexWriter(directory, config)
        try:
            for positions in np.array_split(np.arange(len(vectors)), 3):
                for position in positions:
                    writer.addDocument(
                        runtime._document(
                            int(external_ids[position]), vectors[position]
                        )
                    )
                writer.commit()
            if force_merge:
                writer.forceMerge(1)
                writer.commit()
        finally:
            writer.close()

        reader = runtime.DirectoryReader.open(directory)
        try:
            return int(reader.leaves().size())
        finally:
            reader.close()
    finally:
        directory.close()


def _assert_artifact_provenance(metadata: dict, role: str) -> None:
    assert metadata[f"{role}_cuvs_java_coordinates"] == (
        f"com.nvidia.cuvs:cuvs-java:{_ARTIFACT_VERSION}"
    )
    assert metadata[f"{role}_cuvs_lucene_coordinates"] == (
        f"com.nvidia.cuvs.lucene:cuvs-lucene:{_ARTIFACT_VERSION}"
    )
    for artifact in ("cuvs_java", "cuvs_lucene"):
        assert Path(metadata[f"{role}_{artifact}_jar_path"]).is_file()
        assert re.fullmatch(
            r"[0-9a-f]{64}", metadata[f"{role}_{artifact}_jar_sha256"]
        )


def _assert_timing_contract(
    result, *, query_count: int, requested_batch_size: int, java_timing: bool
) -> None:
    metadata = result.metadata
    assert metadata["timing_contract_version"] == 1
    assert metadata["latency_scope"] == "client_query"
    assert metadata["throughput_scope"] == "query_corpus_wall"
    assert metadata["sample_unit"] == "query"
    assert metadata["execution_model"] == "serial_single_query"
    assert metadata["requested_batch_size"] == requested_batch_size
    assert metadata["effective_search_batch_size"] == 1
    assert metadata["query_count"] == query_count
    assert metadata["client_query_count"] == query_count
    assert metadata["pylucene_search_dispatch_count"] == query_count
    assert metadata["java_timing_available"] is java_timing
    assert metadata["search_dispatch_kind"] == (
        TIMED_BRIDGE_PYLUCENE_DISPATCH
        if java_timing
        else DIRECT_PYLUCENE_DISPATCH
    )
    assert metadata["index_prewarm_bytes"] > 0
    assert metadata["index_prewarm_file_count"] > 0
    assert metadata["query_corpus_wall_ms"] > 0.0
    assert metadata["client_query_mean_ms"] >= 0.0
    assert metadata["pylucene_search_dispatch_mean_ms"] >= 0.0
    assert (
        metadata["client_query_total_ms"] <= metadata["query_corpus_wall_ms"]
    )
    if java_timing:
        assert metadata["java_index_searcher_search_count"] == query_count
        assert metadata["java_index_searcher_search_mean_ms"] >= 0.0
        assert metadata["first_query_java_index_searcher_search_ms"] >= 0.0
        subsequent_java_mean = metadata[
            "subsequent_java_index_searcher_search_mean_ms"
        ]
        if query_count > 1:
            assert subsequent_java_mean >= 0.0
        else:
            assert subsequent_java_mean is None
    else:
        assert "java_index_searcher_search_count" not in metadata
        assert "first_query_java_index_searcher_search_ms" not in metadata
        assert "subsequent_java_index_searcher_search_mean_ms" not in metadata


def test_accelerated_hnsw_builds_on_gpu_and_searches_on_cpu(
    tmp_path,
    capfd,
    request,
):
    dataset, query_ids = _case(ACCELERATED_HNSW_ALGORITHM, 128)
    backend, index = _backend_and_index(
        tmp_path,
        ACCELERATED_HNSW_ALGORITHM,
        ACCELERATED_HNSW_CODEC,
        [{"num_candidates": 64}],
    )
    case_name = request.node.nodeid
    capfd.readouterr()

    with lucene_log_case(case_name):
        build = backend.build(dataset, [index], force=True)
        assert build.success, build.error_message
        result = backend.search(dataset, [index], k=10, batch_size=2)[0]

    captured = capfd.readouterr()
    assert not case_used_cpu_hnsw_fallback(captured.err, case_name), (
        f"{case_name} used Lucene's CPU HNSW writer fallback:\n"
        f"{lucene_case_output(captured.err, case_name)}"
    )
    for warning in _GRAPH_CLAMP_WARNINGS:
        assert (
            warning.casefold() not in (captured.out + captured.err).casefold()
        )

    assert build.metadata["codec"] == ACCELERATED_HNSW_CODEC
    assert build.metadata["persisted_index_kind"] == "hnsw"
    assert build.metadata["build_route_policy"] == (
        "gpu_cagra_or_cpu_hnsw_fallback"
    )
    _assert_artifact_provenance(build.metadata, "build_runtime")
    assert result.success, result.error_message
    assert result.metadata["expected_search_route"] == "cpu_hnsw"
    assert result.metadata["persisted_index_kind"] == "hnsw"
    assert result.metadata["build_route_policy"] == (
        "gpu_cagra_or_cpu_hnsw_fallback"
    )
    _assert_timing_contract(
        result,
        query_count=len(dataset.query_vectors),
        requested_batch_size=2,
        java_timing=True,
    )
    _assert_artifact_provenance(result.metadata, "build_runtime")
    _assert_artifact_provenance(result.metadata, "search_runtime")
    np.testing.assert_array_equal(result.neighbors[:, 0], query_ids)
    assert all(len(set(row)) == len(row) for row in result.neighbors.tolist())
    assert (
        _recall(result.neighbors, dataset.groundtruth_neighbors)
        >= _MINIMUM_RECALL
    )


@pytest.mark.parametrize(
    ("algorithm", "codec", "dimensions", "expected_search_route"),
    [
        pytest.param(
            CAGRA_ALGORITHM,
            CAGRA_CODEC,
            127,
            "gpu_cagra",
            id="gpu-cagra-unaligned-dimensions",
        ),
        pytest.param(
            CAGRA_ALGORITHM,
            CAGRA_CODEC,
            128,
            "gpu_cagra",
            id="gpu-cagra-aligned-dimensions",
        ),
        pytest.param(
            CAGRA_ALGORITHM,
            CAGRA_CODEC,
            129,
            "gpu_cagra",
            id="gpu-cagra-above-alignment-boundary",
        ),
        pytest.param(
            CAGRA_ALGORITHM,
            CAGRA_CODEC,
            1025,
            "gpu_cagra",
            id="gpu-cagra-above-cpu-dimension-limit",
        ),
    ],
)
def test_cagra_build_and_search_verify_persisted_index_and_search_behavior(
    tmp_path,
    capfd,
    algorithm,
    codec,
    dimensions,
    expected_search_route,
):
    dataset, query_ids = _case(algorithm, dimensions)
    backend, index = _backend_and_index(tmp_path, algorithm, codec, [{}])

    build = backend.build(dataset, [index], force=True)
    assert build.success, build.error_message
    assert build.metadata["codec"] == codec
    assert build.metadata["persisted_index_kind"] == (
        "gpu_cagra_only" if algorithm == CAGRA_ALGORITHM else "cpu_hnsw"
    )
    _assert_artifact_provenance(build.metadata, "build_runtime")

    # CAGRA evidence combines verified CAGRA-only payloads, a supported k,
    # result oracles, and the absence of fallback or graph-clamp warnings.
    results = backend.search(dataset, [index], k=10, batch_size=2)
    assert len(results) == 1
    result = results[0]
    assert result.success, result.error_message
    assert result.metadata["expected_search_route"] == expected_search_route
    _assert_timing_contract(
        result,
        query_count=len(dataset.query_vectors),
        requested_batch_size=2,
        java_timing=True,
    )
    _assert_artifact_provenance(result.metadata, "build_runtime")
    _assert_artifact_provenance(result.metadata, "search_runtime")
    np.testing.assert_array_equal(result.neighbors[:, 0], query_ids)
    assert all(len(set(row)) == len(row) for row in result.neighbors.tolist())
    assert (
        _recall(result.neighbors, dataset.groundtruth_neighbors)
        >= _MINIMUM_RECALL
    )

    vectors = dataset.training_vectors
    queries = dataset.query_vectors
    exact_squared_distances = np.sum(
        (queries[:, np.newaxis, :] - vectors[np.newaxis, :, :]) ** 2,
        axis=2,
    )
    returned_distances = np.take_along_axis(
        exact_squared_distances, result.neighbors, axis=1
    )
    np.testing.assert_allclose(
        result.distances, returned_distances, rtol=1e-4, atol=1e-4
    )

    if algorithm == CAGRA_ALGORITHM:
        assert build.metadata["field_count"] == 1
        assert build.metadata["vector_count"] == vectors.shape[0]
        assert build.metadata["dimensions"] == dimensions
        index_files = {path.suffix for path in Path(index.file).iterdir()}
        assert ".cfs" in index_files
        assert ".vcag" not in index_files

    output = "\n".join(capfd.readouterr())
    assert "falling back to a brute force index" not in output
    for warning in _GRAPH_CLAMP_WARNINGS:
        assert warning.casefold() not in output.casefold()


def test_cpu_hnsw_in_fresh_process_without_cuvs_artifacts(tmp_path):
    """Prove the CPU control does not inherit a cuVS-capable process JVM."""
    child_flag = "CUVS_BENCH_LUCENE_CPU_PROBE"
    if child_flag not in os.environ:
        environment = os.environ.copy()
        environment[child_flag] = "1"
        environment.pop("CUVS_LUCENE_CUVS_JAVA_JAR", None)
        environment.pop("CUVS_LUCENE_JAR", None)
        environment.pop("JAVA_LIBRARY_PATH", None)
        node = (
            f"{Path(__file__).resolve()}::"
            "test_cpu_hnsw_in_fresh_process_without_cuvs_artifacts"
        )
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-s",
                    node,
                    "--run-lucene-e2e",
                ],
                cwd=Path(__file__).resolve().parents[4],
                env=environment,
                text=True,
                capture_output=True,
                check=False,
                timeout=_FRESH_PROCESS_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as error:
            pytest.fail(
                "The isolated CPU HNSW probe exceeded "
                f"{_FRESH_PROCESS_TIMEOUT_SECONDS} seconds: {error}"
            )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert "1 passed" in completed.stdout
        return

    dataset, query_ids = _case(CPU_HNSW_ALGORITHM, 32)
    backend, index = _backend_and_index(
        tmp_path,
        CPU_HNSW_ALGORITHM,
        CPU_HNSW_CODEC,
        [{"num_candidates": 32}],
        include_cuvs=False,
    )
    build = backend.build(dataset, [index], force=True)
    assert build.success, build.error_message
    assert build.metadata["persisted_index_kind"] == "cpu_hnsw"
    result = backend.search(dataset, [index], k=10, batch_size=2)[0]
    assert result.success, result.error_message
    assert result.metadata["expected_search_route"] == "cpu_hnsw"
    _assert_timing_contract(
        result,
        query_count=len(dataset.query_vectors),
        requested_batch_size=2,
        java_timing=False,
    )
    np.testing.assert_array_equal(result.neighbors[:, 0], query_ids)
    assert all(len(set(row)) == len(row) for row in result.neighbors.tolist())
    assert (
        _recall(result.neighbors, dataset.groundtruth_neighbors)
        >= _MINIMUM_RECALL
    )


@pytest.mark.parametrize(
    ("force_merge", "expected_segments"),
    (
        pytest.param(False, 3, id="three-segments"),
        pytest.param(True, 1, id="force-merged"),
    ),
)
def test_numeric_document_values_preserve_dataset_ids_across_segments(
    tmp_path: Path, force_merge: bool, expected_segments: int
) -> None:
    rng = np.random.default_rng(2624)
    vectors = rng.standard_normal((12, 16)).astype(np.float32)
    external_ids = np.roll(np.arange(12, dtype=np.int64) * 17 + 1000, 3)
    query_positions = np.asarray([1, 9])
    queries = vectors[query_positions].copy()
    backend, index = _backend_and_index(
        tmp_path,
        CPU_HNSW_ALGORITHM,
        CPU_HNSW_CODEC,
        [{"num_candidates": len(vectors)}],
        include_cuvs=True,
    )
    runtime = backend._get_runtime()

    segment_count = _write_segmented_cpu_index(
        runtime,
        Path(index.file),
        vectors,
        external_ids,
        force_merge=force_merge,
    )
    result = runtime.search_index(
        Path(index.file),
        queries,
        k=4,
        num_candidates=len(vectors),
    )

    assert segment_count == expected_segments
    returned_ids = [
        [hit.document_id for hit in query_hits] for query_hits in result.hits
    ]
    assert [row[0] for row in returned_ids] == external_ids[
        query_positions
    ].tolist()
    assert all(len(set(row)) == len(row) for row in returned_ids)
    valid_ids = set(external_ids.tolist())
    assert all(
        document_id in valid_ids for row in returned_ids for document_id in row
    )


def test_accelerated_hnsw_logs_cpu_fallback_when_gpu_is_hidden(tmp_path):
    """Keep the positive GPU-path assertion from passing vacuously."""
    child_flag = "CUVS_BENCH_LUCENE_GPU_HIDDEN_FALLBACK_PROBE"
    if child_flag not in os.environ:
        environment = os.environ.copy()
        environment[child_flag] = "1"
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
        node = (
            f"{Path(__file__).resolve()}::"
            "test_accelerated_hnsw_logs_cpu_fallback_when_gpu_is_hidden"
        )
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-s",
                    node,
                    "--run-lucene-e2e",
                ],
                cwd=Path(__file__).resolve().parents[4],
                env=environment,
                text=True,
                capture_output=True,
                check=False,
                timeout=_FRESH_PROCESS_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as error:
            pytest.fail(
                "The GPU-hidden accelerated-HNSW probe exceeded "
                f"{_FRESH_PROCESS_TIMEOUT_SECONDS} seconds: {error}"
            )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert "1 passed" in completed.stdout
        assert case_used_cpu_hnsw_fallback(
            completed.stderr, _GPU_HIDDEN_FALLBACK_CASE
        )
        return

    dataset, _query_ids = _case(ACCELERATED_HNSW_ALGORITHM, 32)
    backend, index = _backend_and_index(
        tmp_path,
        ACCELERATED_HNSW_ALGORITHM,
        ACCELERATED_HNSW_CODEC,
        [{"num_candidates": 32}],
    )
    with lucene_log_case(_GPU_HIDDEN_FALLBACK_CASE):
        build = backend.build(dataset, [index], force=True)
    assert build.success, build.error_message
    assert build.metadata["persisted_index_kind"] == "hnsw"


def test_public_cli_builds_searches_and_exports_cpu_hnsw(tmp_path):
    rng = np.random.default_rng(2475)
    vectors = rng.standard_normal((256, 16)).astype(np.float32)
    query_ids = np.asarray([0, 127, 255])
    queries = vectors[query_ids].copy()
    squared_distances = np.sum(
        (queries[:, np.newaxis, :] - vectors[np.newaxis, :, :]) ** 2,
        axis=2,
    )
    ground_truth = np.argsort(squared_distances, axis=1)[:, :10].astype(
        np.int32
    )
    base_file = tmp_path / "base.fbin"
    query_file = tmp_path / "query.fbin"
    ground_truth_file = tmp_path / "ground-truth.ibin"
    _write_bin(base_file, vectors)
    _write_bin(query_file, queries)
    _write_bin(ground_truth_file, ground_truth)
    dataset_name = "lucene-cli-e2e"
    dataset_configuration = tmp_path / "datasets.yaml"
    dataset_configuration.write_text(
        json.dumps(
            [
                {
                    "name": dataset_name,
                    "base_file": str(base_file),
                    "query_file": str(query_file),
                    "groundtruth_neighbors_file": str(ground_truth_file),
                    "dims": 16,
                    "distance": "euclidean",
                }
            ]
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cuvs_bench.run",
            "--backend",
            "lucene",
            "--dataset",
            dataset_name,
            "--dataset-path",
            str(tmp_path),
            "--dataset-configuration",
            str(dataset_configuration),
            "--algorithms",
            CPU_HNSW_ALGORITHM,
            "--groups",
            "test",
            "--batch-size",
            "2",
            "-k",
            "10",
            "--search-mode",
            "latency",
            "--build",
            "--search",
            "--force",
        ],
        cwd=Path(__file__).resolve().parents[4],
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
        timeout=_FRESH_PROCESS_TIMEOUT_SECONDS,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    search_csv = (
        tmp_path
        / dataset_name
        / "result"
        / "search"
        / "lucene_cpu_hnsw,test,k10,bs2,raw.csv"
    )
    with search_csv.open(newline="", encoding="utf-8") as stream:
        [row] = csv.DictReader(stream)
    assert row["index_name"] == f"{CPU_HNSW_ALGORITHM}_test"
    assert float(row["recall"]) >= _MINIMUM_RECALL
    assert float(row["build time"]) >= 0.0
    assert row["persisted_index_kind"] == "cpu_hnsw"
    assert row["expected_search_route"] == "cpu_hnsw"
    assert row["timing_contract_version"] == "1"
    assert row["latency_scope"] == "client_query"
    assert row["throughput_scope"] == "query_corpus_wall"
    assert row["requested_batch_size"] == "2"
    assert row["effective_search_batch_size"] == "1"
    assert row["client_query_count"] == "3"
    assert int(row["index_prewarm_bytes"]) > 0


def test_cagra_build_rejects_lucenes_one_document_brute_force_fallback(
    tmp_path,
):
    vectors = np.zeros((1, 32), dtype=np.float32)
    dataset = Dataset(
        name="lucene-cagra-one-document",
        training_vectors=vectors,
        query_vectors=vectors.copy(),
        distance_metric="euclidean",
    )
    backend, index = _backend_and_index(
        tmp_path, CAGRA_ALGORITHM, CAGRA_CODEC, [{}]
    )

    build = backend.build(dataset, [index], force=True)

    assert not build.success
    assert "CagraVerificationError: Persisted brute-force fallback" in (
        build.error_message
    )
    assert not Path(index.file).exists()


@pytest.mark.parametrize(
    ("algorithm", "codec"),
    (
        pytest.param(
            CPU_HNSW_ALGORITHM,
            CPU_HNSW_CODEC,
            id="cpu-built-hnsw",
        ),
        pytest.param(
            ACCELERATED_HNSW_ALGORITHM,
            ACCELERATED_HNSW_CODEC,
            id="gpu-cagra-built-hnsw",
        ),
    ),
)
def test_hnsw_search_supports_top_k_2000(
    tmp_path,
    capfd,
    request,
    algorithm,
    codec,
):
    rng = np.random.default_rng(2000)
    vectors = rng.standard_normal((2500, 16)).astype(np.float32)
    dataset = Dataset(
        name=f"lucene-{algorithm}-top-k-2000",
        training_vectors=vectors,
        query_vectors=vectors[[2000]].copy(),
        distance_metric="euclidean",
    )
    backend, index = _backend_and_index(
        tmp_path,
        algorithm,
        codec,
        [{"num_candidates": 2500}],
    )
    case_name = request.node.nodeid
    capfd.readouterr()

    with lucene_log_case(case_name):
        build = backend.build(dataset, [index], force=True)
        assert build.success, build.error_message
        result = backend.search(dataset, [index], k=2000, batch_size=1)[0]

    captured = capfd.readouterr()
    if algorithm == ACCELERATED_HNSW_ALGORITHM:
        assert not case_used_cpu_hnsw_fallback(captured.err, case_name), (
            f"{case_name} used Lucene's CPU HNSW writer fallback:\n"
            f"{lucene_case_output(captured.err, case_name)}"
        )

    assert result.success, result.error_message
    assert result.neighbors.shape == (1, 2000)
    assert result.neighbors[0, 0] == 2000
    assert len(set(result.neighbors[0])) == 2000
    assert result.metadata["expected_search_route"] == "cpu_hnsw"
    assert result.metadata["persisted_index_kind"] == (
        "hnsw" if algorithm == ACCELERATED_HNSW_ALGORITHM else "cpu_hnsw"
    )
    java_timing_available = (
        "search_runtime_cuvs_lucene_coordinates" in result.metadata
    )
    if algorithm == ACCELERATED_HNSW_ALGORITHM:
        assert java_timing_available
    _assert_timing_contract(
        result,
        query_count=1,
        requested_batch_size=1,
        java_timing=java_timing_available,
    )
