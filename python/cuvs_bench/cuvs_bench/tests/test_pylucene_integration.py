#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Opt-in integration coverage for the PyLucene benchmark backend."""

import csv
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cuvs_bench._bin_format import write_bin_header
from cuvs_bench.backends._utils import compute_recall
from cuvs_bench.backends.base import Dataset
from cuvs_bench.backends.pylucene import (
    _CAGRA_PROVENANCE_FILE,
    PyLuceneBackend,
    _HNSW_PROVENANCE_FILE,
    _PyLuceneRuntime,
    _ResolvedCodec,
)
from cuvs_bench.orchestrator.config_loaders import IndexConfig

pytestmark = [
    pytest.mark.pylucene,
    pytest.mark.filterwarnings(
        "ignore:builtin type .* has no __module__ attribute:DeprecationWarning"
    ),
]

_OPT_IN_ENV = "CUVS_BENCH_PYLUCENE_INTEGRATION"
_CUVS_JAVA_JAR_ENV = "CUVS_LUCENE_CUVS_JAVA_JAR"
_CUVS_LUCENE_JAR_ENV = "CUVS_LUCENE_JAR"
_CAGRA_CODEC = "CuVS2510GPUSearchCodec"
_HNSW_CODEC = "Lucene101AcceleratedHNSWCodec"


def _write_test_bin(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        write_bin_header(file, data.shape[0], data.shape[1])
        np.ascontiguousarray(data).tofile(file)


def _required_jar(env_name: str) -> Path:
    configured = os.environ.get(env_name)
    if not configured:
        pytest.fail(f"{env_name} must point to the required runtime jar")

    jar = Path(configured).expanduser()
    if not jar.is_file():
        pytest.fail(f"{env_name} does not point to an existing file: {jar}")
    return jar.resolve()


def _assert_cagra_verification(
    metadata, expected_vector_count, expected_dimensions
):
    verification = metadata["cagra_verification"]
    assert verification["status"] == "cagra-only"
    assert verification["segment_count"] >= 1
    assert verification["field_count"] >= 1
    assert verification["vector_count"] == expected_vector_count
    assert verification["dimensions"] == expected_dimensions


def _assert_cagra_provenance(
    metadata, expected_vector_count, expected_dimensions
):
    assert metadata["cagra_provenance"] == {
        "status": "gpu-cagra-provenance",
        "schema_version": 1,
        "codec": _CAGRA_CODEC,
        "writer_path": "gpu-cagra",
        "vector_count": expected_vector_count,
        "dimensions": expected_dimensions,
        "commit_file_count": 1,
    }


def _commit_one_deletion(backend, index_path):
    runtime = backend._get_runtime()
    runtime.attach_current_thread()
    directory = runtime.FSDirectory.open(runtime.Paths.get(str(index_path)))
    writer = None
    reader = None
    try:
        writer_config = runtime.IndexWriterConfig()
        writer_config.setOpenMode(runtime.IndexWriterConfig.OpenMode.APPEND)
        writer = runtime.IndexWriter(directory, writer_config)
        reader = runtime.DirectoryReader.open(writer)
        assert int(writer.tryDeleteDocument(reader, 0)) >= 0
        writer.commit()
    finally:
        try:
            if reader is not None:
                reader.close()
        finally:
            try:
                if writer is not None:
                    writer.close()
            finally:
                directory.close()


def _assert_hnsw_verification(
    metadata, codec, expected_vector_count, expected_dimensions
):
    verification = metadata["hnsw_verification"]
    assert verification == {
        "status": "gpu-hnsw-provenance",
        "schema_version": 1,
        "codec": codec,
        "writer_path": "gpu-hnsw",
        "vector_count": expected_vector_count,
        "dimensions": expected_dimensions,
        "commit_file_count": 1,
    }


@pytest.fixture(scope="module")
def pylucene_runtime_config():
    """Resolve the explicitly configured PyLucene/cuVS runtime."""
    if os.environ.get(_OPT_IN_ENV) != "1":
        pytest.skip(f"set {_OPT_IN_ENV}=1 to run PyLucene integration tests")

    cuvs_java_jar = _required_jar(_CUVS_JAVA_JAR_ENV)
    cuvs_lucene_jar = _required_jar(_CUVS_LUCENE_JAR_ENV)

    java_library_path = os.environ.get("JAVA_LIBRARY_PATH") or os.environ.get(
        "LD_LIBRARY_PATH"
    )
    if not java_library_path:
        pytest.fail(
            "JAVA_LIBRARY_PATH or LD_LIBRARY_PATH must provide the native "
            "cuVS runtime libraries"
        )

    try:
        importlib.import_module("lucene")
    except ImportError as exc:
        pytest.fail(f"PyLucene is not importable: {exc}")

    return {
        "cuvs_java_jar": str(cuvs_java_jar),
        "cuvs_lucene_jar": str(cuvs_lucene_jar),
        "java_library_path": java_library_path,
    }


@pytest.mark.parametrize(
    (
        "algo",
        "codec",
        "expected_writer_telemetry",
        "expected_index_suffix",
    ),
    [
        pytest.param(
            "pylucene_cuvs_hnsw",
            "Lucene101AcceleratedHNSWCodec",
            {
                "writerPath": "gpu-hnsw",
                "hnswLayers": "1",
                "cagraGraphBuildAlgo": "NN_DESCENT",
                "cagraGraphDegree": "64",
                "cagraIntermediateGraphDegree": "128",
            },
            ".vex",
            id="accelerated-hnsw",
        ),
        pytest.param(
            "pylucene_cuvs_hnsw",
            "Lucene101AcceleratedHNSWBaseLayerCodec",
            {
                "writerPath": "gpu-hnsw",
                "hnswLayers": "1",
                "cagraGraphBuildAlgo": "NN_DESCENT",
                "cagraGraphDegree": "32",
                "cagraIntermediateGraphDegree": "64",
            },
            ".vex",
            id="accelerated-hnsw-base-layer",
        ),
        pytest.param(
            "pylucene_cuvs_hnsw",
            "Lucene101AcceleratedHNSWMultiLayerCodec",
            {
                "writerPath": "gpu-hnsw",
                "hnswLayers": "3",
                "cagraGraphBuildAlgo": "NN_DESCENT",
                "cagraGraphDegree": "32",
                "cagraIntermediateGraphDegree": "64",
            },
            ".vex",
            id="accelerated-hnsw-multi-layer",
        ),
        pytest.param(
            "pylucene_cuvs_cagra",
            "CuVS2510GPUSearchCodec",
            {
                "writerPath": "gpu-cagra",
                "cagraStrategy": "HEURISTIC",
                "cagraGraphBuildAlgo": "NN_DESCENT",
                "cagraGraphDegree": "64",
                "cagraIntermediateGraphDegree": "128",
            },
            ".vcag",
            id="cagra",
        ),
    ],
)
def test_build_and_search_with_real_pylucene_runtime(
    tmp_path,
    pylucene_runtime_config,
    algo,
    codec,
    expected_writer_telemetry,
    expected_index_suffix,
):
    """Build and search through Lucene's public codec and query SPI."""
    rng = np.random.default_rng(1907)
    training_vectors = rng.standard_normal((512, 32)).astype(np.float32)
    query_ids = np.asarray([0, 137, 259, 511], dtype=np.int64)
    query_vectors = training_vectors[query_ids].copy()
    k = 5

    squared_distances = np.sum(
        (query_vectors[:, np.newaxis, :] - training_vectors[np.newaxis, :, :])
        ** 2,
        axis=2,
    )
    groundtruth_neighbors = np.argsort(squared_distances, axis=1)[:, :k]
    groundtruth_distances = np.take_along_axis(
        squared_distances, groundtruth_neighbors, axis=1
    )
    dataset = Dataset(
        name="pylucene-integration",
        training_vectors=training_vectors,
        query_vectors=query_vectors,
        groundtruth_neighbors=groundtruth_neighbors.astype(np.int32),
        groundtruth_distances=groundtruth_distances.astype(np.float32),
        distance_metric="euclidean",
    )

    index_path = tmp_path / f"{algo}-index"
    index = IndexConfig(
        name=f"{algo}-integration",
        algo=algo,
        build_param={"codec": codec},
        search_params=[{}],
        file=str(index_path),
    )
    backend = PyLuceneBackend(
        {
            "name": index.name,
            "algo": algo,
            "codec": codec,
            **pylucene_runtime_config,
        }
    )

    try:
        build_result = backend.build(dataset, [index], force=True)
        assert build_result.success, build_result.error_message
        assert build_result.build_time_seconds > 0
        assert build_result.index_size_bytes > 0
        assert build_result.metadata["codec"] == codec
        assert build_result.metadata["pylucene_version"] != "unknown"
        assert (
            build_result.metadata["writer_path"]
            == expected_writer_telemetry["writerPath"]
        )
        assert (
            build_result.metadata["writer_telemetry"]
            == expected_writer_telemetry
        )
        assert any(
            path.suffix == expected_index_suffix
            for path in index_path.iterdir()
        )
        if codec == _CAGRA_CODEC:
            _assert_cagra_provenance(
                build_result.metadata,
                training_vectors.shape[0],
                training_vectors.shape[1],
            )
            _assert_cagra_verification(
                build_result.metadata,
                training_vectors.shape[0],
                training_vectors.shape[1],
            )
            assert (index_path / _CAGRA_PROVENANCE_FILE).is_file()
        else:
            _assert_hnsw_verification(
                build_result.metadata,
                codec,
                training_vectors.shape[0],
                training_vectors.shape[1],
            )
            assert (index_path / _HNSW_PROVENANCE_FILE).is_file()

        reused_result = backend.build(dataset, [index], force=False)
        assert reused_result.success, reused_result.error_message
        assert reused_result.metadata["skipped"] is True
        if codec == _CAGRA_CODEC:
            _assert_cagra_provenance(
                reused_result.metadata,
                training_vectors.shape[0],
                training_vectors.shape[1],
            )
            _assert_cagra_verification(
                reused_result.metadata,
                training_vectors.shape[0],
                training_vectors.shape[1],
            )
        else:
            _assert_hnsw_verification(
                reused_result.metadata,
                codec,
                training_vectors.shape[0],
                training_vectors.shape[1],
            )

        search_result = backend.search(dataset, [index], k=k, batch_size=2)
        assert search_result.success, search_result.error_message
        assert search_result.neighbors.shape == (query_vectors.shape[0], k)
        assert search_result.distances.shape == (query_vectors.shape[0], k)
        np.testing.assert_array_equal(search_result.neighbors[:, 0], query_ids)
        assert np.all(np.isfinite(search_result.distances))
        recall = compute_recall(
            search_result.neighbors, groundtruth_neighbors, k
        )
        assert recall >= 0.75
        returned_distances = np.take_along_axis(
            squared_distances, search_result.neighbors, axis=1
        )
        np.testing.assert_allclose(
            search_result.distances,
            returned_distances,
            rtol=1e-5,
            atol=1e-5,
        )
        assert np.all(np.diff(search_result.distances, axis=1) >= -1e-6)
        assert search_result.search_time_ms > 0
        assert search_result.latency_seconds is not None
        assert search_result.latency_seconds > 0
        assert search_result.queries_per_second > 0
        assert search_result.metadata["codec"] == codec
        assert search_result.metadata["pylucene_version"] != "unknown"
        assert search_result.metadata["num_batches"] == 2
        assert search_result.metadata["mode"] == "latency"
        assert set(search_result.latency_percentiles) == {"p50", "p95", "p99"}
        if codec == _CAGRA_CODEC:
            _assert_cagra_provenance(
                search_result.metadata,
                training_vectors.shape[0],
                training_vectors.shape[1],
            )
            _assert_cagra_verification(
                search_result.metadata,
                training_vectors.shape[0],
                training_vectors.shape[1],
            )

            wrong_count_dataset = Dataset(
                name="pylucene-integration-wrong-count",
                training_vectors=training_vectors[:-1],
                query_vectors=query_vectors,
                distance_metric="euclidean",
            )
            rejected = backend.build(wrong_count_dataset, [index], force=False)
            assert not rejected.success
            assert "vector count does not match the dataset: 512 != 511" in (
                rejected.error_message
            )
            rejected = backend.search(wrong_count_dataset, [index], k=k)
            assert not rejected.success
            assert "vector count does not match the dataset: 512 != 511" in (
                rejected.error_message
            )

            data_path = next(index_path.glob("*.vcag"))
            original_data = data_path.read_bytes()

            data_path.unlink()
            rejected = backend.search(dataset, [index], k=k)
            assert not rejected.success
            assert "cannot read" in rejected.error_message
            data_path.write_bytes(original_data)

            data_path.write_bytes(original_data[:-1])
            rejected = backend.search(dataset, [index], k=k)
            assert not rejected.success
            assert "do not exactly cover" in rejected.error_message
            data_path.write_bytes(original_data)

            corrupted_data = bytearray(original_data)
            corrupted_data[len(corrupted_data) // 2] ^= 0xFF
            original_stat = data_path.stat()
            data_path.write_bytes(corrupted_data)
            os.utime(
                data_path,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
            rejected = backend.search(dataset, [index], k=k)
            assert not rejected.success
            assert "checksum" in rejected.error_message.lower()
            data_path.write_bytes(original_data)

            deleted_index_path = tmp_path / f"{algo}-deleted-index"
            shutil.copytree(index_path, deleted_index_path)
            deleted_index = IndexConfig(
                name=f"{algo}-deleted-integration",
                algo=algo,
                build_param={"codec": codec},
                search_params=[{}],
                file=str(deleted_index_path),
            )
            _commit_one_deletion(backend, deleted_index_path)
            with pytest.raises(RuntimeError, match="committed deletions"):
                backend._get_runtime().verify_cagra_index(deleted_index_path)
            rejected = backend.search(dataset, [deleted_index], k=k)
            assert not rejected.success
            assert "does not match the current Lucene commit" in (
                rejected.error_message
            )

            metadata_path = next(index_path.glob("*.vemc"))
            contents = bytearray(metadata_path.read_bytes())
            contents[-1] ^= 0xFF
            metadata_path.write_bytes(contents)
            with pytest.raises(RuntimeError, match="metadata format v0"):
                backend._get_runtime().verify_cagra_index(index_path)
        else:
            assert "cagra_verification" not in search_result.metadata
            _assert_hnsw_verification(
                search_result.metadata,
                codec,
                training_vectors.shape[0],
                training_vectors.shape[1],
            )
            if codec == _HNSW_CODEC:
                manifest_path = index_path / _HNSW_PROVENANCE_FILE
                original_manifest = manifest_path.read_bytes()
                segment_path = next(index_path.glob("segments_*"))
                original_segment = segment_path.read_bytes()

                manifest_path.unlink()
                rejected = backend.search(dataset, [index], k=k)
                assert not rejected.success
                assert "provenance manifest is missing" in (
                    rejected.error_message
                )
                manifest_path.write_bytes(original_manifest)

                manifest_path.write_text("{")
                rejected = backend.search(dataset, [index], k=k)
                assert not rejected.success
                assert "provenance manifest cannot be read" in (
                    rejected.error_message
                )
                manifest_path.write_bytes(original_manifest)

                segment_path.write_bytes(original_segment + b"stale")
                rejected = backend.search(dataset, [index], k=k)
                assert not rejected.success
                assert "does not match the current Lucene commit" in (
                    rejected.error_message
                )
                segment_path.write_bytes(original_segment)

                payload = json.loads(original_manifest)
                payload["codec"] = "Lucene101AcceleratedHNSWBaseLayerCodec"
                manifest_path.write_text(json.dumps(payload))
                rejected = backend.search(dataset, [index], k=k)
                assert not rejected.success
                assert "does not match the requested codec" in (
                    rejected.error_message
                )
                manifest_path.write_bytes(original_manifest)
    finally:
        backend.cleanup()


def test_real_verifier_rejects_cagra_brute_force_fallback(
    tmp_path, pylucene_runtime_config
):
    """Reject cuVS-Lucene's real one-vector fallback before search."""
    runtime = _PyLuceneRuntime.create(pylucene_runtime_config)
    vectors = np.ones((1, 32), dtype=np.float32)
    index_path = tmp_path / "cagra-fallback-index"
    index_path.mkdir()
    java_codec = runtime.resolve_codec(_CAGRA_CODEC)
    telemetry = runtime.codec_telemetry(_CAGRA_CODEC, java_codec)
    assert telemetry["writerPath"] == "gpu-cagra"
    runtime.build_index(
        index_path,
        vectors,
        _ResolvedCodec(
            codec_name=_CAGRA_CODEC,
            java_codec=java_codec,
            telemetry=telemetry,
            writer_path=telemetry["writerPath"],
        ),
    )

    with pytest.raises(RuntimeError, match="persisted brute-force"):
        runtime.verify_cagra_index(index_path, expected_vector_count=1)

    dataset = Dataset(
        name="pylucene-cagra-fallback",
        training_vectors=vectors,
        query_vectors=vectors.copy(),
        distance_metric="euclidean",
    )
    index = IndexConfig(
        name="pylucene-cagra-fallback",
        algo="pylucene_cuvs_cagra",
        build_param={"codec": _CAGRA_CODEC},
        search_params=[{}],
        file=str(index_path),
    )
    backend = PyLuceneBackend(
        {
            "name": index.name,
            "algo": index.algo,
            "codec": _CAGRA_CODEC,
            **pylucene_runtime_config,
        }
    )
    backend._runtime = runtime
    try:
        result = backend.search(dataset, [index], k=1)
        assert not result.success
        assert "CAGRA provenance manifest is missing" in result.error_message
    finally:
        backend.cleanup()


def test_cli_build_and_search_with_real_pylucene_runtime(
    tmp_path, pylucene_runtime_config
):
    """Exercise the documented module CLI in a fresh PyLucene process."""
    rng = np.random.default_rng(174)
    training_vectors = rng.standard_normal((512, 32)).astype(np.float32)
    query_ids = np.asarray([0, 137, 259, 511], dtype=np.int64)
    query_vectors = training_vectors[query_ids].copy()
    k = 5
    squared_distances = np.sum(
        (query_vectors[:, np.newaxis, :] - training_vectors[np.newaxis, :, :])
        ** 2,
        axis=2,
    )
    groundtruth_neighbors = np.argsort(squared_distances, axis=1)[:, :k]
    groundtruth_distances = np.take_along_axis(
        squared_distances, groundtruth_neighbors, axis=1
    )

    dataset_name = "pylucene-cli-integration"
    dataset_path = tmp_path / "datasets"
    dataset_dir = dataset_path / dataset_name
    _write_test_bin(dataset_dir / "base.fbin", training_vectors)
    _write_test_bin(dataset_dir / "query.fbin", query_vectors)
    _write_test_bin(
        dataset_dir / "groundtruth.neighbors.ibin",
        groundtruth_neighbors.astype(np.int32),
    )
    _write_test_bin(
        dataset_dir / "groundtruth.distances.fbin",
        groundtruth_distances.astype(np.float32),
    )

    dataset_config = tmp_path / "datasets.yaml"
    dataset_config.write_text(
        json.dumps(
            [
                {
                    "name": dataset_name,
                    "base_file": f"{dataset_name}/base.fbin",
                    "query_file": f"{dataset_name}/query.fbin",
                    "groundtruth_neighbors_file": (
                        f"{dataset_name}/groundtruth.neighbors.ibin"
                    ),
                    "groundtruth_distances_file": (
                        f"{dataset_name}/groundtruth.distances.fbin"
                    ),
                    "distance": "euclidean",
                    "dims": training_vectors.shape[1],
                }
            ]
        )
    )
    backend_config = tmp_path / "pylucene-backend.yaml"
    backend_config.write_text(
        json.dumps({"backend": "pylucene", **pylucene_runtime_config})
    )

    environment = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[2])
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (source_root, environment.get("PYTHONPATH"))
        if value
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cuvs_bench.run",
            "--backend-config",
            str(backend_config),
            "--dataset-configuration",
            str(dataset_config),
            "--dataset",
            dataset_name,
            "--dataset-path",
            str(dataset_path),
            "--algorithms",
            "pylucene_cuvs_hnsw",
            "--groups",
            "test",
            "--batch-size",
            "2",
            "-k",
            str(k),
            "-m",
            "latency",
            "--build",
            "--search",
            "--force",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )

    codec = "Lucene101AcceleratedHNSWCodec"
    index_name = f"pylucene_cuvs_hnsw_test.codec{codec}"
    index_path = dataset_dir / "index" / index_name
    assert any(path.suffix == ".vex" for path in index_path.iterdir())
    assert (index_path / _HNSW_PROVENANCE_FILE).is_file()

    result_path = dataset_dir / "result"
    build_json = result_path / "build" / "pylucene_cuvs_hnsw,test.json"
    search_stem = f"pylucene_cuvs_hnsw,test,k{k},bs2"
    search_json = result_path / "search" / f"{search_stem}.json"
    build_rows = json.loads(build_json.read_text())["benchmarks"]
    search_rows = json.loads(search_json.read_text())["benchmarks"]
    assert len(build_rows) == 1
    assert len(search_rows) == 1

    build_row = build_rows[0]
    assert build_row["name"] == f"{index_name}/build"
    assert build_row["real_time"] > 0
    assert build_row["index_size"] > 0
    assert build_row["codec"] == codec
    assert build_row["writer_path"] == "gpu-hnsw"
    _assert_hnsw_verification(
        build_row,
        codec,
        training_vectors.shape[0],
        training_vectors.shape[1],
    )

    search_row = search_rows[0]
    assert search_row["name"] == f"{index_name}/search"
    assert search_row["Recall"] >= 0.75
    assert search_row["items_per_second"] > 0
    assert search_row["Latency"] > 0
    assert search_row["search_time_ms"] > 0
    _assert_hnsw_verification(
        search_row,
        codec,
        training_vectors.shape[0],
        training_vectors.shape[1],
    )

    raw_csv = result_path / "search" / f"{search_stem},raw.csv"
    with raw_csv.open(newline="") as file:
        csv_rows = list(csv.DictReader(file))
    assert len(csv_rows) == 1
    csv_row = csv_rows[0]
    assert csv_row["index_name"] == index_name
    assert float(csv_row["recall"]) >= 0.75
    assert float(csv_row["throughput"]) > 0
    assert float(csv_row["latency"]) > 0
    assert float(csv_row["build time"]) > 0
    assert (result_path / "search" / f"{search_stem},latency.csv").is_file()
    assert (result_path / "search" / f"{search_stem},throughput.csv").is_file()
