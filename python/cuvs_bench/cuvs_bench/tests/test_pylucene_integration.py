#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Opt-in integration coverage for the PyLucene benchmark backend."""

import csv
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from cuvs_bench._bin_format import write_bin_header
from cuvs_bench.backends._utils import compute_recall
from cuvs_bench.backends.base import BuildResult, Dataset
from cuvs_bench.backends.pylucene import (
    _BuildCodec,
    _CAGRA_PROVENANCE_FILE,
    PyLuceneBackend,
    _HNSW_PROVENANCE_FILE,
    _PyLuceneRuntime,
    _validate_pylucene_version,
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
_HNSW_WRITER_POLICY = "gpu-with-cpu-fallback"
_COMPOUND_FILE_POLICY = {
    _HNSW_CODEC: "lucene-default",
    _CAGRA_CODEC: "disabled",
}
_GPU_HNSW_WRITER = "com.nvidia.cuvs.lucene.Lucene99AcceleratedHNSWVectorsWriter"
_CPU_HNSW_WRITER = "org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsWriter"
_WRITER_SELECTION_CODEC = "com.nvidia.cuvs.bench.PyLuceneWriterSelectionCodec"
_WRITER_SELECTION_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "java"
    / "com"
    / "nvidia"
    / "cuvs"
    / "bench"
    / "PyLuceneWriterSelectionCodec.java"
)
_CPU_FALLBACK_PROBE = Path(__file__).with_name("pylucene_cpu_fallback_probe.py")


@dataclass(frozen=True)
class _DatasetCase:
    dataset: Dataset
    query_ids: np.ndarray
    squared_distances: np.ndarray
    k: int


@dataclass(frozen=True)
class _RuntimeFixture:
    backend_config: dict[str, str]
    writer_selection_classes: Path


@dataclass(frozen=True)
class _BaselineIndex:
    algo: str
    codec: str
    writer_policy: str
    index_path: Path
    index: IndexConfig
    build_result: BuildResult
    dataset_case: _DatasetCase
    runtime_config: dict[str, str]


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


def _single_search_result(results):
    assert len(results) == 1
    return results[0]


def _writer_diagnostics(java_codec):
    diagnostics = str(java_codec.knnVectorsFormat())
    match = re.search(r"writerClass=([^,)]+), fieldsWriterCalls=(\d+)", diagnostics)
    assert match is not None, diagnostics
    return match.group(1), int(match.group(2))


def _backend(algo, codec, runtime_config):
    return PyLuceneBackend(
        {
            "name": f"{algo}-integration",
            "algo": algo,
            "codec": codec,
            **runtime_config,
        }
    )


def _index_at(baseline, index_path):
    return IndexConfig(
        name=baseline.index.name,
        algo=baseline.algo,
        build_param={"codec": baseline.codec},
        search_params=[{}],
        file=str(index_path),
    )


def _copy_baseline(baseline, tmp_path):
    index_path = tmp_path / f"{baseline.algo}-index"
    shutil.copytree(baseline.index_path, index_path)
    return index_path, _index_at(baseline, index_path)


def _search_copied_index(baseline, index, dataset=None):
    backend = _backend(baseline.algo, baseline.codec, baseline.runtime_config)
    search_dataset = baseline.dataset_case.dataset if dataset is None else dataset
    try:
        return _single_search_result(
            backend.search(
                search_dataset,
                [index],
                k=baseline.dataset_case.k,
            )
        )
    finally:
        backend.cleanup()


def _compile_writer_selection_codec(
    output_dir: Path,
    cuvs_java_jar: Path,
    cuvs_lucene_jar: Path,
    pylucene_classpath: str,
) -> None:
    environment_javac = Path(sys.prefix) / "lib" / "jvm" / "bin" / "javac"
    javac = shutil.which("javac")
    if javac is None and environment_javac.is_file():
        javac = str(environment_javac)
    if javac is None:
        pytest.fail("javac is required for the PyLucene integration tests")
    if not _WRITER_SELECTION_SOURCE.is_file():
        pytest.fail(
            "PyLucene writer-selection test source is missing: "
            f"{_WRITER_SELECTION_SOURCE}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    compile_classpath = os.pathsep.join(
        (str(cuvs_java_jar), str(cuvs_lucene_jar), pylucene_classpath)
    )
    completed = subprocess.run(
        [
            javac,
            "--release",
            "22",
            "-classpath",
            compile_classpath,
            "-d",
            str(output_dir),
            str(_WRITER_SELECTION_SOURCE),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.fail(
            "Could not compile the PyLucene writer-selection test codec:\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def _assert_cagra_verification(metadata, expected_vector_count, expected_dimensions):
    verification = metadata["cagra_verification"]
    assert verification["status"] == "cagra-only"
    assert verification["segment_count"] >= 1
    assert verification["field_count"] >= 1
    assert verification["vector_count"] == expected_vector_count
    assert verification["dimensions"] == expected_dimensions


def _assert_cagra_provenance(metadata, expected_vector_count, expected_dimensions):
    assert metadata["cagra_provenance"] == {
        "status": "gpu-cagra-provenance",
        "schema_version": 3,
        "codec": _CAGRA_CODEC,
        "writer_policy": "gpu-cagra",
        "compound_file_policy": "disabled",
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
        "status": "gpu-with-cpu-fallback-provenance",
        "schema_version": 3,
        "codec": codec,
        "writer_policy": _HNSW_WRITER_POLICY,
        "compound_file_policy": "lucene-default",
        "vector_count": expected_vector_count,
        "dimensions": expected_dimensions,
        "commit_file_count": 1,
    }


@pytest.fixture(scope="module")
def pylucene_runtime_config(tmp_path_factory):
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
        lucene = importlib.import_module("lucene")
    except ImportError as exc:
        pytest.fail(f"PyLucene is not importable: {exc}")
    _validate_pylucene_version(lucene)

    test_classes = tmp_path_factory.mktemp("pylucene-java-test-classes")
    _compile_writer_selection_codec(
        test_classes,
        cuvs_java_jar,
        cuvs_lucene_jar,
        lucene.CLASSPATH,
    )
    lucene.CLASSPATH = os.pathsep.join((str(test_classes), lucene.CLASSPATH))

    return _RuntimeFixture(
        backend_config={
            "cuvs_java_jar": str(cuvs_java_jar),
            "cuvs_lucene_jar": str(cuvs_lucene_jar),
            "java_library_path": java_library_path,
        },
        writer_selection_classes=test_classes,
    )


@pytest.fixture(scope="module")
def integration_dataset_case():
    rng = np.random.default_rng(1907)
    training_vectors = rng.standard_normal((512, 32)).astype(np.float32)
    query_ids = np.asarray([0, 137, 259, 511], dtype=np.int64)
    query_vectors = training_vectors[query_ids].copy()
    k = 5

    squared_distances = np.sum(
        (query_vectors[:, np.newaxis, :] - training_vectors[np.newaxis, :, :]) ** 2,
        axis=2,
    )
    groundtruth_neighbors = np.argsort(squared_distances, axis=1)[:, :k]
    groundtruth_distances = np.take_along_axis(
        squared_distances, groundtruth_neighbors, axis=1
    )
    return _DatasetCase(
        dataset=Dataset(
            name="pylucene-integration",
            training_vectors=training_vectors,
            query_vectors=query_vectors,
            groundtruth_neighbors=groundtruth_neighbors.astype(np.int32),
            groundtruth_distances=groundtruth_distances.astype(np.float32),
            distance_metric="euclidean",
        ),
        query_ids=query_ids,
        squared_distances=squared_distances,
        k=k,
    )


def _build_baseline(
    tmp_path_factory,
    runtime_fixture,
    dataset_case,
    *,
    algo,
    codec,
    writer_policy,
):
    index_path = tmp_path_factory.mktemp(f"{algo}-baseline") / "index"
    index = IndexConfig(
        name=f"{algo}-integration",
        algo=algo,
        build_param={"codec": codec},
        search_params=[{}],
        file=str(index_path),
    )
    backend = _backend(algo, codec, runtime_fixture.backend_config)
    try:
        build_result = backend.build(dataset_case.dataset, [index], force=True)
    finally:
        backend.cleanup()
    assert build_result.success, build_result.error_message
    return _BaselineIndex(
        algo=algo,
        codec=codec,
        writer_policy=writer_policy,
        index_path=index_path,
        index=index,
        build_result=build_result,
        dataset_case=dataset_case,
        runtime_config=runtime_fixture.backend_config,
    )


@pytest.fixture(scope="module")
def hnsw_baseline(tmp_path_factory, pylucene_runtime_config, integration_dataset_case):
    return _build_baseline(
        tmp_path_factory,
        pylucene_runtime_config,
        integration_dataset_case,
        algo="pylucene_cuvs_hnsw",
        codec=_HNSW_CODEC,
        writer_policy=_HNSW_WRITER_POLICY,
    )


@pytest.fixture(scope="module")
def cagra_baseline(tmp_path_factory, pylucene_runtime_config, integration_dataset_case):
    return _build_baseline(
        tmp_path_factory,
        pylucene_runtime_config,
        integration_dataset_case,
        algo="pylucene_cuvs_cagra",
        codec=_CAGRA_CODEC,
        writer_policy="gpu-cagra",
    )


def _assert_build_contract(baseline):
    result = baseline.build_result
    case = baseline.dataset_case
    vectors = case.dataset.training_vectors
    assert result.build_time_seconds > 0
    assert result.index_size_bytes > 0
    assert result.metadata["codec"] == baseline.codec
    assert result.metadata["pylucene_version"] != "unknown"
    assert result.metadata["writer_policy"] == baseline.writer_policy
    assert (
        result.metadata["compound_file_policy"] == _COMPOUND_FILE_POLICY[baseline.codec]
    )
    if baseline.codec == _CAGRA_CODEC:
        assert any(path.suffix == ".vemc" for path in baseline.index_path.iterdir())
        assert any(path.suffix == ".vcag" for path in baseline.index_path.iterdir())
        _assert_cagra_provenance(result.metadata, vectors.shape[0], vectors.shape[1])
        _assert_cagra_verification(result.metadata, vectors.shape[0], vectors.shape[1])
        assert (baseline.index_path / _CAGRA_PROVENANCE_FILE).is_file()
    else:
        _assert_hnsw_verification(
            result.metadata,
            baseline.codec,
            vectors.shape[0],
            vectors.shape[1],
        )
        assert (baseline.index_path / _HNSW_PROVENANCE_FILE).is_file()


def _assert_search_contract(baseline, result):
    case = baseline.dataset_case
    dataset = case.dataset
    query_vectors = dataset.query_vectors
    training_vectors = dataset.training_vectors
    assert result.success, result.error_message
    assert result.neighbors.shape == (query_vectors.shape[0], case.k)
    assert result.distances.shape == (query_vectors.shape[0], case.k)
    np.testing.assert_array_equal(result.neighbors[:, 0], case.query_ids)
    assert np.all(np.isfinite(result.distances))
    recall = compute_recall(result.neighbors, dataset.groundtruth_neighbors, case.k)
    assert recall >= 0.75
    returned_distances = np.take_along_axis(
        case.squared_distances, result.neighbors, axis=1
    )
    np.testing.assert_allclose(
        result.distances, returned_distances, rtol=1e-5, atol=1e-5
    )
    assert np.all(np.diff(result.distances, axis=1) >= -1e-6)
    assert result.search_time_ms > 0
    assert result.metadata["latency_seconds"] > 0
    assert result.queries_per_second > 0
    assert result.metadata["codec"] == baseline.codec
    assert result.metadata["pylucene_version"] != "unknown"
    assert (
        result.metadata["compound_file_policy"] == _COMPOUND_FILE_POLICY[baseline.codec]
    )
    assert result.metadata["num_batches"] == 2
    assert result.metadata["mode"] == "latency"
    assert set(result.latency_percentiles) == {"p50", "p95", "p99"}
    if baseline.codec == _CAGRA_CODEC:
        _assert_cagra_provenance(
            result.metadata,
            training_vectors.shape[0],
            training_vectors.shape[1],
        )
        _assert_cagra_verification(
            result.metadata,
            training_vectors.shape[0],
            training_vectors.shape[1],
        )
    else:
        assert "cagra_verification" not in result.metadata
        _assert_hnsw_verification(
            result.metadata,
            baseline.codec,
            training_vectors.shape[0],
            training_vectors.shape[1],
        )


def test_build_hnsw_with_real_pylucene_runtime(hnsw_baseline):
    _assert_build_contract(hnsw_baseline)


def test_build_cagra_with_real_pylucene_runtime(cagra_baseline):
    _assert_build_contract(cagra_baseline)


def _assert_reuse_contract(baseline_index):
    backend = _backend(
        baseline_index.algo,
        baseline_index.codec,
        baseline_index.runtime_config,
    )
    try:
        result = backend.build(
            baseline_index.dataset_case.dataset,
            [baseline_index.index],
            force=False,
        )
    finally:
        backend.cleanup()
    assert result.success, result.error_message
    assert result.metadata["skipped"] is True
    assert (
        result.metadata["compound_file_policy"]
        == _COMPOUND_FILE_POLICY[baseline_index.codec]
    )
    vectors = baseline_index.dataset_case.dataset.training_vectors
    if baseline_index.codec == _CAGRA_CODEC:
        _assert_cagra_provenance(result.metadata, vectors.shape[0], vectors.shape[1])
        _assert_cagra_verification(result.metadata, vectors.shape[0], vectors.shape[1])
    else:
        _assert_hnsw_verification(
            result.metadata,
            baseline_index.codec,
            vectors.shape[0],
            vectors.shape[1],
        )


def test_reuse_real_hnsw_index(hnsw_baseline):
    _assert_reuse_contract(hnsw_baseline)


def test_reuse_real_cagra_index(cagra_baseline):
    _assert_reuse_contract(cagra_baseline)


def _assert_baseline_search(baseline_index):
    backend = _backend(
        baseline_index.algo,
        baseline_index.codec,
        baseline_index.runtime_config,
    )
    try:
        result = _single_search_result(
            backend.search(
                baseline_index.dataset_case.dataset,
                [baseline_index.index],
                k=baseline_index.dataset_case.k,
                batch_size=2,
            )
        )
    finally:
        backend.cleanup()
    _assert_search_contract(baseline_index, result)


def test_search_real_hnsw_index(hnsw_baseline):
    _assert_baseline_search(hnsw_baseline)


def test_search_real_cagra_index(cagra_baseline):
    _assert_baseline_search(cagra_baseline)


def test_cagra_rejects_dataset_vector_count_mismatch(tmp_path, cagra_baseline):
    _, index = _copy_baseline(cagra_baseline, tmp_path)
    case = cagra_baseline.dataset_case
    dataset = Dataset(
        name="pylucene-integration",
        training_vectors=case.dataset.training_vectors[:-1],
        query_vectors=case.dataset.query_vectors,
        distance_metric="euclidean",
    )
    backend = _backend(
        cagra_baseline.algo,
        cagra_baseline.codec,
        cagra_baseline.runtime_config,
    )
    try:
        build_result = backend.build(dataset, [index], force=False)
        search_result = _single_search_result(
            backend.search(dataset, [index], k=case.k)
        )
    finally:
        backend.cleanup()
    assert not build_result.success
    assert "vector count does not match the dataset: 512 != 511" in (
        build_result.error_message
    )
    assert not search_result.success
    assert "vector count does not match the dataset: 512 != 511" in (
        search_result.error_message
    )


def test_cagra_rejects_missing_vector_data(tmp_path, cagra_baseline):
    index_path, index = _copy_baseline(cagra_baseline, tmp_path)
    next(index_path.glob("*.vcag")).unlink()
    result = _search_copied_index(cagra_baseline, index)
    assert not result.success
    assert "cannot read" in result.error_message


def test_cagra_rejects_truncated_vector_data(tmp_path, cagra_baseline):
    index_path, index = _copy_baseline(cagra_baseline, tmp_path)
    data_path = next(index_path.glob("*.vcag"))
    data = data_path.read_bytes()
    data_path.write_bytes(data[:-1])
    result = _search_copied_index(cagra_baseline, index)
    assert not result.success
    assert "do not exactly cover" in result.error_message


def test_cagra_rejects_corrupted_vector_data(tmp_path, cagra_baseline):
    index_path, index = _copy_baseline(cagra_baseline, tmp_path)
    data_path = next(index_path.glob("*.vcag"))
    data = bytearray(data_path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    original_stat = data_path.stat()
    data_path.write_bytes(data)
    os.utime(
        data_path,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    result = _search_copied_index(cagra_baseline, index)
    assert not result.success
    assert "checksum" in result.error_message.lower()


def test_cagra_rejects_committed_deletion(tmp_path, cagra_baseline):
    index_path, index = _copy_baseline(cagra_baseline, tmp_path)
    backend = _backend(
        cagra_baseline.algo,
        cagra_baseline.codec,
        cagra_baseline.runtime_config,
    )
    try:
        _commit_one_deletion(backend, index_path)
        with pytest.raises(RuntimeError, match="committed deletions"):
            backend._get_runtime().verify_cagra_index(index_path)
        result = _single_search_result(
            backend.search(
                cagra_baseline.dataset_case.dataset,
                [index],
                k=cagra_baseline.dataset_case.k,
            )
        )
    finally:
        backend.cleanup()
    assert not result.success
    assert "does not match the current Lucene commit" in result.error_message


def test_cagra_rejects_corrupted_metadata(tmp_path, cagra_baseline):
    index_path, _ = _copy_baseline(cagra_baseline, tmp_path)
    metadata_path = next(index_path.glob("*.vemc"))
    contents = bytearray(metadata_path.read_bytes())
    contents[-1] ^= 0xFF
    metadata_path.write_bytes(contents)
    runtime = _PyLuceneRuntime.create(cagra_baseline.runtime_config)
    with pytest.raises(RuntimeError, match="metadata format v0"):
        runtime.verify_cagra_index(index_path)


def test_hnsw_rejects_missing_provenance(tmp_path, hnsw_baseline):
    index_path, index = _copy_baseline(hnsw_baseline, tmp_path)
    (index_path / _HNSW_PROVENANCE_FILE).unlink()
    result = _search_copied_index(hnsw_baseline, index)
    assert not result.success
    assert "provenance manifest is missing" in result.error_message


def test_hnsw_rejects_unreadable_provenance(tmp_path, hnsw_baseline):
    index_path, index = _copy_baseline(hnsw_baseline, tmp_path)
    (index_path / _HNSW_PROVENANCE_FILE).write_text("{")
    result = _search_copied_index(hnsw_baseline, index)
    assert not result.success
    assert "provenance manifest cannot be read" in result.error_message


def test_hnsw_rejects_changed_lucene_commit(tmp_path, hnsw_baseline):
    index_path, index = _copy_baseline(hnsw_baseline, tmp_path)
    segment_path = next(index_path.glob("segments_*"))
    segment_path.write_bytes(segment_path.read_bytes() + b"stale")
    result = _search_copied_index(hnsw_baseline, index)
    assert not result.success
    assert "does not match the current Lucene commit" in result.error_message


def test_hnsw_rejects_wrong_codec_provenance(tmp_path, hnsw_baseline):
    index_path, index = _copy_baseline(hnsw_baseline, tmp_path)
    manifest_path = index_path / _HNSW_PROVENANCE_FILE
    payload = json.loads(manifest_path.read_bytes())
    payload["codec"] = _CAGRA_CODEC
    manifest_path.write_text(json.dumps(payload))
    result = _search_copied_index(hnsw_baseline, index)
    assert not result.success
    assert "does not match the requested codec" in result.error_message


def _build_and_force_merge(runtime, index_path, vectors, writer_config):
    directory = runtime.FSDirectory.open(runtime.Paths.get(str(index_path)))
    writer = None
    reader = None
    try:
        writer = runtime.IndexWriter(directory, writer_config)
        for document_id, vector in enumerate(vectors):
            writer.addDocument(runtime._vector_document(document_id, vector))
        writer.commit()

        reader = runtime.DirectoryReader.open(writer)
        initial_segment_count = int(reader.leaves().size())
        reader.close()
        reader = None

        writer.forceMerge(1)
        writer.commit()
        reader = runtime.DirectoryReader.open(writer)
        final_segment_count = int(reader.leaves().size())
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
    return initial_segment_count, final_segment_count


def _assert_default_merge_scheduler(writer_config):
    assert (
        str(writer_config.getMergeScheduler().getClass().getName())
        == "org.apache.lucene.index.ConcurrentMergeScheduler"
    )


def _committed_segment_compound_flags(runtime, index_path):
    verifier = runtime._cagra_verifier
    runtime.attach_current_thread()
    directory = verifier.FSDirectory.open(verifier.Paths.get(str(index_path)))
    try:
        segment_infos = verifier.SegmentInfos.readLatestCommit(directory)
        return [
            bool(
                verifier.SegmentCommitInfo.cast_(raw_segment).info.getUseCompoundFile()
            )
            for raw_segment in segment_infos
        ]
    finally:
        directory.close()


def test_real_hnsw_codec_selects_gpu_writer(tmp_path, pylucene_runtime_config):
    """Exercise GPU writer selection during default-scheduler merges."""
    runtime = _PyLuceneRuntime.create(pylucene_runtime_config.backend_config)
    runtime.attach_current_thread()
    reflected_codec = runtime.Class.forName(_WRITER_SELECTION_CODEC).newInstance()
    java_codec = runtime.Codec.cast_(reflected_codec)
    index_path = tmp_path / "hnsw-writer-selection-index"
    index_path.mkdir()
    vectors = np.random.default_rng(1907).standard_normal((6, 32)).astype(np.float32)
    default_config = runtime.IndexWriterConfig()
    writer_config = runtime._new_index_writer_config(
        _BuildCodec(
            codec_name=_HNSW_CODEC,
            java_codec=java_codec,
            writer_policy=_HNSW_WRITER_POLICY,
        )
    )
    assert bool(writer_config.getUseCompoundFile()) == bool(
        default_config.getUseCompoundFile()
    )
    assert float(writer_config.getMergePolicy().getNoCFSRatio()) == pytest.approx(
        float(default_config.getMergePolicy().getNoCFSRatio())
    )
    _assert_default_merge_scheduler(writer_config)
    writer_config.setMaxBufferedDocs(2)

    initial_segments, final_segments = _build_and_force_merge(
        runtime, index_path, vectors, writer_config
    )
    assert initial_segments >= 2
    assert final_segments == 1

    writer_class, writer_calls = _writer_diagnostics(java_codec)
    assert writer_class == _GPU_HNSW_WRITER
    assert writer_calls >= 4
    search = runtime.search_index(index_path, vectors[[3]], k=2, batch_size=1)
    assert search.hits[0][0].document_id == 3


def test_real_cagra_codec_merges_without_compound_files(
    tmp_path, pylucene_runtime_config, integration_dataset_case
):
    """Exercise CAGRA flush and merge layout through production configuration."""
    runtime = _PyLuceneRuntime.create(pylucene_runtime_config.backend_config)
    java_codec = runtime.resolve_codec(_CAGRA_CODEC)
    index_path = tmp_path / "cagra-merge-index"
    index_path.mkdir()
    vectors = integration_dataset_case.dataset.training_vectors
    writer_config = runtime._new_index_writer_config(
        _BuildCodec(
            codec_name=_CAGRA_CODEC,
            java_codec=java_codec,
            writer_policy="gpu-cagra",
        )
    )
    assert not bool(writer_config.getUseCompoundFile())
    assert float(writer_config.getMergePolicy().getNoCFSRatio()) == 0.0
    _assert_default_merge_scheduler(writer_config)
    writer_config.setMaxBufferedDocs(128)

    initial_segments, final_segments = _build_and_force_merge(
        runtime, index_path, vectors, writer_config
    )
    assert initial_segments >= 2
    assert final_segments == 1
    assert _committed_segment_compound_flags(runtime, index_path) == [False]

    suffixes = {path.suffix for path in index_path.iterdir()}
    assert ".vemc" in suffixes
    assert ".vcag" in suffixes
    assert suffixes.isdisjoint({".cfs", ".cfe"})
    verification = runtime.verify_cagra_index(
        index_path,
        expected_vector_count=vectors.shape[0],
        expected_dimensions=vectors.shape[1],
    )
    assert verification.segment_count == 1
    search = runtime.search_index(index_path, vectors[[137]], k=1, batch_size=1)
    assert search.hits[0][0].document_id == 137


def test_real_hnsw_codec_falls_back_to_cpu_in_fresh_process(
    pylucene_runtime_config,
):
    """Prove production fallback when CUDA devices are hidden at startup."""
    environment = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[2])
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "CUVS_LUCENE_CUVS_JAVA_JAR": (
                pylucene_runtime_config.backend_config["cuvs_java_jar"]
            ),
            "CUVS_LUCENE_JAR": pylucene_runtime_config.backend_config[
                "cuvs_lucene_jar"
            ],
            "JAVA_LIBRARY_PATH": pylucene_runtime_config.backend_config[
                "java_library_path"
            ],
            "PYLUCENE_WRITER_SELECTION_CLASSES": str(
                pylucene_runtime_config.writer_selection_classes
            ),
            "PYTHONPATH": os.pathsep.join(
                value for value in (source_root, environment.get("PYTHONPATH")) if value
            ),
        }
    )
    completed = subprocess.run(
        [sys.executable, str(_CPU_FALLBACK_PROBE)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    assert f"writerClass={_CPU_HNSW_WRITER}" in completed.stdout
    assert "firstHit=2" in completed.stdout


def test_real_verifier_rejects_cagra_brute_force_fallback(
    tmp_path, pylucene_runtime_config
):
    """Reject cuVS-Lucene's real one-vector fallback before search."""
    runtime = _PyLuceneRuntime.create(pylucene_runtime_config.backend_config)
    vectors = np.ones((1, 32), dtype=np.float32)
    index_path = tmp_path / "cagra-fallback-index"
    index_path.mkdir()
    java_codec = runtime.resolve_codec(_CAGRA_CODEC)
    runtime.build_index(
        index_path,
        vectors,
        _BuildCodec(
            codec_name=_CAGRA_CODEC,
            java_codec=java_codec,
            writer_policy="gpu-cagra",
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
            **pylucene_runtime_config.backend_config,
        }
    )
    backend._runtime = runtime
    try:
        result = _single_search_result(backend.search(dataset, [index], k=1))
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
        (query_vectors[:, np.newaxis, :] - training_vectors[np.newaxis, :, :]) ** 2,
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
        json.dumps(
            {
                "backend": "pylucene",
                **pylucene_runtime_config.backend_config,
            }
        )
    )

    environment = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[2])
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_root, environment.get("PYTHONPATH")) if value
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
    index_name = f"pylucene_cuvs_hnsw[group=test][codec={codec}]"
    index_path = dataset_dir / "index" / index_name
    assert any(index_path.glob("segments_*"))
    assert (index_path / _HNSW_PROVENANCE_FILE).is_file()

    result_path = dataset_dir / "result"
    build_csv = result_path / "build" / "pylucene_cuvs_hnsw,test.csv"
    search_stem = f"pylucene_cuvs_hnsw,test,k{k},bs2"
    with build_csv.open(newline="") as file:
        build_rows = list(csv.DictReader(file))
    assert len(build_rows) == 1

    build_row = build_rows[0]
    assert build_row["index_name"] == index_name
    assert float(build_row["time"]) > 0
    assert build_row["codec"] == codec
    assert build_row["writer_policy"] == _HNSW_WRITER_POLICY
    assert build_row["compound_file_policy"] == "lucene-default"

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
    assert float(csv_row["p50"]) > 0
    assert float(csv_row["p95"]) > 0
    assert float(csv_row["p99"]) > 0
    assert csv_row["codec"] == codec
    assert csv_row["compound_file_policy"] == "lucene-default"
    assert (result_path / "search" / f"{search_stem},latency.csv").is_file()
    assert (result_path / "search" / f"{search_stem},throughput.csv").is_file()
