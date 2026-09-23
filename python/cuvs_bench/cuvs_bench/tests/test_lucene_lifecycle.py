#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Index publication, reuse, and provenance tests for the Lucene backend."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from _lucene_test_support import (
    RecordingRuntime,
    _backend_and_index,
    _dataset,
    _write_fbin,
)
from cuvs_bench.backends._lucene_runtime import CPU_HNSW_CODEC
from cuvs_bench.backends.base import Dataset
from cuvs_bench.backends.lucene import (
    CAGRA_ALGORITHM,
    CPU_HNSW_ALGORITHM,
    LuceneBackend,
    _prewarm_index_files,
    _source_identity,
)
from cuvs_bench.orchestrator.config_loaders import IndexConfig


def test_index_prewarm_reads_every_regular_file(tmp_path: Path) -> None:
    (tmp_path / "segments_1").write_bytes(b"segments")
    (tmp_path / "vectors.vec").write_bytes(b"vector-data")
    (tmp_path / "ignored-directory").mkdir()

    timing = _prewarm_index_files(tmp_path)

    assert timing.file_count == 2
    assert timing.bytes_read == len(b"segments") + len(b"vector-data")
    assert timing.wall_ns >= 0


def test_index_prewarm_reports_the_file_that_could_not_be_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unreadable = tmp_path / "vectors.vec"
    unreadable.write_bytes(b"vector-data")
    original_open = Path.open

    def fail_open(path: Path, *args, **kwargs):
        if path == unreadable:
            raise PermissionError("read denied")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_open)

    with pytest.raises(
        RuntimeError,
        match=f"Failed to prewarm Lucene index file: {unreadable}",
    ) as failure:
        _prewarm_index_files(tmp_path)

    assert isinstance(failure.value.__cause__, PermissionError)


def test_index_prewarm_refuses_symbolic_links(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"index-data")
    link = tmp_path / "linked-index-file"
    link.symlink_to(target)

    with pytest.raises(RuntimeError, match="refuses symbolic links"):
        _prewarm_index_files(tmp_path)


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


def test_search_requests_rebuild_for_the_legacy_stored_id_schema(
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
    manifest["schema_version"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = backend.search(dataset, [index], k=2)[0]

    assert not result.success
    assert "Unsupported Lucene index manifest" in result.error_message
    assert "rerun with --force" in result.error_message
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
        "build_route_policy": "cpu_hnsw",
        "segment_count": 1,
        "field_count": 1,
        "vector_count": 4,
        "dimensions": 2,
    }
    assert len(runtime.build_calls) == 1
    assert runtime.artifact_verification_count == 2
