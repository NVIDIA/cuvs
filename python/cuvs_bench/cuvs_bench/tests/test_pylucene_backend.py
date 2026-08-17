#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Build and search behavior tests for the PyLucene backend."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import cuvs_bench.backends.pylucene as pylucene_backend
from cuvs_bench.backends.pylucene import _SearchHit
from cuvs_bench.tests._pylucene_test_utils import (
    _CAGRA_CODEC,
    _HNSW_CODEC,
    _FakeRuntime,
    _backend,
    _dataset,
    _index,
    _prepare_cagra_index,
    _prepare_hnsw_index,
)


def _only_search_result(results):
    assert len(results) == 1
    return results[0]


@pytest.mark.parametrize(
    ("score", "distance"),
    [(1.0, 0.0), (0.5, 1.0), (0.2, 4.0), (0.0, np.inf)],
)
def test_score_to_squared_euclidean(score, distance):
    assert pylucene_backend._score_to_squared_euclidean(
        score
    ) == pytest.approx(distance)


def test_build_dry_run_does_not_initialize_pylucene(tmp_path):
    backend = _backend()
    result = backend.build(
        _dataset(), [_index(tmp_path / "index")], dry_run=True
    )

    assert result.success
    assert backend._runtime is None
    assert result.metadata["codec"] == _HNSW_CODEC
    assert result.metadata["compound_file_policy"] == "lucene-default"


def test_build_creates_index_and_records_hnsw_writer_policy(tmp_path):
    runtime = _FakeRuntime()
    backend = _backend(runtime)
    index_path = tmp_path / "index"

    result = backend.build(_dataset(), [_index(index_path)], force=True)

    assert result.success
    assert result.index_size_bytes > len(b"index")
    assert result.metadata["writer_policy"] == "gpu-with-cpu-fallback"
    assert result.metadata["compound_file_policy"] == "lucene-default"
    assert result.metadata["hnsw_verification"] == {
        "status": "gpu-with-cpu-fallback-provenance",
        "schema_version": 3,
        "codec": _HNSW_CODEC,
        "writer_policy": "gpu-with-cpu-fallback",
        "compound_file_policy": "lucene-default",
        "vector_count": 10,
        "dimensions": 4,
        "commit_file_count": 1,
    }
    assert (index_path / pylucene_backend._HNSW_PROVENANCE_FILE).is_file()
    assert runtime.resolve_calls == [_HNSW_CODEC]
    assert len(runtime.build_calls) == 1
    assert runtime.build_calls[0][2].writer_policy == ("gpu-with-cpu-fallback")
    assert runtime.verification_calls == []


def test_build_verifies_persisted_cagra_index(tmp_path):
    runtime = _FakeRuntime()
    index_path = tmp_path / "index"

    result = _backend(runtime, codec=_CAGRA_CODEC).build(
        _dataset(),
        [_index(index_path, codec=_CAGRA_CODEC)],
        force=True,
    )

    assert result.success
    assert result.metadata["compound_file_policy"] == "disabled"
    assert runtime.verification_calls == [(index_path, 10, 4)]
    assert result.metadata["cagra_provenance"] == {
        "status": "gpu-cagra-provenance",
        "schema_version": 3,
        "codec": _CAGRA_CODEC,
        "writer_policy": "gpu-cagra",
        "compound_file_policy": "disabled",
        "vector_count": 10,
        "dimensions": 4,
        "commit_file_count": 1,
    }
    assert result.metadata["cagra_verification"] == {
        "status": "cagra-only",
        "segment_count": 1,
        "field_count": 1,
        "vector_count": 10,
        "dimensions": 4,
    }
    assert (index_path / pylucene_backend._CAGRA_PROVENANCE_FILE).is_file()


def test_build_rejects_persisted_cagra_fallback_and_removes_index(tmp_path):
    runtime = _FakeRuntime()
    runtime.verification_error = RuntimeError("persisted brute-force index")
    index_path = tmp_path / "index"

    result = _backend(runtime, codec=_CAGRA_CODEC).build(
        _dataset(),
        [_index(index_path, codec=_CAGRA_CODEC)],
        force=True,
    )

    assert not result.success
    assert "persisted brute-force index" in result.error_message
    assert runtime.verification_calls == [(index_path, 10, 4)]
    assert len(runtime.build_calls) == 1
    assert not index_path.exists()


def test_build_reuses_existing_index_without_runtime(tmp_path):
    index_path = tmp_path / "index"
    _prepare_hnsw_index(index_path)
    runtime = _FakeRuntime()
    backend = _backend(runtime)

    result = backend.build(_dataset(), [_index(index_path)])

    assert result.success
    assert result.metadata["skipped"] is True
    assert result.metadata["hnsw_verification"]["status"] == (
        "gpu-with-cpu-fallback-provenance"
    )
    assert runtime.resolve_calls == []
    assert runtime.build_calls == []
    assert runtime.verification_calls == []


def test_build_reports_reused_index_sizing_failure(tmp_path, monkeypatch):
    index_path = tmp_path / "index"
    _prepare_hnsw_index(index_path)
    runtime = _FakeRuntime()

    def fail_index_size(_index_path):
        raise PermissionError("cannot read index size")

    monkeypatch.setattr(pylucene_backend, "_index_size", fail_index_size)

    result = _backend(runtime).build(_dataset(), [_index(index_path)])

    assert not result.success
    assert "cannot read index size" in result.error_message
    assert runtime.build_calls == []


def test_build_rejects_reused_hnsw_index_without_provenance(tmp_path):
    index_path = tmp_path / "index"
    index_path.mkdir()
    (index_path / "segments_1").write_bytes(b"existing")
    runtime = _FakeRuntime()

    result = _backend(runtime).build(_dataset(), [_index(index_path)])

    assert not result.success
    assert "provenance manifest is missing" in result.error_message
    assert runtime.build_calls == []


def test_build_verifies_reused_cagra_index(tmp_path):
    index_path = tmp_path / "index"
    _prepare_cagra_index(index_path)
    runtime = _FakeRuntime()

    result = _backend(runtime, codec=_CAGRA_CODEC).build(
        _dataset(),
        [_index(index_path, codec=_CAGRA_CODEC)],
    )

    assert result.success
    assert result.metadata["skipped"] is True
    assert runtime.verification_calls == [(index_path, 10, 4)]
    assert result.metadata["cagra_provenance"]["status"] == (
        "gpu-cagra-provenance"
    )
    assert result.metadata["cagra_verification"] == {
        "status": "cagra-only",
        "segment_count": 1,
        "field_count": 1,
        "vector_count": 10,
        "dimensions": 4,
    }
    assert runtime.build_calls == []


def test_build_rejects_reused_cagra_index_that_cannot_be_verified(tmp_path):
    index_path = tmp_path / "index"
    _prepare_cagra_index(index_path)
    segments_file = index_path / "segments_1"
    runtime = _FakeRuntime()
    runtime.verification_error = RuntimeError("persisted brute-force index")

    result = _backend(runtime, codec=_CAGRA_CODEC).build(
        _dataset(),
        [_index(index_path, codec=_CAGRA_CODEC)],
    )

    assert not result.success
    assert "persisted brute-force index" in result.error_message
    assert runtime.verification_calls == [(index_path, 10, 4)]
    assert segments_file.read_bytes() == b"index"
    assert runtime.build_calls == []


def test_build_rejects_reused_cagra_index_without_provenance_before_runtime(
    tmp_path,
):
    index_path = tmp_path / "index"
    index_path.mkdir()
    (index_path / "segments_1").write_bytes(b"existing")
    runtime = _FakeRuntime()

    result = _backend(runtime, codec=_CAGRA_CODEC).build(
        _dataset(),
        [_index(index_path, codec=_CAGRA_CODEC)],
    )

    assert not result.success
    assert "CAGRA provenance manifest is missing" in result.error_message
    assert runtime.verification_calls == []
    assert runtime.search_calls == []


@pytest.mark.parametrize("force", [False, True])
def test_build_rejects_unrelated_existing_directory(tmp_path, force):
    index_path = tmp_path / "index"
    index_path.mkdir()
    unrelated_file = index_path / "unrelated"
    unrelated_file.write_bytes(b"not a Lucene index")
    runtime = _FakeRuntime()

    result = _backend(runtime).build(
        _dataset(), [_index(index_path)], force=force
    )

    assert not result.success
    assert "does not contain a Lucene segments file" in result.error_message
    assert unrelated_file.exists()


def test_build_rejects_existing_index_path_that_is_a_file(tmp_path):
    index_path = tmp_path / "index"
    index_path.write_bytes(b"not a directory")

    result = _backend(_FakeRuntime()).build(
        _dataset(), [_index(index_path)], force=True
    )

    assert not result.success
    assert "not a directory" in result.error_message
    assert index_path.read_bytes() == b"not a directory"


def test_build_force_replaces_existing_index(tmp_path):
    index_path = tmp_path / "index"
    index_path.mkdir()
    (index_path / "segments_1").write_bytes(b"existing")
    old_file = index_path / "old"
    old_file.write_bytes(b"old")
    runtime = _FakeRuntime()

    result = _backend(runtime).build(
        _dataset(), [_index(index_path)], force=True
    )

    assert result.success
    assert not old_file.exists()
    assert (index_path / "segments_1").exists()


def test_build_refuses_index_outside_configured_root(tmp_path):
    allowed_root = tmp_path / "dataset" / "index"
    external_index = tmp_path / "external-index"
    external_index.mkdir()
    (external_index / "segments_1").write_bytes(b"existing")
    old_file = external_index / "old"
    old_file.write_bytes(b"must remain")
    runtime = _FakeRuntime()
    backend = _backend(runtime)
    backend.config["index_root"] = str(allowed_root)

    result = backend.build(_dataset(), [_index(external_index)], force=True)

    assert not result.success
    assert "immediate child" in result.error_message
    assert old_file.read_bytes() == b"must remain"
    assert runtime.resolve_calls == []
    assert runtime.build_calls == []


def test_safe_remove_rechecks_configured_root(tmp_path):
    allowed_root = tmp_path / "allowed"
    external_index = tmp_path / "external-index"
    external_index.mkdir()
    old_file = external_index / "old"
    old_file.write_bytes(b"must remain")

    with pytest.raises(ValueError, match="outside its configured root"):
        pylucene_backend._safe_remove_index(external_index, allowed_root)

    assert old_file.read_bytes() == b"must remain"


def test_build_preserves_existing_index_if_codec_preflight_fails(tmp_path):
    index_path = tmp_path / "index"
    index_path.mkdir()
    (index_path / "segments_1").write_bytes(b"existing")
    old_file = index_path / "old"
    old_file.write_bytes(b"old")
    runtime = _FakeRuntime()
    runtime.resolve_error = RuntimeError("codec unavailable")

    result = _backend(runtime).build(
        _dataset(), [_index(index_path)], force=True
    )

    assert not result.success
    assert "codec unavailable" in result.error_message
    assert old_file.exists()


def test_build_accepts_production_hnsw_fallback_policy(tmp_path):
    runtime = _FakeRuntime()
    index_path = tmp_path / "index"

    result = _backend(runtime).build(
        _dataset(), [_index(index_path)], force=True
    )

    assert result.success
    assert result.metadata["writer_policy"] == "gpu-with-cpu-fallback"
    assert runtime.build_calls[0][2].writer_policy == ("gpu-with-cpu-fallback")


@pytest.mark.parametrize(
    ("dataset", "error"),
    [
        (_dataset(dtype=np.float64), "float32"),
        (_dataset(distance_metric="cosine"), "Euclidean"),
    ],
)
def test_build_rejects_unsupported_dataset(dataset, error, tmp_path):
    result = _backend(_FakeRuntime()).build(
        dataset, [_index(tmp_path / "index")], force=True
    )

    assert not result.success
    assert error in result.error_message


def test_build_rejects_nonfinite_vectors(tmp_path):
    dataset = _dataset()
    dataset.training_vectors[0, 0] = np.nan

    result = _backend(_FakeRuntime()).build(
        dataset, [_index(tmp_path / "index")], force=True
    )

    assert not result.success
    assert "finite" in result.error_message


@pytest.mark.parametrize(
    "codec",
    [
        _HNSW_CODEC,
        _CAGRA_CODEC,
    ],
    ids=["hnsw", "cagra"],
)
def test_build_rejects_single_vector_cuvs_bypass(tmp_path, codec):
    runtime = _FakeRuntime()

    result = _backend(runtime, codec=codec).build(
        _dataset(n_base=1),
        [_index(tmp_path / "index", codec=codec)],
        force=True,
    )

    assert not result.success
    assert "at least two training vectors" in result.error_message
    assert "does not invoke cuVS" in result.error_message
    assert not (tmp_path / "index").exists()


def test_build_rejects_unknown_codec(tmp_path):
    result = _backend(_FakeRuntime()).build(
        _dataset(),
        [_index(tmp_path / "index", codec="UnknownCodec")],
        force=True,
    )

    assert not result.success
    assert "Unsupported PyLucene codec" in result.error_message


@pytest.mark.parametrize(
    "codec",
    [None, "", 0, False],
    ids=["none", "empty", "zero", "false"],
)
def test_explicit_invalid_codec_does_not_fall_back_to_backend_config(
    tmp_path, codec
):
    index = _index(tmp_path / "index")
    index.build_param["codec"] = codec
    backend = _backend(codec=_HNSW_CODEC)

    build_result = backend.build(_dataset(), [index], dry_run=True)
    search_result = _only_search_result(
        backend.search(_dataset(), [index], k=3, dry_run=True)
    )

    for result in (build_result, search_result):
        assert not result.success
        assert f"Unsupported PyLucene codec {codec!r}" in result.error_message


def test_build_rejects_unsupported_parameter(tmp_path):
    index = _index(tmp_path / "index")
    index.build_param["ignored"] = 1

    result = _backend(_FakeRuntime()).build(_dataset(), [index])

    assert not result.success
    assert "Unsupported PyLucene build parameter" in result.error_message


@pytest.mark.parametrize("index_count", [0, 2])
def test_build_requires_exactly_one_index(index_count, tmp_path):
    indexes = [
        _index(tmp_path / f"index-{index_id}")
        for index_id in range(index_count)
    ]

    result = _backend(_FakeRuntime()).build(_dataset(), indexes, force=True)

    assert not result.success
    assert "exactly one" in result.error_message


@pytest.mark.parametrize(
    ("vectors", "error"),
    [
        (np.empty((0, 4), dtype=np.float32), "at least one vector"),
        (np.ones(4, dtype=np.float32), "two-dimensional"),
        (np.ones((1, 4097), dtype=np.float32), "4096"),
    ],
)
def test_build_rejects_invalid_vector_shape(vectors, error, tmp_path):
    dataset = _dataset()
    dataset.training_vectors = vectors

    result = _backend(_FakeRuntime()).build(
        dataset, [_index(tmp_path / "index")], force=True
    )

    assert not result.success
    assert error in result.error_message


def test_build_removes_partial_index_after_runtime_failure(tmp_path):
    runtime = _FakeRuntime()
    runtime.build_error = RuntimeError("Java build failed")
    index_path = tmp_path / "index"

    result = _backend(runtime).build(
        _dataset(), [_index(index_path)], force=True
    )

    assert not result.success
    assert "Java build failed" in result.error_message
    assert not index_path.exists()


def test_build_reports_chained_runtime_failure(tmp_path):
    runtime = _FakeRuntime()
    runtime.build_error = RuntimeError("rollback failed")
    runtime.build_error.__cause__ = RuntimeError("add failed")
    index_path = tmp_path / "index"

    result = _backend(runtime).build(
        _dataset(), [_index(index_path)], force=True
    )

    assert not result.success
    assert "RuntimeError: rollback failed" in result.error_message
    assert "caused by RuntimeError: add failed" in result.error_message
    assert not index_path.exists()


def test_build_removes_partial_index_and_reraises_interrupt(tmp_path):
    runtime = _FakeRuntime()
    runtime.build_error = KeyboardInterrupt()
    index_path = tmp_path / "index"

    with pytest.raises(KeyboardInterrupt):
        _backend(runtime).build(_dataset(), [_index(index_path)], force=True)

    assert not index_path.exists()


def test_build_preserves_interrupt_when_partial_index_cleanup_fails(
    tmp_path, monkeypatch
):
    runtime = _FakeRuntime()
    runtime.build_error = KeyboardInterrupt()
    index_path = tmp_path / "index"

    def fail_cleanup(_index_path, trusted_index_root=None):
        raise PermissionError("cleanup denied")

    monkeypatch.setattr(pylucene_backend, "_safe_remove_index", fail_cleanup)
    with pytest.raises(KeyboardInterrupt) as exc_info:
        _backend(runtime).build(_dataset(), [_index(index_path)], force=True)

    assert exc_info.value.__notes__ == [
        "Failed to remove partial index: PermissionError: cleanup denied"
    ]
    assert index_path.exists()


def test_build_reports_partial_index_cleanup_failure(tmp_path, monkeypatch):
    runtime = _FakeRuntime()
    runtime.build_error = RuntimeError("Java build failed")
    index_path = tmp_path / "index"

    def fail_cleanup(_index_path, trusted_index_root=None):
        raise PermissionError("cleanup denied")

    monkeypatch.setattr(pylucene_backend, "_safe_remove_index", fail_cleanup)
    result = _backend(runtime).build(
        _dataset(), [_index(index_path)], force=True
    )

    assert not result.success
    assert "Java build failed" in result.error_message
    assert "failed to remove partial index" in result.error_message
    assert "cleanup denied" in result.error_message
    assert index_path.exists()


def test_search_dry_run_does_not_initialize_pylucene(tmp_path):
    backend = _backend()

    result = _only_search_result(
        backend.search(
            _dataset(), [_index(tmp_path / "index")], k=3, dry_run=True
        )
    )

    assert result.success
    assert backend._runtime is None
    assert result.neighbors.shape == (0, 3)


def test_search_converts_hits_scores_and_padding(tmp_path):
    index_path = tmp_path / "index"
    _prepare_hnsw_index(index_path)
    runtime = _FakeRuntime(
        hits=[
            [
                _SearchHit(document_id=3, score=1.0),
                _SearchHit(document_id=7, score=0.5),
            ],
            [_SearchHit(document_id=2, score=0.2)],
        ]
    )

    result = _only_search_result(
        _backend(runtime).search(
            _dataset(), [_index(index_path)], k=3, batch_size=1
        )
    )

    assert result.success
    np.testing.assert_array_equal(
        result.neighbors,
        np.array([[3, 7, -1], [2, -1, -1]], dtype=np.int64),
    )
    np.testing.assert_allclose(result.distances[0, :2], [0.0, 1.0])
    assert result.distances[1, 0] == pytest.approx(4.0)
    assert np.isinf(result.distances[:, -1]).all()
    assert result.search_time_ms == pytest.approx(2.0)
    assert result.metadata["latency_seconds"] == pytest.approx(0.001)
    assert result.queries_per_second == pytest.approx(1000.0)
    assert result.latency_percentiles == {
        "p50": 1.0,
        "p95": 1.0,
        "p99": 1.0,
    }
    assert result.metadata["num_batches"] == 2
    assert result.metadata["hnsw_verification"]["status"] == (
        "gpu-with-cpu-fallback-provenance"
    )
    assert "per_search_param_results" not in result.metadata
    assert runtime.search_calls[0][3] == 1
    assert runtime.verification_calls == []


def test_search_end_to_end_timing_starts_before_input_verification(
    tmp_path, monkeypatch
):
    index_path = tmp_path / "index"
    _prepare_hnsw_index(index_path)
    backend = _backend(_FakeRuntime())
    events = []
    times = iter((10.0, 10.012))
    load_search_inputs = backend._load_search_inputs

    def record_time():
        events.append("time")
        return next(times)

    def record_input_loading(dataset):
        events.append("inputs")
        return load_search_inputs(dataset)

    monkeypatch.setattr(pylucene_backend.time, "perf_counter", record_time)
    monkeypatch.setattr(backend, "_load_search_inputs", record_input_loading)

    result = _only_search_result(
        backend.search(_dataset(), [_index(index_path)], k=3, batch_size=2)
    )

    assert result.success
    assert events == ["time", "inputs", "time"]
    assert result.metadata["end_to_end_time_ms"] == pytest.approx(12.0)
    assert result.metadata["non_query_overhead_time_ms"] == pytest.approx(11.0)


@pytest.mark.parametrize(
    ("runtime", "error"),
    [
        (
            _FakeRuntime(document_count=9),
            "document count does not match index provenance",
        ),
        (
            _FakeRuntime(
                hits=[
                    [
                        _SearchHit(document_id=3, score=1.0),
                        _SearchHit(document_id=3, score=0.5),
                    ],
                    [_SearchHit(document_id=2, score=1.0)],
                ]
            ),
            "duplicate stored ID",
        ),
        (
            _FakeRuntime(
                hits=[
                    [_SearchHit(document_id=10, score=1.0)],
                    [_SearchHit(document_id=2, score=1.0)],
                ]
            ),
            "out-of-range stored ID",
        ),
        (
            _FakeRuntime(hits=[[_SearchHit(document_id=0, score=1.0)]]),
            "unexpected number of query results",
        ),
        (
            _FakeRuntime(
                hits=[
                    [_SearchHit(document_id=0, score=float("nan"))],
                    [_SearchHit(document_id=1, score=1.0)],
                ]
            ),
            "non-finite score",
        ),
        (
            _FakeRuntime(
                hits=[
                    [_SearchHit(document_id=0, score=1.1)],
                    [_SearchHit(document_id=1, score=1.0)],
                ]
            ),
            "outside the Euclidean range",
        ),
        (
            _FakeRuntime(
                hits=[
                    [
                        _SearchHit(document_id=0, score=1.0),
                        _SearchHit(document_id=1, score=0.9),
                        _SearchHit(document_id=2, score=0.8),
                        _SearchHit(document_id=3, score=0.7),
                    ],
                    [_SearchHit(document_id=1, score=1.0)],
                ]
            ),
            "too many hits",
        ),
        (
            _FakeRuntime(batch_latencies_ms=[1.0, 2.0]),
            "unexpected number of batch latencies",
        ),
        (
            _FakeRuntime(batch_latencies_ms=[float("nan")]),
            "invalid batch latency",
        ),
    ],
    ids=[
        "document-count",
        "duplicate-id",
        "out-of-range-id",
        "query-count",
        "score",
        "score-range",
        "hit-count",
        "latency-count",
        "batch-latency",
    ],
)
def test_search_rejects_invalid_runtime_results(tmp_path, runtime, error):
    index_path = tmp_path / "index"
    _prepare_hnsw_index(index_path)

    result = _only_search_result(
        _backend(runtime).search(
            _dataset(), [_index(index_path)], k=3, batch_size=2
        )
    )

    assert not result.success
    assert error in result.error_message


def test_search_verifies_cagra_index_before_querying(tmp_path):
    index_path = tmp_path / "index"
    _prepare_cagra_index(index_path)
    runtime = _FakeRuntime()

    result = _only_search_result(
        _backend(runtime, codec=_CAGRA_CODEC).search(
            _dataset(),
            [_index(index_path, codec=_CAGRA_CODEC)],
            k=3,
        )
    )

    assert result.success
    assert runtime.verification_calls == [(index_path, 10, 4)]
    assert len(runtime.search_calls) == 1
    assert result.metadata["cagra_provenance"]["status"] == (
        "gpu-cagra-provenance"
    )
    assert result.metadata["cagra_verification"] == {
        "status": "cagra-only",
        "segment_count": 1,
        "field_count": 1,
        "vector_count": 10,
        "dimensions": 4,
    }


def test_search_rejects_unverified_cagra_index_before_querying(tmp_path):
    index_path = tmp_path / "index"
    _prepare_cagra_index(index_path)
    runtime = _FakeRuntime()
    runtime.verification_error = RuntimeError("persisted brute-force index")

    result = _only_search_result(
        _backend(runtime, codec=_CAGRA_CODEC).search(
            _dataset(),
            [_index(index_path, codec=_CAGRA_CODEC)],
            k=3,
        )
    )

    assert not result.success
    assert "persisted brute-force index" in result.error_message
    assert runtime.verification_calls == [(index_path, 10, 4)]
    assert runtime.search_calls == []


@pytest.mark.parametrize("manifest_state", ["missing", "stale"])
def test_search_rejects_invalid_cagra_provenance_before_runtime(
    tmp_path, manifest_state
):
    index_path = tmp_path / "index"
    manifest_path = _prepare_cagra_index(index_path)
    if manifest_state == "missing":
        manifest_path.unlink()
    else:
        (index_path / "segments_1").write_bytes(b"changed commit")
    runtime = _FakeRuntime()

    result = _only_search_result(
        _backend(runtime, codec=_CAGRA_CODEC).search(
            _dataset(),
            [_index(index_path, codec=_CAGRA_CODEC)],
            k=3,
        )
    )

    assert not result.success
    assert "CAGRA provenance" in result.error_message
    assert runtime.verification_calls == []
    assert runtime.search_calls == []


@pytest.mark.parametrize(
    ("index", "k", "batch_size", "search_threads", "mode", "error"),
    [
        (
            _index(Path("/tmp/index")),
            0,
            10000,
            None,
            "latency",
            "k must be positive",
        ),
        (
            _index(Path("/tmp/index")),
            3,
            10000,
            2,
            "latency",
            "one search thread",
        ),
        (
            _index(Path("/tmp/index")),
            3,
            10000,
            None,
            "throughput",
            "only latency mode",
        ),
        (
            _index(Path("/tmp/index")),
            3,
            0,
            None,
            "latency",
            "batch_size must be positive",
        ),
        (
            _index(Path("/tmp/index"), search_params=[{"search_width": 16}]),
            3,
            10000,
            None,
            "latency",
            "does not expose",
        ),
        (
            _index(Path("/tmp/index"), codec=_CAGRA_CODEC),
            1025,
            10000,
            None,
            "latency",
            "GPU brute-force",
        ),
    ],
)
def test_search_rejects_unsupported_options(
    index, k, batch_size, search_threads, mode, error
):
    result = _only_search_result(
        _backend(_FakeRuntime()).search(
            _dataset(),
            [index],
            k=k,
            batch_size=batch_size,
            mode=mode,
            search_threads=search_threads,
            dry_run=True,
        )
    )

    assert not result.success
    assert error in result.error_message


def test_resolve_search_plan_normalizes_the_complete_request(tmp_path):
    index = _index(tmp_path / "index", search_params=[])

    plan = _backend()._resolve_search_plan(
        index,
        k=3,
        batch_size=7,
        mode="latency",
        search_threads="1",
    )

    assert plan == pylucene_backend._SearchPlan(
        index_path=tmp_path / "index",
        codec_name=_HNSW_CODEC,
        search_params=[{}],
        k=3,
        batch_size=7,
        mode="latency",
    )


def test_search_validation_preserves_first_error_precedence(tmp_path):
    result = _only_search_result(
        _backend().search(
            _dataset(),
            [
                _index(
                    tmp_path / "index",
                    search_params=[{"unsupported": True}],
                )
            ],
            k=0,
            batch_size=0,
            mode="throughput",
            search_threads=2,
            dry_run=True,
        )
    )

    assert not result.success
    assert result.error_message == "k must be positive"


@pytest.mark.parametrize("index_count", [0, 2])
def test_search_requires_exactly_one_index(index_count, tmp_path):
    indexes = [
        _index(tmp_path / f"index-{index_id}")
        for index_id in range(index_count)
    ]

    result = _only_search_result(
        _backend(_FakeRuntime()).search(_dataset(), indexes, k=3)
    )

    assert not result.success
    assert "exactly one" in result.error_message


def test_search_rejects_missing_index(tmp_path):
    result = _only_search_result(
        _backend(_FakeRuntime()).search(
            _dataset(), [_index(tmp_path / "missing")], k=3
        )
    )

    assert not result.success
    assert "does not exist" in result.error_message


def test_search_rejects_empty_query_vectors(tmp_path):
    index_path = tmp_path / "index"
    index_path.mkdir()
    dataset = _dataset()
    dataset.query_vectors = np.empty((0, 4), dtype=np.float32)

    result = _only_search_result(
        _backend(_FakeRuntime()).search(dataset, [_index(index_path)], k=3)
    )

    assert not result.success
    assert "at least one vector" in result.error_message


def test_search_reports_runtime_query_dimension_mismatch(tmp_path):
    index_path = tmp_path / "index"
    _prepare_hnsw_index(index_path)
    runtime = _FakeRuntime(index_dimensions=8)

    result = _only_search_result(
        _backend(runtime).search(
            _dataset(dimensions=4), [_index(index_path)], k=3
        )
    )

    assert not result.success
    assert "dimensions do not match the Lucene index" in result.error_message


def test_search_rejects_runtime_dimensions_that_disagree_with_provenance(
    tmp_path,
):
    index_path = tmp_path / "index"
    _prepare_hnsw_index(index_path, dimensions=4)
    runtime = _FakeRuntime(
        index_dimensions=8,
        validate_query_dimensions=False,
    )

    result = _only_search_result(
        _backend(runtime).search(
            _dataset(dimensions=4), [_index(index_path)], k=3
        )
    )

    assert not result.success
    assert (
        "Lucene index dimensions do not match index provenance"
        in result.error_message
    )
    assert len(runtime.search_calls) == 1


def test_search_rejects_query_dimensions_that_disagree_with_dataset(tmp_path):
    index_path = tmp_path / "index"
    index_path.mkdir()
    dataset = _dataset(dimensions=4)
    dataset.query_vectors = np.zeros((2, 8), dtype=np.float32)
    runtime = _FakeRuntime()

    result = _only_search_result(
        _backend(runtime).search(dataset, [_index(index_path)], k=3)
    )

    assert not result.success
    assert "Query vector dimensions do not match the dataset" in (
        result.error_message
    )
    assert runtime.search_calls == []


def test_search_reports_runtime_failure(tmp_path):
    index_path = tmp_path / "index"
    _prepare_hnsw_index(index_path)
    runtime = _FakeRuntime()
    runtime.search_error = RuntimeError("Java search failed")

    result = _only_search_result(
        _backend(runtime).search(_dataset(), [_index(index_path)], k=3)
    )

    assert not result.success
    assert "Java search failed" in result.error_message


@pytest.mark.parametrize(
    "manifest_state",
    ["missing", "corrupt", "stale", "wrong-codec"],
)
def test_search_rejects_invalid_hnsw_provenance_before_runtime(
    tmp_path, manifest_state
):
    index_path = tmp_path / "index"
    manifest_path = _prepare_hnsw_index(index_path)
    if manifest_state == "missing":
        manifest_path.unlink()
    elif manifest_state == "corrupt":
        manifest_path.write_text("{")
    elif manifest_state == "stale":
        (index_path / "segments_1").write_bytes(b"changed commit")
    else:
        payload = json.loads(manifest_path.read_text())
        payload["codec"] = "DifferentCodec"
        manifest_path.write_text(json.dumps(payload))

    runtime = _FakeRuntime()
    result = _only_search_result(
        _backend(runtime).search(_dataset(), [_index(index_path)], k=3)
    )

    assert not result.success
    assert "provenance" in result.error_message.lower()
    assert runtime.search_calls == []


def test_cleanup_releases_runtime_reference():
    backend = _backend(_FakeRuntime())

    backend.cleanup()

    assert backend._runtime is None
