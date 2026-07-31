#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Index provenance and training-shape tests for PyLucene."""

from __future__ import annotations

import json
import stat

import numpy as np
import pytest

import cuvs_bench.backends.pylucene as pylucene_backend
from cuvs_bench.backends.base import Dataset
from cuvs_bench.tests._pylucene_test_utils import (
    _CAGRA_CODEC,
    _HNSW_CODEC,
    _FakeRuntime,
    _backend,
    _dataset,
    _index,
    _prepare_cagra_index,
    _prepare_hnsw_index,
    _write_test_bin,
)


def test_hnsw_provenance_round_trip(tmp_path):
    index_path = tmp_path / "index"
    manifest_path = _prepare_hnsw_index(index_path)

    verification = pylucene_backend._verify_hnsw_provenance(
        index_path,
        _HNSW_CODEC,
        expected_vector_count=10,
        expected_dimensions=4,
    )

    assert verification.to_metadata() == {
        "status": "gpu-hnsw-provenance",
        "schema_version": 1,
        "codec": _HNSW_CODEC,
        "writer_path": "gpu-hnsw",
        "vector_count": 10,
        "dimensions": 4,
        "commit_file_count": 1,
    }
    payload = json.loads(manifest_path.read_text())
    assert payload["commit_fingerprints"] == [
        {
            "name": "segments_1",
            "sha256": (
                "1bc04b5291c26a46d918139138b992d2de976d6851d0893b"
                "0476b85bfbdfc6e6"
            ),
        }
    ]
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o644
    assert list(index_path.glob(f"{manifest_path.name}.*.tmp")) == []


def test_cagra_provenance_round_trip(tmp_path):
    index_path = tmp_path / "index"
    manifest_path = _prepare_cagra_index(index_path)

    verification = pylucene_backend._verify_cagra_provenance(
        index_path,
        expected_vector_count=10,
        expected_dimensions=4,
    )

    assert verification.to_metadata() == {
        "status": "gpu-cagra-provenance",
        "schema_version": 1,
        "codec": _CAGRA_CODEC,
        "writer_path": "gpu-cagra",
        "vector_count": 10,
        "dimensions": 4,
        "commit_file_count": 1,
    }
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o644


@pytest.mark.parametrize(
    ("expected_vector_count", "expected_dimensions", "error"),
    [
        (11, 4, "vector count"),
        (10, 8, "dimensions"),
    ],
)
def test_hnsw_provenance_rejects_dataset_shape_mismatch(
    tmp_path, expected_vector_count, expected_dimensions, error
):
    index_path = tmp_path / "index"
    _prepare_hnsw_index(index_path)

    with pytest.raises(RuntimeError, match=error):
        pylucene_backend._verify_hnsw_provenance(
            index_path,
            _HNSW_CODEC,
            expected_vector_count=expected_vector_count,
            expected_dimensions=expected_dimensions,
        )


@pytest.mark.parametrize(
    ("manifest_state", "error"),
    [
        ("wrong-writer", "required GPU writer path"),
        ("boolean-count", "positive integer"),
        ("malformed-fingerprint", "malformed Lucene commit fingerprint"),
        ("duplicate-fingerprint", "must be unique and sorted"),
    ],
)
def test_hnsw_provenance_rejects_malformed_manifest_fields(
    tmp_path, manifest_state, error
):
    index_path = tmp_path / "index"
    manifest_path = _prepare_hnsw_index(index_path)
    payload = json.loads(manifest_path.read_text())

    if manifest_state == "wrong-writer":
        payload["writer_path"] = "cpu-hnsw"
    elif manifest_state == "boolean-count":
        payload["vector_count"] = True
    elif manifest_state == "malformed-fingerprint":
        payload["commit_fingerprints"] = [{"name": "segments_1"}]
    else:
        payload["commit_fingerprints"] *= 2
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match=error):
        pylucene_backend._verify_hnsw_provenance(
            index_path,
            _HNSW_CODEC,
        )


@pytest.mark.parametrize(
    ("field_name", "field_value", "error"),
    [
        ("name", 1, "commit filename must be a string"),
        ("name", "commit_1", "must start with 'segments_'"),
        ("name", "segments_1/child", "must not contain a path"),
        ("sha256", 1, "commit SHA-256 must be a string"),
        ("sha256", "0" * 63, "must contain 64 characters"),
        ("sha256", "A" * 64, "must be lowercase hexadecimal"),
    ],
)
def test_hnsw_provenance_reports_invalid_commit_fingerprint_field(
    tmp_path, field_name, field_value, error
):
    index_path = tmp_path / "index"
    manifest_path = _prepare_hnsw_index(index_path)
    payload = json.loads(manifest_path.read_text())
    payload["commit_fingerprints"][0][field_name] = field_value
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match=error):
        pylucene_backend._verify_hnsw_provenance(index_path, _HNSW_CODEC)


def test_expected_training_shape_prefers_explicit_vectors_over_base_file(
    tmp_path,
):
    base_file = tmp_path / "different.fbin"
    _write_test_bin(base_file, np.zeros((3, 8), dtype=np.float32))
    dataset = _dataset(n_base=10, dimensions=4)
    dataset.base_file = str(base_file)

    assert pylucene_backend._expected_training_shape(dataset) == (10, 4)


def test_expected_training_shape_does_not_validate_unused_file_metadata(
    tmp_path,
):
    dataset = _dataset(n_base=10, dimensions=4)
    dataset.base_file = str(tmp_path / "missing.ibin")
    dataset.metadata["subset_size"] = True

    assert pylucene_backend._expected_training_shape(dataset) == (10, 4)


def test_expected_training_shape_returns_none_without_training_source():
    dataset = Dataset(
        name="query-only",
        query_vectors=np.zeros((2, 4), dtype=np.float32),
        distance_metric="euclidean",
    )

    assert pylucene_backend._expected_training_shape(dataset) is None


def test_expected_training_shape_clamps_file_to_valid_subset(tmp_path):
    base_file = tmp_path / "base.fbin"
    _write_test_bin(base_file, np.zeros((10, 4), dtype=np.float32))
    dataset = Dataset(
        name="file-backed",
        query_vectors=np.zeros((2, 4), dtype=np.float32),
        base_file=str(base_file),
        distance_metric="euclidean",
        metadata={"subset_size": 3},
    )

    assert pylucene_backend._expected_training_shape(dataset) == (3, 4)
    assert dataset.loaded_training_vectors is None


def test_build_reuses_file_backed_subset_without_loading_vectors(tmp_path):
    base_file = tmp_path / "base.fbin"
    _write_test_bin(base_file, np.zeros((10, 4), dtype=np.float32))
    dataset = Dataset(
        name="file-backed",
        query_vectors=np.zeros((2, 4), dtype=np.float32),
        base_file=str(base_file),
        distance_metric="euclidean",
        metadata={"subset_size": 3},
    )
    index_path = tmp_path / "index"
    _prepare_hnsw_index(index_path, vector_count=3, dimensions=4)
    backend = _backend()

    result = backend.build(dataset, [_index(index_path)], force=False)

    assert result.success
    assert result.metadata["skipped"] is True
    assert result.metadata["hnsw_verification"]["vector_count"] == 3
    assert dataset.loaded_training_vectors is None
    assert backend._runtime is None


@pytest.mark.parametrize("subset_size", [0, -1, True, "3"])
def test_expected_training_shape_rejects_invalid_file_subset(
    tmp_path, subset_size
):
    base_file = tmp_path / "base.fbin"
    _write_test_bin(base_file, np.zeros((10, 4), dtype=np.float32))
    dataset = Dataset(
        name="file-backed",
        query_vectors=np.zeros((2, 4), dtype=np.float32),
        base_file=str(base_file),
        distance_metric="euclidean",
        metadata={"subset_size": subset_size},
    )

    with pytest.raises(ValueError, match="positive integer"):
        pylucene_backend._expected_training_shape(dataset)


def test_reused_index_rejects_non_float32_base_file(tmp_path):
    base_file = tmp_path / "base.ibin"
    _write_test_bin(base_file, np.zeros((10, 4), dtype=np.int32))
    dataset = Dataset(
        name="integer-data",
        query_vectors=np.zeros((2, 4), dtype=np.float32),
        base_file=str(base_file),
        distance_metric="euclidean",
    )
    index_path = tmp_path / "index"
    _prepare_hnsw_index(index_path)

    result = _backend(_FakeRuntime()).build(
        dataset, [_index(index_path)], force=False
    )

    assert not result.success
    assert "must use float32 values, got int32" in result.error_message
