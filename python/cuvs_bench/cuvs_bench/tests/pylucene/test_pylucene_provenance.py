#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Index provenance and training-shape tests for PyLucene."""

from __future__ import annotations

import hashlib
import json
import stat

import numpy as np
import pytest

import cuvs_bench.backends.pylucene as pylucene_backend
from cuvs_bench.backends.base import Dataset
from cuvs_bench.tests.pylucene._pylucene_test_utils import (
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


_DEFAULT_HNSW_BUILD_PARAMETERS = {
    "codec": _HNSW_CODEC,
    "m": 32,
    "ef_construction": 32,
    "direct_single_segment": False,
}


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
        "status": "gpu-with-cpu-fallback-provenance",
        "schema_version": 4,
        "codec": _HNSW_CODEC,
        "build_parameters": _DEFAULT_HNSW_BUILD_PARAMETERS,
        "writer_policy": "gpu-with-cpu-fallback",
        "compound_file_policy": "lucene-default",
        "vector_count": 10,
        "dimensions": 4,
        "segment_count": 1,
        "commit_file_count": 1,
    }
    payload = json.loads(manifest_path.read_text())
    expected_commit_sha256 = hashlib.sha256(
        (index_path / "segments_1").read_bytes()
    ).hexdigest()
    assert payload["build_parameters"] == _DEFAULT_HNSW_BUILD_PARAMETERS
    assert payload["commit_fingerprints"] == [
        {
            "name": "segments_1",
            "sha256": expected_commit_sha256,
        }
    ]


def test_cagra_provenance_round_trip(tmp_path):
    index_path = tmp_path / "index"
    _prepare_cagra_index(index_path)

    verification = pylucene_backend._verify_cagra_provenance(
        index_path,
        expected_vector_count=10,
        expected_dimensions=4,
    )

    assert verification.to_metadata() == {
        "status": "gpu-cagra-provenance",
        "schema_version": 4,
        "codec": _CAGRA_CODEC,
        "build_parameters": {"codec": _CAGRA_CODEC},
        "writer_policy": "gpu-cagra",
        "compound_file_policy": "disabled",
        "vector_count": 10,
        "dimensions": 4,
        "segment_count": 1,
        "commit_file_count": 1,
    }


@pytest.mark.parametrize(
    "prepare_index",
    [
        pytest.param(_prepare_hnsw_index, id="hnsw"),
        pytest.param(_prepare_cagra_index, id="cagra"),
    ],
)
def test_provenance_manifest_has_owner_write_and_shared_read_mode_bits(
    tmp_path, prepare_index
):
    manifest_path = prepare_index(tmp_path / "index")
    expected_permission_bits = (
        stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
    )

    permission_bits = stat.S_IMODE(manifest_path.stat().st_mode)

    assert permission_bits == expected_permission_bits


@pytest.mark.parametrize(
    "prepare_index",
    [
        pytest.param(_prepare_hnsw_index, id="hnsw"),
        pytest.param(_prepare_cagra_index, id="cagra"),
    ],
)
def test_successful_provenance_write_leaves_no_temporary_file(
    tmp_path, prepare_index
):
    manifest_path = prepare_index(tmp_path / "index")

    temporary_manifests = tuple(
        manifest_path.parent.glob(f"{manifest_path.name}.*.tmp")
    )

    assert not temporary_manifests


def test_hnsw_provenance_rejects_build_parameter_mismatch(tmp_path):
    index_path = tmp_path / "index"
    _prepare_hnsw_index(index_path)

    with pytest.raises(RuntimeError, match="build parameter"):
        pylucene_backend._verify_hnsw_provenance(
            index_path,
            _HNSW_CODEC,
            expected_build_parameters={
                **_DEFAULT_HNSW_BUILD_PARAMETERS,
                "m": 24,
            },
        )


def test_hnsw_provenance_rejects_inconsistent_direct_segment_count(tmp_path):
    index_path = tmp_path / "index"
    manifest_path = _prepare_hnsw_index(index_path)
    build_parameters = {
        **_DEFAULT_HNSW_BUILD_PARAMETERS,
        "direct_single_segment": True,
    }
    payload = json.loads(manifest_path.read_text())
    payload["build_parameters"] = build_parameters
    payload["segment_count"] = 2
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match="must record exactly one segment"):
        pylucene_backend._verify_hnsw_provenance(
            index_path,
            _HNSW_CODEC,
            expected_build_parameters=build_parameters,
        )


@pytest.mark.parametrize(
    "malformed_build_parameters",
    [
        pytest.param(
            {**_DEFAULT_HNSW_BUILD_PARAMETERS, "m": True},
            id="boolean-m",
        ),
        pytest.param(
            {**_DEFAULT_HNSW_BUILD_PARAMETERS, "ef_construction": 513},
            id="ef-construction-above-limit",
        ),
        pytest.param(
            {**_DEFAULT_HNSW_BUILD_PARAMETERS, "direct_single_segment": 0},
            id="non-boolean-direct-single-segment",
        ),
    ],
)
def test_hnsw_provenance_rejects_malformed_build_parameters(
    tmp_path, malformed_build_parameters
):
    index_path = tmp_path / "index"
    manifest_path = _prepare_hnsw_index(index_path)
    payload = json.loads(manifest_path.read_text())
    payload["build_parameters"] = malformed_build_parameters
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match="build parameter"):
        pylucene_backend._verify_hnsw_provenance(index_path, _HNSW_CODEC)


@pytest.mark.parametrize(
    ("expected_vector_count", "expected_dimensions", "error"),
    [
        pytest.param(11, 4, "vector count", id="vector-count"),
        pytest.param(10, 8, "dimensions", id="dimensions"),
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
        pytest.param(
            "wrong-writer-policy",
            "expected writer policy",
            id="wrong-writer-policy",
        ),
        pytest.param(
            "wrong-compound-file-policy",
            "expected compound-file policy",
            id="wrong-compound-file-policy",
        ),
        pytest.param(
            "boolean-vector-count",
            "positive integer",
            id="boolean-vector-count",
        ),
        pytest.param(
            "missing-commit-sha256",
            "malformed Lucene commit fingerprint",
            id="missing-commit-sha256",
        ),
        pytest.param(
            "duplicate-commit-fingerprint",
            "must be unique and sorted",
            id="duplicate-commit-fingerprint",
        ),
    ],
)
def test_hnsw_provenance_rejects_malformed_manifest_fields(
    tmp_path, manifest_state, error
):
    index_path = tmp_path / "index"
    manifest_path = _prepare_hnsw_index(index_path)
    payload = json.loads(manifest_path.read_text())

    if manifest_state == "wrong-writer-policy":
        payload["writer_policy"] = "gpu-cagra"
    elif manifest_state == "wrong-compound-file-policy":
        payload["compound_file_policy"] = "disabled"
    elif manifest_state == "boolean-vector-count":
        payload["vector_count"] = True
    elif manifest_state == "missing-commit-sha256":
        payload["commit_fingerprints"] = [{"name": "segments_1"}]
    elif manifest_state == "duplicate-commit-fingerprint":
        payload["commit_fingerprints"] *= 2
    else:
        raise AssertionError(f"Unhandled malformed manifest: {manifest_state}")
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match=error):
        pylucene_backend._verify_hnsw_provenance(
            index_path,
            _HNSW_CODEC,
        )


@pytest.mark.parametrize(
    ("field_name", "field_value", "error"),
    [
        pytest.param(
            "name", 1, "commit filename must be a string", id="non-string-name"
        ),
        pytest.param(
            "name",
            "commit_1",
            "must start with 'segments_'",
            id="wrong-name-prefix",
        ),
        pytest.param(
            "name",
            "segments_1/child",
            "must not contain a path",
            id="name-containing-path",
        ),
        pytest.param(
            "sha256",
            1,
            "commit SHA-256 must be a string",
            id="non-string-sha256",
        ),
        pytest.param(
            "sha256",
            "0" * 63,
            "must contain 64 characters",
            id="short-sha256",
        ),
        pytest.param(
            "sha256",
            "A" * 64,
            "must be lowercase hexadecimal",
            id="uppercase-sha256",
        ),
    ],
)
def test_hnsw_provenance_rejects_invalid_commit_fingerprint_field(
    tmp_path, field_name, field_value, error
):
    index_path = tmp_path / "index"
    manifest_path = _prepare_hnsw_index(index_path)
    payload = json.loads(manifest_path.read_text())
    payload["commit_fingerprints"][0][field_name] = field_value
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match=error):
        pylucene_backend._verify_hnsw_provenance(index_path, _HNSW_CODEC)


def test_expected_training_shape_uses_loaded_vectors_instead_of_base_file(
    tmp_path,
):
    base_file = tmp_path / "different.fbin"
    _write_test_bin(base_file, np.zeros((3, 8), dtype=np.float32))
    dataset = _dataset(n_base=10, dimensions=4)
    dataset.base_file = str(base_file)

    assert pylucene_backend._expected_training_shape(dataset) == (10, 4)


def test_expected_training_shape_ignores_file_metadata_when_vectors_are_loaded(
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


def test_expected_training_shape_uses_file_header_and_subset_without_loading(
    tmp_path,
):
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


def test_build_reuses_file_subset_without_loading_vectors_or_starting_jvm(
    tmp_path,
):
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


@pytest.mark.parametrize(
    "subset_size",
    [
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(True, id="boolean"),
        pytest.param("3", id="string"),
    ],
)
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
