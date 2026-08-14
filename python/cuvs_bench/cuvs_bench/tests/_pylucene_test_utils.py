#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Shared fakes and data builders for PyLucene unit tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import cuvs_bench.backends.pylucene as pylucene_backend
from cuvs_bench._bin_format import write_bin_header
from cuvs_bench.backends.base import Dataset
from cuvs_bench.backends.pylucene import (
    PyLuceneBackend,
    _CagraIndexVerification,
    _RuntimeSearchResult,
    _SearchHit,
)
from cuvs_bench.orchestrator.config_loaders import IndexConfig

_HNSW_CODEC = "Lucene101AcceleratedHNSWCodec"
_CAGRA_CODEC = "CuVS2510GPUSearchCodec"


class _FakeRuntime:
    pylucene_version = "test"

    def __init__(
        self,
        *,
        index_dimensions: int = 4,
        document_count: int = 10,
        hits: list[list[_SearchHit]] | None = None,
        batch_latencies_ms: list[float] | None = None,
        validate_query_dimensions: bool = True,
    ):
        self.index_dimensions = index_dimensions
        self.document_count = document_count
        self.hits = hits
        self.batch_latencies_ms = batch_latencies_ms
        self.validate_query_dimensions = validate_query_dimensions
        self.resolve_calls = []
        self.build_calls = []
        self.search_calls = []
        self.verification_calls = []
        self.resolve_error = None
        self.build_error = None
        self.search_error = None
        self.verification_error = None

    def resolve_codec(self, codec_name):
        self.resolve_calls.append(codec_name)
        if self.resolve_error is not None:
            raise self.resolve_error
        return codec_name

    def build_index(self, index_path, vectors, build_codec):
        self.build_calls.append((index_path, vectors.copy(), build_codec))
        if self.build_error is not None:
            (index_path / "partial").write_bytes(b"partial")
            raise self.build_error
        self.document_count = int(vectors.shape[0])
        (index_path / "segments_1").write_bytes(b"index")

    def verify_cagra_index(
        self,
        index_path,
        expected_vector_count=None,
        expected_dimensions=None,
    ):
        self.verification_calls.append(
            (index_path, expected_vector_count, expected_dimensions)
        )
        if self.verification_error is not None:
            raise self.verification_error
        return _CagraIndexVerification(
            segment_count=1,
            field_count=1,
            vector_count=(
                expected_vector_count
                if expected_vector_count is not None
                else 10
            ),
            dimensions=(
                expected_dimensions if expected_dimensions is not None else 4
            ),
        )

    def search_index(self, index_path, query_vectors, k, batch_size):
        self.search_calls.append(
            (index_path, query_vectors.copy(), k, batch_size)
        )
        if self.search_error is not None:
            raise self.search_error
        if (
            self.validate_query_dimensions
            and query_vectors.shape[1] != self.index_dimensions
        ):
            raise ValueError(
                "Query vector dimensions do not match the Lucene index"
            )
        hits = self.hits
        if hits is None:
            hits = [
                [_SearchHit(document_id=0, score=1.0)]
                for _ in range(query_vectors.shape[0])
            ]
        batch_latencies_ms = self.batch_latencies_ms
        if batch_latencies_ms is None:
            batch_latencies_ms = [
                1.0 for _ in range(0, query_vectors.shape[0], batch_size)
            ]
        return _RuntimeSearchResult(
            hits=hits,
            batch_latencies_ms=batch_latencies_ms,
            index_dimensions=self.index_dimensions,
            document_count=self.document_count,
        )


def _dataset(
    *,
    n_base: int = 10,
    n_queries: int = 2,
    dimensions: int = 4,
    dtype=np.float32,
    distance_metric: str = "euclidean",
) -> Dataset:
    rng = np.random.default_rng(7)
    base = rng.random((n_base, dimensions)).astype(dtype)
    queries = rng.random((n_queries, dimensions)).astype(dtype)
    return Dataset(
        name="test",
        training_vectors=base,
        query_vectors=queries,
        distance_metric=distance_metric,
    )


def _index(
    index_path: Path,
    *,
    codec: str = _HNSW_CODEC,
    search_params: list[dict] | None = None,
) -> IndexConfig:
    return IndexConfig(
        name="pylucene-test",
        algo="pylucene_cuvs_hnsw",
        build_param={"codec": codec},
        search_params=[{}] if search_params is None else search_params,
        file=str(index_path),
    )


def _backend(
    runtime: _FakeRuntime | None = None,
    *,
    codec: str = _HNSW_CODEC,
) -> PyLuceneBackend:
    backend = PyLuceneBackend(
        {
            "name": "pylucene-test",
            "algo": "pylucene_cuvs_hnsw",
            "codec": codec,
        }
    )
    if runtime is not None:
        backend._runtime = runtime
    return backend


def _prepare_hnsw_index(
    index_path: Path,
    *,
    codec: str = _HNSW_CODEC,
    vector_count: int = 10,
    dimensions: int = 4,
) -> Path:
    index_path.mkdir(parents=True, exist_ok=True)
    (index_path / "segments_1").write_bytes(b"index")
    pylucene_backend._write_hnsw_provenance(
        index_path,
        codec,
        vector_count=vector_count,
        dimensions=dimensions,
    )
    return index_path / pylucene_backend._HNSW_PROVENANCE_FILE


def _prepare_cagra_index(
    index_path: Path,
    *,
    vector_count: int = 10,
    dimensions: int = 4,
) -> Path:
    index_path.mkdir(parents=True, exist_ok=True)
    (index_path / "segments_1").write_bytes(b"index")
    pylucene_backend._write_cagra_provenance(
        index_path,
        vector_count=vector_count,
        dimensions=dimensions,
    )
    return index_path / pylucene_backend._CAGRA_PROVENANCE_FILE


def _write_test_bin(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        write_bin_header(file, data.shape[0], data.shape[1])
        np.ascontiguousarray(data).tofile(file)
