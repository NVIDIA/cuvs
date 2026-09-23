#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Opt-in Lucene backend using stock PyLucene and cuvs-lucene codecs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np

from cuvs_bench._bin_format import read_bin_header

from ..orchestrator.config_loaders import (
    BenchmarkConfig,
    ConfigLoader,
    IndexConfig,
)
from ._lucene_runtime import (
    ACCELERATED_HNSW_CODEC,
    CAGRA_CODEC,
    CPU_HNSW_CODEC,
    DIRECT_PYLUCENE_DISPATCH,
    MAX_CAGRA_TOP_K,
    LuceneRuntime,
    RuntimeBuildResult,
    RuntimeSearchResult,
    RuntimeSearchTiming,
    TIMED_BRIDGE_PYLUCENE_DISPATCH,
)
from ._lucene_runtime_config import resolve_lucene_runtime_config
from ._utils import dtype_from_filename
from .base import BenchmarkBackend, BuildResult, Dataset, SearchResult

CPU_HNSW_ALGORITHM = "lucene_cpu_hnsw"
ACCELERATED_HNSW_ALGORITHM = "lucene_accelerated_hnsw"
CAGRA_ALGORITHM = "lucene_cuvs_cagra"

MAX_CPU_HNSW_DIMENSIONS = 1024
MAX_CAGRA_DIMENSIONS = 4096

_CODEC_BY_ALGORITHM = {
    CPU_HNSW_ALGORITHM: CPU_HNSW_CODEC,
    ACCELERATED_HNSW_ALGORITHM: ACCELERATED_HNSW_CODEC,
    CAGRA_ALGORITHM: CAGRA_CODEC,
}
_MAX_DIMENSIONS_BY_ALGORITHM = {
    CPU_HNSW_ALGORITHM: MAX_CPU_HNSW_DIMENSIONS,
    ACCELERATED_HNSW_ALGORITHM: MAX_CAGRA_DIMENSIONS,
    CAGRA_ALGORITHM: MAX_CAGRA_DIMENSIONS,
}
_CUVS_ALGORITHMS = frozenset((ACCELERATED_HNSW_ALGORITHM, CAGRA_ALGORITHM))
_BUILD_ROUTE_POLICY_BY_ALGORITHM = {
    CPU_HNSW_ALGORITHM: "cpu_hnsw",
    ACCELERATED_HNSW_ALGORITHM: "gpu_cagra_or_cpu_hnsw_fallback",
    CAGRA_ALGORITHM: "gpu_cagra",
}
_SEARCH_ROUTE_BY_ALGORITHM = {
    CPU_HNSW_ALGORITHM: "cpu_hnsw",
    ACCELERATED_HNSW_ALGORITHM: "cpu_hnsw",
    CAGRA_ALGORITHM: "gpu_cagra",
}
_MANIFEST_FILE = ".cuvs-bench-lucene.json"
_MANIFEST_SCHEMA = 2
_SAFE_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_RUNTIME_KEYS = (
    "cuvs_java_jar",
    "cuvs_lucene_jar",
    "java_library_path",
    "jvm_args",
)
_SCORE_ROUNDOFF_TOLERANCE = float(np.spacing(np.float32(1.0)))
_INDEX_PREWARM_BLOCK_BYTES = 1024 * 1024
_TIMING_CONTRACT_VERSION = 1


@dataclass(frozen=True)
class _IndexPrewarmTiming:
    wall_ns: int
    bytes_read: int
    file_count: int


def _nanoseconds_to_milliseconds(value: int) -> float:
    return float(value) / 1_000_000.0


def _nanoseconds_to_seconds(value: int) -> float:
    return float(value) / 1_000_000_000.0


def _timing_statistics(
    scope: str, samples_ns: list[int]
) -> dict[str, float | int]:
    """Summarize one timing sample per measured query."""
    if not samples_ns or any(value < 0 for value in samples_ns):
        raise RuntimeError(f"Lucene returned invalid {scope} timing data")
    samples_ms = np.asarray(samples_ns, dtype=np.float64) / 1_000_000.0
    return {
        f"{scope}_count": len(samples_ns),
        f"{scope}_total_ms": float(samples_ms.sum()),
        f"{scope}_mean_ms": float(samples_ms.mean()),
        f"{scope}_p50_ms": float(np.percentile(samples_ms, 50)),
        f"{scope}_p95_ms": float(np.percentile(samples_ms, 95)),
        f"{scope}_p99_ms": float(np.percentile(samples_ms, 99)),
    }


def _validate_nested_timing(
    parent_scope: str,
    parent_ns: int,
    child_timings_ns: Mapping[str, int],
) -> None:
    """Reject impossible same-clock timing relationships."""
    if parent_ns < 0 or any(value < 0 for value in child_timings_ns.values()):
        raise RuntimeError(
            f"Lucene returned invalid {parent_scope} timing data"
        )
    child_total_ns = sum(child_timings_ns.values())
    if child_total_ns > parent_ns:
        raise RuntimeError(
            f"Lucene returned impossible {parent_scope} timing data: "
            f"child phases total {child_total_ns} ns, exceeding "
            f"the enclosing {parent_ns} ns"
        )


def _prewarm_index_files(index_path: Path) -> _IndexPrewarmTiming:
    """Populate the host page cache by sequentially reading index files."""
    started = time.perf_counter_ns()
    bytes_read = 0
    file_count = 0
    buffer = bytearray(_INDEX_PREWARM_BLOCK_BYTES)
    for path in sorted(index_path.iterdir()):
        if path.is_symlink():
            raise RuntimeError(
                f"Lucene index prewarm refuses symbolic links: {path}"
            )
        if not path.is_file():
            continue
        file_count += 1
        try:
            with path.open("rb", buffering=0) as stream:
                while count := stream.readinto(buffer):
                    bytes_read += count
        except OSError as error:
            raise RuntimeError(
                f"Failed to prewarm Lucene index file: {path}"
            ) from error
    if file_count == 0:
        raise RuntimeError(
            f"Lucene index contains no files to prewarm: {index_path}"
        )
    return _IndexPrewarmTiming(
        wall_ns=time.perf_counter_ns() - started,
        bytes_read=bytes_read,
        file_count=file_count,
    )


def _runtime_build_timing_metadata(
    result: RuntimeBuildResult,
) -> dict[str, float]:
    timing = result.timing
    phases = {
        "runtime_directory_open_seconds": timing.directory_open_ns,
        "runtime_writer_setup_seconds": timing.writer_setup_ns,
        "runtime_document_ingest_seconds": timing.document_ingest_ns,
        "runtime_writer_commit_close_seconds": timing.writer_commit_close_ns,
        "runtime_post_build_reader_seconds": timing.post_build_reader_ns,
        "runtime_directory_close_seconds": timing.directory_close_ns,
    }
    _validate_nested_timing(
        "runtime_build_wall",
        timing.runtime_build_wall_ns,
        phases,
    )
    fields = {
        **phases,
        "runtime_build_wall_seconds": timing.runtime_build_wall_ns,
    }
    return {
        name: _nanoseconds_to_seconds(value) for name, value in fields.items()
    }


def _java_search_timing_metadata(
    timing: RuntimeSearchTiming,
) -> dict[str, Any]:
    """Validate the dispatch route and summarize independent JVM timings."""
    java_samples = [
        item.java_index_searcher_search_ns for item in timing.queries
    ]
    java_available = all(value is not None for value in java_samples)
    if any(value is not None for value in java_samples) and not java_available:
        raise RuntimeError(
            "Lucene returned Java timing data for only some queries"
        )
    valid_dispatch_kinds = {
        DIRECT_PYLUCENE_DISPATCH,
        TIMED_BRIDGE_PYLUCENE_DISPATCH,
    }
    if timing.search_dispatch_kind not in valid_dispatch_kinds:
        raise RuntimeError(
            "Lucene returned an unknown search dispatch kind: "
            f"{timing.search_dispatch_kind!r}"
        )
    expected_java_timing = (
        timing.search_dispatch_kind == TIMED_BRIDGE_PYLUCENE_DISPATCH
    )
    if java_available != expected_java_timing:
        raise RuntimeError(
            "Lucene search dispatch kind does not match Java timing availability"
        )

    metadata: dict[str, Any] = {
        "search_dispatch_kind": timing.search_dispatch_kind,
        "java_timing_available": java_available,
    }
    if not java_available:
        return metadata

    resolved_java_samples = [int(value) for value in java_samples]
    metadata.update(
        _timing_statistics(
            "java_index_searcher_search",
            resolved_java_samples,
        )
    )
    metadata.update(
        {
            "first_query_java_index_searcher_search_ms": (
                _nanoseconds_to_milliseconds(resolved_java_samples[0])
            ),
            "subsequent_java_index_searcher_search_mean_ms": (
                float(np.mean(resolved_java_samples[1:])) / 1_000_000.0
                if len(resolved_java_samples) > 1
                else None
            ),
        }
    )
    return metadata


def _runtime_search_timing_metadata(
    result: RuntimeSearchResult, expected_query_count: int
) -> tuple[dict[str, Any], list[int]]:
    timing = result.timing
    if len(timing.queries) != expected_query_count:
        raise RuntimeError(
            "Lucene returned timing data for "
            f"{len(timing.queries)} queries, expected {expected_query_count}"
        )
    plan_phases = {
        "directory_open_ms": timing.directory_open_ns,
        "reader_searcher_setup_ms": timing.reader_searcher_setup_ns,
        "query_corpus_wall_ms": timing.query_corpus_wall_ns,
        "reader_close_ms": timing.reader_close_ns,
        "directory_close_ms": timing.directory_close_ns,
    }
    _validate_nested_timing(
        "runtime_plan_wall",
        timing.runtime_plan_wall_ns,
        plan_phases,
    )
    if timing.query_corpus_wall_ns <= 0:
        raise RuntimeError("Lucene returned an empty query-corpus duration")

    for query_number, query_timing in enumerate(timing.queries):
        _validate_nested_timing(
            f"client_query[{query_number}]",
            query_timing.client_query_ns,
            {
                "query_prepare": query_timing.query_prepare_ns,
                "pylucene_search_dispatch": (
                    query_timing.pylucene_search_dispatch_ns
                ),
                "result_materialization": (
                    query_timing.result_materialization_ns
                ),
            },
        )
    client_query_total_ns = sum(
        item.client_query_ns for item in timing.queries
    )
    if client_query_total_ns > timing.query_corpus_wall_ns:
        raise RuntimeError(
            "Lucene returned impossible query_corpus_wall timing data: "
            f"client queries total {client_query_total_ns} ns, exceeding "
            f"the enclosing {timing.query_corpus_wall_ns} ns"
        )

    scopes = {
        "query_prepare": [item.query_prepare_ns for item in timing.queries],
        "pylucene_search_dispatch": [
            item.pylucene_search_dispatch_ns for item in timing.queries
        ],
        "result_materialization": [
            item.result_materialization_ns for item in timing.queries
        ],
        "client_query": [item.client_query_ns for item in timing.queries],
    }
    plan_timings = {
        **plan_phases,
        "runtime_plan_wall_ms": timing.runtime_plan_wall_ns,
    }
    metadata: dict[str, Any] = {
        name: _nanoseconds_to_milliseconds(value)
        for name, value in plan_timings.items()
    }
    for scope, samples in scopes.items():
        metadata.update(_timing_statistics(scope, samples))
    metadata.update(_java_search_timing_metadata(timing))

    dispatch_samples = scopes["pylucene_search_dispatch"]
    client_samples = scopes["client_query"]
    metadata.update(
        {
            "first_query_pylucene_search_dispatch_ms": (
                _nanoseconds_to_milliseconds(dispatch_samples[0])
            ),
            "first_query_client_query_ms": _nanoseconds_to_milliseconds(
                client_samples[0]
            ),
            "subsequent_query_count": max(0, len(dispatch_samples) - 1),
            "subsequent_pylucene_search_dispatch_mean_ms": (
                float(np.mean(dispatch_samples[1:])) / 1_000_000.0
                if len(dispatch_samples) > 1
                else None
            ),
            "subsequent_client_query_mean_ms": (
                float(np.mean(client_samples[1:])) / 1_000_000.0
                if len(client_samples) > 1
                else None
            ),
        }
    )
    return metadata, client_samples


def _validate_vectors(
    vectors: Any, label: str, *, maximum_dimensions: int
) -> np.ndarray:
    array = np.asarray(vectors)
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError(f"{label} must be a nonempty two-dimensional array")
    if array.dtype != np.float32:
        raise TypeError(f"{label} must use float32 values, got {array.dtype}")
    if array.shape[1] > maximum_dimensions:
        raise ValueError(
            f"{label} dimensions must not exceed {maximum_dimensions}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite values")
    return np.ascontiguousarray(array)


def _validate_metric(dataset: Dataset) -> None:
    if dataset.distance_metric.casefold() not in {"euclidean", "l2"}:
        raise ValueError(
            "The Lucene backend currently supports only Euclidean/L2 data, "
            f"not {dataset.distance_metric!r}"
        )


def _score_to_squared_euclidean(score: float) -> float:
    """Invert Lucene's ``score = 1 / (1 + squared_distance)`` transform."""
    if (
        not math.isfinite(score)
        or score <= 0.0
        or score > 1.0 + _SCORE_ROUNDOFF_TOLERANCE
    ):
        raise RuntimeError(
            f"Lucene returned an invalid Euclidean score: {score}"
        )
    # Admit one float32 ULP defensively at the mathematical upper boundary.
    score = min(score, 1.0)
    return max(0.0, (1.0 / score) - 1.0)


def _format_exception(error: BaseException) -> str:
    details = [f"{type(error).__name__}: {error}"]
    details.extend(getattr(error, "__notes__", ()))
    return "\n".join(details)


def _safe_label(value: str, kind: str) -> str:
    if not isinstance(value, str) or _SAFE_LABEL.fullmatch(value) is None:
        raise ValueError(f"Unsafe Lucene {kind}: {value!r}")
    return value


def _codec_for(algorithm: str, build_params: Mapping[str, Any]) -> str:
    try:
        expected = _CODEC_BY_ALGORITHM[algorithm]
    except KeyError as error:
        raise ValueError(
            f"Unsupported Lucene algorithm: {algorithm!r}"
        ) from error
    unsupported = set(build_params) - {"codec"}
    if unsupported:
        raise ValueError(
            "Unsupported Lucene build parameters: "
            + ", ".join(sorted(str(name) for name in unsupported))
        )
    actual = build_params.get("codec", expected)
    if actual != expected:
        raise ValueError(
            f"{algorithm} requires codec {expected!r}, got {actual!r}"
        )
    return expected


def _search_parameters(
    algorithm: str, parameters: Mapping[str, Any], k: int
) -> dict[str, int]:
    if not isinstance(parameters, Mapping):
        raise TypeError("Lucene search parameters must be mappings")
    if algorithm == CAGRA_ALGORITHM:
        if parameters:
            raise ValueError(
                "lucene_cuvs_cagra currently uses the codec's fixed search "
                "defaults and accepts no search parameters"
            )
        if k > MAX_CAGRA_TOP_K:
            raise ValueError(
                f"CAGRA search supports k <= {MAX_CAGRA_TOP_K}; k={k} would "
                "use the codec's brute-force route"
            )
        return {"num_candidates": k}
    unsupported = set(parameters) - {"num_candidates"}
    if unsupported:
        raise ValueError(
            "Unsupported Lucene search parameters: "
            + ", ".join(sorted(str(name) for name in unsupported))
        )
    candidates = parameters.get("num_candidates", k)
    if type(candidates) is not int or candidates < k:
        raise ValueError(
            f"num_candidates must be an integer >= k ({k}), got {candidates!r}"
        )
    return {"num_candidates": candidates}


def _source_identity(dataset: Dataset) -> dict[str, Any]:
    source = Path(dataset.base_file).expanduser().resolve()
    stat = source.stat()
    return {
        "path": str(source),
        "device": stat.st_dev,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "inode": stat.st_ino,
    }


def _normalize_subset_size(dataset: Dataset) -> int | None:
    value = dataset.metadata.get("subset_size")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(
            f"subset_size must be a positive integer, got {value!r}"
        )
    normalized = int(value)
    if type(value) is not int:
        dataset.metadata = {**dataset.metadata, "subset_size": normalized}
    return normalized


def _dataset_identity(dataset: Dataset, vectors: np.ndarray) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "name": dataset.name,
        "vector_count": int(vectors.shape[0]),
        "dimensions": int(vectors.shape[1]),
        "subset_size": _normalize_subset_size(dataset),
        "sha256": hashlib.sha256(vectors.view(np.uint8)).hexdigest(),
    }
    if dataset.base_file:
        identity["source"] = _source_identity(dataset)
    return identity


def _file_backed_dataset_identity(
    dataset: Dataset,
    stored: Mapping[str, Any],
    *,
    maximum_dimensions: int,
) -> tuple[dict[str, Any], int, int]:
    """Stream a file source identity without materializing its vectors."""
    subset_size = _normalize_subset_size(dataset)
    source = _source_identity(dataset)
    dtype = np.dtype(dtype_from_filename(source["path"]))
    if dtype != np.dtype(np.float32):
        raise TypeError(
            f"training vectors must use float32 values, got {dtype}"
        )
    rows, dimensions, header_bytes = read_bin_header(
        source["path"], dtype.itemsize
    )
    if dimensions > maximum_dimensions:
        raise ValueError(
            f"training vectors dimensions must not exceed {maximum_dimensions}"
        )
    if subset_size is not None:
        rows = min(rows, subset_size)
    remaining = int(rows) * int(dimensions) * dtype.itemsize
    digest_builder = hashlib.sha256()
    with Path(source["path"]).open("rb") as stream:
        stream.seek(header_bytes)
        while remaining:
            chunk = stream.read(min(remaining, 8 * 1024 * 1024))
            if not chunk:
                raise ValueError(
                    f"training vector file is truncated: {source['path']}"
                )
            digest_builder.update(chunk)
            remaining -= len(chunk)
    confirmed_source = _source_identity(dataset)
    if confirmed_source != source:
        raise RuntimeError(
            "training vector file changed while its identity was being read"
        )
    digest = digest_builder.hexdigest()
    stored_digest = stored.get("sha256")
    if (
        not isinstance(stored_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", stored_digest) is None
    ):
        raise RuntimeError(
            "Lucene index manifest has an invalid dataset digest"
        )
    identity = {
        "name": dataset.name,
        "vector_count": int(rows),
        "dimensions": int(dimensions),
        "subset_size": subset_size,
        "sha256": digest,
        "source": source,
    }
    return identity, int(rows), int(dimensions)


def _manifest_payload(
    dataset: Dataset,
    vectors: np.ndarray,
    algorithm: str,
    codec: str,
    segment_count: int,
    build_runtime_artifacts: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": _MANIFEST_SCHEMA,
        "algorithm": algorithm,
        "codec": codec,
        "dataset": _dataset_identity(dataset, vectors),
        "segment_count": segment_count,
        "build_runtime_artifacts": dict(build_runtime_artifacts),
    }


def _artifact_metadata(
    role: str, provenance: Mapping[str, str]
) -> dict[str, str]:
    return {f"{role}_{key}": value for key, value in provenance.items()}


def _write_manifest(index_path: Path, payload: Mapping[str, Any]) -> None:
    target = index_path / _MANIFEST_FILE
    temporary = index_path / f"{_MANIFEST_FILE}.tmp"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.chmod(0o644)
    os.replace(temporary, target)


def _is_manifest_integer(value: Any, *, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _validate_manifest_dataset(dataset: Any, path: Path) -> None:
    required = {
        "name",
        "vector_count",
        "dimensions",
        "subset_size",
        "sha256",
    }
    if not isinstance(dataset, dict) or set(dataset) not in (
        required,
        required | {"source"},
    ):
        raise RuntimeError(
            f"Lucene index manifest has an invalid dataset: {path}"
        )
    if not isinstance(dataset["name"], str):
        raise RuntimeError(
            f"Lucene index manifest has an invalid dataset: {path}"
        )
    for field in ("vector_count", "dimensions"):
        if not _is_manifest_integer(dataset[field], minimum=1):
            raise RuntimeError(
                f"Lucene index manifest has an invalid dataset {field}: {path}"
            )
    subset_size = dataset["subset_size"]
    if subset_size is not None and not _is_manifest_integer(
        subset_size, minimum=1
    ):
        raise RuntimeError(
            f"Lucene index manifest has an invalid dataset subset_size: {path}"
        )
    digest = dataset["sha256"]
    if (
        not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
    ):
        raise RuntimeError(
            f"Lucene index manifest has an invalid dataset digest: {path}"
        )
    if "source" not in dataset:
        return
    source = dataset["source"]
    source_fields = {
        "path",
        "device",
        "size",
        "mtime_ns",
        "ctime_ns",
        "inode",
    }
    if not isinstance(source, dict) or set(source) != source_fields:
        raise RuntimeError(
            f"Lucene index manifest has an invalid dataset source: {path}"
        )
    if (
        not isinstance(source["path"], str)
        or not Path(source["path"]).is_absolute()
    ):
        raise RuntimeError(
            f"Lucene index manifest has an invalid dataset source path: {path}"
        )
    for field in source_fields - {"path"}:
        if not _is_manifest_integer(source[field]):
            raise RuntimeError(
                "Lucene index manifest has an invalid dataset source "
                f"{field}: {path}"
            )


def _read_manifest(index_path: Path) -> dict[str, Any]:
    path = index_path / _MANIFEST_FILE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Cannot read Lucene index manifest {path}: {error}"
        ) from error
    if (
        not isinstance(payload, dict)
        or type(payload.get("schema_version")) is not int
        or payload["schema_version"] != _MANIFEST_SCHEMA
    ):
        raise RuntimeError(
            f"Unsupported Lucene index manifest: {path}; rerun with --force"
        )
    required = {
        "schema_version",
        "algorithm",
        "codec",
        "dataset",
        "segment_count",
        "build_runtime_artifacts",
    }
    if set(payload) != required:
        raise RuntimeError(f"Lucene index manifest has invalid fields: {path}")
    if not isinstance(payload["algorithm"], str) or not isinstance(
        payload["codec"], str
    ):
        raise RuntimeError(
            f"Lucene index manifest has invalid identifiers: {path}"
        )
    _validate_manifest_dataset(payload["dataset"], path)
    segment_count = payload.get("segment_count")
    if not _is_manifest_integer(segment_count, minimum=1):
        raise RuntimeError(
            f"Lucene index manifest has an invalid segment count: {path}"
        )
    if not isinstance(payload["build_runtime_artifacts"], dict):
        raise RuntimeError(
            "Lucene index manifest has invalid build-runtime artifact "
            f"provenance: {path}"
        )
    return payload


def _index_size(index_path: Path) -> int:
    return sum(
        path.stat().st_size for path in index_path.rglob("*") if path.is_file()
    )


class LuceneConfigLoader(ConfigLoader):
    """Load the fixed initial Lucene algorithms from standard Bench YAML."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or str(
            Path(__file__).resolve().parents[1] / "config"
        )

    @property
    def backend_type(self) -> str:
        return "lucene"

    @staticmethod
    def _split(value: Any) -> list[str] | None:
        if not value:
            return None
        return [part.strip() for part in str(value).split(",") if part.strip()]

    def _discover_algo_groups(
        self,
        dataset_conf: dict,
        dataset: str,
        dataset_path: str,
        **kwargs: Any,
    ) -> list[tuple[str, str, dict, dict]]:
        files = self.gather_algorithm_configs(
            self.config_path, kwargs.get("algorithm_configuration")
        )
        configs = {}
        for path in files:
            config = self.load_yaml_file(path)
            if (
                isinstance(config, dict)
                and config.get("name") in _CODEC_BY_ALGORITHM
            ):
                configs[config["name"]] = config

        algorithms = self._split(kwargs.get("algorithms"))
        groups = self._split(kwargs.get("groups")) or ["base"]
        requested_pairs: dict[str, list[str]] = {}
        if algo_groups := self._split(kwargs.get("algo_groups")):
            for item in algo_groups:
                algorithm, separator, group = item.partition(".")
                if not separator:
                    raise ValueError(
                        f"Lucene --algo-groups entry must be algorithm.group: {item!r}"
                    )
                selected = requested_pairs.setdefault(algorithm, [])
                if group not in selected:
                    selected.append(group)

        requested_algorithms = (
            algorithms or list(requested_pairs) or list(configs)
        )
        result = []
        for algorithm in requested_algorithms:
            if algorithm not in _CODEC_BY_ALGORITHM:
                raise ValueError(
                    f"Unsupported Lucene algorithm: {algorithm!r}"
                )
            config = configs.get(algorithm)
            if config is None:
                raise ValueError(f"No configuration found for {algorithm!r}")
            selected_groups = requested_pairs.get(algorithm, groups)
            for group in selected_groups:
                try:
                    group_config = config["groups"][group]
                except KeyError as error:
                    raise ValueError(
                        f"No Lucene group {group!r} for {algorithm!r}"
                    ) from error
                result.append((algorithm, group, group_config, {}))
        return result

    def _build_benchmark_configs(
        self,
        dataset_config: Any,
        dataset_conf: dict,
        dataset: str,
        dataset_path: str,
        expanded_groups: list[tuple],
        **kwargs: Any,
    ) -> list[BenchmarkConfig]:
        include_cuvs = any(
            item[0] in _CUVS_ALGORITHMS for item in expanded_groups
        )
        runtime_overrides = {
            key: kwargs[key]
            for key in _RUNTIME_KEYS
            if kwargs.get(key) is not None
        }
        _safe_label(dataset, "dataset")
        dataset_base = Path(dataset_path).expanduser().resolve()
        root = (dataset_base / dataset / "index").resolve()
        if not root.is_relative_to(dataset_base):
            raise ValueError(
                f"Lucene index root escapes the dataset path: {root}"
            )
        configs = []
        for (
            algorithm,
            group,
            _raw,
            build_combos,
            search_combos,
            _meta,
        ) in expanded_groups:
            _safe_label(algorithm, "algorithm")
            _safe_label(group, "group")
            for build_params in build_combos:
                codec = _codec_for(algorithm, build_params)
                name = algorithm if group == "base" else f"{algorithm}_{group}"
                index = IndexConfig(
                    name=name,
                    algo=algorithm,
                    build_param={"codec": codec},
                    search_params=[dict(item) for item in search_combos],
                    file=str(root / name),
                )
                configs.append(
                    BenchmarkConfig(
                        indexes=[index],
                        backend_config={
                            "name": name,
                            "algo": algorithm,
                            "group": group,
                            "codec": codec,
                            "index_root": str(root),
                            "requires_cuvs": algorithm in _CUVS_ALGORITHMS,
                            "include_cuvs": include_cuvs,
                            **runtime_overrides,
                        },
                    )
                )
        return configs


class LuceneBackend(BenchmarkBackend):
    """Build and query one immutable Lucene vector index per configuration."""

    def __init__(
        self,
        config: dict[str, Any],
        runtime_factory: Callable[
            [Mapping[str, Any]], LuceneRuntime
        ] = LuceneRuntime.create,
    ):
        super().__init__(config)
        self.algorithm = str(config.get("algo", ""))
        self.codec = str(config.get("codec", ""))
        self.group = _safe_label(str(config.get("group", "base")), "group")
        if _CODEC_BY_ALGORITHM.get(self.algorithm) != self.codec:
            raise ValueError(
                f"Invalid Lucene algorithm/codec pair: {self.algorithm!r}, {self.codec!r}"
            )
        self.maximum_dimensions = _MAX_DIMENSIONS_BY_ALGORITHM[self.algorithm]
        self._runtime_factory = runtime_factory
        self._runtime: LuceneRuntime | None = None

    @property
    def algo(self) -> str:
        """Return the selected cuVS Bench algorithm name."""
        return self.algorithm

    def _get_runtime(self) -> LuceneRuntime:
        if self._runtime is None:
            resolved = resolve_lucene_runtime_config(
                self.config,
                requires_cuvs=bool(self.config.get("requires_cuvs")),
                include_cuvs=bool(self.config.get("include_cuvs")),
            )
            self._runtime = self._runtime_factory(resolved)
        return self._runtime

    def _index(self, indexes: list[IndexConfig]) -> IndexConfig:
        if len(indexes) != 1:
            raise ValueError(
                "The Lucene backend expects one index configuration"
            )
        index = indexes[0]
        if index.algo != self.algorithm:
            raise ValueError(
                f"Index algorithm {index.algo!r} does not match {self.algorithm!r}"
            )
        _codec_for(index.algo, index.build_param)
        return index

    def _index_path(self, index: IndexConfig) -> Path:
        path = Path(os.path.abspath(Path(index.file).expanduser()))
        root = Path(self.config["index_root"]).expanduser().resolve()
        if path.is_symlink():
            raise ValueError(
                f"Lucene index path must not be a symlink: {path}"
            )
        if path.parent.resolve() != root or path == root:
            raise ValueError(
                f"Lucene index path is outside its configured root: {path}"
            )
        return path

    def _failure_build(
        self, index: IndexConfig, error: Exception
    ) -> BuildResult:
        return BuildResult(
            index_path=index.file,
            build_time_seconds=0.0,
            index_size_bytes=0,
            algorithm=index.algo,
            build_params=dict(index.build_param),
            success=False,
            error_message=_format_exception(error),
            metadata={"group": self.group, "index_name": index.name},
        )

    def _validate_manifest_identity(
        self, payload: Mapping[str, Any], dataset_identity: Mapping[str, Any]
    ) -> None:
        build_runtime_artifacts = payload.get("build_runtime_artifacts")
        if not isinstance(build_runtime_artifacts, Mapping):
            raise RuntimeError(
                "Lucene index manifest has invalid build-runtime artifact provenance"
            )
        artifact_keys = {
            "cuvs_java_coordinates",
            "cuvs_java_jar_path",
            "cuvs_java_jar_sha256",
            "cuvs_lucene_coordinates",
            "cuvs_lucene_jar_path",
            "cuvs_lucene_jar_sha256",
        }
        if (
            build_runtime_artifacts
            and set(build_runtime_artifacts) != artifact_keys
        ):
            raise RuntimeError(
                "Lucene index manifest has incomplete build-runtime artifact provenance"
            )
        if self.algorithm in _CUVS_ALGORITHMS and not build_runtime_artifacts:
            raise RuntimeError(
                "cuVS-backed Lucene index manifest has no build-runtime "
                "artifact provenance"
            )
        if build_runtime_artifacts:
            expected_coordinates = {
                "cuvs_java_coordinates": "com.nvidia.cuvs:cuvs-java:",
                "cuvs_lucene_coordinates": (
                    "com.nvidia.cuvs.lucene:cuvs-lucene:"
                ),
            }
            for key, prefix in expected_coordinates.items():
                value = build_runtime_artifacts[key]
                if (
                    not isinstance(value, str)
                    or not value.startswith(prefix)
                    or len(value) == len(prefix)
                ):
                    raise RuntimeError(
                        f"Lucene index manifest has invalid {key}"
                    )
            for key in ("cuvs_java_jar_path", "cuvs_lucene_jar_path"):
                value = build_runtime_artifacts[key]
                if not isinstance(value, str) or not Path(value).is_absolute():
                    raise RuntimeError(
                        f"Lucene index manifest has invalid {key}"
                    )
            for key in (
                "cuvs_java_jar_sha256",
                "cuvs_lucene_jar_sha256",
            ):
                value = build_runtime_artifacts[key]
                if (
                    not isinstance(value, str)
                    or re.fullmatch(r"[0-9a-f]{64}", value) is None
                ):
                    raise RuntimeError(
                        f"Lucene index manifest has invalid {key}"
                    )
        expected_without_segment_count = {
            "schema_version": _MANIFEST_SCHEMA,
            "algorithm": self.algorithm,
            "codec": self.codec,
            "dataset": dict(dataset_identity),
            "build_runtime_artifacts": dict(build_runtime_artifacts),
        }
        actual_without_segment_count = dict(payload)
        actual_without_segment_count.pop("segment_count", None)
        if actual_without_segment_count != expected_without_segment_count:
            raise RuntimeError(
                "Existing Lucene index does not match this dataset and configuration; "
                "rerun with --force"
            )

    @staticmethod
    def _validate_manifest_segment_count(
        payload: Mapping[str, Any], verification: Mapping[str, Any]
    ) -> None:
        stored = payload["segment_count"]
        observed = verification.get("segment_count")
        if not _is_manifest_integer(observed, minimum=1):
            raise RuntimeError(
                "Lucene index verification returned an invalid segment count"
            )
        if stored != observed:
            raise RuntimeError(
                "Lucene index manifest segment count does not match the "
                f"physical index: {stored} != {observed}; rerun with --force"
            )

    def _search_dataset_identity(
        self, dataset: Dataset, payload: Mapping[str, Any]
    ) -> tuple[dict[str, Any], int, int]:
        stored_identity = payload.get("dataset")
        if not isinstance(stored_identity, Mapping):
            raise RuntimeError("Lucene index manifest has no dataset identity")
        if dataset.base_file:
            return _file_backed_dataset_identity(
                dataset,
                stored_identity,
                maximum_dimensions=self.maximum_dimensions,
            )
        _normalize_subset_size(dataset)
        vectors = _validate_vectors(
            dataset.training_vectors,
            "training vectors",
            maximum_dimensions=self.maximum_dimensions,
        )
        return (
            _dataset_identity(dataset, vectors),
            int(vectors.shape[0]),
            int(vectors.shape[1]),
        )

    def _verification_metadata(
        self,
        runtime: LuceneRuntime,
        path: Path,
        vector_count: int,
        dimensions: int,
    ) -> dict[str, Any]:
        verification = runtime.index_verifier.verify(
            path,
            expected_codec=self.codec,
            expected_vector_count=vector_count,
            expected_dimensions=dimensions,
        )
        metadata = verification.metadata()
        metadata["build_route_policy"] = _BUILD_ROUTE_POLICY_BY_ALGORITHM[
            self.algorithm
        ]
        if self.codec == CAGRA_CODEC:
            metadata.update(
                runtime.cagra_verifier.verify(
                    path,
                    expected_vector_count=vector_count,
                    expected_dimensions=dimensions,
                ).metadata()
            )
        return metadata

    @staticmethod
    def _install_staged_index(staged: Path, destination: Path) -> str | None:
        """Publish a validated index and report non-fatal backup cleanup."""
        if not destination.exists():
            staged.rename(destination)
            return None
        backup = destination.parent / (
            f".{destination.name}.backup-{uuid.uuid4().hex}"
        )
        destination.rename(backup)
        try:
            staged.rename(destination)
        except BaseException as install_error:
            try:
                backup.rename(destination)
            except BaseException as restore_error:
                restore_note = (
                    "Failed to restore the previous index; its recoverable "
                    f"backup is {backup}: {type(restore_error).__name__}: "
                    f"{restore_error}"
                )
                if isinstance(
                    restore_error, (KeyboardInterrupt, SystemExit)
                ) and isinstance(install_error, Exception):
                    restore_error.add_note(
                        "Index installation first failed: "
                        f"{type(install_error).__name__}: {install_error}. "
                        f"The previous index remains in {backup}."
                    )
                    raise
                install_error.add_note(restore_note)
            raise
        try:
            shutil.rmtree(backup)
        except OSError as cleanup_error:
            return (
                f"Published the new index, but could not remove old backup "
                f"{backup}: {type(cleanup_error).__name__}: {cleanup_error}"
            )
        return None

    def build(
        self,
        dataset: Dataset,
        indexes: list[IndexConfig],
        force: bool = False,
        dry_run: bool = False,
    ) -> BuildResult:
        index = self._index(indexes)
        try:
            _validate_metric(dataset)
            path = self._index_path(index)
            if dry_run:
                return BuildResult(
                    index_path=str(path),
                    build_time_seconds=0.0,
                    index_size_bytes=0,
                    algorithm=self.algorithm,
                    build_params=dict(index.build_param),
                    metadata={
                        "dry_run": True,
                        "codec": self.codec,
                        "group": self.group,
                        "index_name": index.name,
                    },
                )
            if path.exists() and not force:
                payload = _read_manifest(path)
                dataset_identity, vector_count, dimensions = (
                    self._search_dataset_identity(dataset, payload)
                )
                self._validate_manifest_identity(payload, dataset_identity)
                runtime = self._get_runtime()
                runtime.verify_artifacts()
                metadata = self._verification_metadata(
                    runtime,
                    path,
                    vector_count,
                    dimensions,
                )
                self._validate_manifest_segment_count(payload, metadata)
                return BuildResult(
                    index_path=str(path),
                    build_time_seconds=0.0,
                    index_size_bytes=_index_size(path),
                    algorithm=self.algorithm,
                    build_params=dict(index.build_param),
                    metadata={
                        "skipped": True,
                        "codec": self.codec,
                        "group": self.group,
                        "index_name": index.name,
                        **_artifact_metadata(
                            "build_runtime",
                            payload["build_runtime_artifacts"],
                        ),
                        **metadata,
                    },
                )
            backend_build_started = time.perf_counter_ns()
            dataset_prepare_started = time.perf_counter_ns()
            _normalize_subset_size(dataset)
            vectors = _validate_vectors(
                dataset.training_vectors,
                "training vectors",
                maximum_dimensions=self.maximum_dimensions,
            )
            dataset_prepare_ns = (
                time.perf_counter_ns() - dataset_prepare_started
            )
            if path.exists() and (path.is_symlink() or not path.is_dir()):
                raise ValueError(
                    f"Lucene index path must be a directory: {path}"
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            runtime_setup_started = time.perf_counter_ns()
            runtime = self._get_runtime()
            runtime_setup_ns = time.perf_counter_ns() - runtime_setup_started
            artifact_validation_started = time.perf_counter_ns()
            runtime.verify_artifacts()
            artifact_validation_ns = (
                time.perf_counter_ns() - artifact_validation_started
            )
            staged = Path(
                tempfile.mkdtemp(
                    prefix=f".{path.name}.build-", dir=path.parent
                )
            )
            try:
                build_started = time.perf_counter_ns()
                runtime_build = runtime.build_index(
                    staged, vectors, self.codec
                )
                build_elapsed_ns = time.perf_counter_ns() - build_started

                validation_started = time.perf_counter_ns()
                metadata = self._verification_metadata(
                    runtime,
                    staged,
                    int(vectors.shape[0]),
                    int(vectors.shape[1]),
                )
                observed_segments = int(metadata["segment_count"])
                if runtime_build.segment_count != observed_segments:
                    raise RuntimeError(
                        "Lucene build reported a different segment count from "
                        f"the committed index: {runtime_build.segment_count} != "
                        f"{observed_segments}"
                    )
                payload = _manifest_payload(
                    dataset,
                    vectors,
                    self.algorithm,
                    self.codec,
                    observed_segments,
                    runtime.artifact_provenance,
                )
                _write_manifest(staged, payload)
                validation_elapsed_ns = (
                    time.perf_counter_ns() - validation_started
                )

                install_started = time.perf_counter_ns()
                cleanup_warning = self._install_staged_index(staged, path)
                install_elapsed_ns = time.perf_counter_ns() - install_started
            except BaseException:
                shutil.rmtree(staged, ignore_errors=True)
                raise
            index_size_started = time.perf_counter_ns()
            index_size_bytes = _index_size(path)
            index_size_elapsed_ns = time.perf_counter_ns() - index_size_started
            lifecycle_metadata: dict[str, Any] = {
                "dataset_load_validate_seconds": _nanoseconds_to_seconds(
                    dataset_prepare_ns
                ),
                "runtime_setup_seconds": _nanoseconds_to_seconds(
                    runtime_setup_ns
                ),
                "artifact_validation_seconds": _nanoseconds_to_seconds(
                    artifact_validation_ns
                ),
                "index_build_call_seconds": _nanoseconds_to_seconds(
                    build_elapsed_ns
                ),
                "index_validation_manifest_seconds": (
                    _nanoseconds_to_seconds(validation_elapsed_ns)
                ),
                "index_install_seconds": _nanoseconds_to_seconds(
                    install_elapsed_ns
                ),
                "index_size_measurement_seconds": _nanoseconds_to_seconds(
                    index_size_elapsed_ns
                ),
                "backend_build_total_seconds": _nanoseconds_to_seconds(
                    time.perf_counter_ns() - backend_build_started
                ),
                # Retain the original names for existing result consumers.
                "validation_time_seconds": _nanoseconds_to_seconds(
                    validation_elapsed_ns
                ),
                "install_time_seconds": _nanoseconds_to_seconds(
                    install_elapsed_ns
                ),
                **_runtime_build_timing_metadata(runtime_build),
            }
            if cleanup_warning is not None:
                lifecycle_metadata["cleanup_warning"] = cleanup_warning
            return BuildResult(
                index_path=str(path),
                build_time_seconds=_nanoseconds_to_seconds(build_elapsed_ns),
                index_size_bytes=index_size_bytes,
                algorithm=self.algorithm,
                build_params=dict(index.build_param),
                metadata={
                    "codec": self.codec,
                    "group": self.group,
                    "index_name": index.name,
                    "pylucene_version": runtime.pylucene_version,
                    **_artifact_metadata(
                        "build_runtime", runtime.artifact_provenance
                    ),
                    **lifecycle_metadata,
                    **metadata,
                },
            )
        except Exception as error:
            return self._failure_build(index, error)

    @staticmethod
    def _validate_hits(
        result: RuntimeSearchResult, k: int, expected_query_count: int
    ) -> None:
        if len(result.hits) != expected_query_count:
            raise RuntimeError(
                f"Lucene returned results for {len(result.hits)} queries, "
                f"expected {expected_query_count}"
            )
        for query_number, hits in enumerate(result.hits):
            if len(hits) != k:
                raise RuntimeError(
                    f"Lucene query {query_number} returned {len(hits)} hits, expected {k}"
                )
            ids = [hit.document_id for hit in hits]
            if len(ids) != len(set(ids)):
                raise RuntimeError(
                    f"Lucene query {query_number} returned duplicate IDs"
                )
            if any(
                identifier < 0 or identifier >= result.document_count
                for identifier in ids
            ):
                raise RuntimeError(
                    f"Lucene query {query_number} returned an invalid ID"
                )

    def _successful_search(
        self,
        runtime_result: RuntimeSearchResult,
        parameters: dict[str, int],
        *,
        k: int,
        batch_size: int,
        expected_query_count: int,
        runtime: LuceneRuntime,
        metadata: Mapping[str, Any],
    ) -> SearchResult:
        self._validate_hits(runtime_result, k, expected_query_count)
        timing_metadata, latency_samples_ns = _runtime_search_timing_metadata(
            runtime_result, expected_query_count
        )
        conversion_started = time.perf_counter_ns()
        neighbors = np.asarray(
            [
                [hit.document_id for hit in hits]
                for hits in runtime_result.hits
            ],
            dtype=np.int64,
        )
        distances = np.asarray(
            [
                [_score_to_squared_euclidean(hit.score) for hit in hits]
                for hits in runtime_result.hits
            ],
            dtype=np.float32,
        )
        result_array_conversion_ms = _nanoseconds_to_milliseconds(
            time.perf_counter_ns() - conversion_started
        )
        elapsed_ms = _nanoseconds_to_milliseconds(sum(latency_samples_ns))
        query_count = len(runtime_result.hits)
        query_corpus_seconds = _nanoseconds_to_seconds(
            runtime_result.timing.query_corpus_wall_ns
        )
        if elapsed_ms < 0.0 or query_corpus_seconds <= 0.0:
            raise RuntimeError("Lucene returned invalid query timing data")
        latency_samples_ms = (
            np.asarray(latency_samples_ns, dtype=np.float64) / 1_000_000.0
        )
        return SearchResult(
            neighbors=neighbors,
            distances=distances,
            search_time_ms=elapsed_ms,
            queries_per_second=(query_count / query_corpus_seconds),
            recall=0.0,
            algorithm=self.algorithm,
            search_params=[
                {} if self.algorithm == CAGRA_ALGORITHM else parameters
            ],
            latency_percentiles={
                "p50": float(np.percentile(latency_samples_ms, 50)),
                "p95": float(np.percentile(latency_samples_ms, 95)),
                "p99": float(np.percentile(latency_samples_ms, 99)),
            },
            metadata={
                "codec": self.codec,
                "pylucene_version": runtime.pylucene_version,
                **_artifact_metadata(
                    "search_runtime", runtime.artifact_provenance
                ),
                "timing_contract_version": _TIMING_CONTRACT_VERSION,
                "latency_scope": "client_query",
                "throughput_scope": "query_corpus_wall",
                "sample_unit": "query",
                "execution_model": "serial_single_query",
                "latency_seconds": float(latency_samples_ms.mean()) / 1000.0,
                "requested_batch_size": batch_size,
                "batch_size": batch_size,
                "effective_search_batch_size": 1,
                "query_count": query_count,
                "warmup_query_count": 0,
                "result_array_conversion_ms": result_array_conversion_ms,
                **timing_metadata,
                **metadata,
            },
        )

    def search(
        self,
        dataset: Dataset,
        indexes: list[IndexConfig],
        k: int,
        batch_size: int = 10000,
        mode: str = "latency",
        force: bool = False,
        search_threads: Optional[int] = None,
        dry_run: bool = False,
    ) -> list[SearchResult]:
        index = self._index(indexes)
        try:
            _validate_metric(dataset)
            if type(k) is not int or k < 1:
                raise ValueError("k must be a positive integer")
            if (
                isinstance(batch_size, bool)
                or not isinstance(batch_size, Integral)
                or batch_size < 1
            ):
                raise ValueError("batch_size must be a positive integer")
            batch_size = int(batch_size)
            if mode != "latency":
                raise ValueError(
                    "The initial Lucene backend supports only latency mode"
                )
            if search_threads is not None:
                raise ValueError(
                    "The initial Lucene backend does not yet support "
                    "search_threads"
                )
            path = self._index_path(index)
            validated_parameters = [
                _search_parameters(self.algorithm, parameters, k)
                for parameters in index.search_params
            ]
            if dry_run:
                return [
                    SearchResult(
                        neighbors=np.empty((0, k), dtype=np.int64),
                        distances=np.empty((0, k), dtype=np.float32),
                        search_time_ms=0.0,
                        queries_per_second=0.0,
                        recall=0.0,
                        algorithm=self.algorithm,
                        search_params=[],
                        metadata={
                            "dry_run": True,
                            "codec": self.codec,
                            "group": self.group,
                            "index_name": index.name,
                        },
                    )
                ]
            backend_search_started = time.perf_counter_ns()
            input_prepare_started = time.perf_counter_ns()
            queries = _validate_vectors(
                dataset.query_vectors,
                "query vectors",
                maximum_dimensions=self.maximum_dimensions,
            )
            query_input_prepare_ns = (
                time.perf_counter_ns() - input_prepare_started
            )
            identity_validation_started = time.perf_counter_ns()
            if not path.is_dir():
                raise FileNotFoundError(f"Lucene index does not exist: {path}")
            payload = _read_manifest(path)
            dataset_identity, vector_count, dimensions = (
                self._search_dataset_identity(dataset, payload)
            )
            self._validate_manifest_identity(payload, dataset_identity)
            if int(queries.shape[1]) != dimensions:
                raise ValueError(
                    "query vector dimensions do not match the indexed dataset: "
                    f"{queries.shape[1]} != {dimensions}"
                )
            index_identity_validation_ns = (
                time.perf_counter_ns() - identity_validation_started
            )
            runtime_setup_started = time.perf_counter_ns()
            runtime = self._get_runtime()
            runtime_setup_ns = time.perf_counter_ns() - runtime_setup_started
            artifact_validation_started = time.perf_counter_ns()
            runtime.verify_artifacts()
            artifact_validation_ns = (
                time.perf_counter_ns() - artifact_validation_started
            )
            index_verification_started = time.perf_counter_ns()
            metadata = self._verification_metadata(
                runtime, path, vector_count, dimensions
            )
            self._validate_manifest_segment_count(payload, metadata)
            index_verification_ns = (
                time.perf_counter_ns() - index_verification_started
            )

            results = []
            for parameters in validated_parameters:
                search_plan_started = time.perf_counter_ns()
                prewarm = _prewarm_index_files(path)
                runtime_result = runtime.search_index(
                    path,
                    queries,
                    k=k,
                    num_candidates=parameters["num_candidates"],
                )
                result = self._successful_search(
                    runtime_result,
                    parameters,
                    k=k,
                    batch_size=batch_size,
                    expected_query_count=int(queries.shape[0]),
                    runtime=runtime,
                    metadata={
                        "mode": mode,
                        "group": self.group,
                        "index_name": index.name,
                        **_artifact_metadata(
                            "build_runtime",
                            payload["build_runtime_artifacts"],
                        ),
                        "expected_search_route": (
                            _SEARCH_ROUTE_BY_ALGORITHM[self.algorithm]
                        ),
                        "query_input_prepare_ms": (
                            _nanoseconds_to_milliseconds(
                                query_input_prepare_ns
                            )
                        ),
                        "index_identity_validation_ms": (
                            _nanoseconds_to_milliseconds(
                                index_identity_validation_ns
                            )
                        ),
                        "runtime_setup_ms": _nanoseconds_to_milliseconds(
                            runtime_setup_ns
                        ),
                        "artifact_validation_ms": (
                            _nanoseconds_to_milliseconds(
                                artifact_validation_ns
                            )
                        ),
                        "index_verification_ms": (
                            _nanoseconds_to_milliseconds(index_verification_ns)
                        ),
                        "cache_policy": "sequential_read_all_index_files",
                        "index_prewarm_ms": _nanoseconds_to_milliseconds(
                            prewarm.wall_ns
                        ),
                        "index_prewarm_bytes": prewarm.bytes_read,
                        "index_prewarm_file_count": prewarm.file_count,
                        **metadata,
                    },
                )
                result.metadata["search_plan_total_ms"] = (
                    _nanoseconds_to_milliseconds(
                        time.perf_counter_ns() - search_plan_started
                    )
                )
                results.append(result)
            backend_search_invocation_total_ms = _nanoseconds_to_milliseconds(
                time.perf_counter_ns() - backend_search_started
            )
            for result in results:
                result.metadata["backend_search_invocation_total_ms"] = (
                    backend_search_invocation_total_ms
                )
            return results
        except Exception as error:
            result_width = k if type(k) is int and k > 0 else 0
            return [
                SearchResult(
                    neighbors=np.empty((0, result_width), dtype=np.int64),
                    distances=np.empty((0, result_width), dtype=np.float32),
                    search_time_ms=0.0,
                    queries_per_second=0.0,
                    recall=0.0,
                    algorithm=self.algorithm,
                    search_params=[],
                    success=False,
                    error_message=_format_exception(error),
                    metadata={
                        "group": self.group,
                        "index_name": index.name,
                    },
                )
            ]


def register() -> None:
    """Register the Lucene backend and loader without importing PyLucene."""
    from .registry import (
        _CONFIG_LOADER_REGISTRY,
        get_registry,
        register_backend,
        register_config_loader,
    )

    registry = get_registry()
    if not registry.is_registered("lucene"):
        register_backend("lucene", LuceneBackend)
    if "lucene" not in _CONFIG_LOADER_REGISTRY:
        register_config_loader("lucene", LuceneConfigLoader)
