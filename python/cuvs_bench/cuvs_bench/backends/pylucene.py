#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""PyLucene backend for cuVS-accelerated Lucene codecs."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import numpy as np

from .._bin_format import read_bin_header
from ..orchestrator.config_loaders import (
    BenchmarkConfig,
    ConfigLoader,
    DatasetConfig,
    IndexConfig,
)
from ._utils import dtype_from_filename
from .base import BenchmarkBackend, BuildResult, Dataset, SearchResult

_ID_FIELD = "id"
_VECTOR_FIELD = "vector"
_MAX_DIMENSIONS = 4096

_HNSW_CODECS = frozenset(
    {
        "Lucene101AcceleratedHNSWCodec",
        "Lucene101AcceleratedHNSWBaseLayerCodec",
        "Lucene101AcceleratedHNSWMultiLayerCodec",
    }
)
_CAGRA_CODEC = "CuVS2510GPUSearchCodec"
_SUPPORTED_CODECS = _HNSW_CODECS | {_CAGRA_CODEC}
_SUPPORTED_BUILD_KEYS = frozenset({"codec"})
_EXPECTED_WRITER_PATH = {
    **{codec: "gpu-hnsw" for codec in _HNSW_CODECS},
    _CAGRA_CODEC: "gpu-cagra",
}
_CAGRA_META_EXTENSION = ".vemc"
_CAGRA_META_CODEC_NAME = "Lucene102CuVSVectorsFormatMeta"
_CAGRA_INDEX_EXTENSION = ".vcag"
_CAGRA_INDEX_CODEC_NAME = "Lucene102CuVSVectorsFormatIndex"
_CAGRA_META_VERSION = 0
_CAGRA_INDEX_VERSION = 0
# Avoid caching checksums while coarse filesystem timestamps can still make a
# same-size rewrite look unchanged.
_CAGRA_CACHE_MIN_FILE_AGE_NS = 2_000_000_000
_FLOAT32_ENCODING_ORDINAL = 1
_EUCLIDEAN_SIMILARITY_ORDINAL = 0

_HNSW_PROVENANCE_FILE = ".cuvs-bench-pylucene-hnsw.json"
_CAGRA_PROVENANCE_FILE = ".cuvs-bench-pylucene-cagra.json"
_PROVENANCE_SCHEMA_VERSION = 1
_PROVENANCE_KEYS = frozenset(
    {
        "schema_version",
        "codec",
        "writer_path",
        "vector_count",
        "dimensions",
        "commit_fingerprints",
    }
)
_SHA256_HEX_DIGITS = frozenset("0123456789abcdef")
_LUCENE_CORE_CLASS = "org/apache/lucene/index/IndexWriter.class"

_JVM_INIT_LOCK = threading.Lock()
_INITIALIZED_CLASSPATH: Optional[str] = None
_INITIALIZED_VMARGS: Optional[Tuple[str, ...]] = None


def _attempt_cleanup(
    cleanup: Callable[[], None],
    description: str,
    primary_error: Optional[BaseException],
) -> Optional[BaseException]:
    try:
        cleanup()
    except BaseException as cleanup_error:
        if primary_error is None:
            return cleanup_error
        if isinstance(primary_error, Exception) and not isinstance(
            cleanup_error, Exception
        ):
            cleanup_error.add_note(
                f"Raised while attempting to {description}; prior failure: "
                f"{type(primary_error).__name__}: {primary_error}"
            )
            return cleanup_error
        primary_error.add_note(
            f"Failed to {description}: "
            f"{type(cleanup_error).__name__}: {cleanup_error}"
        )
    return primary_error


class _CleanupStack:
    """Close registered resources without losing the operation's failure."""

    def __init__(self) -> None:
        self._cleanups: List[Tuple[str, Callable[[], None]]] = []

    def __enter__(self) -> _CleanupStack:
        return self

    def add(self, description: str, cleanup: Callable[[], None]) -> None:
        self._cleanups.append((description, cleanup))

    def __exit__(
        self,
        _error_type: Any,
        primary_error: Optional[BaseException],
        _traceback: Any,
    ) -> bool:
        final_error = primary_error
        for description, cleanup in reversed(self._cleanups):
            final_error = _attempt_cleanup(cleanup, description, final_error)
        if final_error is not None and final_error is not primary_error:
            raise final_error
        return False


def _exception_summary(error: Exception) -> str:
    details = [f"{type(error).__name__}: {error}"]
    if error.__cause__ is not None:
        cause = error.__cause__
        details.append(f"caused by {type(cause).__name__}: {cause}")
    details.extend(getattr(error, "__notes__", ()))
    return "; ".join(details)


def _configured_jar(
    config: Dict[str, Any], config_key: str, environment_key: str
) -> Path:
    value = config.get(config_key) or os.environ.get(environment_key)
    if not value:
        raise RuntimeError(
            f"PyLucene backend requires '{config_key}' or "
            f"the {environment_key} environment variable"
        )

    path = Path(os.fspath(value)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{config_key} does not exist: {path}")
    return path


def _reject_bundled_lucene_classes(cuvs_lucene_jar: Path) -> None:
    """Reject fat cuVS-Lucene jars before they can poison the process JVM."""
    if not zipfile.is_zipfile(cuvs_lucene_jar):
        return

    with zipfile.ZipFile(cuvs_lucene_jar) as archive:
        try:
            archive.getinfo(_LUCENE_CORE_CLASS)
        except KeyError:
            return

    raise RuntimeError(
        "cuvs_lucene_jar bundles Lucene classes and is incompatible with "
        "PyLucene's process-wide JVM. Use the standard thin cuvs-lucene JAR, "
        "not a '-jar-with-dependencies' artifact."
    )


def _load_pylucene() -> Any:
    try:
        return importlib.import_module("lucene")
    except ImportError as exc:
        raise ImportError(
            "PyLucene's `lucene` module is required for the pylucene "
            "cuvs-bench backend. PyLucene must be built and installed "
            "separately; see the cuVS Bench installation documentation."
        ) from exc


def _pylucene_classpath(config: Dict[str, Any], lucene: Any) -> str:
    cuvs_java_jar = _configured_jar(
        config, "cuvs_java_jar", "CUVS_LUCENE_CUVS_JAVA_JAR"
    )
    cuvs_lucene_jar = _configured_jar(
        config, "cuvs_lucene_jar", "CUVS_LUCENE_JAR"
    )
    _reject_bundled_lucene_classes(cuvs_lucene_jar)
    return os.pathsep.join(
        [str(cuvs_java_jar), str(cuvs_lucene_jar), lucene.CLASSPATH]
    )


def _pylucene_vmargs(config: Dict[str, Any]) -> List[str]:
    java_library_path = (
        config.get("java_library_path")
        or os.environ.get("JAVA_LIBRARY_PATH")
        or os.environ.get("LD_LIBRARY_PATH")
    )
    vmargs = [
        "--enable-native-access=ALL-UNNAMED",
        "--add-modules=jdk.incubator.vector",
    ]
    if java_library_path:
        vmargs.append(f"-Djava.library.path={java_library_path}")

    extra_vmargs = config.get("jvm_args", [])
    if isinstance(extra_vmargs, (str, bytes)) or not isinstance(
        extra_vmargs, (list, tuple)
    ):
        raise TypeError("jvm_args must be a list or tuple of strings")
    if not all(isinstance(arg, str) for arg in extra_vmargs):
        raise TypeError("every jvm_args entry must be a string")
    vmargs.extend(extra_vmargs)
    return vmargs


def _attach_pylucene_jvm(
    lucene: Any, classpath: str, vmargs: List[str]
) -> None:
    with _JVM_INIT_LOCK:
        vm_environment = lucene.getVMEnv()
        if vm_environment is None:
            vm_environment = _start_pylucene_jvm(lucene, classpath, vmargs)
        else:
            _validate_pylucene_jvm_configuration(classpath, vmargs)

        vm_environment.attachCurrentThread()


def _start_pylucene_jvm(lucene: Any, classpath: str, vmargs: List[str]) -> Any:
    global _INITIALIZED_CLASSPATH, _INITIALIZED_VMARGS
    vm_environment = lucene.initVM(classpath=classpath, vmargs=vmargs)
    if vm_environment is None:
        vm_environment = lucene.getVMEnv()
    if vm_environment is None:
        raise RuntimeError("PyLucene did not return a JVM environment")

    _INITIALIZED_CLASSPATH = classpath
    _INITIALIZED_VMARGS = tuple(vmargs)
    return vm_environment


def _validate_pylucene_jvm_configuration(
    classpath: str, vmargs: List[str]
) -> None:
    if _INITIALIZED_CLASSPATH is None or _INITIALIZED_VMARGS is None:
        raise RuntimeError(
            "PyLucene's process-wide JVM was initialized before the "
            "pylucene backend, so the required cuVS classpath and JVM "
            "arguments cannot be verified. Start a new Python process and "
            "let cuVS Bench initialize PyLucene."
        )
    if _INITIALIZED_CLASSPATH != classpath:
        raise RuntimeError(
            "PyLucene's process-wide JVM is already initialized with "
            "different cuVS Java or cuVS-Lucene jars"
        )
    if _INITIALIZED_VMARGS != tuple(vmargs):
        raise RuntimeError(
            "PyLucene's process-wide JVM is already initialized with "
            "different JVM arguments or native-library paths"
        )


def _initialize_pylucene(config: Dict[str, Any]) -> Any:
    """Initialize PyLucene once and attach the current Python thread."""
    lucene = _load_pylucene()
    classpath = _pylucene_classpath(config, lucene)
    vmargs = _pylucene_vmargs(config)
    _attach_pylucene_jvm(lucene, classpath, vmargs)
    return lucene


def _parse_writer_telemetry(description: str) -> Dict[str, str]:
    """Parse cuVS-Lucene vector-format diagnostics."""
    _, separator, payload = description.partition("(")
    if not separator or not payload.endswith(")"):
        raise RuntimeError(
            f"Malformed cuVS-Lucene writer telemetry: {description!r}"
        )

    telemetry: Dict[str, str] = {}
    for item in payload[:-1].split(";"):
        key, item_separator, value = item.partition("=")
        if not item_separator or not key:
            raise RuntimeError(
                "Malformed cuVS-Lucene writer telemetry item "
                f"{item!r} in {description!r}"
            )
        telemetry[key] = value
    return telemetry


def _score_to_squared_euclidean(score: float) -> float:
    """Convert Lucene's Euclidean score to squared Euclidean distance."""
    if score <= 0.0:
        return float("inf")
    return max(0.0, (1.0 / score) - 1.0)


def _validate_float32_matrix(vectors: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(vectors)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one vector")
    if array.dtype != np.float32:
        raise TypeError(f"{name} must use float32 values, got {array.dtype}")
    if array.shape[1] > _MAX_DIMENSIONS:
        raise ValueError(
            f"{name} dimensions must not exceed {_MAX_DIMENSIONS}, "
            f"got {array.shape[1]}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array)


def _validate_metric(dataset: Dataset) -> None:
    if dataset.distance_metric.lower() not in {"euclidean", "l2"}:
        raise ValueError(
            "PyLucene cuVS codecs currently support only Euclidean/L2 "
            f"datasets, got {dataset.distance_metric!r}"
        )


def _validate_codec(codec_name: Any) -> str:
    if not isinstance(codec_name, str) or codec_name not in _SUPPORTED_CODECS:
        available = ", ".join(sorted(_SUPPORTED_CODECS))
        raise ValueError(
            f"Unsupported PyLucene codec {codec_name!r}. "
            f"Supported codecs: {available}"
        )
    return codec_name


def _configured_codec(
    build_params: Dict[str, Any], backend_config: Dict[str, Any]
) -> str:
    if "codec" in build_params:
        codec_name = build_params["codec"]
    else:
        codec_name = backend_config.get("codec")
    return _validate_codec(codec_name)


def _validate_build_params(build_params: Any) -> Dict[str, Any]:
    if not isinstance(build_params, dict):
        raise TypeError("PyLucene build parameters must be a mapping")
    unsupported = set(build_params) - _SUPPORTED_BUILD_KEYS
    if unsupported:
        names = ", ".join(sorted(str(name) for name in unsupported))
        raise ValueError(f"Unsupported PyLucene build parameter(s): {names}")
    return build_params


def _validate_search_params(search_params: Any) -> List[Dict[str, Any]]:
    if (
        not isinstance(search_params, list)
        or len(search_params) != 1
        or search_params[0] != {}
    ):
        raise ValueError(
            "PyLucene's public Lucene query API does not expose "
            "cuVS-specific search parameters"
        )
    return search_params


def _safe_remove_index(index_path: Path) -> None:
    resolved = index_path.resolve()
    forbidden = {
        Path(resolved.anchor),
        Path.cwd().resolve(),
        Path.home().resolve(),
    }
    if resolved in forbidden:
        raise ValueError(f"Refusing to remove unsafe index path: {resolved}")
    if index_path.is_symlink() or not index_path.is_dir():
        raise ValueError(
            f"PyLucene index path must be a directory: {index_path}"
        )
    shutil.rmtree(index_path)


def _index_size(index_path: Path) -> int:
    return sum(
        path.stat().st_size for path in index_path.rglob("*") if path.is_file()
    )


@dataclass(frozen=True)
class _FileSignature:
    resolved_path: str
    device: int
    inode: int
    size: int
    modified_at_ns: int
    changed_at_ns: int


def _file_signature(path: Path) -> _FileSignature:
    file_stat = path.stat()
    return _FileSignature(
        resolved_path=str(path.resolve()),
        device=file_stat.st_dev,
        inode=file_stat.st_ino,
        size=file_stat.st_size,
        modified_at_ns=file_stat.st_mtime_ns,
        changed_at_ns=file_stat.st_ctime_ns,
    )


def _has_lucene_segments(index_path: Path) -> bool:
    return any(
        path.is_file() and path.name.startswith("segments_")
        for path in index_path.iterdir()
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _commit_fingerprints(index_path: Path) -> List[Dict[str, str]]:
    commit_files = sorted(
        path
        for path in index_path.iterdir()
        if path.name.startswith("segments_")
    )
    if not commit_files:
        raise RuntimeError(
            "Lucene index has no segments_* commit file to fingerprint"
        )

    fingerprints = []
    for path in commit_files:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                "Lucene commit fingerprint target must be a regular file: "
                f"{path}"
            )
        fingerprints.append({"name": path.name, "sha256": _sha256_file(path)})
    return fingerprints


@dataclass(frozen=True)
class _IndexProvenanceVerification:
    codec: str
    writer_path: str
    vector_count: int
    dimensions: int
    commit_file_count: int

    def to_metadata(self) -> Dict[str, Union[str, int]]:
        return {
            "status": f"{self.writer_path}-provenance",
            "schema_version": _PROVENANCE_SCHEMA_VERSION,
            "codec": self.codec,
            "writer_path": self.writer_path,
            "vector_count": self.vector_count,
            "dimensions": self.dimensions,
            "commit_file_count": self.commit_file_count,
        }


@dataclass(frozen=True)
class _ProvenanceExpectation:
    codec: str
    manifest_name: str
    label: str
    vector_count: Optional[int]
    dimensions: Optional[int]


class _IndexProvenanceError(RuntimeError):
    """Raised when an index cannot be proven to be backend-built."""


def _write_index_provenance(
    index_path: Path,
    codec: str,
    vector_count: int,
    dimensions: int,
    manifest_name: str,
) -> None:
    payload = {
        "schema_version": _PROVENANCE_SCHEMA_VERSION,
        "codec": codec,
        "writer_path": _EXPECTED_WRITER_PATH[codec],
        "vector_count": int(vector_count),
        "dimensions": int(dimensions),
        "commit_fingerprints": _commit_fingerprints(index_path),
    }
    manifest_path = index_path / manifest_name
    with _CleanupStack() as cleanups:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=index_path,
            prefix=f"{manifest_name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            cleanups.add(
                "remove temporary provenance file",
                lambda: temporary_path.unlink(missing_ok=True),
            )
            json.dump(
                payload,
                file,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        temporary_path.chmod(0o644)
        temporary_path.replace(manifest_path)


def _write_hnsw_provenance(
    index_path: Path,
    codec: str,
    vector_count: int,
    dimensions: int,
) -> None:
    _write_index_provenance(
        index_path,
        codec,
        vector_count,
        dimensions,
        _HNSW_PROVENANCE_FILE,
    )


def _write_cagra_provenance(
    index_path: Path,
    vector_count: int,
    dimensions: int,
) -> None:
    _write_index_provenance(
        index_path,
        _CAGRA_CODEC,
        vector_count,
        dimensions,
        _CAGRA_PROVENANCE_FILE,
    )


def _require_positive_int(
    value: Any, provenance_label: str, field_name: str
) -> int:
    if type(value) is not int or value < 1:
        raise _IndexProvenanceError(
            f"{provenance_label} field {field_name!r} must be a "
            "positive integer"
        )
    return value


def _read_index_provenance(
    manifest_path: Path, provenance_label: str
) -> Dict[str, Any]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise _IndexProvenanceError(
            f"{provenance_label} manifest is missing: {manifest_path}"
        ) from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _IndexProvenanceError(
            f"{provenance_label} manifest cannot be read: "
            f"{manifest_path}: {exc}"
        ) from exc

    if not isinstance(payload, dict) or set(payload) != _PROVENANCE_KEYS:
        raise _IndexProvenanceError(
            f"{provenance_label} manifest has an unsupported schema"
        )
    return payload


def _validate_provenance_identity(
    payload: Dict[str, Any],
    expected_codec: str,
    provenance_label: str,
) -> str:
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != _PROVENANCE_SCHEMA_VERSION
    ):
        raise _IndexProvenanceError(
            f"{provenance_label} manifest has an unsupported schema version"
        )
    if payload["codec"] != expected_codec:
        raise _IndexProvenanceError(
            f"{provenance_label} codec does not match the requested codec: "
            f"{payload['codec']!r} != {expected_codec!r}"
        )

    writer_path = payload["writer_path"]
    if writer_path != _EXPECTED_WRITER_PATH[expected_codec]:
        raise _IndexProvenanceError(
            f"{provenance_label} does not record the required GPU writer path"
        )
    return writer_path


def _validate_provenance_shape(
    payload: Dict[str, Any],
    provenance_label: str,
    expected_vector_count: Optional[int] = None,
    expected_dimensions: Optional[int] = None,
) -> Tuple[int, int]:
    vector_count = _require_positive_int(
        payload["vector_count"], provenance_label, "vector_count"
    )
    if vector_count < 2:
        raise _IndexProvenanceError(
            f"{provenance_label} vector count must be at least two"
        )
    dimensions = _require_positive_int(
        payload["dimensions"], provenance_label, "dimensions"
    )
    if dimensions > _MAX_DIMENSIONS:
        raise _IndexProvenanceError(
            f"{provenance_label} dimensions exceed the supported maximum: "
            f"{dimensions} > {_MAX_DIMENSIONS}"
        )
    if (
        expected_vector_count is not None
        and vector_count != expected_vector_count
    ):
        raise _IndexProvenanceError(
            f"{provenance_label} vector count does not match the dataset: "
            f"{vector_count} != {expected_vector_count}"
        )
    if expected_dimensions is not None and dimensions != expected_dimensions:
        raise _IndexProvenanceError(
            f"{provenance_label} dimensions do not match the dataset: "
            f"{dimensions} != {expected_dimensions}"
        )
    return vector_count, dimensions


def _validate_commit_fingerprints(
    stored_fingerprints: Any, provenance_label: str
) -> List[Dict[str, str]]:
    if not isinstance(stored_fingerprints, list) or not stored_fingerprints:
        raise _IndexProvenanceError(
            f"{provenance_label} has no Lucene commit fingerprints"
        )
    names = []
    for fingerprint in stored_fingerprints:
        name, _ = _validate_commit_fingerprint(fingerprint, provenance_label)
        names.append(name)
    if names != sorted(set(names)):
        raise _IndexProvenanceError(
            f"{provenance_label} commit fingerprints must be unique and sorted"
        )
    return stored_fingerprints


def _validate_commit_fingerprint(
    fingerprint: Any, provenance_label: str
) -> Tuple[str, str]:
    if not isinstance(fingerprint, dict) or set(fingerprint) != {
        "name",
        "sha256",
    }:
        raise _IndexProvenanceError(
            f"{provenance_label} has a malformed Lucene commit fingerprint"
        )

    name = _validate_commit_filename(fingerprint["name"], provenance_label)
    digest = _validate_sha256_digest(fingerprint["sha256"], provenance_label)
    return name, digest


def _validate_commit_filename(value: Any, provenance_label: str) -> str:
    if not isinstance(value, str):
        raise _IndexProvenanceError(
            f"{provenance_label} commit filename must be a string"
        )
    if not value.startswith("segments_"):
        raise _IndexProvenanceError(
            f"{provenance_label} commit filename must start with 'segments_'"
        )
    if Path(value).name != value:
        raise _IndexProvenanceError(
            f"{provenance_label} commit filename must not contain a path"
        )
    return value


def _validate_sha256_digest(value: Any, provenance_label: str) -> str:
    if not isinstance(value, str):
        raise _IndexProvenanceError(
            f"{provenance_label} commit SHA-256 must be a string"
        )
    if len(value) != 64:
        raise _IndexProvenanceError(
            f"{provenance_label} commit SHA-256 must contain 64 characters"
        )
    if not set(value).issubset(_SHA256_HEX_DIGITS):
        raise _IndexProvenanceError(
            f"{provenance_label} commit SHA-256 must be lowercase hexadecimal"
        )
    return value


def _verify_index_provenance(
    index_path: Path,
    expectation: _ProvenanceExpectation,
) -> _IndexProvenanceVerification:
    payload = _read_index_provenance(
        index_path / expectation.manifest_name, expectation.label
    )
    writer_path = _validate_provenance_identity(
        payload, expectation.codec, expectation.label
    )
    vector_count, dimensions = _validate_provenance_shape(
        payload,
        expectation.label,
        expectation.vector_count,
        expectation.dimensions,
    )
    stored_fingerprints = _validate_commit_fingerprints(
        payload["commit_fingerprints"], expectation.label
    )
    try:
        current_fingerprints = _commit_fingerprints(index_path)
    except (OSError, RuntimeError) as exc:
        raise _IndexProvenanceError(
            f"{expectation.label} cannot fingerprint the Lucene commit: {exc}"
        ) from exc
    if stored_fingerprints != current_fingerprints:
        raise _IndexProvenanceError(
            f"{expectation.label} does not match the current Lucene commit"
        )

    return _IndexProvenanceVerification(
        codec=expectation.codec,
        writer_path=writer_path,
        vector_count=vector_count,
        dimensions=dimensions,
        commit_file_count=len(current_fingerprints),
    )


def _verify_hnsw_provenance(
    index_path: Path,
    expected_codec: str,
    *,
    expected_vector_count: Optional[int] = None,
    expected_dimensions: Optional[int] = None,
) -> _IndexProvenanceVerification:
    return _verify_index_provenance(
        index_path,
        _ProvenanceExpectation(
            codec=expected_codec,
            manifest_name=_HNSW_PROVENANCE_FILE,
            label="HNSW provenance",
            vector_count=expected_vector_count,
            dimensions=expected_dimensions,
        ),
    )


def _verify_cagra_provenance(
    index_path: Path,
    *,
    expected_vector_count: Optional[int] = None,
    expected_dimensions: Optional[int] = None,
) -> _IndexProvenanceVerification:
    return _verify_index_provenance(
        index_path,
        _ProvenanceExpectation(
            codec=_CAGRA_CODEC,
            manifest_name=_CAGRA_PROVENANCE_FILE,
            label="CAGRA provenance",
            vector_count=expected_vector_count,
            dimensions=expected_dimensions,
        ),
    )


def _validate_subset_size(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"subset_size must be a positive integer, got {value!r}"
        )
    return value


def _expected_training_shape(dataset: Dataset) -> Optional[Tuple[int, int]]:
    vectors = dataset.loaded_training_vectors
    if vectors is not None:
        vectors = np.asarray(vectors)
        if vectors.ndim != 2:
            raise ValueError(
                "training_vectors must be a two-dimensional array"
            )
        if vectors.dtype != np.float32:
            raise TypeError(
                "training_vectors must use float32 values, "
                f"got {vectors.dtype}"
            )
        rows, dimensions = vectors.shape
    elif dataset.base_file:
        dtype = np.dtype(dtype_from_filename(dataset.base_file))
        if dtype != np.float32:
            raise TypeError(
                f"training_vectors must use float32 values, got {dtype}"
            )
        rows, dimensions, _ = read_bin_header(
            dataset.base_file, itemsize=dtype.itemsize
        )
        subset_size = _validate_subset_size(
            dataset.metadata.get("subset_size")
        )
        if subset_size is not None:
            rows = min(rows, subset_size)
    else:
        return None

    if rows < 1 or dimensions < 1:
        raise ValueError("training_vectors must contain at least one vector")
    if dimensions > _MAX_DIMENSIONS:
        raise ValueError(
            f"training_vectors dimensions must not exceed {_MAX_DIMENSIONS}, "
            f"got {dimensions}"
        )
    return int(rows), int(dimensions)


@dataclass(frozen=True)
class _SearchHit:
    document_id: int
    score: float


@dataclass(frozen=True)
class _RuntimeSearchResult:
    hits: List[List[_SearchHit]]
    batch_latencies_ms: List[float]
    index_dimensions: int
    document_count: int


@dataclass(frozen=True)
class _ProcessedSearchResult:
    neighbors: np.ndarray
    distances: np.ndarray
    search_time_ms: float
    queries_per_second: float
    latency_seconds: float
    latency_percentiles: Dict[str, float]
    num_batches: int


@dataclass(frozen=True)
class _SearchInputs:
    query_vectors: np.ndarray
    expected_training_shape: Optional[Tuple[int, int]]


@dataclass(frozen=True)
class _SearchPlan:
    index_path: Path
    codec_name: str
    search_params: List[Dict[str, Any]]
    k: int
    batch_size: int
    mode: str


@dataclass(frozen=True)
class _ResolvedCodec:
    codec_name: str
    java_codec: Any
    telemetry: Mapping[str, str]
    writer_path: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "telemetry", MappingProxyType(dict(self.telemetry))
        )


class _ExistingIndexAction(Enum):
    BUILD = "build"
    REUSE = "reuse"
    REJECT = "reject"


@dataclass(frozen=True)
class _ExistingIndexDecision:
    action: _ExistingIndexAction
    result: Optional[BuildResult] = None

    @classmethod
    def build(cls) -> _ExistingIndexDecision:
        return cls(action=_ExistingIndexAction.BUILD)

    @classmethod
    def reuse(cls, result: BuildResult) -> _ExistingIndexDecision:
        return cls(action=_ExistingIndexAction.REUSE, result=result)

    @classmethod
    def reject(cls, result: BuildResult) -> _ExistingIndexDecision:
        return cls(action=_ExistingIndexAction.REJECT, result=result)

    def completed_result(self) -> BuildResult:
        if self.result is None:
            raise RuntimeError(
                "Existing-index decision is missing its build result"
            )
        return self.result


def _validate_runtime_search_result(
    runtime_result: _RuntimeSearchResult,
    provenance: _IndexProvenanceVerification,
    *,
    query_count: int,
) -> None:
    if runtime_result.document_count != provenance.vector_count:
        raise RuntimeError(
            "Lucene document count does not match index provenance: "
            f"{runtime_result.document_count} != {provenance.vector_count}"
        )
    if runtime_result.index_dimensions != provenance.dimensions:
        raise RuntimeError(
            "Lucene index dimensions do not match index provenance: "
            f"{runtime_result.index_dimensions} != {provenance.dimensions}"
        )
    if len(runtime_result.hits) != query_count:
        raise RuntimeError(
            "PyLucene returned an unexpected number of query results: "
            f"{len(runtime_result.hits)} != {query_count}"
        )


def _validate_search_hit(
    hit: _SearchHit,
    *,
    query_id: int,
    document_count: int,
    seen_document_ids: set[int],
) -> None:
    if not 0 <= hit.document_id < document_count:
        raise RuntimeError(
            "PyLucene returned an out-of-range stored ID for "
            f"query {query_id}: {hit.document_id}"
        )
    if hit.document_id in seen_document_ids:
        raise RuntimeError(
            "PyLucene returned a duplicate stored ID for "
            f"query {query_id}: {hit.document_id}"
        )
    if not np.isfinite(hit.score):
        raise RuntimeError(
            "PyLucene returned a non-finite score for "
            f"query {query_id}: {hit.score}"
        )
    if not 0.0 <= hit.score <= 1.0:
        raise RuntimeError(
            "PyLucene returned a score outside the Euclidean range "
            f"[0, 1] for query {query_id}: {hit.score}"
        )
    seen_document_ids.add(hit.document_id)


def _convert_search_hits(
    runtime_result: _RuntimeSearchResult, *, query_count: int, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    neighbors = np.full((query_count, k), -1, dtype=np.int64)
    distances = np.full((query_count, k), np.inf, dtype=np.float32)
    for query_id, hits in enumerate(runtime_result.hits):
        query_neighbors, query_distances = _convert_query_hits(
            hits,
            query_id=query_id,
            document_count=runtime_result.document_count,
            k=k,
        )
        hit_count = len(query_neighbors)
        neighbors[query_id, :hit_count] = query_neighbors
        distances[query_id, :hit_count] = query_distances
    return neighbors, distances


def _convert_query_hits(
    hits: List[_SearchHit],
    *,
    query_id: int,
    document_count: int,
    k: int,
) -> Tuple[List[int], List[float]]:
    if len(hits) > min(k, document_count):
        raise RuntimeError(
            f"PyLucene returned too many hits for query {query_id}: "
            f"{len(hits)}"
        )

    neighbors = []
    distances = []
    seen_document_ids = set()
    for hit in hits:
        _validate_search_hit(
            hit,
            query_id=query_id,
            document_count=document_count,
            seen_document_ids=seen_document_ids,
        )
        neighbors.append(hit.document_id)
        distances.append(_score_to_squared_euclidean(hit.score))
    return neighbors, distances


def _validate_batch_latencies(
    runtime_result: _RuntimeSearchResult,
    *,
    query_count: int,
    batch_size: int,
) -> Tuple[np.ndarray, int]:
    latencies = np.asarray(runtime_result.batch_latencies_ms, dtype=np.float64)
    num_batches = (query_count + batch_size - 1) // batch_size
    if latencies.shape != (num_batches,):
        raise RuntimeError(
            "PyLucene returned an unexpected number of batch latencies: "
            f"{latencies.size} != {num_batches}"
        )
    if not np.isfinite(latencies).all() or np.any(latencies < 0.0):
        raise RuntimeError(
            "PyLucene returned invalid batch latency measurements"
        )
    return latencies, num_batches


def _process_search_result(
    runtime_result: _RuntimeSearchResult,
    provenance: _IndexProvenanceVerification,
    *,
    query_count: int,
    k: int,
    batch_size: int,
) -> _ProcessedSearchResult:
    """Validate the Java boundary result and convert it to benchmark arrays."""
    _validate_runtime_search_result(
        runtime_result, provenance, query_count=query_count
    )
    neighbors, distances = _convert_search_hits(
        runtime_result, query_count=query_count, k=k
    )
    latencies, num_batches = _validate_batch_latencies(
        runtime_result, query_count=query_count, batch_size=batch_size
    )

    search_time_ms = float(latencies.sum())
    return _ProcessedSearchResult(
        neighbors=neighbors,
        distances=distances,
        search_time_ms=search_time_ms,
        queries_per_second=(
            query_count / (search_time_ms / 1000.0)
            if search_time_ms > 0.0
            else 0.0
        ),
        latency_seconds=float(latencies.mean()) / 1000.0,
        latency_percentiles={
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
        },
        num_batches=num_batches,
    )


@dataclass(frozen=True)
class _CagraIndexVerification:
    segment_count: int
    field_count: int
    vector_count: int
    dimensions: int

    def to_metadata(self) -> Dict[str, Union[str, int]]:
        return {
            "status": "cagra-only",
            "segment_count": self.segment_count,
            "field_count": self.field_count,
            "vector_count": self.vector_count,
            "dimensions": self.dimensions,
        }


class _CagraVerificationError(RuntimeError):
    """Raised when persisted metadata cannot prove a CAGRA-only index."""


@dataclass(frozen=True)
class _CagraFieldMetadata:
    field_number: int
    dimensions: int
    vector_count: int
    cagra_offset: int
    cagra_length: int


@dataclass(frozen=True)
class _RawCagraFieldMetadata:
    encoding: int
    similarity: int
    dimensions: int
    vector_count: int
    cagra_offset: int
    cagra_length: int
    brute_force_offset: int
    brute_force_length: int


@dataclass(frozen=True)
class _CagraFieldSource:
    metadata: _CagraFieldMetadata
    metadata_file: str


@dataclass(frozen=True)
class _CagraDataFileContext:
    index_path: Path
    directory: Any
    segment_info: Any
    suffix: str
    metadata_file: str

    @property
    def data_file(self) -> str:
        stem = self.metadata_file[: -len(_CAGRA_META_EXTENSION)]
        return stem + _CAGRA_INDEX_EXTENSION

    @property
    def data_path(self) -> Path:
        return self.index_path / self.data_file


@dataclass(frozen=True)
class _CagraSegmentVerification:
    field_count: int
    vector_count: int
    dimensions: frozenset[int]


class _CagraIndexVerifier:
    """Own persisted CAGRA inspection and its verified-file cache."""

    def __init__(
        self,
        *,
        attach_current_thread: Callable[[], None],
        paths: Any,
        codec_util: Any,
        field_info: Any,
        segment_commit_info: Any,
        segment_infos: Any,
        vector_encoding: Any,
        vector_similarity_function: Any,
        fs_directory: Any,
        io_context: Any,
    ) -> None:
        self._attach_current_thread = attach_current_thread
        self.Paths = paths
        self.CodecUtil = codec_util
        self.FieldInfo = field_info
        self.SegmentCommitInfo = segment_commit_info
        self.SegmentInfos = segment_infos
        self.VectorEncoding = vector_encoding
        self.VectorSimilarityFunction = vector_similarity_function
        self.FSDirectory = fs_directory
        self.IOContext = io_context
        self._verified_data_files: set[_FileSignature] = set()

    @staticmethod
    def _segment_suffix(segment_name: str, metadata_file: str) -> str:
        stem = metadata_file[: -len(_CAGRA_META_EXTENSION)]
        if stem == segment_name:
            return ""

        prefix = f"{segment_name}_"
        if not stem.startswith(prefix) or len(stem) == len(prefix):
            raise _CagraVerificationError(
                "CAGRA-only verification found a metadata filename that "
                f"does not match segment {segment_name!r}: {metadata_file!r}"
            )
        return stem[len(prefix) :]

    @classmethod
    def _read_cagra_field(
        cls,
        metadata_input: Any,
        metadata_file: str,
        field_number: int,
    ) -> Optional[_CagraFieldMetadata]:
        raw_field = cls._decode_cagra_field(metadata_input)
        return cls._validate_cagra_field_metadata(
            raw_field, metadata_file, field_number
        )

    @staticmethod
    def _decode_cagra_field(metadata_input: Any) -> _RawCagraFieldMetadata:
        return _RawCagraFieldMetadata(
            encoding=int(metadata_input.readInt()),
            similarity=int(metadata_input.readInt()),
            dimensions=int(metadata_input.readInt()),
            vector_count=int(metadata_input.readInt()),
            cagra_offset=int(metadata_input.readVLong()),
            cagra_length=int(metadata_input.readVLong()),
            brute_force_offset=int(metadata_input.readVLong()),
            brute_force_length=int(metadata_input.readVLong()),
        )

    @classmethod
    def _validate_cagra_field_metadata(
        cls,
        field: _RawCagraFieldMetadata,
        metadata_file: str,
        field_number: int,
    ) -> Optional[_CagraFieldMetadata]:
        if field.encoding != _FLOAT32_ENCODING_ORDINAL:
            raise _CagraVerificationError(
                "CAGRA-only verification found unsupported vector "
                f"encoding ordinal {field.encoding} in {metadata_file!r}"
            )
        if field.similarity != _EUCLIDEAN_SIMILARITY_ORDINAL:
            raise _CagraVerificationError(
                "CAGRA-only verification found unsupported similarity "
                f"ordinal {field.similarity} in {metadata_file!r}"
            )
        if not 1 <= field.dimensions <= _MAX_DIMENSIONS:
            raise _CagraVerificationError(
                "CAGRA-only verification found invalid field metadata "
                f"in {metadata_file!r}"
            )
        if field.vector_count < 0:
            raise _CagraVerificationError(
                "CAGRA-only verification found invalid field metadata "
                f"in {metadata_file!r}"
            )
        if field.vector_count == 0:
            if cls._empty_cagra_field_has_data(field):
                raise _CagraVerificationError(
                    "CAGRA-only verification found index data for an "
                    f"empty field in {metadata_file!r}"
                )
            return None
        if field.brute_force_length != 0:
            raise _CagraVerificationError(
                "CAGRA-only verification found a persisted brute-force "
                f"index for field {field_number} in {metadata_file!r}"
            )
        if field.cagra_length <= 0:
            raise _CagraVerificationError(
                "CAGRA-only verification found no persisted CAGRA index "
                f"for field {field_number} in {metadata_file!r}"
            )
        return _CagraFieldMetadata(
            field_number=field_number,
            dimensions=field.dimensions,
            vector_count=field.vector_count,
            cagra_offset=field.cagra_offset,
            cagra_length=field.cagra_length,
        )

    @staticmethod
    def _empty_cagra_field_has_data(field: _RawCagraFieldMetadata) -> bool:
        return any(
            (
                field.cagra_offset,
                field.cagra_length,
                field.brute_force_offset,
                field.brute_force_length,
            )
        )

    @classmethod
    def _read_cagra_fields(
        cls, metadata_input: Any, metadata_file: str
    ) -> List[_CagraFieldMetadata]:
        field_numbers = set()
        fields = []
        while True:
            field_number = int(metadata_input.readInt())
            if field_number == -1:
                return fields
            if field_number < 0 or field_number in field_numbers:
                raise _CagraVerificationError(
                    "CAGRA-only verification found an invalid field number "
                    f"{field_number} in {metadata_file!r}"
                )

            field_numbers.add(field_number)
            field = cls._read_cagra_field(
                metadata_input, metadata_file, field_number
            )
            if field is not None:
                fields.append(field)

    def _read_segment_field_infos(
        self, directory: Any, segment_info: Any
    ) -> Any:
        if segment_info.hasFieldUpdates():
            raise _CagraVerificationError(
                "CAGRA-only verification does not accept segments with "
                "field-info updates"
            )
        try:
            return (
                segment_info.info.getCodec()
                .fieldInfosFormat()
                .read(
                    directory,
                    segment_info.info,
                    "",
                    self.IOContext.READONCE,
                )
            )
        except Exception as exc:
            raise _CagraVerificationError(
                "CAGRA-only verification cannot read Lucene field "
                f"metadata for segment {segment_info.info.name!r}: {exc}"
            ) from exc

    @classmethod
    def _verify_cagra_payload_coverage(
        cls,
        fields: List[_CagraFieldMetadata],
        payload_start: int,
        payload_end: int,
        data_file: str,
    ) -> None:
        intervals = []
        for field in fields:
            intervals.append(
                (
                    field.cagra_offset,
                    field.cagra_offset + field.cagra_length,
                )
            )
        intervals.sort()

        if not intervals:
            if payload_start != payload_end:
                raise _CagraVerificationError(
                    "CAGRA data file contains unreferenced payload: "
                    f"{data_file!r}"
                )
            return

        coverage_error = (
            "CAGRA metadata ranges do not exactly cover data "
            f"file {data_file!r}"
        )
        expected_offset = payload_start
        for interval_start, interval_end in intervals:
            if interval_start != expected_offset:
                raise _CagraVerificationError(coverage_error)
            expected_offset = interval_end
        if expected_offset != payload_end:
            raise _CagraVerificationError(coverage_error)

    def _verify_checksum_with_signature_cache(
        self,
        data_path: Path,
        data_input: Any,
        signature_before: Optional[_FileSignature],
        data_file: str,
    ) -> None:
        cache_hit = (
            signature_before is not None
            and signature_before in self._verified_data_files
        )
        if cache_hit:
            self.CodecUtil.retrieveChecksum(data_input)
        else:
            self.CodecUtil.checksumEntireFile(data_input)

        if signature_before is None:
            return
        signature_after = _file_signature(data_path)
        if signature_after != signature_before:
            raise _CagraVerificationError(
                f"CAGRA data file changed during verification: {data_file!r}"
            )
        file_age = time.time_ns() - signature_before.changed_at_ns
        if cache_hit or file_age < _CAGRA_CACHE_MIN_FILE_AGE_NS:
            return

        self._verified_data_files = {
            signature
            for signature in self._verified_data_files
            if signature.resolved_path != signature_before.resolved_path
        }
        self._verified_data_files.add(signature_before)

    def _verify_cagra_data_file(
        self,
        context: _CagraDataFileContext,
        fields: List[_CagraFieldMetadata],
    ) -> None:
        try:
            signature_before = _file_signature(context.data_path)
        except OSError:
            signature_before = None

        try:
            with _CleanupStack() as cleanups:
                data_input = context.directory.openInput(
                    context.data_file, self.IOContext.READONCE
                )
                cleanups.add("close CAGRA data input", data_input.close)
                self.CodecUtil.checkIndexHeader(
                    data_input,
                    _CAGRA_INDEX_CODEC_NAME,
                    _CAGRA_INDEX_VERSION,
                    _CAGRA_INDEX_VERSION,
                    context.segment_info.info.getId(),
                    context.suffix,
                )
                payload_start = int(data_input.getFilePointer())
                payload_end = int(data_input.length()) - int(
                    self.CodecUtil.footerLength()
                )
                if payload_end < payload_start:
                    raise _CagraVerificationError(
                        f"CAGRA data file is truncated: {context.data_file!r}"
                    )
                self._verify_cagra_payload_coverage(
                    fields,
                    payload_start,
                    payload_end,
                    context.data_file,
                )
                self._verify_checksum_with_signature_cache(
                    context.data_path,
                    data_input,
                    signature_before,
                    context.data_file,
                )
        except _CagraVerificationError:
            raise
        except Exception as exc:
            raise _CagraVerificationError(
                "CAGRA-only verification cannot read "
                f"{context.data_file!r}: {exc}"
            ) from exc

    def _verify_field_against_lucene_metadata(
        self,
        field_infos: Any,
        field: _CagraFieldMetadata,
        metadata_file: str,
    ) -> None:
        field_info = field_infos.fieldInfo(field.field_number)
        if field_info is None:
            raise _CagraVerificationError(
                "CAGRA-only verification found an unknown field number "
                f"{field.field_number} in {metadata_file!r}"
            )
        field_name = str(field_info.getName())
        if field_name != _VECTOR_FIELD:
            raise _CagraVerificationError(
                "CAGRA-only verification found data for unexpected field "
                f"{field_name!r} in {metadata_file!r}"
            )
        if int(field_info.getVectorDimension()) != field.dimensions:
            raise _CagraVerificationError(
                "CAGRA-only verification found dimensions inconsistent "
                f"with Lucene field metadata in {metadata_file!r}"
            )
        if field_info.getVectorEncoding() != self.VectorEncoding.FLOAT32:
            raise _CagraVerificationError(
                "CAGRA-only verification found non-FLOAT32 Lucene field "
                f"metadata in {metadata_file!r}"
            )
        if (
            field_info.getVectorSimilarityFunction()
            != self.VectorSimilarityFunction.EUCLIDEAN
        ):
            raise _CagraVerificationError(
                "CAGRA-only verification found non-Euclidean Lucene field "
                f"metadata in {metadata_file!r}"
            )

    @staticmethod
    def _validate_segment_deletions(
        segment_info: Any, segment_name: str
    ) -> None:
        deletion_count = int(segment_info.getDelCount())
        soft_deletion_count = int(segment_info.getSoftDelCount())
        if (
            segment_info.hasDeletions()
            or deletion_count
            or soft_deletion_count
        ):
            raise _CagraVerificationError(
                "CAGRA-only verification does not accept committed "
                f"deletions in segment {segment_name!r}: "
                f"deleted={deletion_count}, "
                f"soft_deleted={soft_deletion_count}"
            )

    @staticmethod
    def _cagra_metadata_files(
        segment_info: Any, segment_name: str
    ) -> List[str]:
        metadata_files = []
        for file_name in segment_info.files():
            file_name = str(file_name)
            if file_name.endswith(_CAGRA_META_EXTENSION):
                metadata_files.append(file_name)
        metadata_files.sort()
        if not metadata_files:
            raise _CagraVerificationError(
                "CAGRA-only verification found no "
                f"{_CAGRA_META_EXTENSION} metadata for segment "
                f"{segment_name!r}"
            )
        return metadata_files

    def _read_and_verify_cagra_metadata_file(
        self,
        context: _CagraDataFileContext,
    ) -> List[_CagraFieldMetadata]:
        metadata_input = context.directory.openChecksumInput(
            context.metadata_file
        )
        try:
            with _CleanupStack() as cleanups:
                cleanups.add(
                    "close CAGRA metadata input", metadata_input.close
                )
                self.CodecUtil.checkIndexHeader(
                    metadata_input,
                    _CAGRA_META_CODEC_NAME,
                    _CAGRA_META_VERSION,
                    _CAGRA_META_VERSION,
                    context.segment_info.info.getId(),
                    context.suffix,
                )
                fields = self._read_cagra_fields(
                    metadata_input, context.metadata_file
                )
                self.CodecUtil.checkFooter(metadata_input)
                return fields
        except _CagraVerificationError:
            raise
        except Exception as exc:
            raise _CagraVerificationError(
                "CAGRA-only verification cannot read "
                f"{context.metadata_file!r} as cuVS-Lucene metadata "
                f"format v{_CAGRA_META_VERSION}: {exc}"
            ) from exc

    def _lucene_vector_field_numbers(self, field_infos: Any) -> set[int]:
        field_numbers = set()
        for raw_field_info in field_infos:
            field_info = self.FieldInfo.cast_(raw_field_info)
            if int(field_info.getVectorDimension()) > 0:
                field_numbers.add(int(field_info.number))
        return field_numbers

    def _verify_cagra_segment(
        self,
        index_path: Path,
        directory: Any,
        segment_info: Any,
    ) -> _CagraSegmentVerification:
        segment_name = str(segment_info.info.name)
        self._validate_segment_deletions(segment_info, segment_name)
        metadata_files = self._cagra_metadata_files(segment_info, segment_name)
        field_infos = self._read_segment_field_infos(directory, segment_info)

        # Verify every metadata/data pair before interpreting fields across
        # the segment. This preserves fail-closed file-validation ordering.
        verified_field_sources = []
        for metadata_file in metadata_files:
            suffix = self._segment_suffix(segment_name, metadata_file)
            context = _CagraDataFileContext(
                index_path=index_path,
                directory=directory,
                segment_info=segment_info,
                suffix=suffix,
                metadata_file=metadata_file,
            )
            fields = self._read_and_verify_cagra_metadata_file(context)
            self._verify_cagra_data_file(context, fields)
            for field in fields:
                verified_field_sources.append(
                    _CagraFieldSource(
                        metadata=field,
                        metadata_file=metadata_file,
                    )
                )

        # Compare the verified fields with Lucene's segment-wide view and
        # aggregate the values needed for index-wide validation.
        field_numbers = set()
        vector_count = 0
        dimensions = set()
        for field_source in verified_field_sources:
            field = field_source.metadata
            if field.field_number in field_numbers:
                raise _CagraVerificationError(
                    "CAGRA-only verification found duplicate "
                    f"field {field.field_number} across metadata "
                    f"files for segment {segment_name!r}"
                )
            field_numbers.add(field.field_number)
            self._verify_field_against_lucene_metadata(
                field_infos, field, field_source.metadata_file
            )
            vector_count += field.vector_count
            dimensions.add(field.dimensions)

        lucene_field_numbers = self._lucene_vector_field_numbers(field_infos)
        if field_numbers != lucene_field_numbers:
            raise _CagraVerificationError(
                "CAGRA-only verification found vector fields without "
                "matching CAGRA metadata in segment "
                f"{segment_name!r}: metadata={sorted(field_numbers)}, "
                f"Lucene={sorted(lucene_field_numbers)}"
            )

        max_documents = int(segment_info.info.maxDoc())
        if vector_count != max_documents:
            raise _CagraVerificationError(
                "CAGRA-only verification found "
                f"{vector_count} vectors for "
                f"{max_documents} documents in segment "
                f"{segment_name!r}"
            )
        return _CagraSegmentVerification(
            field_count=len(field_numbers),
            vector_count=vector_count,
            dimensions=frozenset(dimensions),
        )

    @staticmethod
    def _summarize_cagra_segments(
        segments: List[_CagraSegmentVerification],
        expected_vector_count: Optional[int],
        expected_dimensions: Optional[int],
    ) -> _CagraIndexVerification:
        field_count = 0
        vector_count = 0
        dimensions = set()
        for segment in segments:
            field_count += segment.field_count
            vector_count += segment.vector_count
            dimensions.update(segment.dimensions)
        if not segments or field_count == 0:
            raise _CagraVerificationError(
                "CAGRA-only verification found no nonempty vector fields"
            )
        if len(dimensions) != 1:
            raise _CagraVerificationError(
                "CAGRA-only verification found inconsistent vector "
                f"dimensions: {sorted(dimensions)}"
            )

        index_dimensions = next(iter(dimensions))
        if (
            expected_vector_count is not None
            and vector_count != expected_vector_count
        ):
            raise _CagraVerificationError(
                "CAGRA-only verification found "
                f"{vector_count} vectors; expected {expected_vector_count}"
            )
        if (
            expected_dimensions is not None
            and index_dimensions != expected_dimensions
        ):
            raise _CagraVerificationError(
                "CAGRA-only verification found "
                f"{index_dimensions} dimensions; expected "
                f"{expected_dimensions}"
            )
        return _CagraIndexVerification(
            segment_count=len(segments),
            field_count=field_count,
            vector_count=vector_count,
            dimensions=index_dimensions,
        )

    def verify_index(
        self,
        index_path: Path,
        *,
        expected_vector_count: Optional[int] = None,
        expected_dimensions: Optional[int] = None,
    ) -> _CagraIndexVerification:
        """Verify every committed vector field is persisted as CAGRA only."""
        self._attach_current_thread()
        directory = self.FSDirectory.open(self.Paths.get(str(index_path)))
        verified_segments = []
        with _CleanupStack() as cleanups:
            cleanups.add("close Lucene directory", directory.close)
            segment_infos = self.SegmentInfos.readLatestCommit(directory)
            for raw_segment_info in segment_infos:
                segment_info = self.SegmentCommitInfo.cast_(raw_segment_info)
                verified_segments.append(
                    self._verify_cagra_segment(
                        index_path, directory, segment_info
                    )
                )
        return self._summarize_cagra_segments(
            verified_segments,
            expected_vector_count,
            expected_dimensions,
        )


class _PyLuceneRuntime:
    """Own generated PyLucene/Lucene bindings and index operations."""

    def __init__(self, lucene: Any):
        from java.lang import Class
        from java.nio.file import Paths
        from org.apache.lucene.codecs import Codec, CodecUtil
        from org.apache.lucene.document import (
            Document,
            KnnFloatVectorField,
            StoredField,
        )
        from org.apache.lucene.index import (
            DirectoryReader,
            FieldInfo,
            IndexWriter,
            IndexWriterConfig,
            SegmentCommitInfo,
            SegmentInfos,
            SerialMergeScheduler,
            VectorEncoding,
            VectorSimilarityFunction,
        )
        from org.apache.lucene.search import (
            IndexSearcher,
            KnnFloatVectorQuery,
        )
        from org.apache.lucene.store import FSDirectory, IOContext

        self.lucene = lucene
        self.Class = Class
        self.Paths = Paths
        self.Codec = Codec
        self.Document = Document
        self.KnnFloatVectorField = KnnFloatVectorField
        self.StoredField = StoredField
        self.DirectoryReader = DirectoryReader
        self.IndexWriter = IndexWriter
        self.IndexWriterConfig = IndexWriterConfig
        self.SerialMergeScheduler = SerialMergeScheduler
        self.VectorSimilarityFunction = VectorSimilarityFunction
        self.IndexSearcher = IndexSearcher
        self.KnnFloatVectorQuery = KnnFloatVectorQuery
        self.FSDirectory = FSDirectory
        self.IOContext = IOContext
        self._codec_cache: Dict[str, Any] = {}

        # Resolve the base cuvs-java provider before codec construction so
        # classpath failures have a direct error location.
        self.Class.forName("com.nvidia.cuvs.spi.JDKProvider")
        self._cagra_verifier = _CagraIndexVerifier(
            attach_current_thread=self.attach_current_thread,
            paths=Paths,
            codec_util=CodecUtil,
            field_info=FieldInfo,
            segment_commit_info=SegmentCommitInfo,
            segment_infos=SegmentInfos,
            vector_encoding=VectorEncoding,
            vector_similarity_function=VectorSimilarityFunction,
            fs_directory=FSDirectory,
            io_context=IOContext,
        )

    @classmethod
    def create(cls, config: Dict[str, Any]) -> "_PyLuceneRuntime":
        return cls(_initialize_pylucene(config))

    @property
    def pylucene_version(self) -> str:
        return str(getattr(self.lucene, "VERSION", "unknown"))

    def attach_current_thread(self) -> None:
        vm_environment = self.lucene.getVMEnv()
        if vm_environment is None:
            raise RuntimeError("PyLucene JVM is not initialized")
        vm_environment.attachCurrentThread()

    def resolve_codec(self, codec_name: str) -> Any:
        self.attach_current_thread()
        cached = self._codec_cache.get(codec_name)
        if cached is not None:
            return cached

        available_codecs = self.Codec.availableCodecs()
        if not available_codecs.contains(codec_name):
            available = ", ".join(str(name) for name in available_codecs)
            raise RuntimeError(
                f"{codec_name} was not advertised by Lucene SPI. "
                f"Available codecs: {available}"
            )
        codec = self.Codec.forName(codec_name)
        if str(codec.getName()) != codec_name:
            raise RuntimeError(
                f"Requested codec {codec_name}, got {codec.getName()}"
            )
        self._codec_cache[codec_name] = codec
        return codec

    def codec_telemetry(self, codec_name: str, codec: Any) -> Dict[str, str]:
        """Inspect the selected writer path before indexing starts."""
        vectors_format = codec.knnVectorsFormat()
        if vectors_format is None:
            raise RuntimeError(
                f"{codec_name} did not initialize a Lucene vector format"
            )
        return _parse_writer_telemetry(str(vectors_format))

    def _java_float_array(self, vector: np.ndarray) -> Any:
        return self.lucene.JArray("float")(
            tuple(float(value) for value in vector)
        )

    def verify_cagra_index(
        self,
        index_path: Path,
        expected_vector_count: Optional[int] = None,
        expected_dimensions: Optional[int] = None,
    ) -> _CagraIndexVerification:
        return self._cagra_verifier.verify_index(
            index_path,
            expected_vector_count=expected_vector_count,
            expected_dimensions=expected_dimensions,
        )

    def build_index(
        self,
        index_path: Path,
        vectors: np.ndarray,
        resolved_codec: _ResolvedCodec,
    ) -> Dict[str, str]:
        self.attach_current_thread()
        directory = self.FSDirectory.open(self.Paths.get(str(index_path)))
        with _CleanupStack() as cleanups:
            cleanups.add("close Lucene directory", directory.close)
            writer_config = self.IndexWriterConfig()
            writer_config.setOpenMode(self.IndexWriterConfig.OpenMode.CREATE)
            writer_config.setCodec(resolved_codec.java_codec)
            writer_config.setUseCompoundFile(False)
            writer_config.setMergeScheduler(self.SerialMergeScheduler())
            writer = self.IndexWriter(directory, writer_config)
            self._write_and_close_index(writer, vectors)
            return dict(resolved_codec.telemetry)

    def _write_and_close_index(self, writer: Any, vectors: np.ndarray) -> None:
        try:
            for document_id, vector in enumerate(vectors):
                writer.addDocument(self._vector_document(document_id, vector))
            writer.commit()
        except BaseException as write_error:
            self._rollback_failed_write(writer, write_error)
            raise
        writer.close()

    @staticmethod
    def _rollback_failed_write(
        writer: Any, write_error: BaseException
    ) -> None:
        try:
            writer.rollback()
        except BaseException as rollback_error:
            if isinstance(write_error, Exception):
                raise rollback_error from write_error
            write_error.add_note(
                "IndexWriter rollback also failed: "
                f"{type(rollback_error).__name__}: {rollback_error}"
            )

    def _vector_document(self, document_id: int, vector: np.ndarray) -> Any:
        document = self.Document()
        document.add(self.StoredField(_ID_FIELD, str(document_id)))
        document.add(
            self.KnnFloatVectorField(
                _VECTOR_FIELD,
                self._java_float_array(vector),
                self.VectorSimilarityFunction.EUCLIDEAN,
            )
        )
        return document

    @staticmethod
    def _index_vector_dimensions(reader: Any) -> int:
        dimensions = set()
        for leaf_context in reader.leaves():
            values = leaf_context.reader().getFloatVectorValues(_VECTOR_FIELD)
            if values is not None and values.size() > 0:
                dimensions.add(int(values.dimension()))
        if not dimensions:
            raise RuntimeError(
                f"Lucene index contains no {_VECTOR_FIELD!r} vectors"
            )
        if len(dimensions) != 1:
            raise RuntimeError(
                "Lucene index has inconsistent vector dimensions: "
                f"{dimensions}"
            )
        return dimensions.pop()

    def _search_vector(
        self,
        searcher: Any,
        stored_fields: Any,
        vector: np.ndarray,
        k: int,
    ) -> List[_SearchHit]:
        query = self.KnnFloatVectorQuery(
            _VECTOR_FIELD,
            self._java_float_array(vector),
            k,
        )
        score_docs = searcher.search(query, k).scoreDocs
        hits = []
        for score_doc in score_docs:
            stored_id = stored_fields.document(score_doc.doc).get(_ID_FIELD)
            if stored_id is None:
                raise RuntimeError(
                    f"Lucene document {score_doc.doc} has no stored ID"
                )
            hits.append(
                _SearchHit(
                    document_id=int(stored_id),
                    score=float(score_doc.score),
                )
            )
        return hits

    def search_index(
        self,
        index_path: Path,
        query_vectors: np.ndarray,
        k: int,
        batch_size: int,
    ) -> _RuntimeSearchResult:
        self.attach_current_thread()
        directory = self.FSDirectory.open(self.Paths.get(str(index_path)))
        with _CleanupStack() as cleanups:
            cleanups.add("close Lucene directory", directory.close)
            reader = self.DirectoryReader.open(directory)
            cleanups.add("close Lucene index reader", reader.close)
            index_dimensions = self._index_vector_dimensions(reader)
            if query_vectors.shape[1] != index_dimensions:
                raise ValueError(
                    "Query vector dimensions do not match the Lucene index: "
                    f"{query_vectors.shape[1]} != {index_dimensions}"
                )

            document_count = int(reader.numDocs())
            if document_count < 1:
                raise RuntimeError("Lucene index contains no documents")
            lucene_k = min(k, document_count)
            searcher = self.IndexSearcher(reader)
            stored_fields = searcher.storedFields()

            all_hits: List[List[_SearchHit]] = []
            batch_latencies_ms: List[float] = []
            for batch_start in range(0, query_vectors.shape[0], batch_size):
                start = time.perf_counter()
                batch_end = min(
                    batch_start + batch_size, query_vectors.shape[0]
                )
                all_hits.extend(
                    self._search_vectors(
                        searcher,
                        stored_fields,
                        query_vectors[batch_start:batch_end],
                        lucene_k,
                    )
                )
                batch_latencies_ms.append(
                    (time.perf_counter() - start) * 1000.0
                )

            return _RuntimeSearchResult(
                hits=all_hits,
                batch_latencies_ms=batch_latencies_ms,
                index_dimensions=index_dimensions,
                document_count=document_count,
            )

    def _search_vectors(
        self,
        searcher: Any,
        stored_fields: Any,
        query_vectors: np.ndarray,
        k: int,
    ) -> List[List[_SearchHit]]:
        hits = []
        for vector in query_vectors:
            hits.append(
                self._search_vector(searcher, stored_fields, vector, k)
            )
        return hits


class _SelectedAlgorithmGroup(NamedTuple):
    algorithm: str
    group: str
    configuration: dict
    metadata: dict


@dataclass(frozen=True)
class _GlobalSelectionScope:
    algorithms: frozenset[str]
    groups: frozenset[str]


@dataclass(frozen=True)
class _AlgorithmSelection:
    algorithms: Optional[frozenset[str]]
    groups: Optional[frozenset[str]]
    explicit_groups: frozenset[Tuple[str, str]]

    def _resolve_global_scope(
        self, algorithm_configs: Dict[str, Dict[str, Any]]
    ) -> Optional[_GlobalSelectionScope]:
        available_algorithms = frozenset(algorithm_configs)
        if self.algorithms is not None:
            unknown_algorithms = self.algorithms - available_algorithms
            if unknown_algorithms:
                names = ", ".join(sorted(unknown_algorithms))
                raise ValueError(
                    f"Unknown PyLucene algorithm selector(s): {names}"
                )

        candidate_algorithms = (
            self.algorithms
            if self.algorithms is not None
            else available_algorithms
        )
        if self.groups is not None:
            available_groups = set()
            for algorithm in candidate_algorithms:
                available_groups.update(algorithm_configs[algorithm])
            unknown_groups = self.groups - available_groups
            if unknown_groups:
                names = ", ".join(sorted(unknown_groups))
                raise ValueError(
                    f"Unknown PyLucene group selector(s): {names}"
                )

        has_global_selector = (
            self.algorithms is not None or self.groups is not None
        )
        if not has_global_selector and self.explicit_groups:
            return None
        return _GlobalSelectionScope(
            algorithms=candidate_algorithms,
            groups=(
                self.groups if self.groups is not None else frozenset({"base"})
            ),
        )

    def _validate_explicit_groups(
        self, algorithm_configs: Dict[str, Dict[str, Any]]
    ) -> None:
        for algorithm, group in sorted(self.explicit_groups):
            if algorithm not in algorithm_configs:
                raise ValueError(
                    f"Unknown PyLucene algorithm in --algo-groups: {algorithm}"
                )
            if group not in algorithm_configs[algorithm]:
                raise ValueError(
                    f"Unknown PyLucene group for {algorithm}: {group}"
                )

    def _selected_pairs(
        self,
        algorithm_configs: Dict[str, Dict[str, Any]],
        global_scope: Optional[_GlobalSelectionScope],
    ) -> set[Tuple[str, str]]:
        selected_pairs = set(self.explicit_groups)
        if global_scope is None:
            return selected_pairs

        for algorithm, groups in algorithm_configs.items():
            if algorithm not in global_scope.algorithms:
                continue
            for group in groups:
                if group in global_scope.groups:
                    selected_pairs.add((algorithm, group))
        return selected_pairs

    def resolve(
        self, algorithm_configs: Dict[str, Dict[str, Any]]
    ) -> List[_SelectedAlgorithmGroup]:
        global_scope = self._resolve_global_scope(algorithm_configs)
        self._validate_explicit_groups(algorithm_configs)
        selected_pairs = self._selected_pairs(algorithm_configs, global_scope)

        selected_groups = []
        for algorithm, groups in algorithm_configs.items():
            for group, group_config in groups.items():
                if (algorithm, group) not in selected_pairs:
                    continue
                selected_groups.append(
                    _SelectedAlgorithmGroup(
                        algorithm=algorithm,
                        group=group,
                        configuration=group_config,
                        metadata={},
                    )
                )
        return selected_groups


@dataclass(frozen=True)
class _BenchmarkConfigContext:
    dataset: str
    dataset_path: str
    subset_scope: Optional[str]
    runtime_config: Dict[str, Any]
    neighbor_count: int
    batch_size: int


class PyLuceneConfigLoader(ConfigLoader):
    """Load PyLucene algorithm configurations."""

    def __init__(self, config_path: Optional[Union[str, os.PathLike]] = None):
        self.config_path = (
            os.fspath(config_path)
            if config_path is not None
            else os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "../config"
            )
        )

    @property
    def backend_type(self) -> str:
        return "pylucene"

    @staticmethod
    def _parse_name_filter(
        value: Optional[str],
    ) -> Optional[frozenset[str]]:
        if not value:
            return None

        names = set()
        for raw_name in value.split(","):
            name = raw_name.strip()
            if name:
                names.add(name)
        return frozenset(names) or None

    @staticmethod
    def _parse_algorithm_group_filter(
        value: Optional[str],
    ) -> frozenset[Tuple[str, str]]:
        selections = set()
        if not value:
            return frozenset()

        for selection in value.split(","):
            algorithm, separator, group = selection.strip().partition(".")
            if not separator or not algorithm or not group:
                raise ValueError(
                    "algo_groups entries must use <algorithm>.<group>, "
                    f"got {selection!r}"
                )
            selections.add((algorithm, group))
        return frozenset(selections)

    @classmethod
    def _selection_from_options(
        cls, options: Dict[str, Any]
    ) -> _AlgorithmSelection:
        return _AlgorithmSelection(
            algorithms=cls._parse_name_filter(options.get("algorithms")),
            groups=cls._parse_name_filter(options.get("groups")),
            explicit_groups=cls._parse_algorithm_group_filter(
                options.get("algo_groups")
            ),
        )

    def _match_backend_algorithm_config(
        self, config: Any
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        if not isinstance(config, dict):
            return None

        algorithm = config.get("name")
        if not isinstance(algorithm, str) or not algorithm:
            return None

        declared_backend = config.get("backend")
        if declared_backend is None:
            belongs_to_backend = algorithm.startswith("pylucene_")
        else:
            belongs_to_backend = declared_backend == self.backend_type
        if not belongs_to_backend:
            return None
        return algorithm, config.get("groups", {})

    def _load_algorithm_configs(
        self, algorithm_files: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        algorithm_configs = {}
        for algorithm_file in algorithm_files:
            matched_config = self._match_backend_algorithm_config(
                self.load_yaml_file(algorithm_file)
            )
            if matched_config is None:
                continue
            algorithm, groups = matched_config
            # Later files are explicit overrides and determine output order.
            algorithm_configs.pop(algorithm, None)
            algorithm_configs[algorithm] = groups
        return algorithm_configs

    def _discover_algo_groups(
        self,
        dataset_conf: dict,
        dataset: str,
        dataset_path: str,
        **kwargs,
    ) -> List[Tuple[str, str, dict, dict]]:
        algorithm_files = self.gather_algorithm_configs(
            self.config_path, kwargs.get("algorithm_configuration")
        )
        algorithm_configs = self._load_algorithm_configs(algorithm_files)
        selection = self._selection_from_options(kwargs)
        return selection.resolve(algorithm_configs)

    @staticmethod
    def _runtime_config(options: Dict[str, Any]) -> Dict[str, Any]:
        runtime_keys = (
            "cuvs_java_jar",
            "cuvs_lucene_jar",
            "java_library_path",
            "jvm_args",
        )
        runtime_config = {}
        for key in runtime_keys:
            value = options.get(key)
            if value is not None:
                runtime_config[key] = value
        return runtime_config

    @staticmethod
    def _subset_scope(subset_size: Any) -> Optional[str]:
        subset_size = _validate_subset_size(subset_size)
        if subset_size is None:
            return None
        return f"subset{subset_size}"

    @staticmethod
    def _effective_parameters(
        build_combinations: List[Dict[str, Any]],
        search_combinations: List[Dict[str, Any]],
        *,
        tune_mode: bool,
        tune_build_params: Optional[Dict[str, Any]],
        tune_search_params: Optional[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        use_tuned_parameters = tune_mode and tune_build_params is not None
        if not use_tuned_parameters:
            return build_combinations, search_combinations
        search_params = [tune_search_params] if tune_search_params else [{}]
        return [tune_build_params], search_params

    @staticmethod
    def _index_label(
        algorithm: str,
        group: str,
        subset_scope: Optional[str],
        build_params: Dict[str, Any],
    ) -> str:
        prefix = algorithm if group == "base" else f"{algorithm}_{group}"
        label_parts = [prefix]
        if subset_scope is not None:
            label_parts.append(subset_scope)
        for key, value in build_params.items():
            label_parts.append(f"{key}{value}")
        return ".".join(label_parts)

    @classmethod
    def _benchmark_config(
        cls,
        context: _BenchmarkConfigContext,
        algorithm: str,
        group: str,
        build_params: Dict[str, Any],
        search_params: List[Dict[str, Any]],
    ) -> BenchmarkConfig:
        index_label = cls._index_label(
            algorithm, group, context.subset_scope, build_params
        )
        index_path = os.path.join(
            context.dataset_path,
            context.dataset,
            "index",
            index_label,
        )
        result_stem = f"{algorithm},{group}"
        if context.subset_scope is not None:
            result_stem = f"{result_stem},{context.subset_scope}"

        index_config = IndexConfig(
            name=index_label,
            algo=algorithm,
            build_param=build_params,
            search_params=search_params,
            file=index_path,
        )
        backend_config = {
            "name": index_label,
            "algo": algorithm,
            "codec": build_params.get("codec"),
            "requires_gpu": True,
            "output_filename": (
                result_stem,
                f"{result_stem},k{context.neighbor_count},"
                f"bs{context.batch_size}",
            ),
            **context.runtime_config,
        }
        return BenchmarkConfig(
            indexes=[index_config],
            backend_config=backend_config,
        )

    @classmethod
    def _group_benchmark_configs(
        cls,
        context: _BenchmarkConfigContext,
        algorithm: str,
        group: str,
        build_params: List[Dict[str, Any]],
        search_params: List[Dict[str, Any]],
    ) -> List[BenchmarkConfig]:
        _validate_search_params(search_params)
        benchmark_configs = []
        for params in build_params:
            _validate_build_params(params)
            benchmark_configs.append(
                cls._benchmark_config(
                    context,
                    algorithm,
                    group,
                    params,
                    search_params,
                )
            )
        return benchmark_configs

    def _build_benchmark_configs(
        self,
        dataset_config: DatasetConfig,
        dataset_conf: dict,
        dataset: str,
        dataset_path: str,
        expanded_groups: List[Tuple[str, str, dict, List, List, dict]],
        **kwargs,
    ) -> List[BenchmarkConfig]:
        context = _BenchmarkConfigContext(
            dataset=dataset,
            dataset_path=dataset_path,
            subset_scope=self._subset_scope(dataset_config.subset_size),
            runtime_config=self._runtime_config(kwargs),
            neighbor_count=kwargs.get("count", 10),
            batch_size=kwargs.get("batch_size", 10000),
        )
        tune_mode = kwargs.get("_tune_mode", False)
        tune_build_params = kwargs.get("_tune_build_params")
        tune_search_params = kwargs.get("_tune_search_params")

        benchmark_configs = []
        for (
            algorithm,
            group,
            _group_config,
            build_combinations,
            search_combinations,
            _group_metadata,
        ) in expanded_groups:
            build_params, search_params = self._effective_parameters(
                build_combinations,
                search_combinations,
                tune_mode=tune_mode,
                tune_build_params=tune_build_params,
                tune_search_params=tune_search_params,
            )
            benchmark_configs.extend(
                self._group_benchmark_configs(
                    context,
                    algorithm,
                    group,
                    build_params,
                    search_params,
                )
            )

        return benchmark_configs


class PyLuceneBackend(BenchmarkBackend):
    """Build and search cuVS-Lucene indexes through PyLucene."""

    orchestrator_persists_results = True

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._runtime: Optional[_PyLuceneRuntime] = None

    @property
    def algo(self) -> str:
        return self.config.get("algo", "pylucene")

    def cleanup(self) -> None:
        """Release Python references; the process-wide JVM remains active."""
        self._runtime = None

    def _get_runtime(self) -> _PyLuceneRuntime:
        if self._runtime is None:
            self._runtime = _PyLuceneRuntime.create(self.config)
        return self._runtime

    def _failed_build_result(
        self,
        error_message: str,
        index_path: str = "",
        build_params: Optional[Dict[str, Any]] = None,
    ) -> BuildResult:
        return BuildResult(
            index_path=index_path,
            build_time_seconds=0.0,
            index_size_bytes=0,
            algorithm=self.algo,
            build_params=build_params or {},
            success=False,
            error_message=error_message,
        )

    def _failed_search_result(
        self,
        k: int,
        error_message: str,
        search_params: Optional[List[Dict[str, Any]]] = None,
    ) -> SearchResult:
        result_k = max(0, k)
        return SearchResult(
            neighbors=np.empty((0, result_k), dtype=np.int64),
            distances=np.empty((0, result_k), dtype=np.float32),
            search_time_ms=0.0,
            queries_per_second=0.0,
            recall=0.0,
            algorithm=self.algo,
            search_params=search_params or [],
            success=False,
            error_message=error_message,
        )

    def _verify_existing_index(
        self,
        index_path: Path,
        codec_name: str,
        *,
        expected_vector_count: Optional[int] = None,
        expected_dimensions: Optional[int] = None,
    ) -> Tuple[_IndexProvenanceVerification, Dict[str, Any]]:
        """Verify backend ownership and codec-specific persisted data."""
        if codec_name == _CAGRA_CODEC:
            provenance = _verify_cagra_provenance(
                index_path,
                expected_vector_count=expected_vector_count,
                expected_dimensions=expected_dimensions,
            )
            cagra_verification = self._get_runtime().verify_cagra_index(
                index_path,
                expected_vector_count=provenance.vector_count,
                expected_dimensions=provenance.dimensions,
            )
            metadata = {
                "cagra_provenance": provenance.to_metadata(),
                "cagra_verification": cagra_verification.to_metadata(),
            }
        else:
            provenance = _verify_hnsw_provenance(
                index_path,
                codec_name,
                expected_vector_count=expected_vector_count,
                expected_dimensions=expected_dimensions,
            )
            metadata = {"hnsw_verification": provenance.to_metadata()}
        return provenance, metadata

    def _dry_run_build_result(
        self,
        index_path: Path,
        codec_name: str,
        build_params: Dict[str, Any],
    ) -> BuildResult:
        print(
            f"[dry_run] Would build PyLucene index '{index_path}' "
            f"with codec={codec_name}"
        )
        return BuildResult(
            index_path=str(index_path),
            build_time_seconds=0.0,
            index_size_bytes=0,
            algorithm=self.algo,
            build_params=build_params,
            metadata={"codec": codec_name},
            success=True,
        )

    def _decide_existing_index(
        self,
        dataset: Dataset,
        index_path: Path,
        codec_name: str,
        build_params: Dict[str, Any],
        force: bool,
    ) -> _ExistingIndexDecision:
        if not index_path.exists():
            return _ExistingIndexDecision.build()
        if not index_path.is_dir():
            return _ExistingIndexDecision.reject(
                self._failed_build_result(
                    f"PyLucene index path is not a directory: {index_path}",
                    str(index_path),
                    build_params,
                )
            )
        if not _has_lucene_segments(index_path):
            return _ExistingIndexDecision.reject(
                self._failed_build_result(
                    "Existing PyLucene index directory does not contain "
                    f"a Lucene segments file: {index_path}",
                    str(index_path),
                    build_params,
                )
            )
        if force:
            return _ExistingIndexDecision.build()

        metadata: Dict[str, Any] = {
            "codec": codec_name,
            "skipped": True,
        }
        try:
            _validate_metric(dataset)
            expected_shape = _expected_training_shape(dataset)
            _, verification_metadata = self._verify_existing_index(
                index_path,
                codec_name,
                expected_vector_count=(
                    expected_shape[0] if expected_shape else None
                ),
                expected_dimensions=(
                    expected_shape[1] if expected_shape else None
                ),
            )
            metadata.update(verification_metadata)
            result = BuildResult(
                index_path=str(index_path),
                build_time_seconds=0.0,
                index_size_bytes=_index_size(index_path),
                algorithm=self.algo,
                build_params=build_params,
                metadata=metadata,
                success=True,
            )
        except Exception as exc:
            return _ExistingIndexDecision.reject(
                self._failed_build_result(
                    _exception_summary(exc),
                    str(index_path),
                    build_params,
                )
            )

        return _ExistingIndexDecision.reuse(result)

    @staticmethod
    def _training_vectors_for_build(
        dataset: Dataset, codec_name: str
    ) -> np.ndarray:
        _validate_metric(dataset)
        vectors = _validate_float32_matrix(
            dataset.training_vectors, "training_vectors"
        )
        if vectors.shape[0] < 2:
            raise ValueError(
                f"{codec_name} requires at least two training vectors; "
                "cuVS-Lucene does not invoke cuVS for a single-vector index"
            )
        return vectors

    @staticmethod
    def _require_gpu_writer(
        runtime: _PyLuceneRuntime, codec_name: str
    ) -> _ResolvedCodec:
        java_codec = runtime.resolve_codec(codec_name)
        telemetry = runtime.codec_telemetry(codec_name, java_codec)
        writer_path = telemetry.get("writerPath")
        expected_writer_path = _EXPECTED_WRITER_PATH[codec_name]
        if writer_path != expected_writer_path:
            raise RuntimeError(
                f"{codec_name} uses writerPath={writer_path!r}; "
                f"expected {expected_writer_path!r}. Refusing to run "
                "a silent CPU or alternate-index fallback as a cuVS build."
            )
        return _ResolvedCodec(
            codec_name=codec_name,
            java_codec=java_codec,
            telemetry=telemetry,
            writer_path=writer_path,
        )

    @staticmethod
    def _verify_and_persist_built_index_provenance(
        runtime: _PyLuceneRuntime,
        index_path: Path,
        codec_name: str,
        vectors: np.ndarray,
    ) -> Dict[str, Any]:
        vector_count = int(vectors.shape[0])
        dimensions = int(vectors.shape[1])
        if codec_name == _CAGRA_CODEC:
            cagra_verification = runtime.verify_cagra_index(
                index_path,
                expected_vector_count=vector_count,
                expected_dimensions=dimensions,
            )
            _write_cagra_provenance(
                index_path,
                vector_count=vector_count,
                dimensions=dimensions,
            )
            provenance = _verify_cagra_provenance(
                index_path,
                expected_vector_count=vector_count,
                expected_dimensions=dimensions,
            )
            return {
                "cagra_provenance": provenance.to_metadata(),
                "cagra_verification": cagra_verification.to_metadata(),
            }

        _write_hnsw_provenance(
            index_path,
            codec_name,
            vector_count=vector_count,
            dimensions=dimensions,
        )
        verification = _verify_hnsw_provenance(
            index_path,
            codec_name,
            expected_vector_count=vector_count,
            expected_dimensions=dimensions,
        )
        return {"hnsw_verification": verification.to_metadata()}

    @staticmethod
    def _cleanup_partial_index(
        index_path: Path, created_for_build: bool
    ) -> Optional[Exception]:
        if not created_for_build or not index_path.exists():
            return None
        try:
            _safe_remove_index(index_path)
        except Exception as exc:
            return exc
        return None

    def build(
        self,
        dataset: Dataset,
        indexes: List[IndexConfig],
        force: bool = False,
        dry_run: bool = False,
    ) -> BuildResult:
        """Build one local Lucene index with the selected cuVS codec."""
        if len(indexes) != 1:
            return self._failed_build_result(
                "PyLucene backend requires exactly one index configuration"
            )

        index_config = indexes[0]
        build_params = index_config.build_param
        index_path = Path(index_config.file)
        try:
            _validate_build_params(build_params)
            codec_name = _configured_codec(build_params, self.config)
        except Exception as exc:
            return self._failed_build_result(
                str(exc), str(index_path), build_params
            )

        if dry_run:
            return self._dry_run_build_result(
                index_path, codec_name, build_params
            )

        existing_index_decision = self._decide_existing_index(
            dataset, index_path, codec_name, build_params, force
        )
        if existing_index_decision.action is _ExistingIndexAction.BUILD:
            return self._build_new_index(
                dataset, index_path, codec_name, build_params
            )

        return existing_index_decision.completed_result()

    def _build_new_index(
        self,
        dataset: Dataset,
        index_path: Path,
        codec_name: str,
        build_params: Dict[str, Any],
    ) -> BuildResult:
        """Build a validated index that is not eligible for reuse."""

        created_for_build = False
        try:
            vectors = self._training_vectors_for_build(dataset, codec_name)
            runtime = self._get_runtime()
            # Preflight must succeed before an existing index is replaced.
            resolved_codec = self._require_gpu_writer(runtime, codec_name)

            if index_path.exists():
                _safe_remove_index(index_path)
            index_path.mkdir(parents=True)
            created_for_build = True

            start = time.perf_counter()
            telemetry = runtime.build_index(
                index_path, vectors, resolved_codec
            )
            build_time = time.perf_counter() - start

            metadata = {
                "codec": codec_name,
                "pylucene_version": runtime.pylucene_version,
                "writer_path": resolved_codec.writer_path,
                "writer_telemetry": telemetry,
            }
            metadata.update(
                self._verify_and_persist_built_index_provenance(
                    runtime, index_path, codec_name, vectors
                )
            )

            return BuildResult(
                index_path=str(index_path),
                build_time_seconds=build_time,
                index_size_bytes=_index_size(index_path),
                algorithm=self.algo,
                build_params=build_params,
                metadata=metadata,
                success=True,
            )
        except Exception as exc:
            cleanup_error = self._cleanup_partial_index(
                index_path, created_for_build
            )
            error_message = _exception_summary(exc)
            if cleanup_error is not None:
                error_message += (
                    "; failed to remove partial index: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            return self._failed_build_result(
                error_message,
                str(index_path),
                build_params,
            )
        except BaseException as exc:
            cleanup_error = self._cleanup_partial_index(
                index_path, created_for_build
            )
            if cleanup_error is not None:
                exc.add_note(
                    "Failed to remove partial index: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise

    def _resolve_search_plan(
        self,
        index_config: IndexConfig,
        *,
        k: int,
        batch_size: int,
        mode: str,
        search_threads: Optional[Union[int, str]],
    ) -> _SearchPlan:
        search_params = index_config.search_params or [{}]
        _validate_build_params(index_config.build_param)
        codec_name = _configured_codec(index_config.build_param, self.config)
        if k < 1:
            raise ValueError("k must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if mode != "latency":
            raise ValueError(
                "PyLucene backend currently supports only latency mode"
            )
        if search_threads not in (None, 1, "1"):
            raise ValueError(
                "PyLucene backend currently supports only one search thread"
            )
        _validate_search_params(search_params)
        if codec_name == _CAGRA_CODEC and k > 1024:
            raise ValueError(
                "CuVS2510GPUSearchCodec benchmarks support k <= 1024 "
                "to avoid cuVS-Lucene search paths that can use GPU "
                "brute-force search above that limit"
            )
        return _SearchPlan(
            index_path=Path(index_config.file),
            codec_name=codec_name,
            search_params=search_params,
            k=k,
            batch_size=batch_size,
            mode=mode,
        )

    def _dry_run_search_result(
        self,
        plan: _SearchPlan,
    ) -> SearchResult:
        print(
            f"[dry_run] Would search PyLucene index '{plan.index_path}' "
            f"with codec={plan.codec_name}, k={plan.k}, "
            f"batch_size={plan.batch_size}"
        )
        return SearchResult(
            neighbors=np.empty((0, plan.k), dtype=np.int64),
            distances=np.empty((0, plan.k), dtype=np.float32),
            search_time_ms=0.0,
            queries_per_second=0.0,
            recall=0.0,
            algorithm=self.algo,
            search_params=plan.search_params,
            metadata={"codec": plan.codec_name},
            success=True,
        )

    @staticmethod
    def _load_search_inputs(dataset: Dataset) -> _SearchInputs:
        _validate_metric(dataset)
        query_vectors = _validate_float32_matrix(
            dataset.query_vectors, "query_vectors"
        )
        expected_shape = _expected_training_shape(dataset)
        if (
            expected_shape is not None
            and expected_shape[1] != query_vectors.shape[1]
        ):
            raise ValueError(
                "Query vector dimensions do not match the dataset: "
                f"{query_vectors.shape[1]} != {expected_shape[1]}"
            )
        return _SearchInputs(
            query_vectors=query_vectors,
            expected_training_shape=expected_shape,
        )

    def _execute_search(
        self, dataset: Dataset, plan: _SearchPlan
    ) -> SearchResult:
        end_to_end_start = time.perf_counter()
        inputs = self._load_search_inputs(dataset)
        query_vectors = inputs.query_vectors
        expected_shape = inputs.expected_training_shape
        provenance, verification_metadata = self._verify_existing_index(
            plan.index_path,
            plan.codec_name,
            expected_vector_count=(
                expected_shape[0] if expected_shape else None
            ),
            expected_dimensions=int(query_vectors.shape[1]),
        )
        runtime = self._get_runtime()

        runtime_result = runtime.search_index(
            plan.index_path,
            query_vectors,
            plan.k,
            plan.batch_size,
        )
        end_to_end_time_ms = (time.perf_counter() - end_to_end_start) * 1000.0
        processed = _process_search_result(
            runtime_result,
            provenance,
            query_count=int(query_vectors.shape[0]),
            k=plan.k,
            batch_size=plan.batch_size,
        )
        metadata = {
            "codec": plan.codec_name,
            "pylucene_version": runtime.pylucene_version,
            "index_dimensions": runtime_result.index_dimensions,
            "document_count": runtime_result.document_count,
            "batch_size": plan.batch_size,
            "num_batches": processed.num_batches,
            "mode": plan.mode,
            "end_to_end_time_ms": end_to_end_time_ms,
            "non_query_overhead_time_ms": max(
                0.0, end_to_end_time_ms - processed.search_time_ms
            ),
        }
        metadata.update(verification_metadata)

        return SearchResult(
            neighbors=processed.neighbors,
            distances=processed.distances,
            search_time_ms=processed.search_time_ms,
            queries_per_second=processed.queries_per_second,
            recall=0.0,
            algorithm=self.algo,
            search_params=plan.search_params,
            latency_seconds=processed.latency_seconds,
            latency_percentiles=processed.latency_percentiles,
            metadata=metadata,
            success=True,
        )

    def search(
        self,
        dataset: Dataset,
        indexes: List[IndexConfig],
        k: int,
        batch_size: int = 10000,
        mode: str = "latency",
        force: bool = False,
        search_threads: Optional[int] = None,
        dry_run: bool = False,
    ) -> SearchResult:
        """Search one local Lucene index and report per-batch latency."""
        if len(indexes) != 1:
            return self._failed_search_result(
                k, "PyLucene backend requires exactly one index configuration"
            )

        index_config = indexes[0]
        try:
            plan = self._resolve_search_plan(
                index_config,
                k=k,
                batch_size=batch_size,
                mode=mode,
                search_threads=search_threads,
            )
        except Exception as exc:
            return self._failed_search_result(
                k,
                str(exc),
                index_config.search_params or [{}],
            )

        if dry_run:
            return self._dry_run_search_result(plan)

        if not plan.index_path.is_dir():
            return self._failed_search_result(
                k,
                f"PyLucene index directory does not exist: {plan.index_path}",
                plan.search_params,
            )

        try:
            return self._execute_search(dataset, plan)
        except Exception as exc:
            return self._failed_search_result(
                k,
                _exception_summary(exc),
                plan.search_params,
            )


__all__ = ["PyLuceneBackend", "PyLuceneConfigLoader"]
