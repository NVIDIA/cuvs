#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Locate the preinstalled artifacts used by the optional Lucene backend."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

_JAVA_JAR_ENV = "CUVS_LUCENE_CUVS_JAVA_JAR"
_LUCENE_JAR_ENV = "CUVS_LUCENE_JAR"
_MAVEN_REPOSITORY_ENV = "MAVEN_LOCAL_REPO"

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_VERSION_FILE = _PACKAGE_ROOT / "VERSION"


def maven_artifact_version() -> str:
    """Return the package version using Maven's non-zero-padded spelling."""
    raw_version = _VERSION_FILE.read_text(encoding="utf-8").strip()
    release_version = raw_version.partition("a")[0]
    components = release_version.split(".")
    if len(components) != 3 or not all(part.isdigit() for part in components):
        raise RuntimeError(
            f"Cannot derive a Maven artifact version from {raw_version!r}"
        )
    return ".".join(str(int(part)) for part in components)


def _maven_repository() -> Path:
    configured = os.environ.get(_MAVEN_REPOSITORY_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".m2" / "repository"


def _artifact_candidates(kind: str) -> tuple[Path, ...]:
    version = maven_artifact_version()
    repository = _maven_repository()
    if kind == "cuvs_java_jar":
        return (
            _REPOSITORY_ROOT
            / "java"
            / "cuvs-java"
            / "target"
            / f"cuvs-java-{version}.jar",
            repository
            / "com"
            / "nvidia"
            / "cuvs"
            / "cuvs-java"
            / version
            / f"cuvs-java-{version}.jar",
        )
    if kind == "cuvs_lucene_jar":
        return (
            _REPOSITORY_ROOT
            / "java"
            / "cuvs-lucene"
            / "target"
            / f"cuvs-lucene-{version}.jar",
            repository
            / "com"
            / "nvidia"
            / "cuvs"
            / "lucene"
            / "cuvs-lucene"
            / version
            / f"cuvs-lucene-{version}.jar",
        )
    raise ValueError(f"Unknown Lucene artifact kind: {kind}")


def _configured_path(value: Any, key: str) -> Path:
    try:
        path = Path(os.fspath(value)).expanduser().resolve()
    except TypeError as error:
        raise TypeError(
            f"{key} must be a filesystem path, got {value!r}"
        ) from error
    if not path.is_file():
        raise FileNotFoundError(f"{key} does not exist: {path}")
    return path


def _explicit_artifact_pair(
    config: Mapping[str, Any],
    *,
    include_environment: bool = True,
) -> tuple[Path, Path] | None:
    configured = (
        config.get("cuvs_java_jar"),
        config.get("cuvs_lucene_jar"),
    )
    if configured[0] or configured[1]:
        if bool(configured[0]) != bool(configured[1]):
            raise RuntimeError(
                "Lucene backend configuration must provide both cuvs_java_jar "
                "and cuvs_lucene_jar or neither"
            )
        values = configured
    elif include_environment:
        environment = (
            os.environ.get(_JAVA_JAR_ENV),
            os.environ.get(_LUCENE_JAR_ENV),
        )
        if bool(environment[0]) != bool(environment[1]):
            raise RuntimeError(
                "Lucene artifact environment overrides must provide both "
                f"{_JAVA_JAR_ENV} and {_LUCENE_JAR_ENV} or neither"
            )
        values = environment
    else:
        values = (None, None)
    if not values[0]:
        return None
    return (
        _configured_path(values[0], "cuvs_java_jar"),
        _configured_path(values[1], "cuvs_lucene_jar"),
    )


def _conventional_artifact_pair() -> tuple[Path, Path] | None:
    for java_candidate, lucene_candidate in zip(
        _artifact_candidates("cuvs_java_jar"),
        _artifact_candidates("cuvs_lucene_jar"),
    ):
        if java_candidate.is_file() and lucene_candidate.is_file():
            return java_candidate.resolve(), lucene_candidate.resolve()
    return None


def _required_artifact_pair(
    config: Mapping[str, Any],
) -> tuple[Path, Path]:
    pair = _explicit_artifact_pair(config) or _conventional_artifact_pair()
    if pair is not None:
        return pair
    searched = ", ".join(
        f"({java_path}, {lucene_path})"
        for java_path, lucene_path in zip(
            _artifact_candidates("cuvs_java_jar"),
            _artifact_candidates("cuvs_lucene_jar"),
        )
    )
    raise RuntimeError(
        "cuVS-backed Lucene algorithms require matching standard cuvs-java "
        "and thin cuvs-lucene JARs. Build both artifacts or set "
        f"{_JAVA_JAR_ENV} and {_LUCENE_JAR_ENV}. Searched: {searched}"
    )


def _python_prefixes() -> Iterable[Path]:
    prefixes: list[Path] = []

    def add(path: Path) -> None:
        if path not in prefixes:
            prefixes.append(path)

    add(Path(sys.prefix))
    add(Path(sys.base_prefix))
    if conda_prefix := os.environ.get("CONDA_PREFIX"):
        add(Path(conda_prefix))
    for entry in sys.path:
        path = Path(entry) if entry else None
        if path is not None and path.name in {
            "site-packages",
            "dist-packages",
        }:
            python_dir = path.parent
            if python_dir.parent.name == "lib":
                add(python_dir.parent.parent)
    yield from prefixes


def _cuda_library_directories() -> tuple[Path, ...]:
    configured = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    candidates = [Path("/usr/local/cuda/lib64")]
    if configured:
        candidates.insert(0, Path(configured) / "lib64")
    return tuple(candidates)


def _native_library_groups() -> Iterable[tuple[Path, ...]]:
    cuda_directories = _cuda_library_directories()
    if cuvs_home := os.environ.get("CUVS_HOME"):
        build = Path(cuvs_home) / "cpp" / "build"
        yield (build / "c", build, *cuda_directories)
    source_build = _REPOSITORY_ROOT / "cpp" / "build"
    yield (source_build / "c", source_build, *cuda_directories)
    for prefix in _python_prefixes():
        yield (
            prefix / "lib",
            prefix / "targets" / "x86_64-linux" / "lib",
            *cuda_directories,
        )


def _contains_library(directory: Path, pattern: str) -> bool:
    return (
        directory.is_dir() and next(directory.glob(pattern), None) is not None
    )


def _library_path_directories(value: str) -> list[Path]:
    return [
        Path(component).expanduser().resolve()
        for component in value.split(os.pathsep)
        if component
    ]


def _has_unversioned_cuvs_c(directories: Iterable[Path]) -> bool:
    return any(
        (directory / "libcuvs_c.so").is_file() for directory in directories
    )


def _validated_explicit_library_path(value: Any) -> str:
    try:
        path_value = os.fspath(value)
    except TypeError as error:
        raise TypeError(
            f"java_library_path must be path-like, got {value!r}"
        ) from error
    directories = _library_path_directories(path_value)
    if not _has_unversioned_cuvs_c(directories):
        raise RuntimeError(
            "java_library_path does not contain the required unversioned "
            "libcuvs_c.so"
        )
    return os.pathsep.join(str(directory) for directory in directories)


def _discover_native_library_path() -> str | None:
    if configured := os.environ.get("JAVA_LIBRARY_PATH"):
        return _validated_explicit_library_path(configured)
    ambient = _library_path_directories(os.environ.get("LD_LIBRARY_PATH", ""))
    if _has_unversioned_cuvs_c(ambient):
        return os.pathsep.join(str(path) for path in ambient)
    for group in _native_library_groups():
        directories = []
        seen = set()
        for candidate in group:
            resolved = candidate.resolve()
            if resolved in seen or not resolved.is_dir():
                continue
            if not any(
                _contains_library(resolved, pattern)
                for pattern in (
                    "libcuvs.so*",
                    "libcuvs_c.so*",
                    "libcudart.so*",
                )
            ):
                continue
            seen.add(resolved)
            directories.append(resolved)
        if _has_unversioned_cuvs_c(directories):
            combined = list(directories)
            combined.extend(path for path in ambient if path not in combined)
            return os.pathsep.join(str(path) for path in combined)
    return None


def resolve_lucene_runtime_config(
    config: Mapping[str, Any] | None = None,
    *,
    requires_cuvs: bool,
    include_cuvs: bool = False,
) -> dict[str, Any]:
    """Resolve explicit overrides, then matching artifacts from known locations."""
    options = dict(config or {})
    explicit_pair = _explicit_artifact_pair(
        options, include_environment=requires_cuvs or include_cuvs
    )
    if requires_cuvs:
        pair = explicit_pair or _required_artifact_pair(options)
    elif include_cuvs:
        pair = explicit_pair or _conventional_artifact_pair()
    else:
        pair = explicit_pair
    resolved: dict[str, Any] = {"requires_cuvs": requires_cuvs}
    if pair is not None:
        resolved["cuvs_java_jar"] = str(pair[0])
        resolved["cuvs_lucene_jar"] = str(pair[1])

    library_path = options.get("java_library_path")
    if library_path is not None and pair is not None:
        library_path = _validated_explicit_library_path(library_path)
    if library_path is None and pair is not None:
        library_path = _discover_native_library_path()
    if library_path:
        resolved["java_library_path"] = os.fspath(library_path)
    if "jvm_args" in options:
        resolved["jvm_args"] = options["jvm_args"]
    return resolved
