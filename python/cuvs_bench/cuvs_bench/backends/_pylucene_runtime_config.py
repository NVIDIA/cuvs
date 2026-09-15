#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Resolve the pre-installed artifacts needed by the PyLucene backend."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

_CUVS_JAVA_JAR_ENV = "CUVS_LUCENE_CUVS_JAVA_JAR"
_CUVS_LUCENE_JAR_ENV = "CUVS_LUCENE_JAR"
_MAVEN_REPOSITORY_ENV = "MAVEN_LOCAL_REPO"

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_VERSION_FILE = _PACKAGE_ROOT / "VERSION"


def _maven_artifact_version() -> str:
    """Translate the RAPIDS package version into its Maven spelling."""
    raw_version = _VERSION_FILE.read_text(encoding="utf-8").strip()
    release_version = raw_version.partition("a")[0]
    components = release_version.split(".")
    if len(components) != 3 or not all(
        component.isdigit() for component in components
    ):
        raise RuntimeError(
            f"Cannot derive a Maven artifact version from {raw_version!r}"
        )
    return ".".join(str(int(component)) for component in components)


def _maven_repository() -> Path:
    configured = os.environ.get(_MAVEN_REPOSITORY_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".m2" / "repository"


def _artifact_candidates(config_key: str) -> tuple[Path, ...]:
    version = _maven_artifact_version()
    maven_repository = _maven_repository()
    if config_key == "cuvs_java_jar":
        return (
            _REPOSITORY_ROOT
            / "java"
            / "cuvs-java"
            / "target"
            / f"cuvs-java-{version}.jar",
            maven_repository
            / "com"
            / "nvidia"
            / "cuvs"
            / "cuvs-java"
            / version
            / f"cuvs-java-{version}.jar",
        )
    if config_key == "cuvs_lucene_jar":
        return (
            _REPOSITORY_ROOT
            / "java"
            / "cuvs-lucene"
            / "target"
            / f"cuvs-lucene-{version}.jar",
            maven_repository
            / "com"
            / "nvidia"
            / "cuvs"
            / "lucene"
            / "cuvs-lucene"
            / version
            / f"cuvs-lucene-{version}.jar",
        )
    raise ValueError(f"Unknown PyLucene artifact key: {config_key}")


def _resolve_artifact(
    config: Mapping[str, Any], config_key: str, environment_key: str
) -> Path:
    configured = config.get(config_key)
    if configured is None:
        configured = os.environ.get(environment_key) or None
    if configured is not None:
        try:
            path = Path(os.fspath(configured)).expanduser().resolve()
        except TypeError as error:
            raise TypeError(
                f"{config_key} must be a filesystem path, got {configured!r}"
            ) from error
        if not path.is_file():
            raise FileNotFoundError(f"{config_key} does not exist: {path}")
        return path

    candidates = _artifact_candidates(config_key)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    searched = ", ".join(str(path) for path in candidates)
    raise RuntimeError(
        f"PyLucene backend could not find {config_key}. Build the matching "
        f"cuVS Java artifacts or set {environment_key}. Searched: {searched}"
    )


def _has_artifact_override(
    config: Mapping[str, Any], config_key: str, environment_key: str
) -> bool:
    return config.get(config_key) is not None or bool(
        os.environ.get(environment_key)
    )


def _resolve_artifact_pair(config: Mapping[str, Any]) -> tuple[Path, Path]:
    has_java_override = _has_artifact_override(
        config, "cuvs_java_jar", _CUVS_JAVA_JAR_ENV
    )
    has_lucene_override = _has_artifact_override(
        config, "cuvs_lucene_jar", _CUVS_LUCENE_JAR_ENV
    )
    if has_java_override != has_lucene_override:
        raise RuntimeError(
            "PyLucene Java artifact overrides must provide both "
            f"{_CUVS_JAVA_JAR_ENV} and {_CUVS_LUCENE_JAR_ENV} (or both "
            "corresponding backend-config paths) to keep artifact selection "
            "explicit. Build both JARs from the same checkout."
        )
    if has_java_override:
        return (
            _resolve_artifact(config, "cuvs_java_jar", _CUVS_JAVA_JAR_ENV),
            _resolve_artifact(config, "cuvs_lucene_jar", _CUVS_LUCENE_JAR_ENV),
        )

    java_candidates = _artifact_candidates("cuvs_java_jar")
    lucene_candidates = _artifact_candidates("cuvs_lucene_jar")
    for java_candidate, lucene_candidate in zip(
        java_candidates, lucene_candidates
    ):
        if java_candidate.is_file() and lucene_candidate.is_file():
            return java_candidate.resolve(), lucene_candidate.resolve()

    searched = ", ".join(
        f"({java_path}, {lucene_path})"
        for java_path, lucene_path in zip(java_candidates, lucene_candidates)
    )
    raise RuntimeError(
        "PyLucene backend could not find a complete, matching cuVS Java "
        "artifact pair. Build both artifacts together or set "
        f"{_CUVS_JAVA_JAR_ENV} and {_CUVS_LUCENE_JAR_ENV}. "
        f"Searched pairs: {searched}"
    )


def _python_environment_prefixes() -> Iterable[Path]:
    prefixes: list[Path] = []

    def add(prefix: Path) -> None:
        if prefix not in prefixes:
            prefixes.append(prefix)

    add(Path(sys.prefix))
    add(Path(sys.base_prefix))
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        add(Path(conda_prefix))

    for entry in sys.path:
        if not entry:
            continue
        path = Path(entry)
        if path.name in {"site-packages", "dist-packages"}:
            python_directory = path.parent
            if python_directory.parent.name == "lib":
                add(python_directory.parent.parent)

    yield from prefixes


def _cuda_library_directories() -> tuple[Path, ...]:
    configured = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    candidates = []
    if configured:
        candidates.append(Path(configured) / "lib64")
    candidates.append(Path("/usr/local/cuda/lib64"))
    return tuple(candidates)


def _native_library_groups() -> Iterable[tuple[Path, ...]]:
    cuda_directories = _cuda_library_directories()
    cuvs_home = os.environ.get("CUVS_HOME")
    if cuvs_home:
        build = Path(cuvs_home) / "cpp" / "build"
        yield (build / "c", build, *cuda_directories)

    source_build = _REPOSITORY_ROOT / "cpp" / "build"
    yield (source_build / "c", source_build, *cuda_directories)

    for prefix in _python_environment_prefixes():
        yield (
            prefix / "lib",
            prefix / "targets" / "x86_64-linux" / "lib",
            *cuda_directories,
        )


def _contains_runtime_library(directory: Path) -> bool:
    library_patterns = ("libcuvs.so*", "libcuvs_c.so*", "libcudart.so*")
    return directory.is_dir() and any(
        next(directory.glob(pattern), None) is not None
        for pattern in library_patterns
    )


def _contains_cuvs_c(directory: Path) -> bool:
    return (
        directory.is_dir()
        and next(directory.glob("libcuvs_c.so*"), None) is not None
    )


def _discover_native_library_path() -> str | None:
    configured = os.environ.get("JAVA_LIBRARY_PATH") or os.environ.get(
        "LD_LIBRARY_PATH"
    )
    if configured:
        return configured

    for group in _native_library_groups():
        directories = []
        seen = set()
        for candidate in group:
            resolved = candidate.resolve()
            if resolved in seen or not _contains_runtime_library(resolved):
                continue
            seen.add(resolved)
            directories.append(resolved)
        if any(_contains_cuvs_c(directory) for directory in directories):
            return os.pathsep.join(str(directory) for directory in directories)
    return None


def resolve_pylucene_runtime_config(
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve explicit overrides first, then conventional local installs."""
    options = dict(config or {})
    cuvs_java_jar, cuvs_lucene_jar = _resolve_artifact_pair(options)
    resolved = {
        "cuvs_java_jar": str(cuvs_java_jar),
        "cuvs_lucene_jar": str(cuvs_lucene_jar),
    }
    java_library_path = (
        options.get("java_library_path") or _discover_native_library_path()
    )
    if java_library_path:
        resolved["java_library_path"] = os.fspath(java_library_path)

    if "jvm_args" in options:
        resolved["jvm_args"] = options["jvm_args"]
    return resolved
