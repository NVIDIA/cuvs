#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Compile the no-argument codec adapter required by stock PyLucene."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Optional, Union

CONFIGURED_HNSW_CODEC_CLASS = (
    "com.nvidia.cuvs.bench.PyLuceneConfiguredHnswCodec"
)
M_PROPERTY = "com.nvidia.cuvs.bench.pylucene.hnsw.m"
EF_CONSTRUCTION_PROPERTY = "com.nvidia.cuvs.bench.pylucene.hnsw.efConstruction"

_SOURCE_FILE = (
    Path(__file__).with_name("_java") / "PyLuceneConfiguredHnswCodec.java"
)
_COMPILE_LOCK = threading.Lock()
_COMPILED_CLASSPATH: Optional[str] = None
_CLASSES_DIRECTORY: Optional[Path] = None
_TEMPORARY_DIRECTORY: Optional[tempfile.TemporaryDirectory] = None


def configured_codec_classes_path(
    cuvs_java_jar: Path,
    cuvs_lucene_jar: Path,
    pylucene_classpath: str,
) -> Path:
    """Compile the codec adapter once and return its stable classes path.

    This function must be called before initializing PyLucene's process-wide
    JVM so that the returned directory can be included in the JVM classpath.
    """

    compile_classpath = _compile_classpath(
        cuvs_java_jar, cuvs_lucene_jar, pylucene_classpath
    )
    with _COMPILE_LOCK:
        if _CLASSES_DIRECTORY is not None:
            _require_same_classpath(compile_classpath)
            return _CLASSES_DIRECTORY
        return _compile(compile_classpath)


def _compile_classpath(
    cuvs_java_jar: Union[str, os.PathLike[str]],
    cuvs_lucene_jar: Union[str, os.PathLike[str]],
    pylucene_classpath: str,
) -> str:
    if not isinstance(pylucene_classpath, str) or not pylucene_classpath:
        raise ValueError("pylucene_classpath must be a non-empty string")

    jar_paths = (
        Path(cuvs_java_jar).resolve(),
        Path(cuvs_lucene_jar).resolve(),
    )
    for jar_path in jar_paths:
        if not jar_path.is_file():
            raise FileNotFoundError(
                f"Cannot compile the PyLucene codec adapter; JAR not found: {jar_path}"
            )
    return os.pathsep.join((*map(str, jar_paths), pylucene_classpath))


def _require_same_classpath(compile_classpath: str) -> None:
    if compile_classpath != _COMPILED_CLASSPATH:
        raise RuntimeError(
            "The PyLucene codec adapter was already compiled with a different "
            "dependency classpath. PyLucene JVM dependencies are process-wide."
        )


def _compile(compile_classpath: str) -> Path:
    global _CLASSES_DIRECTORY, _COMPILED_CLASSPATH, _TEMPORARY_DIRECTORY

    if not _SOURCE_FILE.is_file():
        raise RuntimeError(
            f"The packaged PyLucene codec adapter source is missing: {_SOURCE_FILE}"
        )

    javac = _find_javac()
    temporary_directory = tempfile.TemporaryDirectory(
        prefix="cuvs-bench-pylucene-codec-"
    )
    classes_directory = Path(temporary_directory.name) / "classes"
    classes_directory.mkdir()
    try:
        completed = subprocess.run(
            [
                javac,
                "--release",
                "22",
                "-classpath",
                compile_classpath,
                "-d",
                str(classes_directory),
                str(_SOURCE_FILE),
            ],
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError as error:
        temporary_directory.cleanup()
        raise RuntimeError(
            f"Could not run JDK 22 javac at {javac}: {error}"
        ) from error

    if completed.returncode != 0:
        temporary_directory.cleanup()
        details = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            "Could not compile the PyLucene codec adapter with JDK 22 javac. "
            "Verify that cuvs-java, PyLucene 10.2, and the thin cuvs-lucene "
            "JAR are compatible. The cuvs-lucene JAR must include the "
            "in-tree PyLucene 10.2 codec support and current HNSW heuristic "
            "API. "
            f"javac output:\n{details}"
        )

    _TEMPORARY_DIRECTORY = temporary_directory
    _CLASSES_DIRECTORY = classes_directory
    _COMPILED_CLASSPATH = compile_classpath
    return classes_directory


def _find_javac() -> str:
    candidates = []
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        candidates.append(Path(java_home) / "bin" / "javac")
    candidates.append(Path(sys.prefix) / "lib" / "jvm" / "bin" / "javac")

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    javac = shutil.which("javac")
    if javac is not None:
        return javac
    raise RuntimeError(
        "Configurable PyLucene HNSW builds require JDK 22 javac. Set "
        "JAVA_HOME to a JDK 22 installation or put its javac on PATH."
    )
