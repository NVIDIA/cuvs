# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared configuration for opt-in PyLucene integration tests."""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

OPT_IN_ENV = "CUVS_BENCH_PYLUCENE_INTEGRATION"
CUVS_JAVA_JAR_ENV = "CUVS_LUCENE_CUVS_JAVA_JAR"
CUVS_LUCENE_JAR_ENV = "CUVS_LUCENE_JAR"
PYLUCENE_TEST_CLASSES_ENV = "PYLUCENE_TEST_CLASSES"

PYTHON_PACKAGE_ROOT = Path(__file__).resolve().parents[3]
JAVA_TEST_SOURCE_ROOT = PYTHON_PACKAGE_ROOT / "tests" / "java"

_EXPECTED_NESTED_CLASSES = {
    "PyLuceneTestSupport.java": (
        "CagraBuiltHnswBaseLayerCodec",
        "CagraBuiltHnswThreeLayerCodec",
        "CagraSearchCodec",
        "CagraSearchQuery",
        "CpuHnswCodec",
        "DiagnosticKnnVectorsFormat",
        "HnswGraphVerifyingExactQuery",
        "HnswGraphVerifyingQuery",
        "QueryProperties",
    ),
    "PyLuceneWriterSelectionCodec.java": ("WriterSelectionFormat",),
}


@dataclass(frozen=True)
class PyLuceneRuntimeFixture:
    """Resolved runtime inputs shared by every live PyLucene test."""

    backend_config: dict[str, str]
    test_classes: Path


def _required_jar(env_name: str) -> Path:
    configured = os.environ.get(env_name)
    if not configured:
        pytest.fail(f"{env_name} must point to the required runtime jar")

    jar = Path(configured).expanduser()
    if not jar.is_file():
        pytest.fail(f"{env_name} does not point to an existing file: {jar}")
    return jar.resolve()


def _find_javac() -> str:
    javac = shutil.which("javac")
    environment_javac = Path(sys.prefix) / "lib" / "jvm" / "bin" / "javac"
    if javac is None and environment_javac.is_file():
        javac = str(environment_javac)
    if javac is None:
        pytest.fail("javac is required for the PyLucene integration tests")
    return javac


def _expected_class_files(java_sources: tuple[Path, ...]) -> tuple[Path, ...]:
    expected = []
    for source in java_sources:
        relative_class = source.relative_to(JAVA_TEST_SOURCE_ROOT).with_suffix(
            ".class"
        )
        expected.append(relative_class)
        for nested_class in _EXPECTED_NESTED_CLASSES.get(source.name, ()):
            expected.append(
                relative_class.with_name(
                    f"{relative_class.stem}${nested_class}.class"
                )
            )
    return tuple(expected)


def compile_test_java_sources(
    output_dir: Path,
    cuvs_java_jar: Path,
    cuvs_lucene_jar: Path,
    pylucene_classpath: str,
) -> None:
    """Compile every PyLucene test adapter together before JVM startup."""
    java_sources = tuple(sorted(JAVA_TEST_SOURCE_ROOT.rglob("*.java")))
    if not java_sources:
        pytest.fail(
            "No PyLucene Java test sources were found under "
            f"{JAVA_TEST_SOURCE_ROOT}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    compile_classpath = os.pathsep.join(
        (str(cuvs_java_jar), str(cuvs_lucene_jar), pylucene_classpath)
    )
    completed = subprocess.run(
        [
            _find_javac(),
            "--release",
            "22",
            "-classpath",
            compile_classpath,
            "-d",
            str(output_dir),
            *(str(source) for source in java_sources),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.fail(
            "Could not compile the PyLucene Java test adapters:\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )

    missing_classes = tuple(
        relative
        for relative in _expected_class_files(java_sources)
        if not (output_dir / relative).is_file()
    )
    if missing_classes:
        missing = ", ".join(str(path) for path in missing_classes)
        pytest.fail(
            "Compiled PyLucene test adapters are incomplete under "
            f"{output_dir}: missing {missing}"
        )


def _load_uninitialized_pylucene() -> ModuleType:
    try:
        lucene = importlib.import_module("lucene")
    except ImportError as exc:
        pytest.fail(f"PyLucene is not importable: {exc}")

    from cuvs_bench.backends.pylucene import _validate_pylucene_version

    _validate_pylucene_version(lucene)
    if lucene.getVMEnv() is not None:
        pytest.fail(
            "PyLucene's process-wide JVM was initialized before the shared "
            "integration-test bootstrap; run the suite in a fresh process"
        )
    return lucene


def configure_pylucene_runtime(
    tmp_path_factory: pytest.TempPathFactory,
) -> PyLuceneRuntimeFixture:
    """Resolve inputs and prepare the sole live-test JVM classpath."""
    if os.environ.get(OPT_IN_ENV) != "1":
        pytest.skip(f"set {OPT_IN_ENV}=1 to run PyLucene integration tests")

    cuvs_java_jar = _required_jar(CUVS_JAVA_JAR_ENV)
    cuvs_lucene_jar = _required_jar(CUVS_LUCENE_JAR_ENV)
    java_library_path = os.environ.get("JAVA_LIBRARY_PATH") or os.environ.get(
        "LD_LIBRARY_PATH"
    )
    if not java_library_path:
        pytest.fail(
            "JAVA_LIBRARY_PATH or LD_LIBRARY_PATH must provide the native "
            "cuVS runtime libraries"
        )

    lucene = _load_uninitialized_pylucene()
    test_classes = tmp_path_factory.mktemp("pylucene-java-test-classes")
    compile_test_java_sources(
        test_classes,
        cuvs_java_jar,
        cuvs_lucene_jar,
        str(lucene.CLASSPATH),
    )
    classpath_entries = str(lucene.CLASSPATH).split(os.pathsep)
    if str(test_classes) in classpath_entries:
        pytest.fail(
            "The temporary PyLucene test classes directory was already on "
            "the process classpath before bootstrap"
        )
    lucene.CLASSPATH = os.pathsep.join(
        (str(test_classes), str(lucene.CLASSPATH))
    )

    return PyLuceneRuntimeFixture(
        backend_config={
            "cuvs_java_jar": str(cuvs_java_jar),
            "cuvs_lucene_jar": str(cuvs_lucene_jar),
            "java_library_path": java_library_path,
        },
        test_classes=test_classes,
    )
