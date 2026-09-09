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

from cuvs_bench.backends._pylucene_runtime_config import (
    resolve_pylucene_runtime_config,
)

OPT_IN_ENV = "CUVS_BENCH_PYLUCENE_INTEGRATION"
PYLUCENE_TEST_CLASSES_ENV = "PYLUCENE_TEST_CLASSES"

PYTHON_PACKAGE_ROOT = Path(__file__).resolve().parents[3]
JAVA_TEST_SOURCE_ROOT = PYTHON_PACKAGE_ROOT / "tests" / "java"


@dataclass(frozen=True)
class PyLuceneRuntimeFixture:
    """Resolved runtime inputs shared by every live PyLucene test."""

    backend_config: dict[str, str]
    test_classes: Path


def _find_javac() -> str:
    javac = shutil.which("javac")
    environment_javac = Path(sys.prefix) / "lib" / "jvm" / "bin" / "javac"
    if javac is None and environment_javac.is_file():
        javac = str(environment_javac)
    if javac is None:
        pytest.fail(
            "JDK 22 javac is required for the PyLucene integration tests; "
            "activate the JDK environment or add its bin directory to PATH"
        )
    return javac


def _expected_class_files(java_sources: tuple[Path, ...]) -> tuple[Path, ...]:
    return tuple(
        source.relative_to(JAVA_TEST_SOURCE_ROOT).with_suffix(".class")
        for source in java_sources
    )


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
    javac = _find_javac()
    try:
        completed = subprocess.run(
            [
                javac,
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
    except OSError as error:
        pytest.fail(
            "Could not execute JDK 22 javac for the PyLucene integration "
            f"tests ({javac}): {error}"
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
    *,
    run_requested: bool = False,
) -> PyLuceneRuntimeFixture:
    """Resolve inputs and prepare the sole live-test JVM classpath."""
    if not run_requested and os.environ.get(OPT_IN_ENV) != "1":
        pytest.skip("pass --run-pylucene to run PyLucene integration tests")

    try:
        backend_config = resolve_pylucene_runtime_config()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        pytest.fail(f"Could not configure the PyLucene runtime: {error}")

    cuvs_java_jar = Path(backend_config["cuvs_java_jar"])
    cuvs_lucene_jar = Path(backend_config["cuvs_lucene_jar"])

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
        backend_config=backend_config,
        test_classes=test_classes,
    )
