#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Check PyLucene production resources in the built cuVS Bench wheel."""

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
_REQUIRED_WHEEL_ENTRIES = {
    "cuvs_bench/backends/_java/PyLuceneConfiguredHnswCodec.java",
    "cuvs_bench/config/algos/pylucene_cuvs_cagra.yaml",
    "cuvs_bench/config/algos/pylucene_cuvs_hnsw.yaml",
}


def test_built_wheel_includes_resources_without_java_test_support(tmp_path):
    source_root = tmp_path / "source"
    project_copy = source_root / "python" / "cuvs_bench"
    shutil.copytree(
        _PROJECT_ROOT,
        project_copy,
        ignore=shutil.ignore_patterns(
            "*.egg-info", "__pycache__", "build", "dist"
        ),
    )
    shutil.copy2(
        _REPOSITORY_ROOT / "dependencies.yaml",
        source_root / "dependencies.yaml",
    )
    wheel_directory = tmp_path / "wheel"
    wheel_directory.mkdir()

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-build-isolation",
            "--no-deps",
            "--no-index",
            "--config-settings",
            "rapidsai.disable-cuda=true",
            "--wheel-dir",
            str(wheel_directory),
            str(project_copy),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, (
        "Could not build the cuVS Bench wheel:\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    wheels = tuple(wheel_directory.glob("*.whl"))
    assert len(wheels) == 1, f"expected one wheel, found: {wheels}"

    with zipfile.ZipFile(wheels[0]) as wheel:
        entries = set(wheel.namelist())

    assert _REQUIRED_WHEEL_ENTRIES <= entries
    assert not any("/tests/java/" in f"/{entry}" for entry in entries)
    assert not any(entry.endswith(".class") for entry in entries)
