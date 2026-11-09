#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pinned PyLucene source builder."""

import hashlib
import io
import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from cuvs_bench.backends._lucene_compat import LUCENE_COMPATIBILITY


_REPOSITORY_ROOT = Path(__file__).parents[4]
_RECIPE_DIRECTORY = _REPOSITORY_ROOT / "conda" / "recipes" / "cuvs-bench"
_BUILD_HELPER = _RECIPE_DIRECTORY / "build_pylucene_10_2.sh"
_COMPATIBILITY_PATCH = _RECIPE_DIRECTORY / "pylucene-10.2.0.patch"
_BUILD_RECIPE = _RECIPE_DIRECTORY / "pylucene_10_2_recipe.sh"


def test_source_builder_is_self_describing_from_another_directory(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        ["bash", str(_BUILD_HELPER), "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert os.access(_BUILD_HELPER, os.X_OK)
    assert _COMPATIBILITY_PATCH.is_file()
    assert completed.returncode == 0, completed.stderr
    assert "Build an isolated PyLucene 10.2.0 environment" in completed.stdout


def test_build_recipe_matches_the_supported_runtime() -> None:
    completed = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; printf "%s\\n" "${PYLUCENE_VERSION}" '
            '"${PYLUCENE_SCAFFOLD_VERSION}" "${JCC_VERSION}" "${JDK_VERSION}" '
            '"${COMPATIBILITY_PATCH}" "${PYLUCENE_SMOKE_CLASSES[@]}"',
            "recipe-check",
            str(_BUILD_RECIPE),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    values = completed.stdout.splitlines()
    assert values == [
        "10.2.0",
        "10.0.0",
        "3.15",
        "22",
        "pylucene-10.2.0.patch",
        "org.apache.lucene.codecs.lucene101.Lucene101Codec",
        "org.apache.lucene.codecs.lucene102.Lucene102HnswBinaryQuantizedVectorsFormat",
    ]
    assert values[0] == LUCENE_COMPATIBILITY.pylucene_version


def _write_source_archive(path: Path, files: dict[str, bytes]) -> bytes:
    """Create tiny source-shaped inputs; this fixture does not compile Lucene."""
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(content)
            archive.addfile(entry, io.BytesIO(content))
    return path.read_bytes()


@pytest.fixture
def cached_source_builder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Use the real preparation script with locally pinned miniature archives."""
    script = tmp_path / _BUILD_HELPER.name
    shutil.copyfile(_BUILD_HELPER, script)
    recipe = tmp_path / _BUILD_RECIPE.name
    (tmp_path / _COMPATIBILITY_PATCH.name).write_text("", encoding="utf-8")
    build_root = tmp_path / "build"
    downloads = build_root / "downloads"
    downloads.mkdir(parents=True)

    makefile = b"fixture makefile\n"
    settings = b"fixture settings\n"
    wrapper = b"fixture wrapper\n"
    scaffold = _write_source_archive(
        downloads / "pylucene-10.0.0-src.tar.gz",
        {
            "pylucene-10.0.0/Makefile": makefile,
            "pylucene-10.0.0/extensions/build.gradle": b"fixture extension\n",
        },
    )
    lucene = _write_source_archive(
        downloads / "lucene-10.2.0-src.tgz",
        {
            "lucene-10.2.0/settings.gradle": settings,
            "lucene-10.2.0/gradle/wrapper/gradle-wrapper.properties": wrapper,
            "lucene-10.2.0/lucene/core/build.gradle": b"fixture core\n",
        },
    )
    pins = {
        "PYLUCENE_ARCHIVE_SHA256": hashlib.sha256(scaffold).hexdigest(),
        "LUCENE_ARCHIVE_SHA512": hashlib.sha512(lucene).hexdigest(),
    }
    for name, content in (
        ("MAKEFILE", makefile),
        ("SETTINGS_GRADLE", settings),
        ("GRADLE_WRAPPER", wrapper),
    ):
        for stage in ("UPSTREAM", "PREPARED"):
            pins[f"{stage}_{name}_SHA256"] = hashlib.sha256(
                content
            ).hexdigest()
    recipe_text = _BUILD_RECIPE.read_text(encoding="utf-8")
    for name, value in pins.items():
        recipe_text, replaced = re.subn(
            rf'^readonly {name}="[^"]*"$',
            f'readonly {name}="{value}"',
            recipe_text,
            flags=re.MULTILINE,
        )
        assert replaced == 1, name
    recipe.write_text(recipe_text, encoding="utf-8")

    # Missing cached inputs must fail rather than fetch anything from the network.
    curl = tmp_path / "curl"
    curl.write_text(
        "#!/bin/sh\necho 'network forbidden' >&2\nexit 1\n", encoding="utf-8"
    )
    curl.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    return script, recipe, build_root


def test_source_cache_rejects_recipe_changes_without_rewriting_manifest(
    cached_source_builder,
) -> None:
    script, recipe, build_root = cached_source_builder
    command = [
        "bash",
        str(script),
        "--build-root",
        str(build_root),
        "--prepare-only",
    ]
    original_recipe = recipe.read_text(encoding="utf-8")
    subprocess.run(command, check=True, capture_output=True, text=True)
    manifest = (
        build_root / "src/pylucene-10.2.0/.cuvs-pylucene-source-manifest"
    )
    original_manifest = manifest.read_bytes()
    subprocess.run(command, check=True, capture_output=True, text=True)

    # Change only recipe bytes: the separately recorded pins remain identical.
    recipe.write_text(
        original_recipe + "\n# recipe changed\n", encoding="utf-8"
    )
    rejected = subprocess.run(
        command, check=False, capture_output=True, text=True
    )

    assert rejected.returncode != 0
    assert "prepared sources use different inputs" in rejected.stderr
    assert manifest.read_bytes() == original_manifest
    recipe.write_text(original_recipe, encoding="utf-8")
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert manifest.read_bytes() == original_manifest


@pytest.mark.parametrize(
    "unsafe_character",
    [
        pytest.param(" ", id="space"),
        pytest.param(":", id="colon"),
        pytest.param(";", id="semicolon"),
        pytest.param("$", id="dollar"),
        pytest.param("`", id="backtick"),
        pytest.param("&", id="ampersand"),
        pytest.param("#", id="hash"),
        pytest.param("|", id="pipe"),
    ],
)
def test_source_builder_rejects_unsafe_build_roots(
    tmp_path: Path, unsafe_character: str
) -> None:
    build_root = tmp_path / f"unsafe{unsafe_character}root"
    completed = subprocess.run(
        [
            "bash",
            str(_BUILD_HELPER),
            "--build-root",
            str(build_root),
            "--prepare-only",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "--build-root may contain only" in completed.stderr
    assert not build_root.exists()


def test_source_builder_rejects_unsafe_resolved_build_root(
    tmp_path: Path,
) -> None:
    unsafe_target = tmp_path / "unsafe;target"
    unsafe_target.mkdir()
    build_root = tmp_path / "safe-link"
    build_root.symlink_to(unsafe_target, target_is_directory=True)

    completed = subprocess.run(
        [
            "bash",
            str(_BUILD_HELPER),
            "--build-root",
            str(build_root),
            "--prepare-only",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "--build-root may contain only" in completed.stderr
    assert not (unsafe_target / ".build.lock").exists()
