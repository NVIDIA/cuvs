#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for locating the optional Lucene runtime artifact pair."""

from pathlib import Path

import pytest

from cuvs_bench.backends import _lucene_runtime_config
from cuvs_bench.backends._lucene_runtime_config import (
    resolve_lucene_runtime_config,
)


_ARTIFACT_ENVIRONMENT = (
    "CUVS_LUCENE_CUVS_JAVA_JAR",
    "CUVS_LUCENE_JAR",
)


def _empty_jar_pair(directory: Path, prefix: str = "") -> tuple[Path, Path]:
    java_jar = directory / f"{prefix}cuvs-java.jar"
    lucene_jar = directory / f"{prefix}cuvs-lucene.jar"
    java_jar.touch()
    lucene_jar.touch()
    return java_jar, lucene_jar


def _native_directory(directory: Path) -> Path:
    native = directory / "native"
    native.mkdir(exist_ok=True)
    (native / "libcuvs_c.so").touch()
    return native


def test_explicit_artifact_pair_is_resolved_together(tmp_path: Path) -> None:
    java_jar, lucene_jar = _empty_jar_pair(tmp_path)
    native = _native_directory(tmp_path)

    resolved = resolve_lucene_runtime_config(
        {
            "cuvs_java_jar": java_jar,
            "cuvs_lucene_jar": lucene_jar,
            "java_library_path": native,
            "jvm_args": ["-Xmx1g"],
        },
        requires_cuvs=True,
    )

    assert resolved == {
        "requires_cuvs": True,
        "cuvs_java_jar": str(java_jar.resolve()),
        "cuvs_lucene_jar": str(lucene_jar.resolve()),
        "java_library_path": str(native.resolve()),
        "jvm_args": ["-Xmx1g"],
    }


@pytest.mark.parametrize(
    "configured_key",
    ("cuvs_java_jar", "cuvs_lucene_jar"),
)
def test_partial_artifact_override_names_both_required_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, configured_key: str
) -> None:
    for variable in _ARTIFACT_ENVIRONMENT:
        monkeypatch.delenv(variable, raising=False)
    artifact = tmp_path / "one.jar"
    artifact.touch()

    with pytest.raises(RuntimeError) as error:
        resolve_lucene_runtime_config(
            {configured_key: artifact}, requires_cuvs=True
        )

    assert "both cuvs_java_jar and cuvs_lucene_jar" in str(error.value)


def test_environment_artifact_pair_is_used_when_config_has_no_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    java_jar, lucene_jar = _empty_jar_pair(tmp_path)
    native = _native_directory(tmp_path)
    monkeypatch.setenv("CUVS_LUCENE_CUVS_JAVA_JAR", str(java_jar))
    monkeypatch.setenv("CUVS_LUCENE_JAR", str(lucene_jar))

    resolved = resolve_lucene_runtime_config(
        {"java_library_path": str(native)}, requires_cuvs=True
    )

    assert resolved["cuvs_java_jar"] == str(java_jar.resolve())
    assert resolved["cuvs_lucene_jar"] == str(lucene_jar.resolve())
    assert resolved["java_library_path"] == str(native.resolve())


@pytest.mark.parametrize("configured_environment", ("neither", "java", "both"))
def test_cpu_only_resolution_ignores_ambient_cuvs_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured_environment: str,
) -> None:
    for variable in _ARTIFACT_ENVIRONMENT:
        monkeypatch.delenv(variable, raising=False)
    java_jar, lucene_jar = _empty_jar_pair(tmp_path)
    if configured_environment in {"java", "both"}:
        monkeypatch.setenv("CUVS_LUCENE_CUVS_JAVA_JAR", str(java_jar))
    if configured_environment == "both":
        monkeypatch.setenv("CUVS_LUCENE_JAR", str(lucene_jar))

    resolved = resolve_lucene_runtime_config(
        requires_cuvs=False, include_cuvs=False
    )

    assert resolved == {"requires_cuvs": False}


def test_cpu_only_resolution_still_rejects_partial_explicit_artifacts(
    tmp_path: Path,
) -> None:
    java_jar, _lucene_jar = _empty_jar_pair(tmp_path)

    with pytest.raises(RuntimeError, match="both cuvs_java_jar"):
        resolve_lucene_runtime_config(
            {"cuvs_java_jar": java_jar},
            requires_cuvs=False,
            include_cuvs=False,
        )


def test_backend_config_artifacts_take_precedence_over_environment_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured_java, configured_lucene = _empty_jar_pair(
        tmp_path, "configured-"
    )
    environment_java, environment_lucene = _empty_jar_pair(
        tmp_path, "environment-"
    )
    native = _native_directory(tmp_path)
    monkeypatch.setenv("CUVS_LUCENE_CUVS_JAVA_JAR", str(environment_java))
    monkeypatch.setenv("CUVS_LUCENE_JAR", str(environment_lucene))

    resolved = resolve_lucene_runtime_config(
        {
            "cuvs_java_jar": configured_java,
            "cuvs_lucene_jar": configured_lucene,
            "java_library_path": str(native),
        },
        requires_cuvs=True,
    )

    assert resolved["cuvs_java_jar"] == str(configured_java.resolve())
    assert resolved["cuvs_lucene_jar"] == str(configured_lucene.resolve())


def test_complete_backend_pair_ignores_an_incomplete_environment_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured_java, configured_lucene = _empty_jar_pair(tmp_path)
    native = _native_directory(tmp_path)
    unrelated_environment_jar = tmp_path / "environment-java.jar"
    unrelated_environment_jar.touch()
    monkeypatch.setenv(
        "CUVS_LUCENE_CUVS_JAVA_JAR", str(unrelated_environment_jar)
    )
    monkeypatch.delenv("CUVS_LUCENE_JAR", raising=False)

    resolved = resolve_lucene_runtime_config(
        {
            "cuvs_java_jar": configured_java,
            "cuvs_lucene_jar": configured_lucene,
            "java_library_path": native,
        },
        requires_cuvs=True,
    )

    assert resolved["cuvs_java_jar"] == str(configured_java.resolve())
    assert resolved["cuvs_lucene_jar"] == str(configured_lucene.resolve())


def test_artifacts_from_configuration_and_environment_cannot_form_a_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured_lucene = tmp_path / "configured-lucene.jar"
    environment_java = tmp_path / "environment-java.jar"
    configured_lucene.touch()
    environment_java.touch()
    monkeypatch.setenv("CUVS_LUCENE_CUVS_JAVA_JAR", str(environment_java))

    with pytest.raises(
        RuntimeError, match="both cuvs_java_jar and cuvs_lucene_jar"
    ):
        resolve_lucene_runtime_config(
            {"cuvs_lucene_jar": configured_lucene}, requires_cuvs=True
        )


def test_explicit_native_path_must_contain_unversioned_cuvs_c(
    tmp_path: Path,
) -> None:
    java_jar, lucene_jar = _empty_jar_pair(tmp_path)
    empty_native = tmp_path / "empty-native"
    empty_native.mkdir()

    with pytest.raises(RuntimeError, match="unversioned libcuvs_c.so"):
        resolve_lucene_runtime_config(
            {
                "cuvs_java_jar": java_jar,
                "cuvs_lucene_jar": lucene_jar,
                "java_library_path": empty_native,
            },
            requires_cuvs=True,
        )


def test_cuda_only_ld_library_path_is_augmented_with_discovered_cuvs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    java_jar, lucene_jar = _empty_jar_pair(tmp_path)
    cuvs_native = _native_directory(tmp_path)
    cuda_native = tmp_path / "cuda"
    cuda_native.mkdir()
    (cuda_native / "libcudart.so").touch()
    monkeypatch.setenv("LD_LIBRARY_PATH", str(cuda_native))
    monkeypatch.delenv("JAVA_LIBRARY_PATH", raising=False)
    monkeypatch.setattr(
        _lucene_runtime_config,
        "_native_library_groups",
        lambda: iter(((cuvs_native,),)),
    )

    resolved = resolve_lucene_runtime_config(
        {"cuvs_java_jar": java_jar, "cuvs_lucene_jar": lucene_jar},
        requires_cuvs=True,
    )

    assert resolved["java_library_path"].split(":") == [
        str(cuvs_native.resolve()),
        str(cuda_native.resolve()),
    ]
