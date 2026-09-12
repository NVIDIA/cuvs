# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for deterministic PyLucene runtime artifact discovery."""

import os
from pathlib import Path

import pytest

from cuvs_bench.backends import _pylucene_runtime_config as runtime_config


def _write_artifact(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test artifact")
    return path


def _configure_test_installation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    repository = tmp_path / "source"
    maven_repository = tmp_path / "maven"
    version_file = tmp_path / "VERSION"
    version_file.write_text("26.10.00\n")
    monkeypatch.setattr(runtime_config, "_REPOSITORY_ROOT", repository)
    monkeypatch.setattr(runtime_config, "_VERSION_FILE", version_file)
    monkeypatch.setenv("MAVEN_LOCAL_REPO", str(maven_repository))
    monkeypatch.delenv("CUVS_LUCENE_CUVS_JAVA_JAR", raising=False)
    monkeypatch.delenv("CUVS_LUCENE_JAR", raising=False)
    return repository, maven_repository


def test_maven_artifact_version_normalizes_rapids_patch_number(
    tmp_path, monkeypatch
):
    version_file = tmp_path / "VERSION"
    version_file.write_text("26.10.00a1\n")
    monkeypatch.setattr(runtime_config, "_VERSION_FILE", version_file)

    assert runtime_config._maven_artifact_version() == "26.10.0"


def test_explicit_artifacts_take_precedence_over_environment(
    tmp_path, monkeypatch
):
    explicit_java = _write_artifact(tmp_path / "explicit-cuvs-java.jar")
    explicit_lucene = _write_artifact(tmp_path / "explicit-cuvs-lucene.jar")
    monkeypatch.setenv(
        "CUVS_LUCENE_CUVS_JAVA_JAR",
        str(_write_artifact(tmp_path / "environment-cuvs-java.jar")),
    )
    monkeypatch.setenv(
        "CUVS_LUCENE_JAR",
        str(_write_artifact(tmp_path / "environment-cuvs-lucene.jar")),
    )

    resolved = runtime_config.resolve_pylucene_runtime_config(
        {
            "cuvs_java_jar": explicit_java,
            "cuvs_lucene_jar": explicit_lucene,
        }
    )

    assert resolved["cuvs_java_jar"] == str(explicit_java.resolve())
    assert resolved["cuvs_lucene_jar"] == str(explicit_lucene.resolve())


@pytest.mark.parametrize(
    ("config", "environment"),
    [
        ({"cuvs_java_jar": "configured.jar"}, {}),
        ({"cuvs_lucene_jar": "configured.jar"}, {}),
        ({}, {"CUVS_LUCENE_CUVS_JAVA_JAR": "configured.jar"}),
        ({}, {"CUVS_LUCENE_JAR": "configured.jar"}),
    ],
    ids=[
        "config-java-only",
        "config-lucene-only",
        "environment-java-only",
        "environment-lucene-only",
    ],
)
def test_artifact_override_requires_an_explicit_pair(
    config, environment, tmp_path, monkeypatch
):
    _configure_test_installation(tmp_path, monkeypatch)
    resolved_config = {
        key: _write_artifact(tmp_path / value) for key, value in config.items()
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, str(_write_artifact(tmp_path / value)))

    with pytest.raises(RuntimeError, match="overrides must provide both"):
        runtime_config.resolve_pylucene_runtime_config(resolved_config)


def test_config_and_environment_may_supply_the_explicit_pair(
    tmp_path, monkeypatch
):
    _configure_test_installation(tmp_path, monkeypatch)
    cuvs_java = _write_artifact(tmp_path / "configured-cuvs-java.jar")
    cuvs_lucene = _write_artifact(tmp_path / "configured-cuvs-lucene.jar")
    monkeypatch.setenv("CUVS_LUCENE_JAR", str(cuvs_lucene))

    resolved = runtime_config.resolve_pylucene_runtime_config(
        {"cuvs_java_jar": cuvs_java}
    )

    assert resolved["cuvs_java_jar"] == str(cuvs_java.resolve())
    assert resolved["cuvs_lucene_jar"] == str(cuvs_lucene.resolve())


@pytest.mark.parametrize(
    "config",
    [
        {"cuvs_java_jar": "configured.jar", "cuvs_lucene_jar": None},
        {"cuvs_java_jar": None, "cuvs_lucene_jar": "configured.jar"},
    ],
    ids=["null-lucene-partner", "null-java-partner"],
)
def test_null_artifact_path_does_not_complete_an_explicit_pair(
    config, tmp_path, monkeypatch
):
    _configure_test_installation(tmp_path, monkeypatch)
    resolved_config = {
        key: _write_artifact(tmp_path / value) if value else None
        for key, value in config.items()
    }

    with pytest.raises(RuntimeError, match="overrides must provide both"):
        runtime_config.resolve_pylucene_runtime_config(resolved_config)


def test_null_config_path_does_not_mask_environment_override(
    tmp_path, monkeypatch
):
    _configure_test_installation(tmp_path, monkeypatch)
    cuvs_java = _write_artifact(tmp_path / "environment-cuvs-java.jar")
    cuvs_lucene = _write_artifact(tmp_path / "environment-cuvs-lucene.jar")
    monkeypatch.setenv("CUVS_LUCENE_CUVS_JAVA_JAR", str(cuvs_java))
    monkeypatch.setenv("CUVS_LUCENE_JAR", str(cuvs_lucene))

    resolved = runtime_config.resolve_pylucene_runtime_config(
        {"cuvs_java_jar": None, "cuvs_lucene_jar": None}
    )

    assert resolved["cuvs_java_jar"] == str(cuvs_java.resolve())
    assert resolved["cuvs_lucene_jar"] == str(cuvs_lucene.resolve())


def test_source_build_artifacts_take_precedence_over_maven_repository(
    tmp_path, monkeypatch
):
    repository, maven_repository = _configure_test_installation(
        tmp_path, monkeypatch
    )
    source_java, maven_java = runtime_config._artifact_candidates(
        "cuvs_java_jar"
    )
    source_lucene, maven_lucene = runtime_config._artifact_candidates(
        "cuvs_lucene_jar"
    )
    assert source_java.is_relative_to(repository)
    assert maven_java.is_relative_to(maven_repository)
    _write_artifact(source_java)
    _write_artifact(source_lucene)
    _write_artifact(maven_java)
    _write_artifact(maven_lucene)

    resolved = runtime_config.resolve_pylucene_runtime_config()

    assert resolved["cuvs_java_jar"] == str(source_java.resolve())
    assert resolved["cuvs_lucene_jar"] == str(source_lucene.resolve())


def test_maven_artifacts_are_used_when_source_build_is_absent(
    tmp_path, monkeypatch
):
    _, _ = _configure_test_installation(tmp_path, monkeypatch)
    _, maven_java = runtime_config._artifact_candidates("cuvs_java_jar")
    _, maven_lucene = runtime_config._artifact_candidates("cuvs_lucene_jar")
    _write_artifact(maven_java)
    _write_artifact(maven_lucene)

    resolved = runtime_config.resolve_pylucene_runtime_config()

    assert resolved["cuvs_java_jar"] == str(maven_java.resolve())
    assert resolved["cuvs_lucene_jar"] == str(maven_lucene.resolve())


def test_partial_source_build_does_not_mix_with_maven_artifacts(
    tmp_path, monkeypatch
):
    _configure_test_installation(tmp_path, monkeypatch)
    source_java, maven_java = runtime_config._artifact_candidates(
        "cuvs_java_jar"
    )
    _, maven_lucene = runtime_config._artifact_candidates("cuvs_lucene_jar")
    _write_artifact(source_java)
    _write_artifact(maven_java)
    _write_artifact(maven_lucene)

    resolved = runtime_config.resolve_pylucene_runtime_config()

    assert resolved["cuvs_java_jar"] == str(maven_java.resolve())
    assert resolved["cuvs_lucene_jar"] == str(maven_lucene.resolve())


def test_missing_artifact_error_lists_the_supported_resolution_paths(
    tmp_path, monkeypatch
):
    _configure_test_installation(tmp_path, monkeypatch)

    with pytest.raises(RuntimeError) as error:
        runtime_config.resolve_pylucene_runtime_config()

    message = str(error.value)
    assert "could not find a complete, matching" in message
    assert "CUVS_LUCENE_CUVS_JAVA_JAR" in message
    assert "java/cuvs-java/target" in message
    assert ".m2" not in message
    assert str(tmp_path / "maven") in message


def test_native_library_discovery_uses_active_python_environment(
    tmp_path, monkeypatch
):
    environment_prefix = tmp_path / "environment"
    native_directory = environment_prefix / "lib"
    _write_artifact(native_directory / "libcuvs_c.so")
    monkeypatch.setattr(runtime_config.sys, "prefix", str(environment_prefix))
    monkeypatch.setattr(
        runtime_config.sys, "base_prefix", str(environment_prefix)
    )
    monkeypatch.setattr(runtime_config.sys, "path", [])
    for name in (
        "JAVA_LIBRARY_PATH",
        "LD_LIBRARY_PATH",
        "CONDA_PREFIX",
        "CUVS_HOME",
        "CUDA_HOME",
        "CUDA_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        runtime_config, "_REPOSITORY_ROOT", tmp_path / "empty-source"
    )

    discovered = runtime_config._discover_native_library_path()

    assert discovered is not None
    discovered_directories = discovered.split(os.pathsep)
    assert discovered_directories[0] == str(native_directory.resolve())
    assert all(
        Path(directory) == native_directory.resolve()
        or next(Path(directory).glob("libcudart.so*"), None) is not None
        for directory in discovered_directories
    )


def test_native_library_discovery_requires_java_entrypoint_library(
    tmp_path, monkeypatch
):
    environment_prefix = tmp_path / "environment"
    _write_artifact(environment_prefix / "lib" / "libcuvs.so")
    monkeypatch.setattr(runtime_config.sys, "prefix", str(environment_prefix))
    monkeypatch.setattr(
        runtime_config.sys, "base_prefix", str(environment_prefix)
    )
    monkeypatch.setattr(runtime_config.sys, "path", [])
    monkeypatch.setattr(
        runtime_config, "_REPOSITORY_ROOT", tmp_path / "empty-source"
    )
    for name in (
        "JAVA_LIBRARY_PATH",
        "LD_LIBRARY_PATH",
        "CONDA_PREFIX",
        "CUVS_HOME",
        "CUDA_HOME",
        "CUDA_PATH",
    ):
        monkeypatch.delenv(name, raising=False)

    assert runtime_config._discover_native_library_path() is None


def test_cuvs_home_native_group_does_not_mix_lower_priority_installations(
    tmp_path, monkeypatch
):
    cuvs_home = tmp_path / "selected"
    selected_c = cuvs_home / "cpp" / "build" / "c"
    selected_cpp = cuvs_home / "cpp" / "build"
    _write_artifact(selected_c / "libcuvs_c.so")
    _write_artifact(selected_cpp / "libcuvs.so")
    source_build = tmp_path / "source" / "cpp" / "build"
    _write_artifact(source_build / "c" / "libcuvs_c.so")
    environment_prefix = tmp_path / "environment"
    _write_artifact(environment_prefix / "lib" / "libcuvs_c.so")
    monkeypatch.setenv("CUVS_HOME", str(cuvs_home))
    monkeypatch.delenv("JAVA_LIBRARY_PATH", raising=False)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setattr(
        runtime_config, "_REPOSITORY_ROOT", tmp_path / "source"
    )
    monkeypatch.setattr(runtime_config.sys, "prefix", str(environment_prefix))
    monkeypatch.setattr(
        runtime_config.sys, "base_prefix", str(environment_prefix)
    )
    monkeypatch.setattr(runtime_config.sys, "path", [])

    discovered = runtime_config._discover_native_library_path()

    assert discovered is not None
    discovered_directories = discovered.split(os.pathsep)
    assert discovered_directories[:2] == [
        str(selected_c.resolve()),
        str(selected_cpp.resolve()),
    ]
    assert str(source_build.resolve()) not in discovered_directories
    assert str((environment_prefix / "lib").resolve()) not in (
        discovered_directories
    )


def test_native_library_configuration_precedence(tmp_path, monkeypatch):
    explicit_java_path = tmp_path / "java-native"
    explicit_ld_path = tmp_path / "ld-native"
    monkeypatch.setenv("JAVA_LIBRARY_PATH", str(explicit_java_path))
    monkeypatch.setenv("LD_LIBRARY_PATH", str(explicit_ld_path))

    assert runtime_config._discover_native_library_path() == str(
        explicit_java_path
    )

    monkeypatch.delenv("JAVA_LIBRARY_PATH")
    assert runtime_config._discover_native_library_path() == str(
        explicit_ld_path
    )
