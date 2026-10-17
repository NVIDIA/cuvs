# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise real Maven layouts without building GPU libraries."""

import zipfile

import pytest

from java.ci.assemble_maven_repo import GROUPS, assemble


def make_jar(path, contents):
    """Write distinct, structurally valid artifacts for selection tests."""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("index.html", contents)


def make_build(root, module, cuda, version="26.12.0", dependency="26.12.0"):
    """Create a Maven target directory with the same layout as CI outputs."""
    build = root / f"{module}-cuda{cuda}"
    build.mkdir(parents=True)
    dependencies = ""
    if module == "cuvs-lucene":
        dependencies = (
            "<dependencies><dependency>"
            "<groupId>com.nvidia.cuvs</groupId>"
            "<artifactId>cuvs-java</artifactId>"
            f"<version>{dependency}</version>"
            "</dependency></dependencies>"
        )
    (build / "pom.xml").write_text(
        '<project xmlns="http://maven.apache.org/POM/4.0.0">'
        f"<groupId>{GROUPS[module]}</groupId>"
        f"<artifactId>{module}</artifactId><version>{version}</version>"
        f"{dependencies}</project>"
    )
    for suffix in (".jar", "-sources.jar", "-javadoc.jar", "-tests.jar"):
        make_jar(build / f"{module}-{version}{suffix}", cuda)
    if module == "cuvs-java":
        make_jar(
            build / f"{module}-{version}-x86_64-cuda{cuda.split('.')[0]}.jar",
            cuda,
        )
    else:
        make_jar(build / f"{module}-{version}-jar-with-dependencies.jar", cuda)
    (build / "unrelated.txt").write_text("not a Maven artifact")
    return build


def test_native_classifiers_and_explicit_primary_selection(tmp_path):
    """Preserve both CUDA classifiers while selecting one common artifact."""
    builds = tmp_path / "builds"
    make_build(builds, "cuvs-java", "12.9.2")
    primary = make_build(builds, "cuvs-java", "13.3.0")
    destination = assemble(
        "cuvs-java", "v26.12.00", builds, tmp_path / "maven"
    )
    assert destination.relative_to(tmp_path / "maven").as_posix() == (
        "com/nvidia/cuvs/cuvs-java/26.12.0"
    )
    assert (destination / "cuvs-java-26.12.0.jar").read_bytes() == (
        primary / "cuvs-java-26.12.0.jar"
    ).read_bytes()
    assert {path.name for path in destination.iterdir()} == {
        "cuvs-java-26.12.0.pom",
        "cuvs-java-26.12.0.jar",
        "cuvs-java-26.12.0-sources.jar",
        "cuvs-java-26.12.0-javadoc.jar",
        "cuvs-java-26.12.0-x86_64-cuda12.jar",
        "cuvs-java-26.12.0-x86_64-cuda13.jar",
    }


def test_lucene_layout_and_dependency_assembly(tmp_path):
    """Keep Lucene in its own coordinate with its attached assembly jar."""
    builds = tmp_path / "builds"
    make_build(builds, "cuvs-lucene", "12.9.2")
    make_build(builds, "cuvs-lucene", "13.3.0")
    destination = assemble(
        "cuvs-lucene", "v26.12.0", builds, tmp_path / "maven"
    )
    assert destination.relative_to(tmp_path / "maven").as_posix() == (
        "com/nvidia/cuvs/lucene/cuvs-lucene/26.12.0"
    )
    assert (destination / "cuvs-lucene-26.12.0.pom").is_file()
    assert (
        destination / "cuvs-lucene-26.12.0-jar-with-dependencies.jar"
    ).is_file()
    assert not (destination / "cuvs-lucene-26.12.0-tests.jar").exists()


@pytest.mark.parametrize("module", GROUPS)
def test_pom_tag_mismatch_is_rejected(tmp_path, module):
    """Fail before creating output when a build belongs to another version."""
    builds = tmp_path / "builds"
    make_build(builds, module, "13.3.0", version="26.10.0")
    with pytest.raises(ValueError, match="POM does not match"):
        assemble(module, "v26.12.0", builds, tmp_path / "maven")
    assert not (tmp_path / "maven").exists()


def test_native_classifier_collision_is_rejected(tmp_path):
    """Reject byte-different jars targeting the same native coordinate."""
    builds = tmp_path / "builds"
    make_build(builds, "cuvs-java", "13.2.0")
    make_build(builds, "cuvs-java", "13.3.0")
    with pytest.raises(ValueError, match="Conflicting native classifier"):
        assemble("cuvs-java", "v26.12.0", builds, tmp_path / "maven")


@pytest.mark.parametrize(
    "suffix", ["-sources.jar", "-javadoc.jar", "-x86_64-cuda13.jar"]
)
def test_missing_release_artifact_is_rejected(tmp_path, suffix):
    """Require the complete publication set before any upload."""
    builds = tmp_path / "builds"
    primary = make_build(builds, "cuvs-java", "13.3.0")
    (primary / f"cuvs-java-26.12.0{suffix}").unlink()
    with pytest.raises(ValueError):
        assemble("cuvs-java", "v26.12.0", builds, tmp_path / "maven")
    assert not (tmp_path / "maven").exists()


def test_lucene_dependency_version_must_match(tmp_path):
    """Reject a Lucene release that points at a different Java version."""
    builds = tmp_path / "builds"
    make_build(builds, "cuvs-lucene", "13.3.0", dependency="26.10.0")
    with pytest.raises(ValueError, match="matching Java release"):
        assemble("cuvs-lucene", "v26.12.0", builds, tmp_path / "maven")


def test_pom_entity_declaration_is_rejected(tmp_path):
    """Reject XML entity declarations before the XML parser is called."""
    builds = tmp_path / "builds"
    primary = make_build(builds, "cuvs-java", "13.3.0")
    pom = primary / "pom.xml"
    pom.write_text(
        '<!DOCTYPE project [<!ENTITY x "value">]>' + pom.read_text()
    )
    with pytest.raises(ValueError, match="entity declarations"):
        assemble("cuvs-java", "v26.12.0", builds, tmp_path / "maven")
