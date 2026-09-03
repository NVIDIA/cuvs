# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package-resource checks for the PyLucene backend."""

from importlib.resources import files
from pathlib import PurePosixPath


_REQUIRED_PYLUCENE_RESOURCES = (
    PurePosixPath("backends/_java/PyLuceneConfiguredHnswCodec.java"),
    PurePosixPath("config/algos/pylucene_cuvs_cagra.yaml"),
    PurePosixPath("config/algos/pylucene_cuvs_hnsw.yaml"),
)


def _resource_files(root, relative=PurePosixPath()):
    for child in root.iterdir():
        child_relative = relative / child.name
        if child.is_dir():
            yield from _resource_files(child, child_relative)
        elif child.is_file():
            yield child_relative


def test_pylucene_production_resources_are_packaged():
    package_root = files("cuvs_bench")

    for relative in _REQUIRED_PYLUCENE_RESOURCES:
        resource = package_root.joinpath(*relative.parts)
        assert resource.is_file(), f"missing package resource: {relative}"
        assert resource.read_bytes(), f"empty package resource: {relative}"


def test_pylucene_test_java_and_classes_are_not_packaged():
    payload = tuple(_resource_files(files("cuvs_bench")))
    test_java = tuple(
        relative
        for relative in payload
        if relative.parts[:2] == ("tests", "java")
    )
    class_files = tuple(
        relative for relative in payload if relative.suffix == ".class"
    )

    assert not test_java, (
        f"test-only Java resources were packaged: {test_java}"
    )
    assert not class_files, (
        f"compiled Java classes were packaged: {class_files}"
    )
