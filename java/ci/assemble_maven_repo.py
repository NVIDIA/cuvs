#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gather cuVS Java build outputs into one Maven coordinate per module."""

import argparse
import filecmp
import re
import shutil
import xml.etree.ElementTree as ET  # nosec B405: DTDs are rejected below.
import zipfile
from pathlib import Path

GROUPS = {
    "cuvs-java": "com.nvidia.cuvs",
    "cuvs-lucene": "com.nvidia.cuvs.lucene",
}
NS = {"m": "http://maven.apache.org/POM/4.0.0"}


def read_pom(pom):
    """Read direct POM coordinates without accepting entity declarations."""
    text = pom.read_text()
    if "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
        raise ValueError("POM entity declarations are unsupported")
    return ET.fromstring(text)  # nosec B314: Entity declarations rejected.


def read_coordinates(pom):
    """Read the module's coordinates rather than a dependency's coordinates."""
    root = read_pom(pom)
    return tuple(
        root.findtext(f"m:{key}", namespaces=NS)
        for key in ("groupId", "artifactId", "version")
    )


def assemble(module, tag, artifacts_dir, output_dir):
    """Select common jars explicitly and preserve all native classifiers."""
    match = re.fullmatch(r"v([0-9]{2})\.([0-9]{2})\.([0-9]+)", tag)
    if not match or module not in GROUPS:
        raise ValueError("A registered module and release tag are required")
    major, minor, patch = match.groups()
    version = f"{major}.{minor}.{int(patch)}"
    builds = []
    for path in artifacts_dir.iterdir():
        cuda = re.fullmatch(
            rf"{re.escape(module)}-cuda([0-9]+(?:\.[0-9]+)*)", path.name
        )
        if not path.is_dir() or not cuda:
            raise ValueError(f"Unexpected build artifact: {path.name}")
        cuda_version = tuple(map(int, cuda[1].split(".")))
        if read_coordinates(path / "pom.xml") != (
            GROUPS[module],
            module,
            version,
        ):
            raise ValueError(f"POM does not match {module}:{version}")
        builds.append((cuda_version, path))
        if module == "cuvs-lucene":
            dependencies = [
                dependency
                for dependency in read_pom(path / "pom.xml").findall(
                    "m:dependencies/m:dependency", NS
                )
                if dependency.findtext("m:groupId", namespaces=NS)
                == GROUPS["cuvs-java"]
                and dependency.findtext("m:artifactId", namespaces=NS)
                == "cuvs-java"
            ]
            if (
                len(dependencies) != 1
                or dependencies[0].findtext("m:version", namespaces=NS)
                != version
            ):
                raise ValueError(
                    "Lucene must depend on the matching Java release"
                )
    if not builds:
        raise ValueError("No module build artifacts were downloaded")
    # Keep the established higher-CUDA selection for the unclassified release
    # artifact, sources, javadoc, and POM. Never merge same-named jars blindly.
    builds.sort(key=lambda entry: entry[0])
    primary = builds[-1][1]
    selected = {}
    for suffix in (".jar", "-sources.jar", "-javadoc.jar"):
        name = f"{module}-{version}{suffix}"
        selected[name] = primary / name
    selected[f"{module}-{version}.pom"] = primary / "pom.xml"
    if module == "cuvs-java":
        for cuda, path in builds:
            natives = list(
                path.glob(f"{module}-{version}-*-cuda{cuda[0]}.jar")
            )
            if len(natives) != 1:
                raise ValueError(f"Expected one native classifier in {path}")
            native = natives[0]
            if not re.fullmatch(
                rf"{module}-{re.escape(version)}-"
                rf"(x86_64|aarch64)-cuda{cuda[0]}\.jar",
                native.name,
            ):
                raise ValueError(f"Invalid native classifier: {native.name}")
            previous = selected.get(native.name)
            if previous and not filecmp.cmp(previous, native, shallow=False):
                raise ValueError(
                    f"Conflicting native classifier: {native.name}"
                )
            selected[native.name] = native
    else:
        assembled = primary / f"{module}-{version}-jar-with-dependencies.jar"
        if assembled.is_file():
            selected[assembled.name] = assembled
    for name, source in selected.items():
        if not source.is_file():
            raise ValueError(f"Missing release artifact: {source}")
        if name.endswith(".jar") and not zipfile.is_zipfile(source):
            raise ValueError(f"Invalid JAR: {source}")
    if output_dir.exists():
        raise ValueError("Output directory must not already exist")
    destination = output_dir / GROUPS[module].replace(".", "/")
    destination = destination / module / version
    destination.mkdir(parents=True)
    try:
        for name, source in selected.items():
            shutil.copyfile(source, destination / name)
    except Exception:
        shutil.rmtree(output_dir)
        raise
    print(f"Primary build: {primary.name}")
    print(f"Assembled {GROUPS[module]}:{module}:{version}")
    return destination


def main():
    """Assemble artifacts downloaded from the current release workflow run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", choices=GROUPS, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    assemble(args.module, args.tag, args.artifacts_dir, args.output_dir)


if __name__ == "__main__":
    main()
