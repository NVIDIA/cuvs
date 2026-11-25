#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gather cuVS Java build outputs into one Maven coordinate per module."""

import argparse
import filecmp
import re
import shutil
import xml.etree.ElementTree as ET
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
    return ET.fromstring(text)  # noqa: S314 - Entity declarations rejected.


def read_coordinates(root):
    """Read the module's coordinates rather than a dependency's coordinates."""
    return tuple(
        root.findtext(f"m:{key}", namespaces=NS)
        for key in ("groupId", "artifactId", "version")
    )


def cuda_builds(module, artifacts_dir):
    """Find module builds in numeric CUDA order."""
    builds = []
    for path in artifacts_dir.iterdir():
        match = re.fullmatch(rf"{module}-cuda([0-9]+(?:\.[0-9]+)*)", path.name)
        if not path.is_dir() or not match:
            raise ValueError(f"Unexpected build artifact: {path.name}")
        builds.append((tuple(map(int, match[1].split("."))), path))
    if not builds:
        raise ValueError("No module build artifacts were downloaded")
    return sorted(builds)


def copy_repository(files, output_dir):
    """Copy selected artifacts without leaving a partial repository on failure."""
    if output_dir.exists():
        raise ValueError("Output directory must not already exist")
    try:
        for relative, source in files.items():
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def assemble(module, tag, artifacts_dir, output_dir):
    """Select common jars explicitly and preserve all native classifiers."""
    match = re.fullmatch(r"v([0-9]{2})\.([0-9]{2})\.([0-9]+)", tag)
    if not match or module not in GROUPS:
        raise ValueError("A registered module and release tag are required")
    major, minor, patch = match.groups()
    version = f"{major}.{minor}.{int(patch)}"
    if module == "cuvs-lucene":
        return assemble_lucene(version, artifacts_dir, output_dir)
    builds = cuda_builds(module, artifacts_dir)
    for _, path in builds:
        if read_coordinates(read_pom(path / "pom.xml")) != (
            GROUPS[module],
            module,
            version,
        ):
            raise ValueError(f"POM does not match {module}:{version}")
    # Keep the established higher-CUDA selection for the unclassified release
    # artifact, sources, javadoc, and POM. Never merge same-named jars blindly.
    primary = builds[-1][1]
    selected = {}
    for suffix in (".jar", "-sources.jar", "-javadoc.jar"):
        name = f"{module}-{version}{suffix}"
        selected[name] = primary / name
    selected[f"{module}-{version}.pom"] = primary / "pom.xml"
    for cuda, path in builds:
        natives = list(path.glob(f"{module}-{version}-*-cuda{cuda[0]}.jar"))
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
            raise ValueError(f"Conflicting native classifier: {native.name}")
        selected[native.name] = native
    for name, source in selected.items():
        if not source.is_file():
            raise ValueError(f"Missing release artifact: {source}")
        if name.endswith(".jar") and not zipfile.is_zipfile(source):
            raise ValueError(f"Invalid JAR: {source}")
    relative = Path(GROUPS[module].replace(".", "/")) / module / version
    copy_repository(
        {relative / name: source for name, source in selected.items()},
        output_dir,
    )
    print(f"Primary build: {primary.name}")
    print(f"Assembled {GROUPS[module]}:{module}:{version}")
    return output_dir / relative


def assemble_lucene(version, artifacts_dir, output_dir):
    """Publish the complete reactor, including parent and relocation POMs."""
    group = GROUPS["cuvs-lucene"]
    builds = cuda_builds("cuvs-lucene", artifacts_dir)
    # Flattened POMs omit the reactor list. Use the release source as
    # the authoritative module inventory so a missing CI output fails.
    modules = read_pom(
        Path(__file__).resolve().parents[1] / "cuvs-lucene/pom.xml"
    ).findall("m:modules/m:module", NS)
    names = [element.text for element in modules]
    if (
        not names
        or any(
            name != "relocation"
            and not re.fullmatch(r"lucene-[0-9]+\.[0-9]+", name or "")
            for name in names
        )
        or len(set(names)) != len(names)
    ):
        raise ValueError("Invalid Lucene reactor modules")
    inventory = {"cuvs-lucene-parent", "cuvs-lucene"} | {
        f"cuvs-{name}" for name in names if name != "relocation"
    }
    if "relocation" not in names:
        raise ValueError("Missing Lucene relocation module")
    selected = {}
    for _, build in builds:
        targets = [("cuvs-lucene-parent", build / "target")]
        targets.extend(
            (
                "cuvs-lucene" if name == "relocation" else f"cuvs-{name}",
                build / name / "target",
            )
            for name in names
        )
        current = {}
        for artifact, target in targets:
            pom = target / "pom.xml"
            root = read_pom(pom)
            if read_coordinates(root) != (group, artifact, version):
                raise ValueError(f"POM does not match {artifact}:{version}")
            packaging = root.findtext(
                "m:packaging", default="jar", namespaces=NS
            )
            files = {f"{artifact}-{version}.pom": pom}
            if artifact in {"cuvs-lucene-parent", "cuvs-lucene"}:
                if packaging != "pom":
                    raise ValueError(f"Expected POM-only artifact: {artifact}")
            else:
                if packaging != "jar":
                    raise ValueError(f"Expected JAR artifact: {artifact}")
                for suffix in (".jar", "-sources.jar", "-javadoc.jar"):
                    name = f"{artifact}-{version}{suffix}"
                    files[name] = target / name
                assembly = (
                    target / f"{artifact}-{version}-jar-with-dependencies.jar"
                )
                files[assembly.name] = assembly
            if artifact != "cuvs-lucene":
                dependencies = root.findall("m:dependencies/m:dependency", NS)
                if not any(
                    dependency.findtext("m:groupId", namespaces=NS)
                    == GROUPS["cuvs-java"]
                    and dependency.findtext("m:artifactId", namespaces=NS)
                    == "cuvs-java"
                    and dependency.findtext("m:version", namespaces=NS)
                    == version
                    for dependency in dependencies
                ):
                    raise ValueError(
                        "Lucene must depend on the matching Java release"
                    )
            else:
                relocation = root.find(
                    "m:distributionManagement/m:relocation", NS
                )
                if relocation is None or (
                    relocation.findtext("m:artifactId", namespaces=NS)
                    not in inventory - {"cuvs-lucene", "cuvs-lucene-parent"}
                    or relocation.findtext(
                        "m:groupId", default=group, namespaces=NS
                    )
                    != group
                    or relocation.findtext(
                        "m:version", default=version, namespaces=NS
                    )
                    != version
                ):
                    raise ValueError("Invalid Lucene relocation target")
            for name, source in files.items():
                if not source.is_file():
                    raise ValueError(f"Missing release artifact: {source}")
                if name.endswith(".jar") and not zipfile.is_zipfile(source):
                    raise ValueError(f"Invalid JAR: {source}")
                current[
                    Path(group.replace(".", "/")) / artifact / version / name
                ] = source
        selected = current
    copy_repository(selected, output_dir)
    return output_dir / group.replace(".", "/")


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
