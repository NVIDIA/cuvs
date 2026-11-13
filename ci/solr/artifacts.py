# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage nightly Maven artifacts and reject skipped downstream GPU tests."""

import argparse
import hashlib
import json
import re
import shutil
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path


def stage_jar(source, repository, group, artifact):
    """Install an artifact's flattened POM and plain JAR without rebuilding."""
    pom = source / "pom.xml"
    root = ET.parse(pom).getroot()
    ns = {"m": "http://maven.apache.org/POM/4.0.0"}
    coordinates = [
        root.findtext(f"m:{key}", namespaces=ns)
        for key in ("groupId", "artifactId", "version")
    ]
    if coordinates[:2] != [group, artifact]:
        raise ValueError(
            f"Unexpected Maven coordinates in {pom}: {coordinates}"
        )
    version = coordinates[2]
    if not version or not re.fullmatch(r"[A-Za-z0-9_.+-]+", version):
        raise ValueError(f"Missing or unresolved version in {pom}: {version}")
    if root.find("m:parent", ns) is not None:
        raise ValueError(f"Expected a standalone flattened POM: {pom}")
    jar = source / f"{artifact}-{version}.jar"
    destination = repository / group.replace(".", "/") / artifact / version
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pom, destination / f"{artifact}-{version}.pom")
    shutil.copyfile(jar, destination / jar.name)
    return {
        "coordinate": ":".join(coordinates),
        "version": version,
        "jar": str(destination / jar.name),
        "sha256": hashlib.sha256(jar.read_bytes()).hexdigest(),
    }


def stage(solr, java, lucene, repository, reports):
    """Choose the codec matching Solr's Lucene minor and stage both JARs."""
    with (solr / "gradle/libs.versions.toml").open("rb") as stream:
        catalog = tomllib.load(stream)
    version = catalog["versions"]["apache-lucene"]
    match = re.fullmatch(r"(\d+\.\d+)\.\d+(?:[-.].*)?", version)
    if not match:
        raise ValueError(f"Unrecognized Solr Lucene version: {version}")
    minor = match[1]
    config = {
        "repository": str(repository.resolve()),
        "reports": str(reports.resolve()),
        "solr_lucene_version": version,
        "java": stage_jar(java, repository, "com.nvidia.cuvs", "cuvs-java"),
        "lucene": stage_jar(
            lucene / f"lucene-{minor}" / "target",
            repository,
            "com.nvidia.cuvs.lucene",
            f"cuvs-lucene-{minor}",
        ),
    }
    if config["java"]["version"] != config["lucene"]["version"]:
        raise ValueError("cuVS Java and Lucene artifact versions differ")
    return config


def verify_tests(results):
    """Require a successful indexing test; assumptions/skips are failures."""
    cases = []
    for path in results.glob("TEST-*.xml"):
        root = ET.parse(path).getroot()
        if (
            root.find(".//failure") is not None
            or root.find(".//error") is not None
        ):
            raise ValueError(f"Failing Solr test report: {path}")
        cases.extend(
            case
            for case in root.iter("testcase")
            if case.get("classname")
            == "org.apache.solr.cuvs.TestCuVSCodecSupportIT"
            and case.get("name", "").split("[")[0] == "testIndexAndSearch"
        )
    if not cases or any(case.find("skipped") is not None for case in cases):
        raise ValueError("Solr GPU indexing test was missing or skipped")
    for path in (results / "outputs").glob("OUTPUT-*.txt"):
        if "GPU based indexing not supported" in path.read_text(
            errors="replace"
        ):
            raise ValueError(f"Solr used CPU fallback: {path}")
    return {"indexing_tests_passed": len(cases), "skipped": 0}


def verify_distribution(config, distribution):
    """Check that Solr packages exactly the two staged cuVS JARs."""
    expected = {
        Path(config[key]["jar"]).name: config[key]["sha256"]
        for key in ("java", "lucene")
    }
    jars = list(distribution.rglob("cuvs-*.jar"))
    if sorted(path.name for path in jars) != sorted(expected):
        raise ValueError(f"Unexpected cuVS JARs in Solr distribution: {jars}")
    for jar in jars:
        if hashlib.sha256(jar.read_bytes()).hexdigest() != expected[jar.name]:
            raise ValueError(f"Solr packaged a different cuVS JAR: {jar}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("stage")
    for name in ("solr", "java", "lucene", "repository", "reports"):
        prepare.add_argument(f"--{name}", type=Path, required=True)
    check = sub.add_parser("verify-tests")
    check.add_argument("results", type=Path)
    check_dist = sub.add_parser("verify-distribution")
    check_dist.add_argument("config", type=Path)
    check_dist.add_argument("distribution", type=Path)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "stage":
        print(json.dumps(stage(**args), indent=2))
    elif command == "verify-tests":
        print(json.dumps(verify_tests(args["results"]), indent=2))
    else:
        verify_distribution(
            json.loads(args["config"].read_text()), args["distribution"]
        )


if __name__ == "__main__":
    main()
