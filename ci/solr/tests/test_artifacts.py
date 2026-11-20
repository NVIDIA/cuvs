# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise artifact selection and checks that prevent false-green GPU CI."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "artifacts", Path(__file__).parents[1] / "artifacts.py"
)
artifacts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(artifacts)


class ArtifactsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.solr = self.root / "solr"
        (self.solr / "gradle").mkdir(parents=True)
        (self.solr / "gradle/libs.versions.toml").write_text(
            '[versions]\napache-lucene = "10.4.0"\n'
        )
        self.java = self.root / "java"
        self.lucene = self.root / "lucene"
        self.repo = self.root / "maven"
        self.results = self.root / "results"
        self.results.mkdir()
        self.jar(self.java, "com.nvidia.cuvs", "cuvs-java")
        for minor in ("10.2", "10.4"):
            self.jar(
                self.lucene / f"lucene-{minor}/target",
                "com.nvidia.cuvs.lucene",
                f"cuvs-lucene-{minor}",
            )

    def jar(self, directory, group, name, version="26.12.0"):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "pom.xml").write_text(
            '<project xmlns="http://maven.apache.org/POM/4.0.0">'
            f"<groupId>{group}</groupId><artifactId>{name}</artifactId>"
            f"<version>{version}</version></project>"
        )
        (directory / f"{name}-{version}.jar").write_bytes(name.encode())
        (directory / f"{name}-{version}-sources.jar").write_bytes(
            b"not a binary"
        )

    def stage(self):
        return artifacts.stage(
            self.solr, self.java, self.lucene, self.repo, self.results
        )

    def test_selects_matching_minor_and_plain_java_jar(self):
        config = self.stage()
        self.assertIn(":cuvs-lucene-10.4:", config["lucene"]["coordinate"])
        self.assertEqual(
            Path(config["java"]["jar"]).read_bytes(), b"cuvs-java"
        )

    def test_missing_matching_minor_fails_instead_of_using_legacy(self):
        (self.lucene / "lucene-10.4/target/pom.xml").unlink()
        with self.assertRaises(FileNotFoundError):
            self.stage()

    def test_mismatched_java_lucene_versions_fail(self):
        self.jar(self.java, "com.nvidia.cuvs", "cuvs-java", "26.10.0")
        with self.assertRaisesRegex(ValueError, "versions differ"):
            self.stage()

    def report(
        self, body="", classname="org.apache.solr.cuvs.TestCuVSCodecSupportIT"
    ):
        (self.results / "TEST-cuvs.xml").write_text(
            f'<testsuite><testcase classname="{classname}" '
            f'name="testIndexAndSearch">{body}</testcase></testsuite>'
        )

    def test_passed_indexing_test(self):
        self.report()
        self.assertEqual(
            artifacts.verify_tests(self.results)["indexing_tests_passed"], 1
        )

    def test_missing_skipped_failed_and_unrelated_tests_fail(self):
        with self.assertRaises(ValueError):
            artifacts.verify_tests(self.results)
        for body in ('<skipped message="No GPU"/>', "<failure/>", "<error/>"):
            with self.subTest(body=body):
                self.report(body)
                with self.assertRaises(ValueError):
                    artifacts.verify_tests(self.results)
        self.report(classname="UnrelatedTest")
        with self.assertRaises(ValueError):
            artifacts.verify_tests(self.results)

    def test_cpu_fallback_fails(self):
        self.report()
        (self.results / "outputs").mkdir()
        (self.results / "outputs/OUTPUT-cuvs.txt").write_text(
            "GPU based indexing not supported, falling back to CPU"
        )
        with self.assertRaisesRegex(ValueError, "CPU fallback"):
            artifacts.verify_tests(self.results)

    def test_cpu_fallback_in_junit_output_fails(self):
        self.report()
        report = self.results / "TEST-cuvs.xml"
        report.write_text(
            report.read_text().replace(
                "</testsuite>",
                "<system-out>GPU based indexing not supported</system-out></testsuite>",
            )
        )
        with self.assertRaisesRegex(ValueError, "CPU fallback"):
            artifacts.verify_tests(self.results)

    def test_distribution_rejects_missing_extra_or_stale_jars(self):
        config = self.stage()
        distribution = self.root / "distribution"
        distribution.mkdir()
        with self.assertRaises(ValueError):
            artifacts.verify_distribution(config, distribution)
        for key in ("java", "lucene"):
            jar = Path(config[key]["jar"])
            (distribution / jar.name).write_bytes(jar.read_bytes())
        artifacts.verify_distribution(config, distribution)
        extra = distribution / "cuvs-java-old.jar"
        extra.write_bytes(b"old")
        with self.assertRaises(ValueError):
            artifacts.verify_distribution(config, distribution)
        extra.unlink()
        (distribution / Path(config["java"]["jar"]).name).write_bytes(b"old")
        with self.assertRaisesRegex(ValueError, "different cuVS JAR"):
            artifacts.verify_distribution(config, distribution)


if __name__ == "__main__":
    unittest.main()
