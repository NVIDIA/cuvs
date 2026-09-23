#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for immutable JVM setup and Java artifact provenance."""

import hashlib
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cuvs_bench.backends import _lucene_runtime
from cuvs_bench.backends._lucene_runtime import (
    _CleanupStack,
    _load_pylucene,
    _rollback_writer,
    _validate_artifacts,
    CagraIndexVerifier,
    initialize_pylucene,
    LuceneRuntime,
)
from cuvs_bench.backends._lucene_runtime_config import maven_artifact_version


_ARTIFACT_VERSION = maven_artifact_version()


def _properties(group: str, artifact: str, version: str) -> bytes:
    return (
        f"groupId={group}\nartifactId={artifact}\nversion={version}\n"
    ).encode()


def _write_artifacts(
    directory: Path,
    *,
    java_version: str = _ARTIFACT_VERSION,
    lucene_version: str = _ARTIFACT_VERSION,
) -> tuple[Path, Path]:
    java_jar = directory / "cuvs-java.jar"
    with zipfile.ZipFile(java_jar, "w") as archive:
        archive.writestr("META-INF/MANIFEST.MF", "Multi-Release: true\n")
        archive.writestr("com/nvidia/cuvs/CagraIndex.class", b"")
        archive.writestr("com/nvidia/cuvs/CuVSResources.class", b"")
        archive.writestr(
            "META-INF/versions/22/com/nvidia/cuvs/spi/JDKProvider.class", b""
        )
        archive.writestr(
            "META-INF/maven/com.nvidia.cuvs/cuvs-java/pom.properties",
            _properties("com.nvidia.cuvs", "cuvs-java", java_version),
        )

    lucene_jar = directory / "cuvs-lucene.jar"
    with zipfile.ZipFile(lucene_jar, "w") as archive:
        archive.writestr(
            "com/nvidia/cuvs/lucene/CuVS2510GPUVectorsFormat.class", b""
        )
        archive.writestr(
            "com/nvidia/cuvs/lucene/CuVS2510GPUSearchCodec.class", b""
        )
        archive.writestr(
            "com/nvidia/cuvs/lucene/IndexSearcherTimingBridge.class", b""
        )
        archive.writestr(
            "com/nvidia/cuvs/lucene/Lucene101AcceleratedHNSWCodec.class", b""
        )
        archive.writestr(
            "META-INF/services/org.apache.lucene.codecs.Codec",
            "com.nvidia.cuvs.lucene.CuVS2510GPUSearchCodec\n"
            "com.nvidia.cuvs.lucene.Lucene101AcceleratedHNSWCodec\n",
        )
        archive.writestr(
            "META-INF/maven/com.nvidia.cuvs.lucene/cuvs-lucene/pom.properties",
            _properties(
                "com.nvidia.cuvs.lucene", "cuvs-lucene", lucene_version
            ),
        )
    return java_jar, lucene_jar


class _FakeEnvironment:
    def attachCurrentThread(self) -> None:
        pass


class _FakeLucene:
    VERSION = "10.2.0"
    CLASSPATH = "/pylucene/lucene-core.jar"

    def __init__(self) -> None:
        self.environment = None
        self.initializations: list[tuple[str, tuple[str, ...]]] = []

    def getVMEnv(self):
        return self.environment

    def initVM(self, *, classpath, vmargs):
        self.initializations.append((classpath, tuple(vmargs)))
        self.environment = _FakeEnvironment()
        return self.environment


def _reset_jvm_state(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "_INITIALIZED_CLASSPATH",
        "_INITIALIZED_VMARGS",
        "_INITIALIZED_ARTIFACT_PROVENANCE",
        "_INITIALIZED_ARTIFACT_TOKENS",
    ):
        monkeypatch.setattr(_lucene_runtime, name, None)


def test_java_search_timer_reports_class_loading_failure() -> None:
    class MissingBridgeClass:
        @staticmethod
        def forName(_name: str):
            raise RuntimeError("unsupported bridge bytecode")

    runtime = object.__new__(LuceneRuntime)
    runtime.Class = MissingBridgeClass

    with pytest.raises(RuntimeError) as failure:
        runtime._load_java_search_timer()

    assert str(failure.value) == (
        "Could not load or adapt "
        "com.nvidia.cuvs.lucene.IndexSearcherTimingBridge through "
        "PyLucene/JCC: RuntimeError: unsupported bridge bytecode"
    )
    assert isinstance(failure.value.__cause__, RuntimeError)


def test_java_search_timer_reports_jcc_adaptation_failure() -> None:
    class LoadedBridgeClass:
        @staticmethod
        def newInstance():
            return object()

    class LoadableClass:
        @staticmethod
        def forName(_name: str):
            return LoadedBridgeClass()

    class UnsupportedFunction:
        @staticmethod
        def cast_(_instance):
            raise TypeError("Function.cast_ rejected bridge")

    runtime = object.__new__(LuceneRuntime)
    runtime.Class = LoadableClass
    runtime.Function = UnsupportedFunction

    with pytest.raises(RuntimeError) as failure:
        runtime._load_java_search_timer()

    assert "TypeError: Function.cast_ rejected bridge" in str(failure.value)
    assert isinstance(failure.value.__cause__, TypeError)


def test_float32_vectors_are_converted_to_a_jcc_compatible_sequence() -> None:
    received = []

    class RecordingLucene:
        @staticmethod
        def JArray(element_type: str):
            assert element_type == "float"

            def record(values):
                received.append(values)
                return values

            return record

    runtime = object.__new__(LuceneRuntime)
    runtime.lucene = RecordingLucene()
    vector = np.asarray([1.25, -2.5], dtype=np.float32)

    converted = runtime._java_vector(vector)

    assert converted == [1.25, -2.5]
    assert received == [[1.25, -2.5]]
    assert all(type(value) is float for value in received[0])


def test_numeric_document_ids_preserve_score_order_across_leaves() -> None:
    class ForwardOnlyDocumentIds:
        def __init__(self, values: dict[int, int]) -> None:
            self.values = values
            self.current = -1
            self.visited = []

        def advanceExact(self, document_id: int) -> bool:
            assert document_id >= self.current
            self.current = document_id
            self.visited.append(document_id)
            return document_id in self.values

        def longValue(self) -> int:
            return self.values[self.current]

    class LeafReader:
        def __init__(
            self, max_doc: int, document_ids: ForwardOnlyDocumentIds
        ) -> None:
            self.max_doc = max_doc
            self.document_ids = document_ids

        def maxDoc(self) -> int:
            return self.max_doc

        def getNumericDocValues(self, field: str):
            assert field == "id"
            return self.document_ids

    class Leaf:
        def __init__(self, doc_base: int, reader: LeafReader) -> None:
            self.docBase = doc_base
            self._reader = reader

        def reader(self) -> LeafReader:
            return self._reader

    class Reader:
        def __init__(self, leaves) -> None:
            self._leaves = leaves

        def leaves(self):
            return self._leaves

    first_ids = ForwardOnlyDocumentIds({2: 102, 5: 105})
    second_ids = ForwardOnlyDocumentIds({2: 108})
    reader = Reader(
        [
            Leaf(0, LeafReader(6, first_ids)),
            Leaf(6, LeafReader(4, second_ids)),
        ]
    )
    score_docs = [
        SimpleNamespace(doc=8, score=0.9),
        SimpleNamespace(doc=2, score=0.8),
        SimpleNamespace(doc=5, score=0.7),
    ]

    hits = LuceneRuntime._materialize_hits(reader, score_docs)

    assert first_ids.visited == [2, 5]
    assert second_ids.visited == [2]
    assert hits == [
        _lucene_runtime.SearchHit(108, 0.9),
        _lucene_runtime.SearchHit(102, 0.8),
        _lucene_runtime.SearchHit(105, 0.7),
    ]


def test_missing_numeric_document_id_requests_an_index_rebuild() -> None:
    class MissingDocumentIds:
        @staticmethod
        def advanceExact(_document_id: int) -> bool:
            return False

    class LeafReader:
        @staticmethod
        def maxDoc() -> int:
            return 4

        @staticmethod
        def getNumericDocValues(_field: str):
            return MissingDocumentIds()

    class Leaf:
        docBase = 0

        @staticmethod
        def reader():
            return LeafReader()

    class Reader:
        @staticmethod
        def leaves():
            return [Leaf()]

    with pytest.raises(RuntimeError, match="rebuild the index with --force"):
        LuceneRuntime._materialize_hits(
            Reader(), [SimpleNamespace(doc=3, score=1.0)]
        )


def test_missing_numeric_document_id_field_requests_an_index_rebuild() -> None:
    class LeafReader:
        @staticmethod
        def maxDoc() -> int:
            return 1

        @staticmethod
        def getNumericDocValues(_field: str):
            return None

    class Leaf:
        docBase = 0

        @staticmethod
        def reader():
            return LeafReader()

    class Reader:
        @staticmethod
        def leaves():
            return [Leaf()]

    with pytest.raises(RuntimeError, match="rebuild the index with --force"):
        LuceneRuntime._materialize_hits(
            Reader(), [SimpleNamespace(doc=0, score=1.0)]
        )


def test_cagra_verifier_selects_only_the_current_noncompound_segment() -> None:
    class RootDirectory:
        @staticmethod
        def listAll():
            return ("_c.vemc", "_n.vemc")

    class SegmentInfo:
        @staticmethod
        def files():
            return ("_c.si", "_c.vcag", "_c.vemc")

    class Segment:
        info = SegmentInfo()

    assert CagraIndexVerifier._metadata_files(
        RootDirectory(), Segment(), compound=False
    ) == ["_c.vemc"]


def test_cagra_verifier_reads_metadata_inside_a_compound_segment() -> None:
    class CompoundDirectory:
        @staticmethod
        def listAll():
            return ("_c.fnm", "_c.vcag", "_c.vemc")

    class SegmentInfo:
        @staticmethod
        def files():
            return ("_c.cfe", "_c.cfs", "_c.si")

    class Segment:
        info = SegmentInfo()

    assert CagraIndexVerifier._metadata_files(
        CompoundDirectory(), Segment(), compound=True
    ) == ["_c.vemc"]


def test_missing_pylucene_reports_the_required_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_module(_name: str):
        raise ImportError("no lucene module")

    monkeypatch.setattr(
        _lucene_runtime.importlib, "import_module", missing_module
    )

    with pytest.raises(ImportError, match="requires PyLucene 10.2.0"):
        _load_pylucene()


def test_incompatible_pylucene_version_fails_before_vm_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_lucene = _FakeLucene()
    fake_lucene.VERSION = "10.1.0"
    _reset_jvm_state(monkeypatch)
    monkeypatch.setattr(_lucene_runtime, "_load_pylucene", lambda: fake_lucene)

    with pytest.raises(RuntimeError, match="expected 10.2.0, found 10.1.0"):
        initialize_pylucene({})

    assert fake_lucene.initializations == []


def test_artifact_validation_returns_exact_coordinates_paths_and_hashes(
    tmp_path: Path,
) -> None:
    java_jar, lucene_jar = _write_artifacts(tmp_path)

    provenance, tokens = _validate_artifacts(java_jar, lucene_jar)

    assert provenance == {
        "cuvs_java_coordinates": (
            f"com.nvidia.cuvs:cuvs-java:{_ARTIFACT_VERSION}"
        ),
        "cuvs_java_jar_path": str(java_jar),
        "cuvs_java_jar_sha256": hashlib.sha256(
            java_jar.read_bytes()
        ).hexdigest(),
        "cuvs_lucene_coordinates": (
            f"com.nvidia.cuvs.lucene:cuvs-lucene:{_ARTIFACT_VERSION}"
        ),
        "cuvs_lucene_jar_path": str(lucene_jar),
        "cuvs_lucene_jar_sha256": hashlib.sha256(
            lucene_jar.read_bytes()
        ).hexdigest(),
    }
    assert set(tokens) == {str(java_jar), str(lucene_jar)}


def test_artifact_validation_rejects_mismatched_versions(
    tmp_path: Path,
) -> None:
    java_jar, lucene_jar = _write_artifacts(
        tmp_path, lucene_version=f"{_ARTIFACT_VERSION}-mismatch"
    )

    with pytest.raises(RuntimeError, match="JAR versions differ"):
        _validate_artifacts(java_jar, lucene_jar)


def test_artifact_validation_rejects_a_native_assembled_jar(
    tmp_path: Path,
) -> None:
    java_jar, lucene_jar = _write_artifacts(tmp_path)
    with zipfile.ZipFile(java_jar, "a") as archive:
        archive.writestr("native/linux-x86_64/libcuvs.so", b"native")

    with pytest.raises(RuntimeError, match="embeds native libraries"):
        _validate_artifacts(java_jar, lucene_jar)


def test_artifact_validation_rejects_an_unreadable_jar(tmp_path: Path) -> None:
    java_jar, lucene_jar = _write_artifacts(tmp_path)
    java_jar.write_bytes(b"not a zip archive")

    with pytest.raises(RuntimeError, match="not a readable JAR"):
        _validate_artifacts(java_jar, lucene_jar)


def test_artifact_validation_rejects_a_lucene_assembled_jar(
    tmp_path: Path,
) -> None:
    java_jar, lucene_jar = _write_artifacts(tmp_path)
    with zipfile.ZipFile(lucene_jar, "a") as archive:
        archive.writestr("org/apache/lucene/index/IndexReader.class", b"")

    with pytest.raises(RuntimeError, match="bundles Lucene classes"):
        _validate_artifacts(java_jar, lucene_jar)


@pytest.mark.parametrize("control_error", (KeyboardInterrupt, SystemExit))
def test_cleanup_process_control_takes_precedence_over_an_ordinary_failure(
    control_error: type[BaseException],
) -> None:
    def interrupt_cleanup() -> None:
        raise control_error("cleanup interrupted")

    with pytest.raises(control_error) as failure:
        with _CleanupStack() as cleanups:
            cleanups.add("close test resource", interrupt_cleanup)
            raise RuntimeError("operation failed")

    assert failure.value.__notes__ == [
        "Resource handling first failed: RuntimeError: operation failed"
    ]


@pytest.mark.parametrize("control_error", (KeyboardInterrupt, SystemExit))
def test_writer_rollback_process_control_takes_precedence(
    control_error: type[BaseException],
) -> None:
    class InterruptingWriter:
        @staticmethod
        def rollback() -> None:
            raise control_error("rollback interrupted")

    with pytest.raises(control_error) as failure:
        _rollback_writer(InterruptingWriter(), RuntimeError("write failed"))

    assert failure.value.__notes__ == [
        "Lucene writer first failed: RuntimeError: write failed"
    ]


def test_reused_jvm_reports_the_initialized_artifact_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    java_jar, lucene_jar = _write_artifacts(tmp_path)
    fake_lucene = _FakeLucene()
    _reset_jvm_state(monkeypatch)
    monkeypatch.setattr(_lucene_runtime, "_load_pylucene", lambda: fake_lucene)
    config = {
        "cuvs_java_jar": str(java_jar),
        "cuvs_lucene_jar": str(lucene_jar),
    }

    first = initialize_pylucene(config)
    second = initialize_pylucene(config)

    assert len(fake_lucene.initializations) == 1
    assert second[1:] == first[1:]


def test_reused_jvm_rejects_a_same_path_artifact_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    java_jar, lucene_jar = _write_artifacts(tmp_path)
    fake_lucene = _FakeLucene()
    _reset_jvm_state(monkeypatch)
    monkeypatch.setattr(_lucene_runtime, "_load_pylucene", lambda: fake_lucene)
    config = {
        "cuvs_java_jar": str(java_jar),
        "cuvs_lucene_jar": str(lucene_jar),
    }
    initialize_pylucene(config)
    with zipfile.ZipFile(java_jar, "a") as archive:
        archive.writestr("replacement", b"changed")

    with pytest.raises(RuntimeError, match="changed after"):
        initialize_pylucene(config)


def test_reused_jvm_hashes_artifacts_when_stat_tokens_collide(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    java_jar, lucene_jar = _write_artifacts(tmp_path)
    fake_lucene = _FakeLucene()
    _reset_jvm_state(monkeypatch)
    monkeypatch.setattr(_lucene_runtime, "_load_pylucene", lambda: fake_lucene)
    config = {
        "cuvs_java_jar": str(java_jar),
        "cuvs_lucene_jar": str(lucene_jar),
    }
    initialize_pylucene(config)
    original_tokens = dict(_lucene_runtime._INITIALIZED_ARTIFACT_TOKENS or {})
    replacement = bytearray(java_jar.read_bytes())
    replacement[len(replacement) // 2] ^= 1
    java_jar.write_bytes(replacement)
    monkeypatch.setattr(
        _lucene_runtime,
        "_artifact_stat_token",
        lambda path: original_tokens[str(path)],
    )

    with pytest.raises(RuntimeError, match="changed after"):
        initialize_pylucene(config)


def test_artifact_hashing_occurs_once_per_operation_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = object.__new__(LuceneRuntime)
    runtime._artifact_tokens = {}
    runtime.artifact_provenance = {}
    runtime.lucene = _FakeLucene()
    runtime.lucene.environment = _FakeEnvironment()
    hash_verifications = []
    monkeypatch.setattr(
        _lucene_runtime,
        "_verify_artifact_tokens",
        lambda _tokens, _provenance: hash_verifications.append(True),
    )

    runtime.verify_artifacts()
    runtime.attach_current_thread()
    runtime.attach_current_thread()

    assert hash_verifications == [True]


def test_reused_jvm_rejects_different_vm_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    java_jar, lucene_jar = _write_artifacts(tmp_path)
    fake_lucene = _FakeLucene()
    _reset_jvm_state(monkeypatch)
    monkeypatch.setattr(_lucene_runtime, "_load_pylucene", lambda: fake_lucene)
    config = {
        "cuvs_java_jar": str(java_jar),
        "cuvs_lucene_jar": str(lucene_jar),
    }
    initialize_pylucene(config)

    with pytest.raises(
        RuntimeError, match="different classpath or JVM arguments"
    ):
        initialize_pylucene({**config, "jvm_args": ["-Xmx1g"]})
