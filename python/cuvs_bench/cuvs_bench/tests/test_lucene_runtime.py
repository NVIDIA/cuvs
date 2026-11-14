#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for immutable JVM setup and Java artifact provenance."""

import hashlib
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from cuvs_bench.backends import _lucene_runtime
from cuvs_bench.backends._lucene_runtime import (
    _CleanupStack,
    _load_pylucene,
    _rollback_writer,
    _validate_artifacts,
    CagraIndexVerifier,
    CagraVerificationError,
    initialize_pylucene,
    LuceneRuntime,
)
from cuvs_bench.backends._lucene_runtime_config import maven_artifact_version


_ARTIFACT_VERSION = maven_artifact_version()


@pytest.mark.parametrize("num_candidates", [-1, 0, 9])
@pytest.mark.parametrize(
    "query_count", [0, 1], ids=["empty-queries", "one-query"]
)
def test_search_rejects_fewer_candidates_than_requested_hits_before_using_java(
    tmp_path,
    num_candidates,
    query_count,
):
    # Only Java attachment is faked; call the real runtime entry point.
    runtime = object.__new__(LuceneRuntime)
    runtime.attach_current_thread = Mock()
    queries = np.zeros((query_count, 16), dtype=np.float32)

    with pytest.raises(ValueError) as failure:
        runtime.search_index(
            tmp_path, queries, k=10, num_candidates=num_candidates
        )

    assert (
        str(failure.value)
        == f"num_candidates={num_candidates} must be at least k=10"
    )
    runtime.attach_current_thread.assert_not_called()


def _properties(group: str, artifact: str, version: str) -> bytes:
    return (
        f"groupId={group}\nartifactId={artifact}\nversion={version}\n"
    ).encode()


def _write_artifacts(
    directory: Path,
    *,
    java_version: str = _ARTIFACT_VERSION,
    lucene_version: str = _ARTIFACT_VERSION,
    lucene_artifact: str = "cuvs-lucene-10.2",
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
            f"META-INF/maven/com.nvidia.cuvs.lucene/{lucene_artifact}/pom.properties",
            _properties(
                "com.nvidia.cuvs.lucene", lucene_artifact, lucene_version
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
    received_inputs = []
    fake_converter_output_object = object()

    class FakeLucene:
        @staticmethod
        def JArray(element_type: str):
            assert element_type == "float"

            def fake_converter(values):
                received_inputs.append(values)
                return fake_converter_output_object

            return fake_converter

    # Bypass normal runtime initialization and inject the fake JCC interface.
    runtime = object.__new__(LuceneRuntime)
    runtime.lucene = FakeLucene()
    vector = np.asarray([1.25, -2.5], dtype=np.float32)

    # Call the actual production function with the fake JCC converter.
    converted = runtime._java_vector(vector)

    # Check the numbers and Python float types passed into the converter.
    assert received_inputs == [[1.25, -2.5]]
    assert all(type(value) is float for value in received_inputs[0])
    # Ensure the production function returns the converter's output object.
    assert converted is fake_converter_output_object


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


def test_cagra_verifier_selects_metadata_filenames_from_compound_directory() -> (
    None
):
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


def _verify_cagra_payload_ranges(ranges: list[tuple[int, int]]) -> None:
    """Model recorded (offset, length) pairs relative to a 100-byte payload."""
    header_length, payload_length, footer_length = 32, 100, 16
    data_input = Mock()
    data_input.getFilePointer.return_value = header_length
    data_input.length.return_value = (
        header_length + payload_length + footer_length
    )
    directory = Mock()
    directory.openInput.return_value = data_input
    codec_util = Mock()
    codec_util.footerLength.return_value = footer_length
    runtime = SimpleNamespace(
        IOContext=SimpleNamespace(READONCE=object()), CodecUtil=codec_util
    )
    segment = SimpleNamespace(info=Mock(name="segment_info"))
    segment.info.name = "_0"
    fields = [
        SimpleNamespace(
            cagra_offset=header_length + offset, cagra_length=length
        )
        for offset, length in ranges
    ]

    try:
        CagraIndexVerifier(runtime)._verify_data(
            directory, segment, "_0.vemc", fields
        )
    finally:
        data_input.close.assert_called_once()

    codec_util.checksumEntireFile.assert_called_once_with(data_input)


@pytest.mark.parametrize(
    "ranges",
    [
        pytest.param([(0, 100)], id="contiguous"),
        pytest.param([(8, 92)], id="abandoned-merge-prefix"),
        pytest.param([(0, 40), (48, 52)], id="gap-between-fields"),
        pytest.param([(0, 92)], id="unreferenced-tail"),
    ],
)
def test_cagra_payload_verification_allows_unreferenced_bytes(ranges) -> None:
    # A failed native merge can leave bytes behind before the vector-based
    # rebuild writes its successful CAGRA payload at a later offset.
    _verify_cagra_payload_ranges(ranges)


@pytest.mark.parametrize(
    "ranges",
    [
        pytest.param([(-1, 101)], id="starts-in-header"),
        pytest.param([(0, 101)], id="extends-into-footer"),
        pytest.param([(0, 60), (50, 50)], id="overlapping-fields"),
        pytest.param([(0, 0)], id="empty-range"),
    ],
)
def test_cagra_payload_verification_rejects_invalid_ranges(ranges) -> None:
    with pytest.raises(CagraVerificationError):
        _verify_cagra_payload_ranges(ranges)


def test_missing_pylucene_reports_the_required_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_module(_name: str):
        raise ImportError("no lucene module")

    monkeypatch.setattr(
        _lucene_runtime.importlib, "import_module", missing_module
    )

    with pytest.raises(ImportError) as error:
        _load_pylucene()

    message = str(error.value)
    assert "requires the custom PyLucene 10.2.0 runtime" in message
    assert "Install and activate the optional Lucene runtime" in message
    # Keep documentation URLs and checkout-specific setup paths out of errors.
    assert "http" not in message
    assert ".md" not in message
    assert ".sh" not in message
    assert "PyLucene import failed: no lucene module" in message


@pytest.mark.parametrize("version", ("10.1.0", "10.3.2", "10.4.0", "10.5.0"))
def test_incompatible_pylucene_version_fails_before_vm_initialization(
    monkeypatch: pytest.MonkeyPatch,
    version: str,
) -> None:
    fake_lucene = _FakeLucene()
    fake_lucene.VERSION = version
    _reset_jvm_state(monkeypatch)
    monkeypatch.setattr(_lucene_runtime, "_load_pylucene", lambda: fake_lucene)

    with pytest.raises(RuntimeError) as error:
        initialize_pylucene({})

    message = str(error.value)
    assert f"expected 10.2.0, found {version}" in message
    assert "Install and activate a compatible PyLucene runtime" in message
    assert "http" not in message
    assert ".md" not in message
    assert ".sh" not in message
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
            f"com.nvidia.cuvs.lucene:cuvs-lucene-10.2:{_ARTIFACT_VERSION}"
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


@pytest.mark.parametrize("artifact", ("cuvs-lucene", "cuvs-lucene-10.3"))
def test_runtime_requires_the_lucene_10_2_artifact(
    tmp_path: Path,
    artifact: str,
) -> None:
    java_jar, lucene_jar = _write_artifacts(tmp_path, lucene_artifact=artifact)

    with pytest.raises(RuntimeError, match="cuvs-lucene-10.2/pom.properties"):
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

    _, first_provenance, first_tokens = initialize_pylucene(config)
    _, reused_provenance, reused_tokens = initialize_pylucene(config)

    assert len(fake_lucene.initializations) == 1
    assert reused_provenance == first_provenance
    assert reused_tokens == first_tokens


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


def test_thread_attachment_does_not_rehash_artifacts(
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
