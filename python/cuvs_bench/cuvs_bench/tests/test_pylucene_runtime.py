#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""PyLucene JVM and Java-boundary unit tests."""

from __future__ import annotations

import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

import cuvs_bench.backends._pylucene_java as pylucene_java
import cuvs_bench.backends.pylucene as pylucene_backend
from cuvs_bench.backends.pylucene import _BuildCodec, _IndexTopology
from cuvs_bench.tests._pylucene_test_utils import _CAGRA_CODEC, _HNSW_CODEC


def _build_codec(
    java_codec=None,
    codec_name=_HNSW_CODEC,
    *,
    build_parameters=None,
) -> _BuildCodec:
    if build_parameters is None:
        build_parameters = (
            {"codec": _CAGRA_CODEC}
            if codec_name == _CAGRA_CODEC
            else {
                "codec": _HNSW_CODEC,
                "m": 32,
                "ef_construction": 32,
                "direct_single_segment": False,
            }
        )
    return _BuildCodec(
        codec_name=codec_name,
        java_codec=java_codec if java_codec is not None else object(),
        writer_policy=(
            "gpu-cagra"
            if codec_name == _CAGRA_CODEC
            else "gpu-with-cpu-fallback"
        ),
        build_parameters=build_parameters,
    )


class _FakeVMEnvironment:
    def __init__(self):
        self.attach_count = 0

    def attachCurrentThread(self):
        self.attach_count += 1


class _FakeLuceneModule:
    CLASSPATH = "/pylucene/lucene-core.jar"
    VERSION = pylucene_backend._REQUIRED_PYLUCENE_VERSION

    def __init__(self):
        self.environment = None
        self.init_calls = []

    def getVMEnv(self):
        return self.environment

    def initVM(self, **kwargs):
        self.init_calls.append(kwargs)
        self.environment = _FakeVMEnvironment()
        return self.environment


class _FakeIndexWriter:
    def __init__(self, error_at=None):
        self.error_at = error_at
        self.documents = []
        self.committed = False
        self.rollback_called = False
        self.close_called = False
        self.force_merge_calls = []

    def addDocument(self, document):
        if self.error_at in {"interrupt", "interrupt-rollback"}:
            raise KeyboardInterrupt
        if self.error_at in {"add", "rollback"}:
            raise RuntimeError("add failed")
        self.documents.append(document)

    def commit(self):
        if self.error_at == "commit":
            raise RuntimeError("commit failed")
        self.committed = True

    def rollback(self):
        self.rollback_called = True
        if self.error_at in {"rollback", "interrupt-rollback"}:
            raise RuntimeError("rollback failed")

    def close(self):
        self.close_called = True
        if self.error_at == "close":
            raise RuntimeError("close failed")

    def forceMerge(self, segment_count):
        self.force_merge_calls.append(segment_count)


class _FakeMergePolicy:
    def __init__(self):
        self.no_cfs_ratio = None

    def setNoCFSRatio(self, ratio):
        self.no_cfs_ratio = ratio


class _FakeIndexWriterConfig:
    OpenMode = SimpleNamespace(CREATE=object())
    DISABLE_AUTO_FLUSH = -1

    def __init__(self):
        self.use_compound_file = None
        self.merge_policy = _FakeMergePolicy()
        self.max_buffered_docs = None
        self.ram_buffer_size_mb = None

    def setOpenMode(self, _mode):
        pass

    def setCodec(self, _codec):
        pass

    def setUseCompoundFile(self, enabled):
        self.use_compound_file = enabled

    def getMergePolicy(self):
        return self.merge_policy

    def setMaxBufferedDocs(self, max_buffered_docs):
        self.max_buffered_docs = max_buffered_docs

    def setRAMBufferSizeMB(self, ram_buffer_size_mb):
        self.ram_buffer_size_mb = ram_buffer_size_mb

    def setMergePolicy(self, merge_policy):
        self.merge_policy = merge_policy

    def setMergeScheduler(self, _scheduler):
        raise AssertionError("PyLucene must use Lucene's default scheduler")


class _FakeAvailableCodecs:
    def __init__(self, names):
        self.names = names

    def contains(self, name):
        return name in self.names

    def __iter__(self):
        return iter(self.names)


def _fake_codec_runtime(codec, available_names=(_HNSW_CODEC,)):
    registry = SimpleNamespace(
        calls=[],
        availableCodecs=lambda: _FakeAvailableCodecs(available_names),
    )

    def for_name(codec_name):
        registry.calls.append(codec_name)
        return codec

    registry.forName = for_name
    runtime = pylucene_backend._PyLuceneRuntime.__new__(
        pylucene_backend._PyLuceneRuntime
    )
    runtime.attach_current_thread = lambda: None
    runtime.Codec = registry
    runtime._codec_cache = {}
    return runtime, registry


def _fake_configured_codec_runtime(diagnostics, initial_properties=None):
    properties = dict(initial_properties or {})
    property_snapshots = []
    class_names = []

    class _System:
        @staticmethod
        def getProperty(name):
            return properties.get(name)

        @staticmethod
        def setProperty(name, value):
            properties[name] = value

        @staticmethod
        def clearProperty(name):
            properties.pop(name, None)

    class _Codec:
        @staticmethod
        def getName():
            return _HNSW_CODEC

        @staticmethod
        def knnVectorsFormat():
            return object()

        def __str__(self):
            return diagnostics

    codec = _Codec()

    def new_instance():
        property_snapshots.append(dict(properties))
        return codec

    def for_name(class_name):
        class_names.append(class_name)
        return SimpleNamespace(newInstance=new_instance)

    runtime = pylucene_backend._PyLuceneRuntime.__new__(
        pylucene_backend._PyLuceneRuntime
    )
    runtime.attach_current_thread = lambda: None
    runtime.System = _System
    runtime.Class = SimpleNamespace(forName=for_name)
    runtime.Codec = SimpleNamespace(cast_=lambda reflected: reflected)
    return runtime, properties, property_snapshots, class_names, codec


def _fake_index_writer_runtime(error_at=None, directory_close_error=None):
    writer = _FakeIndexWriter(error_at=error_at)
    directory = SimpleNamespace(closed=False)

    def close_directory():
        directory.closed = True
        if directory_close_error is not None:
            raise directory_close_error

    directory.close = close_directory
    runtime = pylucene_backend._PyLuceneRuntime.__new__(
        pylucene_backend._PyLuceneRuntime
    )
    runtime.attach_current_thread = lambda: None
    runtime.resolve_codec = lambda _codec_name: object()
    runtime.Paths = SimpleNamespace(get=lambda path: path)
    runtime.FSDirectory = SimpleNamespace(open=lambda _path: directory)
    runtime.IndexWriterConfig = _FakeIndexWriterConfig
    runtime.NoMergePolicy = SimpleNamespace(INSTANCE=object())

    def create_writer(_directory, config):
        runtime._test_writer_config = config
        return writer

    runtime.IndexWriter = create_writer
    runtime._vector_document = lambda document_id, vector: (
        document_id,
        vector.copy(),
    )
    runtime._test_writer = writer
    runtime._test_directory = directory
    runtime._test_codec = object()
    runtime._index_topology = lambda _directory: _IndexTopology(
        segment_document_counts=(len(writer.documents),),
        segment_vector_counts=(len(writer.documents),),
    )
    return runtime


@pytest.fixture(autouse=True)
def _reset_jvm_tracking(monkeypatch):
    monkeypatch.setattr(pylucene_backend, "_INITIALIZED_CLASSPATH", None)
    monkeypatch.setattr(pylucene_backend, "_INITIALIZED_VMARGS", None)
    monkeypatch.setattr(
        pylucene_backend,
        "configured_codec_classes_path",
        lambda *_args: "/configured-codec-classes",
    )


def test_initialize_pylucene_uses_verified_classpath_and_vmargs(
    tmp_path, monkeypatch
):
    cuvs_java = tmp_path / "cuvs-java.jar"
    cuvs_lucene = tmp_path / "cuvs-lucene.jar"
    cuvs_java.touch()
    cuvs_lucene.touch()
    fake_lucene = _FakeLuceneModule()
    monkeypatch.setattr(
        pylucene_backend.importlib,
        "import_module",
        lambda name: fake_lucene,
    )

    returned = pylucene_backend._initialize_pylucene(
        {
            "cuvs_java_jar": cuvs_java,
            "cuvs_lucene_jar": cuvs_lucene,
            "java_library_path": "/native",
            "jvm_args": ["-Xms1g"],
        }
    )

    assert returned is fake_lucene
    assert len(fake_lucene.init_calls) == 1
    init_call = fake_lucene.init_calls[0]
    assert init_call["classpath"].split(":") == [
        "/configured-codec-classes",
        str(cuvs_java),
        str(cuvs_lucene),
        fake_lucene.CLASSPATH,
    ]
    assert init_call["vmargs"] == [
        "--enable-native-access=ALL-UNNAMED",
        "--add-modules=jdk.incubator.vector",
        "-Djava.library.path=/native",
        "-Xms1g",
    ]
    assert fake_lucene.environment.attach_count == 1

    pylucene_backend._initialize_pylucene(
        {
            "cuvs_java_jar": cuvs_java,
            "cuvs_lucene_jar": cuvs_lucene,
            "java_library_path": "/native",
            "jvm_args": ["-Xms1g"],
        }
    )
    assert len(fake_lucene.init_calls) == 1
    assert fake_lucene.environment.attach_count == 2


def test_configured_codec_compile_error_identifies_required_cuvs_lucene_api(
    monkeypatch,
):
    completed = SimpleNamespace(
        returncode=1,
        stderr="cannot find symbol: withHnswHeuristicType",
        stdout="",
    )
    monkeypatch.setattr(pylucene_java, "_find_javac", lambda: "/jdk/bin/javac")
    monkeypatch.setattr(
        pylucene_java.subprocess,
        "run",
        lambda *_args, **_kwargs: completed,
    )

    with pytest.raises(
        RuntimeError,
        match="PyLucene 10.2 codec support.*HNSW heuristic API",
    ):
        pylucene_java._compile("/dependencies")


@pytest.mark.parametrize(
    "lucene",
    [SimpleNamespace(VERSION="10.0.0"), SimpleNamespace()],
    ids=["mismatched", "missing"],
)
def test_validate_pylucene_version_rejects_incompatible_binding(lucene):
    with pytest.raises(
        RuntimeError,
        match="expected 10[.]2[.]0, found .*Activate a matching PyLucene build",
    ):
        pylucene_backend._validate_pylucene_version(lucene)


def test_initialize_pylucene_checks_version_before_starting_jvm(monkeypatch):
    fake_lucene = _FakeLuceneModule()
    fake_lucene.VERSION = "10.0.0"
    monkeypatch.setattr(
        pylucene_backend.importlib,
        "import_module",
        lambda _name: fake_lucene,
    )

    with pytest.raises(RuntimeError, match="expected 10[.]2[.]0"):
        pylucene_backend._initialize_pylucene({})

    assert fake_lucene.init_calls == []


@pytest.mark.parametrize(
    ("library_paths", "expected_library_path"),
    [
        (
            {
                "JAVA_LIBRARY_PATH": "/java-native",
                "LD_LIBRARY_PATH": "/ld-native",
            },
            "/java-native",
        ),
        ({"LD_LIBRARY_PATH": "/ld-native"}, "/ld-native"),
    ],
    ids=["java-library-path-precedence", "ld-library-path-fallback"],
)
def test_initialize_pylucene_uses_environment_runtime_config(
    library_paths,
    expected_library_path,
    tmp_path,
    monkeypatch,
):
    cuvs_java = tmp_path / "cuvs-java.jar"
    cuvs_lucene = tmp_path / "cuvs-lucene.jar"
    cuvs_java.touch()
    cuvs_lucene.touch()
    fake_lucene = _FakeLuceneModule()
    monkeypatch.setattr(
        pylucene_backend.importlib,
        "import_module",
        lambda _name: fake_lucene,
    )
    monkeypatch.setenv("CUVS_LUCENE_CUVS_JAVA_JAR", str(cuvs_java))
    monkeypatch.setenv("CUVS_LUCENE_JAR", str(cuvs_lucene))
    monkeypatch.delenv("JAVA_LIBRARY_PATH", raising=False)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    for name, value in library_paths.items():
        monkeypatch.setenv(name, value)

    pylucene_backend._initialize_pylucene({})

    init_call = fake_lucene.init_calls[0]
    assert init_call["classpath"].split(":") == [
        "/configured-codec-classes",
        str(cuvs_java),
        str(cuvs_lucene),
        fake_lucene.CLASSPATH,
    ]
    assert init_call["vmargs"] == [
        "--enable-native-access=ALL-UNNAMED",
        "--add-modules=jdk.incubator.vector",
        f"-Djava.library.path={expected_library_path}",
    ]


def test_initialize_pylucene_rejects_fat_cuvs_lucene_jar_before_init(
    tmp_path, monkeypatch
):
    cuvs_java = tmp_path / "cuvs-java.jar"
    cuvs_lucene = tmp_path / "cuvs-lucene-jar-with-dependencies.jar"
    cuvs_java.touch()
    with zipfile.ZipFile(cuvs_lucene, "w") as archive:
        archive.writestr(
            pylucene_backend._LUCENE_CORE_CLASS,
            b"bundled Lucene bytecode",
        )
    fake_lucene = _FakeLuceneModule()
    monkeypatch.setattr(
        pylucene_backend.importlib,
        "import_module",
        lambda _name: fake_lucene,
    )

    with pytest.raises(RuntimeError, match="standard thin cuvs-lucene JAR"):
        pylucene_backend._initialize_pylucene(
            {
                "cuvs_java_jar": cuvs_java,
                "cuvs_lucene_jar": cuvs_lucene,
            }
        )

    assert fake_lucene.init_calls == []


def test_initialize_pylucene_rejects_externally_started_jvm(
    tmp_path, monkeypatch
):
    cuvs_java = tmp_path / "cuvs-java.jar"
    cuvs_lucene = tmp_path / "cuvs-lucene.jar"
    cuvs_java.touch()
    cuvs_lucene.touch()
    fake_lucene = _FakeLuceneModule()
    fake_lucene.environment = _FakeVMEnvironment()
    monkeypatch.setattr(
        pylucene_backend.importlib,
        "import_module",
        lambda name: fake_lucene,
    )

    with pytest.raises(RuntimeError, match="initialized before"):
        pylucene_backend._initialize_pylucene(
            {
                "cuvs_java_jar": cuvs_java,
                "cuvs_lucene_jar": cuvs_lucene,
            }
        )


def test_initialize_pylucene_rejects_different_jars_after_start(
    tmp_path, monkeypatch
):
    first_java = tmp_path / "first-java.jar"
    first_lucene = tmp_path / "first-lucene.jar"
    second_java = tmp_path / "second-java.jar"
    for path in (first_java, first_lucene, second_java):
        path.touch()
    fake_lucene = _FakeLuceneModule()
    monkeypatch.setattr(
        pylucene_backend.importlib,
        "import_module",
        lambda name: fake_lucene,
    )

    pylucene_backend._initialize_pylucene(
        {
            "cuvs_java_jar": first_java,
            "cuvs_lucene_jar": first_lucene,
        }
    )

    with pytest.raises(RuntimeError, match="different"):
        pylucene_backend._initialize_pylucene(
            {
                "cuvs_java_jar": second_java,
                "cuvs_lucene_jar": first_lucene,
            }
        )


def test_initialize_pylucene_rejects_different_vmargs_after_start(
    tmp_path, monkeypatch
):
    cuvs_java = tmp_path / "cuvs-java.jar"
    cuvs_lucene = tmp_path / "cuvs-lucene.jar"
    cuvs_java.touch()
    cuvs_lucene.touch()
    fake_lucene = _FakeLuceneModule()
    monkeypatch.setattr(
        pylucene_backend.importlib,
        "import_module",
        lambda name: fake_lucene,
    )

    pylucene_backend._initialize_pylucene(
        {
            "cuvs_java_jar": cuvs_java,
            "cuvs_lucene_jar": cuvs_lucene,
            "java_library_path": "/first",
        }
    )

    with pytest.raises(RuntimeError, match="different JVM arguments"):
        pylucene_backend._initialize_pylucene(
            {
                "cuvs_java_jar": cuvs_java,
                "cuvs_lucene_jar": cuvs_lucene,
                "java_library_path": "/second",
            }
        )


def test_initialize_pylucene_reports_missing_binding(monkeypatch):
    def missing_import(_name):
        raise ImportError("missing")

    monkeypatch.setattr(
        pylucene_backend.importlib, "import_module", missing_import
    )

    with pytest.raises(ImportError, match="must be built"):
        pylucene_backend._initialize_pylucene({})


def test_initialize_pylucene_reports_missing_jar(monkeypatch):
    monkeypatch.delenv("CUVS_LUCENE_CUVS_JAVA_JAR", raising=False)
    monkeypatch.delenv("CUVS_LUCENE_JAR", raising=False)
    monkeypatch.setattr(
        pylucene_backend.importlib,
        "import_module",
        lambda _name: _FakeLuceneModule(),
    )

    with pytest.raises(RuntimeError, match="cuvs_java_jar"):
        pylucene_backend._initialize_pylucene({})


def test_resolve_codec_validates_and_caches_initialized_vector_format():
    vectors_format = object()
    codec = SimpleNamespace(
        getName=lambda: _HNSW_CODEC,
        knnVectorsFormat=lambda: vectors_format,
    )
    runtime, registry = _fake_codec_runtime(codec)

    assert runtime.resolve_codec(_HNSW_CODEC) is codec
    assert runtime.resolve_codec(_HNSW_CODEC) is codec
    assert registry.calls == [_HNSW_CODEC]


@pytest.mark.parametrize(
    ("available_names", "returned_name", "vectors_format", "error"),
    [
        ((), _HNSW_CODEC, object(), "was not advertised by Lucene SPI"),
        ((_HNSW_CODEC,), "DifferentCodec", object(), "Requested codec"),
        (
            (_HNSW_CODEC,),
            _HNSW_CODEC,
            None,
            "did not initialize a Lucene vector format",
        ),
    ],
    ids=["missing-spi", "wrong-name", "missing-vector-format"],
)
def test_resolve_codec_rejects_unusable_codec(
    available_names, returned_name, vectors_format, error
):
    codec = SimpleNamespace(
        getName=lambda: returned_name,
        knnVectorsFormat=lambda: vectors_format,
    )
    runtime, _ = _fake_codec_runtime(codec, available_names)

    with pytest.raises(RuntimeError, match=error):
        runtime.resolve_codec(_HNSW_CODEC)


def test_resolve_configured_hnsw_codec_passes_parameters_and_restores_state():
    diagnostics = "PyLuceneConfiguredHnswCodec(m=24, efConstruction=96)"
    runtime, properties, snapshots, class_names, codec = (
        _fake_configured_codec_runtime(
            diagnostics,
            {pylucene_backend.M_PROPERTY: "previous-m"},
        )
    )

    assert runtime.resolve_configured_hnsw_codec(24, 96) is codec
    assert snapshots == [
        {
            pylucene_backend.M_PROPERTY: "24",
            pylucene_backend.EF_CONSTRUCTION_PROPERTY: "96",
        }
    ]
    assert properties == {pylucene_backend.M_PROPERTY: "previous-m"}
    assert class_names == [pylucene_backend.CONFIGURED_HNSW_CODEC_CLASS]


def test_resolve_configured_hnsw_codec_rejects_diagnostic_mismatch_and_restores_state():
    runtime, properties, snapshots, _, _ = _fake_configured_codec_runtime(
        "PyLuceneConfiguredHnswCodec(m=16, efConstruction=48)",
        {pylucene_backend.EF_CONSTRUCTION_PROPERTY: "previous-ef"},
    )

    with pytest.raises(RuntimeError, match="did not retain the requested"):
        runtime.resolve_configured_hnsw_codec(24, 96)

    assert snapshots == [
        {
            pylucene_backend.M_PROPERTY: "24",
            pylucene_backend.EF_CONSTRUCTION_PROPERTY: "96",
        }
    ]
    assert properties == {
        pylucene_backend.EF_CONSTRUCTION_PROPERTY: "previous-ef"
    }


@pytest.mark.parametrize("jvm_args", ["-Xmx1g", ["-Xmx1g", 1]])
def test_initialize_pylucene_rejects_invalid_jvm_args(
    jvm_args, tmp_path, monkeypatch
):
    cuvs_java = tmp_path / "cuvs-java.jar"
    cuvs_lucene = tmp_path / "cuvs-lucene.jar"
    cuvs_java.touch()
    cuvs_lucene.touch()
    monkeypatch.setattr(
        pylucene_backend.importlib,
        "import_module",
        lambda _name: _FakeLuceneModule(),
    )

    with pytest.raises(TypeError, match="jvm_args"):
        pylucene_backend._initialize_pylucene(
            {
                "cuvs_java_jar": cuvs_java,
                "cuvs_lucene_jar": cuvs_lucene,
                "jvm_args": jvm_args,
            }
        )


def test_runtime_build_index_commits_and_closes_writer_and_directory(tmp_path):
    runtime = _fake_index_writer_runtime()
    vectors = np.zeros((2, 4), dtype=np.float32)

    result = runtime.build_index(
        tmp_path,
        vectors,
        _build_codec(runtime._test_codec),
    )

    assert result == _IndexTopology(
        segment_document_counts=(2,),
        segment_vector_counts=(2,),
    )
    assert len(runtime._test_writer.documents) == 2
    assert runtime._test_writer.committed is True
    assert runtime._test_writer.rollback_called is False
    assert runtime._test_writer.close_called is True
    assert runtime._test_directory.closed is True
    assert runtime._test_writer_config.use_compound_file is None
    assert runtime._test_writer_config.merge_policy.no_cfs_ratio is None
    assert runtime._test_writer.force_merge_calls == []


@pytest.mark.parametrize(
    ("codec_name", "use_compound_file", "no_cfs_ratio"),
    [
        (_HNSW_CODEC, None, None),
        (_CAGRA_CODEC, False, 0.0),
    ],
)
def test_runtime_writer_config_applies_codec_compound_file_policy(
    codec_name, use_compound_file, no_cfs_ratio
):
    runtime = _fake_index_writer_runtime()

    config = runtime._new_index_writer_config(
        _build_codec(runtime._test_codec, codec_name),
        vector_count=10,
    )

    assert config.use_compound_file is use_compound_file
    assert config.merge_policy.no_cfs_ratio == no_cfs_ratio


def test_runtime_writer_config_builds_one_segment_without_force_merge():
    runtime = _fake_index_writer_runtime()
    build_parameters = {
        "codec": _HNSW_CODEC,
        "m": 24,
        "ef_construction": 96,
        "direct_single_segment": True,
    }

    config = runtime._new_index_writer_config(
        _build_codec(
            runtime._test_codec,
            build_parameters=build_parameters,
        ),
        vector_count=1_000_000,
    )

    assert config.max_buffered_docs == 1_000_001
    assert config.ram_buffer_size_mb == config.DISABLE_AUTO_FLUSH
    assert type(config.ram_buffer_size_mb) is float
    assert config.merge_policy is runtime.NoMergePolicy.INSTANCE


def test_runtime_direct_single_segment_build_does_not_force_merge(tmp_path):
    runtime = _fake_index_writer_runtime()
    vectors = np.zeros((2, 4), dtype=np.float32)
    build_parameters = {
        "codec": _HNSW_CODEC,
        "m": 32,
        "ef_construction": 32,
        "direct_single_segment": True,
    }

    topology = runtime.build_index(
        tmp_path,
        vectors,
        _build_codec(
            runtime._test_codec,
            build_parameters=build_parameters,
        ),
    )

    assert topology.segment_count == 1
    assert runtime._test_writer.force_merge_calls == []
    assert runtime._test_writer.close_called is True
    assert runtime._test_directory.closed is True


def test_runtime_reads_committed_segment_topology_and_closes_reader():
    reader = SimpleNamespace(closed=False)

    def close_reader():
        reader.closed = True

    def leaf(document_count, vector_count):
        leaf_reader = SimpleNamespace(
            numDocs=lambda: document_count,
            getFloatVectorValues=lambda _field: SimpleNamespace(
                size=lambda: vector_count
            ),
        )
        return SimpleNamespace(reader=lambda: leaf_reader)

    reader.close = close_reader
    reader.leaves = lambda: [leaf(4, 4), leaf(6, 6)]
    runtime = pylucene_backend._PyLuceneRuntime.__new__(
        pylucene_backend._PyLuceneRuntime
    )
    runtime.DirectoryReader = SimpleNamespace(open=lambda _directory: reader)

    topology = runtime._index_topology(object())

    assert topology == _IndexTopology(
        segment_document_counts=(4, 6),
        segment_vector_counts=(4, 6),
    )
    assert reader.closed is True


def test_runtime_topology_rejects_segment_without_vectors_and_closes_reader():
    leaf_reader = SimpleNamespace(
        numDocs=lambda: 2,
        getFloatVectorValues=lambda _field: None,
    )
    reader = SimpleNamespace(
        leaves=lambda: [SimpleNamespace(reader=lambda: leaf_reader)],
        closed=False,
    )

    def close_reader():
        reader.closed = True

    reader.close = close_reader
    runtime = pylucene_backend._PyLuceneRuntime.__new__(
        pylucene_backend._PyLuceneRuntime
    )
    runtime.DirectoryReader = SimpleNamespace(open=lambda _directory: reader)

    with pytest.raises(RuntimeError, match="contains no vector values"):
        runtime._index_topology(object())

    assert reader.closed is True


@pytest.mark.parametrize(
    ("topology", "direct_single_segment", "error"),
    [
        (_IndexTopology((), ()), False, "no committed segments"),
        (
            _IndexTopology((1,), (1,)),
            False,
            "document counts do not match",
        ),
        (
            _IndexTopology((2,), (1,)),
            False,
            "document and vector counts do not match",
        ),
        (
            _IndexTopology((1, 1), (1, 1)),
            True,
            "requested one committed Lucene segment",
        ),
    ],
    ids=[
        "no-segments",
        "wrong-document-count",
        "wrong-vector-count",
        "multiple-direct-segments",
    ],
)
def test_runtime_build_rejects_invalid_topology_and_closes_resources(
    tmp_path, topology, direct_single_segment, error
):
    runtime = _fake_index_writer_runtime()
    runtime._index_topology = lambda _directory: topology
    vectors = np.zeros((2, 4), dtype=np.float32)
    build_parameters = {
        "codec": _HNSW_CODEC,
        "m": 32,
        "ef_construction": 32,
        "direct_single_segment": direct_single_segment,
    }

    with pytest.raises(RuntimeError, match=error):
        runtime.build_index(
            tmp_path,
            vectors,
            _build_codec(
                runtime._test_codec,
                build_parameters=build_parameters,
            ),
        )

    assert runtime._test_writer.close_called is True
    assert runtime._test_directory.closed is True


@pytest.mark.parametrize(
    ("error_at", "expected_error", "expect_rollback", "expect_close"),
    [
        ("add", "add failed", True, False),
        ("commit", "commit failed", True, False),
        ("rollback", "rollback failed", True, False),
        ("close", "close failed", False, True),
    ],
)
def test_runtime_build_index_preserves_transaction_cleanup(
    tmp_path, error_at, expected_error, expect_rollback, expect_close
):
    runtime = _fake_index_writer_runtime(error_at=error_at)
    vectors = np.zeros((2, 4), dtype=np.float32)

    with pytest.raises(RuntimeError, match=expected_error):
        runtime.build_index(
            tmp_path,
            vectors,
            _build_codec(runtime._test_codec),
        )

    assert runtime._test_writer.rollback_called is expect_rollback
    assert runtime._test_writer.close_called is expect_close
    assert runtime._test_directory.closed is True


def test_runtime_build_index_rolls_back_on_interrupt(tmp_path):
    runtime = _fake_index_writer_runtime(error_at="interrupt")
    vectors = np.zeros((2, 4), dtype=np.float32)

    with pytest.raises(KeyboardInterrupt):
        runtime.build_index(
            tmp_path,
            vectors,
            _build_codec(runtime._test_codec),
        )

    assert runtime._test_writer.rollback_called is True
    assert runtime._test_writer.close_called is False
    assert runtime._test_directory.closed is True


def test_runtime_build_index_preserves_interrupt_when_rollback_fails(tmp_path):
    runtime = _fake_index_writer_runtime(error_at="interrupt-rollback")
    vectors = np.zeros((2, 4), dtype=np.float32)

    with pytest.raises(KeyboardInterrupt) as exc_info:
        runtime.build_index(
            tmp_path,
            vectors,
            _build_codec(runtime._test_codec),
        )

    assert exc_info.value.__notes__ == [
        "IndexWriter rollback also failed: RuntimeError: rollback failed"
    ]
    assert runtime._test_writer.rollback_called is True
    assert runtime._test_directory.closed is True


def test_runtime_build_index_preserves_interrupt_when_directory_close_fails(
    tmp_path,
):
    runtime = _fake_index_writer_runtime(
        error_at="interrupt",
        directory_close_error=RuntimeError("directory close failed"),
    )
    vectors = np.zeros((2, 4), dtype=np.float32)

    with pytest.raises(KeyboardInterrupt) as exc_info:
        runtime.build_index(
            tmp_path,
            vectors,
            _build_codec(runtime._test_codec),
        )

    assert exc_info.value.__notes__ == [
        "Failed to close Lucene directory: RuntimeError: directory close failed"
    ]
    assert runtime._test_writer.rollback_called is True
    assert runtime._test_directory.closed is True


def test_runtime_search_uses_candidates_for_query_and_top_k_for_results():
    query_calls = []
    search_calls = []
    query = object()

    def new_query(field, vector, num_candidates):
        query_calls.append((field, vector, num_candidates))
        return query

    def search(received_query, top_k):
        search_calls.append((received_query, top_k))
        return SimpleNamespace(scoreDocs=[SimpleNamespace(doc=4, score=0.5)])

    runtime = pylucene_backend._PyLuceneRuntime.__new__(
        pylucene_backend._PyLuceneRuntime
    )
    runtime.KnnFloatVectorQuery = new_query
    runtime._java_float_array = lambda vector: tuple(vector.tolist())
    searcher = SimpleNamespace(search=search)
    stored_fields = SimpleNamespace(
        document=lambda _document_id: SimpleNamespace(get=lambda _field: "17")
    )

    hits = runtime._search_vector(
        searcher,
        stored_fields,
        np.array([1.0, 2.0], dtype=np.float32),
        k=150,
        num_candidates=300,
    )

    assert query_calls == [(pylucene_backend._VECTOR_FIELD, (1.0, 2.0), 300)]
    assert search_calls == [(query, 150)]
    assert hits == [pylucene_backend._SearchHit(document_id=17, score=0.5)]


def test_search_cleanup_attempts_directory_after_reader_close_fails():
    close_calls = []

    def fail_reader_close():
        close_calls.append("reader")
        raise RuntimeError("reader close failed")

    def fail_directory_close():
        close_calls.append("directory")
        raise RuntimeError("directory close failed")

    reader = SimpleNamespace(close=fail_reader_close)
    directory = SimpleNamespace(close=fail_directory_close)

    with pytest.raises(RuntimeError, match="reader close failed") as exc_info:
        with pylucene_backend._CleanupStack() as cleanups:
            cleanups.add("close Lucene directory", directory.close)
            cleanups.add("close Lucene index reader", reader.close)

    assert close_calls == ["reader", "directory"]
    assert exc_info.value.__notes__ == [
        "Failed to close Lucene directory: RuntimeError: directory close failed"
    ]


def test_cleanup_stack_ignores_unrelated_handled_exception():
    def fail_close():
        raise RuntimeError("close failed")

    unrelated_error = None
    try:
        raise ValueError("unrelated")
    except ValueError as exc:
        unrelated_error = exc
        with pytest.raises(RuntimeError, match="close failed"):
            with pylucene_backend._CleanupStack() as cleanups:
                cleanups.add("close test resource", fail_close)

    assert getattr(unrelated_error, "__notes__", []) == []


def test_cleanup_stack_prioritizes_cleanup_interrupt_over_operation_error():
    def interrupt_close():
        raise KeyboardInterrupt("stop")

    with pytest.raises(KeyboardInterrupt, match="stop") as exc_info:
        with pylucene_backend._CleanupStack() as cleanups:
            cleanups.add("close test resource", interrupt_close)
            raise RuntimeError("operation failed")

    assert exc_info.value.__notes__ == [
        "Raised while attempting to close test resource; prior failure: "
        "RuntimeError: operation failed"
    ]
