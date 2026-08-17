# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import struct
import tempfile
import xml.etree.ElementTree as ElementTree
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[3]
ID_FIELD = "id"
VECTOR_FIELD = "vector"
FILTER_FIELD = "filter-status"
FILTER_INCLUDED_VALUE = "included"
FILTER_EXCLUDED_VALUE = "excluded"

CAGRA_TEST_CODEC_CLASS = (
    "com.nvidia.cuvs.lucene.PyLuceneTestSupport$CagraSearchCodec"
)
CPU_HNSW_TEST_CODEC_CLASS = (
    "com.nvidia.cuvs.lucene.PyLuceneTestSupport$CpuHnswCodec"
)
CAGRA_BUILT_HNSW_BASE_LAYER_TEST_CODEC_CLASS = (
    "com.nvidia.cuvs.lucene."
    "PyLuceneTestSupport$CagraBuiltHnswBaseLayerCodec"
)
CAGRA_BUILT_HNSW_THREE_LAYER_TEST_CODEC_CLASS = (
    "com.nvidia.cuvs.lucene."
    "PyLuceneTestSupport$CagraBuiltHnswThreeLayerCodec"
)
CAGRA_TEST_QUERY_CLASS = (
    "com.nvidia.cuvs.lucene.PyLuceneTestSupport$CagraSearchQuery"
)
HNSW_GRAPH_VERIFYING_QUERY_CLASS = (
    "com.nvidia.cuvs.lucene.PyLuceneTestSupport$HnswGraphVerifyingQuery"
)
CAGRA_VECTOR_READER_CLASS = "com.nvidia.cuvs.lucene.CuVS2510GPUVectorsReader"

QUERY_PROPERTIES = {
    "field": "cuvs.lucene.pylucene.query.field",
    "target": "cuvs.lucene.pylucene.query.target",
    "k": "cuvs.lucene.pylucene.query.k",
    "i_top_k": "cuvs.lucene.pylucene.query.iTopK",
    "search_width": "cuvs.lucene.pylucene.query.searchWidth",
    "expected_hnsw_m": "cuvs.lucene.pylucene.query.expectedHnswM",
    "filter_field": "cuvs.lucene.pylucene.query.filterField",
    "filter_value": "cuvs.lucene.pylucene.query.filterValue",
}

_FLOAT32 = struct.Struct("!f")


@dataclass(frozen=True)
class IndexScenario:
    name: str
    codec_name: str
    document_count: int
    dimensions: int
    top_k: int
    segment_count: int = 1
    force_merge_segment_count: int = 0
    disable_automatic_merges: bool = True
    document_ids_without_vectors: frozenset[int] = frozenset()
    document_ids_to_delete: frozenset[int] = frozenset()
    additional_query_document_ids: tuple[int, ...] = ()
    document_ids_accepted_by_filter: frozenset[int] = frozenset()
    filter_query_document_id: int | None = None
    codec_factory_class: str = ""
    use_cagra_search_query: bool = False
    expected_hnsw_m: int = 0
    search_width: int = 1
    i_top_k: int = 64


class QueryDocumentState(Enum):
    SEARCHABLE = "searchable"
    WITHOUT_VECTOR = "without-vector"
    DELETED = "deleted"


@dataclass(frozen=True)
class QueryObservation:
    query_document_id: int
    hit_ids: tuple[str, ...]
    query_class: str
    query_document_state: QueryDocumentState


@dataclass(frozen=True)
class FilteredQueryObservation:
    query_document_id: int
    hit_ids: tuple[str, ...]
    query_class: str


@dataclass(frozen=True)
class VectorReaderMetadata:
    vector_count: int
    dimensions: tuple[int, ...]
    reader_classes: tuple[str, ...]
    hnsw_layer_counts: tuple[int, ...]


@dataclass(frozen=True)
class IndexRun:
    index_files: tuple[str, ...]
    pre_merge_segment_count: int
    segment_count: int
    live_document_count: int
    max_document_count: int
    vector_count: int
    vector_dimensions: tuple[int, ...]
    vector_reader_classes: tuple[str, ...]
    hnsw_layer_counts: tuple[int, ...]
    searchable_vector_document_ids: tuple[int, ...]
    document_ids_without_vectors: frozenset[int]
    deleted_document_ids: frozenset[int]
    query_observations: tuple[QueryObservation, ...]
    filtered_query_observation: FilteredQueryObservation | None
    writer_telemetry: dict[str, str]


@dataclass
class PyLuceneContext:
    cuvs_lucene_jar: Path
    cuvs_java_jar: Path
    test_classes: Path
    codec_class: Any
    query_class: Any
    class_class: Any
    system_class: Any
    jarray: Any
    codec_cache: dict[str, Any]


def find_cuvs_lucene_jar() -> Path:
    configured = os.environ.get("CUVS_LUCENE_JAR")
    if configured:
        jar = Path(configured).resolve()
        if not jar.is_file():
            raise FileNotFoundError(f"Configured cuvs-lucene jar does not exist: {jar}")
        return jar

    jars = sorted(
        jar
        for jar in (REPO_ROOT / "target").glob("cuvs-lucene-*.jar")
        if not any(
            marker in jar.name
            for marker in ("-jar-with-", "-sources.jar", "-javadoc.jar")
        )
    )
    if not jars:
        raise FileNotFoundError(
            "No cuvs-lucene jar found under target/. "
            "Run `mvn clean package -DskipTests` first."
        )
    return jars[-1].resolve()


def find_cuvs_java_jar() -> Path:
    configured = os.environ.get("CUVS_LUCENE_CUVS_JAVA_JAR") or os.environ.get(
        "CUVS_JAVA_JAR"
    )
    if configured:
        jar = Path(configured).resolve()
        if not jar.is_file():
            raise FileNotFoundError(f"Configured cuvs-java jar does not exist: {jar}")
        return jar

    m2_repo = (
        Path.home() / ".m2" / "repository" / "com" / "nvidia" / "cuvs" / "cuvs-java"
    )
    if not m2_repo.exists():
        raise FileNotFoundError(
            "Unable to find cuvs-java in ~/.m2. Set "
            "CUVS_LUCENE_CUVS_JAVA_JAR to the base cuvs-java jar."
        )

    required_version = _maven_dependency_version(
        "com.nvidia.cuvs", "cuvs-java"
    )
    jar = m2_repo / required_version / f"cuvs-java-{required_version}.jar"
    if not jar.is_file():
        raise FileNotFoundError(
            f"Unable to find the POM-matching base cuvs-java {required_version} "
            f"jar at {jar}. Set CUVS_LUCENE_CUVS_JAVA_JAR explicitly."
        )
    return jar.resolve()


def _maven_dependency_version(group_id: str, artifact_id: str) -> str:
    namespace = {"maven": "http://maven.apache.org/POM/4.0.0"}
    root = ElementTree.parse(REPO_ROOT / "pom.xml").getroot()
    for dependency in root.findall(".//maven:dependency", namespace):
        dependency_group = dependency.findtext("maven:groupId", namespaces=namespace)
        dependency_artifact = dependency.findtext(
            "maven:artifactId", namespaces=namespace
        )
        if dependency_group == group_id and dependency_artifact == artifact_id:
            version = dependency.findtext("maven:version", namespaces=namespace)
            if version and not version.startswith("${"):
                return version
            raise RuntimeError(
                f"Dependency {group_id}:{artifact_id} has no literal version in pom.xml"
            )
    raise RuntimeError(f"Dependency {group_id}:{artifact_id} is absent from pom.xml")


def find_test_classes() -> Path:
    test_classes = Path(
        os.environ.get(
            "CUVS_LUCENE_PYLUCENE_TEST_CLASSES",
            REPO_ROOT / "target" / "test-classes",
        )
    ).resolve()
    required_classes = (
        "PyLuceneTestSupport.class",
        "PyLuceneTestSupport$CpuHnswCodec.class",
        "PyLuceneTestSupport$CagraSearchCodec.class",
        "PyLuceneTestSupport$CagraBuiltHnswBaseLayerCodec.class",
        "PyLuceneTestSupport$CagraBuiltHnswThreeLayerCodec.class",
        "PyLuceneTestSupport$CagraSearchQuery.class",
        "PyLuceneTestSupport$HnswGraphVerifyingQuery.class",
    )
    package_dir = test_classes / "com" / "nvidia" / "cuvs" / "lucene"
    missing = [name for name in required_classes if not (package_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Compiled PyLucene test bridge is incomplete under {test_classes}: "
            f"missing {', '.join(missing)}. Run Maven test compilation first."
        )
    return test_classes


@lru_cache(maxsize=None)
def deterministic_float32_vector(
    document_id: int, dimensions: int
) -> tuple[float, ...]:
    value = ((document_id + 1) * 2654435761) & 0xFFFFFFFF
    vector = []
    for dimension in range(dimensions):
        value = (
            1664525 * value + 1013904223 + dimension * 17
        ) & 0xFFFFFFFF
        component = (value / 4294967295.0) * 2.0 - 1.0
        vector.append(_FLOAT32.unpack(_FLOAT32.pack(component))[0])
    return tuple(vector)


def to_java_float_array(jarray: Any, values: tuple[float, ...]) -> Any:
    return jarray("float")(values)


def validate_pylucene_version(
    actual_version: str, expected_version: str
) -> None:
    if actual_version != expected_version:
        raise RuntimeError(
            "PyLucene must be generated against the same Lucene version as "
            f"cuvs-lucene: expected {expected_version}, found {actual_version}. "
            f"Activate a PyLucene build generated against Lucene {expected_version} "
            "before initializing the JVM."
        )


def _init_vm(cuvs_java_jar: Path, cuvs_lucene_jar: Path, test_classes: Path) -> Any:
    import lucene

    validate_pylucene_version(
        str(getattr(lucene, "VERSION", "<missing>")),
        _maven_dependency_version("org.apache.lucene", "lucene-core"),
    )

    java_library_path = os.environ.get("JAVA_LIBRARY_PATH") or os.environ.get(
        "LD_LIBRARY_PATH"
    )
    vmargs = [
        "--enable-native-access=ALL-UNNAMED",
        "--add-modules=jdk.incubator.vector",
        (
            "-Djava.util.logging.config.file="
            f"{REPO_ROOT / 'src' / 'main' / 'resources' / 'logging.properties'}"
        ),
    ]
    if java_library_path:
        vmargs.append(f"-Djava.library.path={java_library_path}")

    lucene.initVM(
        classpath=os.pathsep.join(
            [
                str(cuvs_java_jar),
                str(cuvs_lucene_jar),
                str(test_classes),
                lucene.CLASSPATH,
            ]
        ),
        vmargs=vmargs,
    )
    return lucene


def initialize_pylucene_context(
    expected_spi_codecs: tuple[str, ...],
) -> PyLuceneContext:
    cuvs_lucene_jar = find_cuvs_lucene_jar()
    cuvs_java_jar = find_cuvs_java_jar()
    test_classes = find_test_classes()
    lucene = _init_vm(cuvs_java_jar, cuvs_lucene_jar, test_classes)

    from java.lang import Class, System
    from org.apache.lucene.codecs import Codec
    from org.apache.lucene.search import Query

    # Resolve classpath failures before constructing codecs or indexes.
    Class.forName("com.nvidia.cuvs.spi.JDKProvider")
    Class.forName(CPU_HNSW_TEST_CODEC_CLASS)
    Class.forName(CAGRA_TEST_CODEC_CLASS)
    Class.forName(CAGRA_BUILT_HNSW_BASE_LAYER_TEST_CODEC_CLASS)
    Class.forName(CAGRA_BUILT_HNSW_THREE_LAYER_TEST_CODEC_CLASS)
    Class.forName(CAGRA_TEST_QUERY_CLASS)
    Class.forName(HNSW_GRAPH_VERIFYING_QUERY_CLASS)

    available_codecs = Codec.availableCodecs()
    for codec_name in expected_spi_codecs:
        if not available_codecs.contains(codec_name):
            raise RuntimeError(
                f"{codec_name} was not advertised by Lucene SPI; "
                f"available codecs: {available_codecs}"
            )

    return PyLuceneContext(
        cuvs_lucene_jar=cuvs_lucene_jar,
        cuvs_java_jar=cuvs_java_jar,
        test_classes=test_classes,
        codec_class=Codec,
        query_class=Query,
        class_class=Class,
        system_class=System,
        jarray=lucene.JArray,
        codec_cache={},
    )


def _codec_for_scenario(scenario: IndexScenario, context: PyLuceneContext) -> Any:
    cache_key = scenario.codec_factory_class or f"spi:{scenario.codec_name}"
    cached = context.codec_cache.get(cache_key)
    if cached is not None:
        return cached

    if scenario.codec_factory_class:
        reflected = context.class_class.forName(
            scenario.codec_factory_class
        ).newInstance()
        codec = context.codec_class.cast_(reflected)
    else:
        available_codecs = context.codec_class.availableCodecs()
        if not available_codecs.contains(scenario.codec_name):
            raise RuntimeError(
                f"{scenario.codec_name} was not advertised by Lucene SPI; "
                f"available codecs: {available_codecs}"
            )
        codec = context.codec_class.forName(scenario.codec_name)

    if codec.getName() != scenario.codec_name:
        raise RuntimeError(
            f"Expected codec {scenario.codec_name}, got {codec.getName()}"
        )
    context.codec_cache[cache_key] = codec
    return codec


def _writer_telemetry(codec: Any) -> dict[str, str]:
    description = str(codec.knnVectorsFormat())
    _, separator, payload = description.partition("(")
    if not separator or not payload.endswith(")"):
        raise RuntimeError(f"Malformed vector-format diagnostics: {description!r}")

    telemetry: dict[str, str] = {}
    for item in payload[:-1].split(";"):
        key, item_separator, value = item.partition("=")
        if not item_separator or not key:
            raise RuntimeError(
                f"Malformed vector-format diagnostic item {item!r} in {description!r}"
            )
        telemetry[key] = value
    return telemetry


@contextmanager
def _temporary_system_properties(
    system_class: Any, properties: dict[str, str]
) -> Iterator[None]:
    previous = {name: system_class.getProperty(name) for name in properties}
    try:
        for name, value in properties.items():
            system_class.setProperty(name, value)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                system_class.clearProperty(name)
            else:
                system_class.setProperty(name, value)


def _new_cagra_search_query(
    context: PyLuceneContext,
    target: tuple[float, ...],
    top_k: int,
    i_top_k: int,
    search_width: int,
    filter_field: str = "",
    filter_value: str = "",
) -> Any:
    properties = {
        QUERY_PROPERTIES["field"]: VECTOR_FIELD,
        QUERY_PROPERTIES["target"]: ",".join(f"{value:.9g}" for value in target),
        QUERY_PROPERTIES["k"]: str(top_k),
        QUERY_PROPERTIES["i_top_k"]: str(max(top_k, i_top_k)),
        QUERY_PROPERTIES["search_width"]: str(search_width),
    }
    if filter_field:
        properties[QUERY_PROPERTIES["filter_field"]] = filter_field
        properties[QUERY_PROPERTIES["filter_value"]] = filter_value
    with _temporary_system_properties(context.system_class, properties):
        reflected = context.class_class.forName(
            CAGRA_TEST_QUERY_CLASS
        ).newInstance()
    return context.query_class.cast_(reflected)


def _new_hnsw_graph_verifying_query(
    context: PyLuceneContext,
    target: tuple[float, ...],
    top_k: int,
    expected_m: int,
    filter_field: str = "",
    filter_value: str = "",
) -> Any:
    properties = {
        QUERY_PROPERTIES["field"]: VECTOR_FIELD,
        QUERY_PROPERTIES["target"]: ",".join(f"{value:.9g}" for value in target),
        QUERY_PROPERTIES["k"]: str(top_k),
        QUERY_PROPERTIES["expected_hnsw_m"]: str(expected_m),
    }
    if filter_field:
        properties[QUERY_PROPERTIES["filter_field"]] = filter_field
        properties[QUERY_PROPERTIES["filter_value"]] = filter_value
    with _temporary_system_properties(context.system_class, properties):
        reflected = context.class_class.forName(
            HNSW_GRAPH_VERIFYING_QUERY_CLASS
        ).newInstance()
    return context.query_class.cast_(reflected)


def _validate_document_ids(scenario: IndexScenario) -> None:
    valid_document_ids = set(range(scenario.document_count))
    referenced_document_ids = (
        set(scenario.document_ids_without_vectors)
        | set(scenario.document_ids_to_delete)
        | set(scenario.additional_query_document_ids)
        | set(scenario.document_ids_accepted_by_filter)
    )
    if scenario.filter_query_document_id is not None:
        referenced_document_ids.add(scenario.filter_query_document_id)
    invalid_document_ids = referenced_document_ids - valid_document_ids
    if invalid_document_ids:
        raise ValueError(
            f"{scenario.name}: document IDs are outside the index: "
            f"{sorted(invalid_document_ids)}"
        )

    conflicting_document_ids = (
        scenario.document_ids_without_vectors & scenario.document_ids_to_delete
    )
    if conflicting_document_ids:
        raise ValueError(
            f"{scenario.name}: documents cannot be both vectorless and deleted: "
            f"{sorted(conflicting_document_ids)}"
        )

    if scenario.filter_query_document_id is None:
        if scenario.document_ids_accepted_by_filter:
            raise ValueError(
                f"{scenario.name}: accepted filter documents require a "
                "filtered query"
            )
        return

    if not scenario.document_ids_accepted_by_filter:
        raise ValueError(
            f"{scenario.name}: filtered query has no accepted documents"
        )
    if scenario.filter_query_document_id in (
        scenario.document_ids_without_vectors | scenario.document_ids_to_delete
    ):
        raise ValueError(
            f"{scenario.name}: filtered query document must have a live "
            f"vector: {scenario.filter_query_document_id}"
        )
    if (
        scenario.filter_query_document_id
        in scenario.document_ids_accepted_by_filter
    ):
        raise ValueError(
            f"{scenario.name}: filtered query document must be rejected by "
            "the filter"
        )


def _representative_query_document_ids(
    scenario: IndexScenario, searchable_vector_document_ids: tuple[int, ...]
) -> tuple[int, ...]:
    targets = (0, scenario.document_count // 2, scenario.document_count - 1)
    query_document_ids: list[int] = []
    for target in targets:
        nearest = min(
            searchable_vector_document_ids,
            key=lambda document_id: abs(document_id - target),
        )
        if nearest not in query_document_ids:
            query_document_ids.append(nearest)
    return tuple(query_document_ids)


def _query_document_state(
    query_document_id: int,
    document_ids_without_vectors: frozenset[int],
    deleted_document_ids: frozenset[int],
) -> QueryDocumentState:
    if query_document_id in document_ids_without_vectors:
        return QueryDocumentState.WITHOUT_VECTOR
    if query_document_id in deleted_document_ids:
        return QueryDocumentState.DELETED
    return QueryDocumentState.SEARCHABLE


def segment_document_id_ranges(
    document_count: int, segment_count: int
) -> tuple[range, ...]:
    effective_segment_count = max(1, min(segment_count, document_count))
    documents_per_segment, remainder = divmod(
        document_count, effective_segment_count
    )
    ranges = []
    start_document_id = 0
    for segment_id in range(effective_segment_count):
        documents_in_segment = documents_per_segment + (
            1 if segment_id < remainder else 0
        )
        end_document_id = start_document_id + documents_in_segment
        ranges.append(range(start_document_id, end_document_id))
        start_document_id = end_document_id
    return tuple(ranges)


def _segment_end_document_ids(scenario: IndexScenario) -> set[int]:
    return {
        document_ids.stop - 1
        for document_ids in segment_document_id_ranges(
            scenario.document_count, scenario.segment_count
        )
    }


def _segment_count(directory: Any) -> int:
    from org.apache.lucene.index import DirectoryReader

    reader = DirectoryReader.open(directory)
    try:
        return sum(1 for _ in reader.leaves())
    finally:
        reader.close()


def _new_writer_config(codec: Any, suppress_merges: bool) -> Any:
    from org.apache.lucene.index import (
        IndexWriterConfig,
        LogDocMergePolicy,
        NoMergePolicy,
        SerialMergeScheduler,
    )

    config = IndexWriterConfig()
    config.setCodec(codec)
    config.setUseCompoundFile(False)
    config.setMergeScheduler(SerialMergeScheduler())
    if suppress_merges:
        config.setMergePolicy(NoMergePolicy.INSTANCE)
    else:
        merge_policy = LogDocMergePolicy()
        merge_policy.setMergeFactor(1000)
        config.setMergePolicy(merge_policy)
    return config


def _hit_ids(stored_fields: Any, hits: Any) -> tuple[str, ...]:
    return tuple(stored_fields.document(hit.doc).get(ID_FIELD) for hit in hits)


def _vector_reader_observations(
    reader: Any,
) -> VectorReaderMetadata:
    from org.apache.lucene.codecs.hnsw import HnswGraphProvider
    from org.apache.lucene.index import SegmentReader

    vector_count = 0
    dimensions: list[int] = []
    reader_classes: list[str] = []
    hnsw_layer_counts: list[int] = []

    for leaf_reader_context in reader.leaves():
        leaf_reader = leaf_reader_context.reader()
        values = leaf_reader.getFloatVectorValues(VECTOR_FIELD)
        if values is not None:
            vector_count += values.size()
            dimensions.append(values.dimension())

        if not SegmentReader.instance_(leaf_reader):
            reader_classes.append(str(leaf_reader.getClass().getName()))
            continue

        segment_reader = SegmentReader.cast_(leaf_reader)
        vector_reader = segment_reader.getVectorReader()
        reader_classes.append(str(vector_reader.getClass().getName()))
        if HnswGraphProvider.instance_(vector_reader):
            graph_provider = HnswGraphProvider.cast_(vector_reader)
            hnsw_layer_counts.append(
                graph_provider.getGraph(VECTOR_FIELD).numLevels()
            )

    return VectorReaderMetadata(
        vector_count=vector_count,
        dimensions=tuple(dimensions),
        reader_classes=tuple(reader_classes),
        hnsw_layer_counts=tuple(hnsw_layer_counts),
    )


def _write_index_and_apply_deletions(
    directory: Any,
    codec: Any,
    scenario: IndexScenario,
    context: PyLuceneContext,
    document_ids_without_vectors: frozenset[int],
    document_ids_to_delete: frozenset[int],
) -> None:
    from org.apache.lucene.document import (
        Document,
        Field,
        KnnFloatVectorField,
        StringField,
    )
    from org.apache.lucene.index import (
        IndexWriter,
        Term,
        VectorSimilarityFunction,
    )

    writer_config = _new_writer_config(
        codec, suppress_merges=scenario.disable_automatic_merges
    )
    writer = IndexWriter(directory, writer_config)
    try:
        segment_end_document_ids = _segment_end_document_ids(scenario)
        for document_id in range(scenario.document_count):
            document = Document()
            document.add(
                StringField(ID_FIELD, f"doc-{document_id}", Field.Store.YES)
            )
            if document_id not in document_ids_without_vectors:
                vector = deterministic_float32_vector(document_id, scenario.dimensions)
                document.add(
                    KnnFloatVectorField(
                        VECTOR_FIELD,
                        to_java_float_array(context.jarray, vector),
                        VectorSimilarityFunction.EUCLIDEAN,
                    )
                )
            if scenario.filter_query_document_id is not None:
                filter_value = (
                    FILTER_INCLUDED_VALUE
                    if document_id
                    in scenario.document_ids_accepted_by_filter
                    else FILTER_EXCLUDED_VALUE
                )
                document.add(
                    StringField(
                        FILTER_FIELD,
                        filter_value,
                        Field.Store.NO,
                    )
                )
            writer.addDocument(document)
            if document_id in segment_end_document_ids:
                writer.commit()

        for document_id in document_ids_to_delete:
            writer.deleteDocuments(Term(ID_FIELD, f"doc-{document_id}"))
        if document_ids_to_delete:
            writer.commit()
    finally:
        writer.close()


def _force_merge_if_requested(
    directory: Any, codec: Any, force_merge_segment_count: int
) -> None:
    if not force_merge_segment_count:
        return

    from org.apache.lucene.index import IndexWriter

    writer = IndexWriter(
        directory, _new_writer_config(codec, suppress_merges=False)
    )
    try:
        writer.forceMerge(force_merge_segment_count)
        writer.commit()
    finally:
        writer.close()


def _new_vector_query(
    scenario: IndexScenario,
    context: PyLuceneContext,
    query_document_id: int,
    top_k: int,
    filter_field: str = "",
    filter_value: str = "",
) -> Any:
    from org.apache.lucene.index import Term
    from org.apache.lucene.search import KnnFloatVectorQuery, TermQuery

    target = deterministic_float32_vector(query_document_id, scenario.dimensions)
    if scenario.use_cagra_search_query:
        return _new_cagra_search_query(
            context,
            target,
            top_k,
            scenario.i_top_k,
            scenario.search_width,
            filter_field,
            filter_value,
        )
    if scenario.expected_hnsw_m:
        return _new_hnsw_graph_verifying_query(
            context,
            target,
            top_k,
            scenario.expected_hnsw_m,
            filter_field,
            filter_value,
        )
    filter_query = (
        TermQuery(Term(filter_field, filter_value)) if filter_field else None
    )
    if filter_query is None:
        return KnnFloatVectorQuery(
            VECTOR_FIELD,
            to_java_float_array(context.jarray, target),
            top_k,
        )
    return KnnFloatVectorQuery(
        VECTOR_FIELD,
        to_java_float_array(context.jarray, target),
        top_k,
        filter_query,
    )


def _run_filtered_query(
    searcher: Any,
    stored_fields: Any,
    context: PyLuceneContext,
    scenario: IndexScenario,
    query_document_id: int,
    accepted_document_count: int,
) -> FilteredQueryObservation:
    top_k = min(scenario.top_k, accepted_document_count)
    query = _new_vector_query(
        scenario,
        context,
        query_document_id,
        top_k,
        FILTER_FIELD,
        FILTER_INCLUDED_VALUE,
    )
    hits = searcher.search(query, top_k).scoreDocs
    return FilteredQueryObservation(
        query_document_id=query_document_id,
        hit_ids=_hit_ids(stored_fields, hits),
        query_class=str(query.getClass().getName()),
    )


def run_index_scenario(
    scenario: IndexScenario, context: PyLuceneContext
) -> IndexRun:
    from java.nio.file import Paths
    from org.apache.lucene.index import DirectoryReader
    from org.apache.lucene.search import IndexSearcher
    from org.apache.lucene.store import FSDirectory

    codec = _codec_for_scenario(scenario, context)
    _validate_document_ids(scenario)
    document_ids_without_vectors = scenario.document_ids_without_vectors
    document_ids_to_delete = scenario.document_ids_to_delete
    searchable_vector_document_ids = tuple(
        document_id
        for document_id in range(scenario.document_count)
        if document_id not in document_ids_without_vectors
        and document_id not in document_ids_to_delete
    )
    if not searchable_vector_document_ids:
        raise RuntimeError(
            f"{scenario.name}: no searchable vectors are available"
        )
    filter_accepted_searchable_document_ids = tuple(
        document_id
        for document_id in searchable_vector_document_ids
        if document_id in scenario.document_ids_accepted_by_filter
    )
    if (
        scenario.filter_query_document_id is not None
        and not filter_accepted_searchable_document_ids
    ):
        raise RuntimeError(
            f"{scenario.name}: filter accepts no searchable vectors"
        )
    representative_query_document_ids = _representative_query_document_ids(
        scenario, searchable_vector_document_ids
    )

    with ExitStack() as stack:
        index_path = stack.enter_context(
            tempfile.TemporaryDirectory(
                prefix=f"cuvs-lucene-pylucene-{scenario.name}-"
            )
        )
        directory = FSDirectory.open(Paths.get(index_path))
        try:
            _write_index_and_apply_deletions(
                directory,
                codec,
                scenario,
                context,
                document_ids_without_vectors,
                document_ids_to_delete,
            )

            pre_merge_segment_count = _segment_count(directory)
            _force_merge_if_requested(
                directory, codec, scenario.force_merge_segment_count
            )

            index_files = tuple(
                sorted(path.name for path in Path(index_path).iterdir())
            )
            reader = DirectoryReader.open(directory)
            try:
                vector_reader_metadata = _vector_reader_observations(reader)
                searcher = IndexSearcher(reader)
                stored_fields = searcher.storedFields()
                top_k = min(scenario.top_k, len(searchable_vector_document_ids))
                observations = []
                query_document_ids = tuple(
                    dict.fromkeys(
                        representative_query_document_ids
                        + scenario.additional_query_document_ids
                    )
                )
                for query_document_id in query_document_ids:
                    query = _new_vector_query(
                        scenario, context, query_document_id, top_k
                    )
                    hits = searcher.search(query, top_k).scoreDocs
                    observations.append(
                        QueryObservation(
                            query_document_id=query_document_id,
                            hit_ids=_hit_ids(stored_fields, hits),
                            query_class=str(query.getClass().getName()),
                            query_document_state=_query_document_state(
                                query_document_id,
                                document_ids_without_vectors,
                                document_ids_to_delete,
                            ),
                        )
                    )

                filtered_query_observation = (
                    _run_filtered_query(
                        searcher,
                        stored_fields,
                        context,
                        scenario,
                        scenario.filter_query_document_id,
                        len(filter_accepted_searchable_document_ids),
                    )
                    if scenario.filter_query_document_id is not None
                    else None
                )

                result = IndexRun(
                    index_files=index_files,
                    pre_merge_segment_count=pre_merge_segment_count,
                    segment_count=sum(1 for _ in reader.leaves()),
                    live_document_count=reader.numDocs(),
                    max_document_count=reader.maxDoc(),
                    vector_count=vector_reader_metadata.vector_count,
                    vector_dimensions=vector_reader_metadata.dimensions,
                    vector_reader_classes=vector_reader_metadata.reader_classes,
                    hnsw_layer_counts=vector_reader_metadata.hnsw_layer_counts,
                    searchable_vector_document_ids=searchable_vector_document_ids,
                    document_ids_without_vectors=document_ids_without_vectors,
                    deleted_document_ids=document_ids_to_delete,
                    query_observations=tuple(observations),
                    filtered_query_observation=filtered_query_observation,
                    writer_telemetry=_writer_telemetry(codec),
                )
            finally:
                reader.close()
        finally:
            directory.close()

    return result
