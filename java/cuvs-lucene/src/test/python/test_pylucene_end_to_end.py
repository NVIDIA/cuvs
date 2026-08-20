# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyLucene end-to-end coverage for CPU HNSW and GPU cuVS search paths.

The parametrized cases cover segment and force-merge topologies, CAGRA search
widths, persisted HNSW layer counts, deletions, filters, and brute-force recall.
GPU-required cases assert that cuVS ran and did not silently fall back to the CPU path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from zipfile import ZipFile

import pytest

from pylucene_test_support import (
    CAGRA_BUILT_HNSW_BASE_LAYER_TEST_CODEC_CLASS,
    CAGRA_BUILT_HNSW_THREE_LAYER_TEST_CODEC_CLASS,
    CAGRA_TEST_CODEC_CLASS,
    CAGRA_TEST_QUERY_CLASS,
    CAGRA_VECTOR_READER_CLASS,
    CPU_HNSW_TEST_CODEC_CLASS,
    HNSW_GRAPH_VERIFYING_QUERY_CLASS,
    IndexRun,
    IndexScenario,
    PyLuceneContext,
    QueryDocumentState,
    deterministic_float32_vector,
    find_cuvs_lucene_jar,
    initialize_pylucene_context,
    run_index_scenario,
    segment_document_id_ranges,
    validate_pylucene_version,
)

# Python 3.14 reports one deprecation per JCC-generated PyLucene builtin type.
# This narrow third-party filter keeps native/cuVS warnings fully visible.
pytestmark = pytest.mark.filterwarnings(
    "ignore:builtin type .* has no __module__ attribute:DeprecationWarning"
)

# TODO(https://github.com/NVIDIA/cuvs/issues/2407): Add multithreaded concurrency coverage.

HNSW_CODEC = "Lucene101AcceleratedHNSWCodec"
CAGRA_CODEC = "CuVS2510GPUSearchCodec"

CPU_HNSW_WRITER_CLASS = (
    "org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsWriter"
)
GPU_CAGRA_BUILT_HNSW_WRITER_CLASS = (
    "com.nvidia.cuvs.lucene.Lucene99AcceleratedHNSWVectorsWriter"
)
GPU_CAGRA_SEARCH_WRITER_CLASS = (
    "com.nvidia.cuvs.lucene.CuVS2510GPUVectorsWriter"
)

EXPECTED_SPI_CODECS = (
    HNSW_CODEC,
    CAGRA_CODEC,
    "Lucene101AcceleratedHNSWBinaryQuantizedCodec",
    "Lucene101AcceleratedHNSWScalarQuantizedCodec",
)
CODEC_SERVICE = "META-INF/services/org.apache.lucene.codecs.Codec"
VECTOR_FORMAT_SERVICE = (
    "META-INF/services/org.apache.lucene.codecs.KnnVectorsFormat"
)
EXPECTED_CODEC_PROVIDERS = frozenset(
    {
        "com.nvidia.cuvs.lucene.Lucene101AcceleratedHNSWCodec",
        "com.nvidia.cuvs.lucene.CuVS2510GPUSearchCodec",
        (
            "com.nvidia.cuvs.lucene."
            "LuceneAcceleratedHNSWBinaryQuantizedCodec"
        ),
        (
            "com.nvidia.cuvs.lucene."
            "LuceneAcceleratedHNSWScalarQuantizedCodec"
        ),
    }
)
EXPECTED_VECTOR_FORMAT_PROVIDERS = frozenset(
    {
        "com.nvidia.cuvs.lucene.CuVS2510GPUVectorsFormat",
        "com.nvidia.cuvs.lucene.Lucene99AcceleratedHNSWVectorsFormat",
        (
            "com.nvidia.cuvs.lucene."
            "LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat"
        ),
        (
            "com.nvidia.cuvs.lucene."
            "LuceneAcceleratedHNSWScalarQuantizedVectorsFormat"
        ),
    }
)
REQUIRED_CUVS_LUCENE_CLASSES = frozenset(
    provider.replace(".", "/") + ".class"
    for provider in EXPECTED_CODEC_PROVIDERS | EXPECTED_VECTOR_FORMAT_PROVIDERS
)

CAGRA_GRAPH_DEGREE = 32
CAGRA_INTERMEDIATE_GRAPH_DEGREE = 64
# cuVS expands NN-Descent's requested degree by 1.5 and reduces it with a
# warning when that internal degree is at least the number of input vectors.
NN_DESCENT_INTERNAL_GRAPH_DEGREE = CAGRA_INTERMEDIATE_GRAPH_DEGREE * 3 // 2
MIN_VECTORS_PER_CAGRA_BUILD = NN_DESCENT_INTERNAL_GRAPH_DEGREE + 1
CAGRA_HNSW_M = CAGRA_GRAPH_DEGREE // 2
MIN_VECTORS_FOR_THREE_HNSW_LAYERS = (
    MIN_VECTORS_PER_CAGRA_BUILD * CAGRA_HNSW_M**2
)
DEFAULT_MIN_RECALL = 0.75


class ExecutionPath(Enum):
    CPU_HNSW = (
        "CPU HNSW build -> HNSW search",
        "cpu-hnsw",
        CPU_HNSW_WRITER_CLASS,
    )
    GPU_CAGRA_BUILT_HNSW = (
        "GPU CAGRA build -> HNSW search",
        "gpu-cagra-built-hnsw",
        GPU_CAGRA_BUILT_HNSW_WRITER_CLASS,
    )
    GPU_CAGRA_SEARCH = (
        "GPU CAGRA build -> CAGRA search",
        "gpu-cagra-search",
        GPU_CAGRA_SEARCH_WRITER_CLASS,
    )

    def __init__(
        self, label: str, selector: str, expected_writer_class: str
    ) -> None:
        self.label = label
        self.selector = selector
        self.expected_writer_class = expected_writer_class

    @property
    def requires_gpu(self) -> bool:
        return self is not ExecutionPath.CPU_HNSW


class DocumentSetup(Enum):
    ALL_SEARCHABLE = "all-searchable"
    ONE_DELETED = "one-deleted"
    SINGLE_LIVE = "single-live"


@dataclass(frozen=True)
class SuiteSettings:
    requested_document_count: int
    dimensions: int
    top_k: int
    minimum_recall: float


@dataclass(frozen=True)
class DocumentConfiguration:
    document_ids_without_vectors: frozenset[int] = frozenset()
    document_ids_to_delete: frozenset[int] = frozenset()
    additional_query_document_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class DocumentFilterConfiguration:
    query_document_id: int | None = None
    accepted_document_ids: frozenset[int] = frozenset()


@dataclass(frozen=True)
class EndToEndCase:
    selector: str
    scenario: IndexScenario
    execution_path: ExecutionPath
    expected_index_file_suffixes: tuple[str, ...]
    expected_hnsw_layers: int = 0
    min_recall: float = DEFAULT_MIN_RECALL


def _positive_int_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _boolean_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def _minimum_recall_from_env() -> float:
    value = float(
        os.environ.get("CUVS_LUCENE_PYLUCENE_MIN_RECALL", DEFAULT_MIN_RECALL)
    )
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            "CUVS_LUCENE_PYLUCENE_MIN_RECALL must be between 0 and 1, "
            f"got {value}"
        )
    return value


def _suite_settings() -> SuiteSettings:
    requested_document_count = _positive_int_env(
        "CUVS_LUCENE_PYLUCENE_ROWS", 2000
    )
    dimensions = _positive_int_env("CUVS_LUCENE_PYLUCENE_DIMS", 32)
    top_k = _positive_int_env("CUVS_LUCENE_PYLUCENE_TOPK", 20)
    if dimensions > 4096:
        raise ValueError(
            f"CUVS_LUCENE_PYLUCENE_DIMS must be at most 4096, got {dimensions}"
        )
    return SuiteSettings(
        requested_document_count=requested_document_count,
        dimensions=dimensions,
        top_k=top_k,
        minimum_recall=_minimum_recall_from_env(),
    )


def _document_configuration(
    document_count: int, setup: DocumentSetup
) -> DocumentConfiguration:
    middle_document_id = document_count // 2
    if setup is DocumentSetup.SINGLE_LIVE:
        deleted_document_ids = frozenset(
            document_id
            for document_id in range(document_count)
            if document_id != middle_document_id
        )
        return DocumentConfiguration(
            document_ids_to_delete=deleted_document_ids,
            additional_query_document_ids=(0,),
        )
    if setup is DocumentSetup.ONE_DELETED:
        return DocumentConfiguration(
            document_ids_to_delete=frozenset({middle_document_id}),
            additional_query_document_ids=(middle_document_id,),
        )
    return DocumentConfiguration()


def _minimum_document_count_for_selective_filter(
    segment_count: int, top_k: int
) -> int:
    return segment_count * 4 * (top_k + 1)


def _selective_document_filter_configuration(
    document_count: int, segment_count: int, top_k: int
) -> DocumentFilterConfiguration:
    accepted_document_ids = set()
    for segment_document_ids in segment_document_id_ranges(
        document_count, segment_count
    ):
        accepted_document_count = max(
            top_k + 1, len(segment_document_ids) // 4
        )
        first_accepted_document_id = (
            segment_document_ids.stop - accepted_document_count
        )
        accepted_document_ids.update(
            range(
                first_accepted_document_id,
                segment_document_ids.stop,
            )
        )

    query_document_id = 0
    if query_document_id in accepted_document_ids:
        raise ValueError(
            "selective filter must reject its query document"
        )
    return DocumentFilterConfiguration(
        query_document_id=query_document_id,
        accepted_document_ids=frozenset(accepted_document_ids),
    )


def _minimum_document_count(
    execution_path: ExecutionPath,
    segment_count: int,
    hnsw_layers: int,
    document_setup: DocumentSetup,
) -> int:
    if execution_path is ExecutionPath.CPU_HNSW:
        required_document_count = segment_count
    elif hnsw_layers == 3:
        required_document_count = MIN_VECTORS_FOR_THREE_HNSW_LAYERS
    else:
        required_document_count = (
            segment_count * MIN_VECTORS_PER_CAGRA_BUILD
        )

    if document_setup is DocumentSetup.ONE_DELETED:
        required_document_count = max(required_document_count, 2)
    return required_document_count


def _cpu_hnsw_case(
    selector: str,
    *,
    segment_count: int = 1,
    force_merge_segment_count: int = 0,
    document_setup: DocumentSetup = DocumentSetup.ALL_SEARCHABLE,
    selective_filter: bool = False,
) -> EndToEndCase:
    settings = _suite_settings()
    minimum_document_count = _minimum_document_count(
        ExecutionPath.CPU_HNSW,
        segment_count,
        0,
        document_setup,
    )
    if selective_filter:
        minimum_document_count = max(
            minimum_document_count,
            _minimum_document_count_for_selective_filter(
                segment_count, settings.top_k
            ),
        )
    document_count = max(
        settings.requested_document_count, minimum_document_count
    )
    documents = _document_configuration(document_count, document_setup)
    document_filter = (
        _selective_document_filter_configuration(
            document_count, segment_count, settings.top_k
        )
        if selective_filter
        else DocumentFilterConfiguration()
    )
    scenario = IndexScenario(
        name=selector,
        codec_name=HNSW_CODEC,
        codec_factory_class=CPU_HNSW_TEST_CODEC_CLASS,
        document_count=document_count,
        dimensions=settings.dimensions,
        top_k=settings.top_k,
        segment_count=segment_count,
        force_merge_segment_count=force_merge_segment_count,
        document_ids_without_vectors=documents.document_ids_without_vectors,
        document_ids_to_delete=documents.document_ids_to_delete,
        additional_query_document_ids=documents.additional_query_document_ids,
        document_ids_accepted_by_filter=document_filter.accepted_document_ids,
        filter_query_document_id=document_filter.query_document_id,
        expected_hnsw_m=32,
    )
    return EndToEndCase(
        selector=selector,
        scenario=scenario,
        execution_path=ExecutionPath.CPU_HNSW,
        expected_index_file_suffixes=(".vex", ".vem"),
        min_recall=settings.minimum_recall,
    )


def _cagra_built_hnsw_case(
    selector: str,
    *,
    hnsw_layers: int = 1,
    segment_count: int = 1,
    force_merge_segment_count: int = 0,
    document_setup: DocumentSetup = DocumentSetup.ALL_SEARCHABLE,
    selective_filter: bool = False,
) -> EndToEndCase:
    settings = _suite_settings()
    minimum_document_count = _minimum_document_count(
        ExecutionPath.GPU_CAGRA_BUILT_HNSW,
        segment_count,
        hnsw_layers,
        document_setup,
    )
    if selective_filter:
        minimum_document_count = max(
            minimum_document_count,
            _minimum_document_count_for_selective_filter(
                segment_count, settings.top_k
            ),
        )
    document_count = max(
        settings.requested_document_count, minimum_document_count
    )
    documents = _document_configuration(document_count, document_setup)
    document_filter = (
        _selective_document_filter_configuration(
            document_count, segment_count, settings.top_k
        )
        if selective_filter
        else DocumentFilterConfiguration()
    )
    codec_factory_class = (
        CAGRA_BUILT_HNSW_THREE_LAYER_TEST_CODEC_CLASS
        if hnsw_layers == 3
        else CAGRA_BUILT_HNSW_BASE_LAYER_TEST_CODEC_CLASS
    )
    scenario = IndexScenario(
        name=selector,
        codec_name=HNSW_CODEC,
        codec_factory_class=codec_factory_class,
        document_count=document_count,
        dimensions=settings.dimensions,
        top_k=settings.top_k,
        segment_count=segment_count,
        force_merge_segment_count=force_merge_segment_count,
        document_ids_without_vectors=documents.document_ids_without_vectors,
        document_ids_to_delete=documents.document_ids_to_delete,
        additional_query_document_ids=documents.additional_query_document_ids,
        document_ids_accepted_by_filter=document_filter.accepted_document_ids,
        filter_query_document_id=document_filter.query_document_id,
        expected_hnsw_m=16,
    )
    return EndToEndCase(
        selector=selector,
        scenario=scenario,
        execution_path=ExecutionPath.GPU_CAGRA_BUILT_HNSW,
        expected_index_file_suffixes=(".vex", ".vem"),
        expected_hnsw_layers=hnsw_layers,
        min_recall=settings.minimum_recall,
    )


def _cagra_search_case(
    selector: str,
    *,
    segment_count: int = 1,
    force_merge_segment_count: int = 0,
    document_setup: DocumentSetup = DocumentSetup.ALL_SEARCHABLE,
    search_width: int = 1,
    selective_filter: bool = False,
) -> EndToEndCase:
    settings = _suite_settings()
    if settings.top_k > 1024:
        raise ValueError(
            "GPU CAGRA search cases require "
            "CUVS_LUCENE_PYLUCENE_TOPK <= 1024"
        )
    minimum_document_count = _minimum_document_count(
        ExecutionPath.GPU_CAGRA_SEARCH,
        segment_count,
        0,
        document_setup,
    )
    if selective_filter:
        minimum_document_count = max(
            minimum_document_count,
            _minimum_document_count_for_selective_filter(
                segment_count, settings.top_k
            ),
        )
    document_count = max(
        settings.requested_document_count, minimum_document_count
    )
    documents = _document_configuration(document_count, document_setup)
    document_filter = (
        _selective_document_filter_configuration(
            document_count, segment_count, settings.top_k
        )
        if selective_filter
        else DocumentFilterConfiguration()
    )
    scenario = IndexScenario(
        name=selector,
        codec_name=CAGRA_CODEC,
        codec_factory_class=CAGRA_TEST_CODEC_CLASS,
        document_count=document_count,
        dimensions=settings.dimensions,
        top_k=settings.top_k,
        segment_count=segment_count,
        force_merge_segment_count=force_merge_segment_count,
        document_ids_without_vectors=documents.document_ids_without_vectors,
        document_ids_to_delete=documents.document_ids_to_delete,
        additional_query_document_ids=documents.additional_query_document_ids,
        document_ids_accepted_by_filter=document_filter.accepted_document_ids,
        filter_query_document_id=document_filter.query_document_id,
        use_cagra_search_query=True,
        search_width=search_width,
        i_top_k=max(64, settings.top_k),
    )
    return EndToEndCase(
        selector=selector,
        scenario=scenario,
        execution_path=ExecutionPath.GPU_CAGRA_SEARCH,
        expected_index_file_suffixes=(".vcag", ".vemc"),
        min_recall=settings.minimum_recall,
    )


SEGMENT_CASES = (
    _cpu_hnsw_case(
        "cpu-hnsw-1-segment",
    ),
    _cpu_hnsw_case(
        "cpu-hnsw-10-segments",
        segment_count=10,
    ),
    _cagra_built_hnsw_case(
        "gpu-cagra-built-hnsw-1-segment",
    ),
    _cagra_built_hnsw_case(
        "gpu-cagra-built-hnsw-10-segments",
        segment_count=10,
    ),
    _cagra_search_case(
        "gpu-cagra-search-10-segments",
        segment_count=10,
    ),
)

SINGLE_LIVE_DOCUMENT_CASES = (
    _cagra_search_case(
        "gpu-cagra-search-single-live-doc",
        document_setup=DocumentSetup.SINGLE_LIVE,
    ),
)

FORCE_MERGE_CASES = (
    _cpu_hnsw_case(
        "cpu-hnsw-10-to-1-force-merge",
        segment_count=10,
        force_merge_segment_count=1,
    ),
    _cpu_hnsw_case(
        "cpu-hnsw-100-to-10-force-merge",
        segment_count=100,
        force_merge_segment_count=10,
    ),
    _cagra_built_hnsw_case(
        "gpu-cagra-built-hnsw-10-to-1-force-merge",
        segment_count=10,
        force_merge_segment_count=1,
    ),
    _cagra_built_hnsw_case(
        "gpu-cagra-built-hnsw-100-to-10-force-merge",
        segment_count=100,
        force_merge_segment_count=10,
    ),
    _cagra_search_case(
        "gpu-cagra-search-10-to-1-force-merge",
        segment_count=10,
        force_merge_segment_count=1,
    ),
    _cagra_search_case(
        "gpu-cagra-search-100-to-10-force-merge",
        segment_count=100,
        force_merge_segment_count=10,
    ),
)

HNSW_LAYER_CASES = (
    _cagra_built_hnsw_case(
        "gpu-cagra-built-hnsw-3-layers",
        hnsw_layers=3,
    ),
)

CAGRA_SEARCH_WIDTH_CASES = (
    _cagra_search_case(
        "gpu-cagra-search-1-segment",
    ),
    _cagra_search_case(
        "gpu-cagra-search-width-16",
        search_width=16,
    ),
    _cagra_search_case(
        "gpu-cagra-search-width-32",
        search_width=32,
    ),
)

DELETED_DOCUMENT_CASES = (
    _cagra_search_case(
        "gpu-cagra-search-deleted-documents",
        document_setup=DocumentSetup.ONE_DELETED,
    ),
)

DOCUMENT_FILTER_CASES = (
    _cpu_hnsw_case(
        "cpu-hnsw-selective-filter",
        selective_filter=True,
    ),
    _cagra_built_hnsw_case(
        "gpu-cagra-built-hnsw-selective-filter",
        selective_filter=True,
    ),
    _cagra_search_case(
        "gpu-cagra-search-selective-filter-10-segments",
        segment_count=10,
        selective_filter=True,
    ),
)


def _case_parameter(case: EndToEndCase) -> object:
    return pytest.param(case, id=case.selector)


def _case_parameters(
    cases: tuple[EndToEndCase, ...],
) -> tuple[object, ...]:
    return tuple(_case_parameter(case) for case in cases)


def _service_providers(
    archive: ZipFile, service_path: str
) -> frozenset[str]:
    lines = archive.read(service_path).decode("utf-8").splitlines()
    return frozenset(
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    )


def test_published_jar_has_expected_lucene_services() -> None:
    """Verify the published jar exposes every expected Lucene SPI service."""
    cuvs_lucene_jar = find_cuvs_lucene_jar()
    with ZipFile(cuvs_lucene_jar) as archive:
        entries = frozenset(archive.namelist())
        expected_services = frozenset(
            {CODEC_SERVICE, VECTOR_FORMAT_SERVICE}
        )
        assert expected_services <= entries, (
            f"{cuvs_lucene_jar.name} is missing service descriptors: "
            f"{sorted(expected_services - entries)}"
        )
        assert REQUIRED_CUVS_LUCENE_CLASSES <= entries, (
            f"{cuvs_lucene_jar.name} is missing classes: "
            f"{sorted(REQUIRED_CUVS_LUCENE_CLASSES - entries)}"
        )

        lucene_service_descriptors = {
            entry
            for entry in entries
            if entry.startswith("META-INF/services/org.apache.lucene.")
        }
        assert lucene_service_descriptors == expected_services

        codec_providers = _service_providers(archive, CODEC_SERVICE)
        vector_format_providers = _service_providers(
            archive, VECTOR_FORMAT_SERVICE
        )
        assert EXPECTED_CODEC_PROVIDERS <= codec_providers
        assert EXPECTED_VECTOR_FORMAT_PROVIDERS <= vector_format_providers
        assert not any(
            provider.startswith("org.apache.lucene.")
            for provider in codec_providers | vector_format_providers
        )


def test_published_jar_does_not_bundle_lucene_or_cuvs_java() -> None:
    """Keep PyLucene's Lucene classes and the base cuVS jar external."""
    cuvs_lucene_jar = find_cuvs_lucene_jar()
    with ZipFile(cuvs_lucene_jar) as archive:
        entries = frozenset(archive.namelist())

    bundled_lucene_classes = sorted(
        entry
        for entry in entries
        if entry.startswith("org/apache/lucene/") and not entry.endswith("/")
    )
    assert not bundled_lucene_classes, (
        "PyLucene must supply Lucene classes; the cuvs-lucene jar contains "
        f"{bundled_lucene_classes}"
    )

    bundled_cuvs_java_classes = sorted(
        entry
        for entry in entries
        if entry.startswith("com/nvidia/cuvs/")
        and not entry.startswith("com/nvidia/cuvs/lucene/")
        and not entry.endswith("/")
    )
    assert not bundled_cuvs_java_classes, (
        "The base cuvs-java jar must remain separate; cuvs-lucene contains "
        f"{bundled_cuvs_java_classes}"
    )

    bundled_multi_release_classes = sorted(
        entry
        for entry in entries
        if entry.startswith("META-INF/versions/")
        and "/com/nvidia/cuvs/" in entry
        and not entry.endswith("/")
    )
    assert not bundled_multi_release_classes, (
        "The base multi-release cuvs-java jar must remain separate; "
        f"cuvs-lucene contains {bundled_multi_release_classes}"
    )


def test_published_jar_excludes_pylucene_test_support() -> None:
    """Keep the Java test bridge out of the published artifact."""
    cuvs_lucene_jar = find_cuvs_lucene_jar()
    with ZipFile(cuvs_lucene_jar) as archive:
        entries = frozenset(archive.namelist())

    test_support_prefix = "com/nvidia/cuvs/lucene/PyLuceneTestSupport"
    published_test_classes = sorted(
        entry for entry in entries if entry.startswith(test_support_prefix)
    )
    assert not published_test_classes


def test_pylucene_version_must_match_cuvs_lucene() -> None:
    validate_pylucene_version("10.2.0", "10.2.0")
    with pytest.raises(
        RuntimeError,
        match=r"expected 10\.2\.0, found 10\.0\.0",
    ):
        validate_pylucene_version("10.0.0", "10.2.0")


@pytest.fixture(scope="session")
def pylucene_context() -> PyLuceneContext:
    expected_codecs = (
        EXPECTED_SPI_CODECS
        if _boolean_env("CUVS_LUCENE_VERIFY_ALL_CODECS", True)
        else ()
    )
    return initialize_pylucene_context(expected_codecs)


def _squared_distance(
    left: tuple[float, ...], right: tuple[float, ...]
) -> float:
    return sum(
        (left_value - right_value) ** 2
        for left_value, right_value in zip(left, right)
    )


def _brute_force_neighbor_ids(
    case: EndToEndCase,
    result: IndexRun,
    query_document_id: int,
    candidate_document_ids: tuple[int, ...] | None = None,
) -> tuple[str, ...]:
    query_vector = deterministic_float32_vector(
        query_document_id, case.scenario.dimensions
    )
    candidates = (
        result.searchable_vector_document_ids
        if candidate_document_ids is None
        else candidate_document_ids
    )
    ordered_ids = sorted(
        candidates,
        key=lambda document_id: (
            _squared_distance(
                query_vector,
                deterministic_float32_vector(
                    document_id, case.scenario.dimensions
                ),
            ),
            document_id,
        ),
    )
    return tuple(
        f"doc-{document_id}"
        for document_id in ordered_ids[: case.scenario.top_k]
    )


def _assert_index_files_and_segments(
    case: EndToEndCase, result: IndexRun
) -> None:
    for suffix in case.expected_index_file_suffixes:
        assert any(name.endswith(suffix) for name in result.index_files), (
            f"{case.selector}: no index file ending with {suffix}; "
            f"files={result.index_files}"
        )

    expected_initial_segments = min(
        case.scenario.segment_count, case.scenario.document_count
    )
    assert result.pre_merge_segment_count == expected_initial_segments, (
        f"{case.selector}: expected {expected_initial_segments} segments "
        f"before forceMerge, got {result.pre_merge_segment_count}"
    )
    expected_final_segments = (
        case.scenario.force_merge_segment_count or expected_initial_segments
    )
    assert result.segment_count == expected_final_segments, (
        f"{case.selector}: expected {expected_final_segments} final segments, "
        f"got {result.segment_count}"
    )


def _assert_index_metadata(case: EndToEndCase, result: IndexRun) -> None:
    expected_live_documents = (
        case.scenario.document_count - len(result.deleted_document_ids)
    )
    assert result.live_document_count == expected_live_documents
    expected_max_documents = (
        expected_live_documents
        if case.scenario.force_merge_segment_count
        else case.scenario.document_count
    )
    assert result.max_document_count == expected_max_documents

    expected_vector_count = (
        len(result.searchable_vector_document_ids)
        if case.scenario.force_merge_segment_count
        else (
            case.scenario.document_count
            - len(result.document_ids_without_vectors)
        )
    )
    assert result.vector_count == expected_vector_count, (
        f"{case.selector}: expected {expected_vector_count} vector values, "
        f"got {result.vector_count}"
    )
    assert result.vector_dimensions
    assert set(result.vector_dimensions) == {case.scenario.dimensions}


def _assert_execution_path(case: EndToEndCase, result: IndexRun) -> None:
    telemetry = result.writer_telemetry
    assert telemetry.get("configuredPath") == case.execution_path.selector
    assert telemetry.get("writerClass") == case.execution_path.expected_writer_class

    if case.execution_path is ExecutionPath.GPU_CAGRA_SEARCH:
        assert set(result.vector_reader_classes) == {CAGRA_VECTOR_READER_CLASS}
        assert all(
            observation.query_class == CAGRA_TEST_QUERY_CLASS
            for observation in result.query_observations
        )
        if result.filtered_query_observation is not None:
            assert (
                result.filtered_query_observation.query_class
                == CAGRA_TEST_QUERY_CLASS
            )
        assert not result.hnsw_layer_counts
        return

    assert result.vector_reader_classes
    assert all(
        observation.query_class == HNSW_GRAPH_VERIFYING_QUERY_CLASS
        for observation in result.query_observations
    )
    if result.filtered_query_observation is not None:
        assert (
            result.filtered_query_observation.query_class
            == HNSW_GRAPH_VERIFYING_QUERY_CLASS
        )
    assert all(
        "Lucene99HnswVectorsReader" in reader_class
        for reader_class in result.vector_reader_classes
    )
    assert all(
        not reader_class.startswith("com.nvidia.cuvs.lucene")
        for reader_class in result.vector_reader_classes
    )


def _assert_graph_configuration(
    case: EndToEndCase, result: IndexRun
) -> None:
    if case.expected_hnsw_layers:
        assert result.hnsw_layer_counts
        assert set(result.hnsw_layer_counts) == {
            case.expected_hnsw_layers
        }, (
            f"{case.selector}: expected {case.expected_hnsw_layers} HNSW "
            f"layers, got {result.hnsw_layer_counts}"
        )

    if not case.execution_path.requires_gpu:
        return

    telemetry = result.writer_telemetry
    assert telemetry.get("cagraGraphBuildAlgo") == "NN_DESCENT"
    assert telemetry.get("cagraGraphDegree") == "32"
    assert telemetry.get("cagraIntermediateGraphDegree") == "64"
    if case.execution_path is ExecutionPath.GPU_CAGRA_SEARCH:
        assert telemetry.get("cagraStrategy") == "CUSTOM"


def _assert_search_results(
    case: EndToEndCase, result: IndexRun
) -> tuple[float, ...]:
    documents_without_vectors = {
        f"doc-{document_id}"
        for document_id in result.document_ids_without_vectors
    }
    deleted_documents = {
        f"doc-{document_id}" for document_id in result.deleted_document_ids
    }
    expected_hit_count = min(
        case.scenario.top_k, len(result.searchable_vector_document_ids)
    )

    recalls = []
    for observation in result.query_observations:
        hits = observation.hit_ids
        queried_document = f"doc-{observation.query_document_id}"
        if (
            observation.query_document_state
            is QueryDocumentState.SEARCHABLE
        ):
            assert len(hits) == expected_hit_count, (
                f"{case.selector}: query {queried_document} expected "
                f"{expected_hit_count} hits, got {hits}"
            )
            assert hits[0] == queried_document, (
                f"{case.selector}: queried document must rank first; "
                f"expected {queried_document}, got {hits}"
            )
        else:
            assert queried_document not in hits, (
                f"{case.selector}: {observation.query_document_state.value} "
                f"document was returned: {hits}"
            )

        assert len(hits) == len(set(hits)), (
            f"{case.selector}: duplicate hits returned: {hits}"
        )
        returned_vectorless_documents = tuple(
            document
            for document in hits
            if document in documents_without_vectors
        )
        assert not returned_vectorless_documents, (
            f"{case.selector}: live documents without vectors were returned: "
            f"{returned_vectorless_documents}"
        )
        returned_deleted_documents = tuple(
            document for document in hits if document in deleted_documents
        )
        assert not returned_deleted_documents, (
            f"{case.selector}: deleted documents were returned: "
            f"{returned_deleted_documents}"
        )

        if (
            observation.query_document_state
            is not QueryDocumentState.SEARCHABLE
        ):
            continue

        expected_neighbors = _brute_force_neighbor_ids(
            case, result, observation.query_document_id
        )
        recall = (
            len(set(hits) & set(expected_neighbors))
            / len(expected_neighbors)
        )
        assert recall >= case.min_recall, (
            f"{case.selector}: query {queried_document} recall "
            f"{recall:.3f} is below floor {case.min_recall:.3f}; "
            f"expected={expected_neighbors}, actual={hits}"
        )
        recalls.append(recall)

    return tuple(recalls)


def _print_result(
    case: EndToEndCase, result: IndexRun, recalls: tuple[float, ...]
) -> None:
    report_label = case.selector.removeprefix(
        f"{case.execution_path.selector}-"
    )
    segment_summary = (
        f"{result.pre_merge_segment_count}->{result.segment_count}"
        if case.scenario.force_merge_segment_count
        else str(result.segment_count)
    )
    details = [
        f"documents={case.scenario.document_count}",
        f"liveDocuments={result.live_document_count}",
        f"segments={segment_summary}",
        f"topK={case.scenario.top_k}",
        f"recall(min/avg)={min(recalls):.3f}/"
        f"{sum(recalls) / len(recalls):.3f}",
    ]
    if result.document_ids_without_vectors:
        details.append(
            f"documentsWithoutVectors={len(result.document_ids_without_vectors)}"
        )
    if result.deleted_document_ids:
        details.append(f"deletedDocuments={len(result.deleted_document_ids)}")
    if case.execution_path is ExecutionPath.GPU_CAGRA_SEARCH:
        details.append(f"searchWidth={case.scenario.search_width}")
    else:
        details.append(
            f"hnswLayers={sorted(set(result.hnsw_layer_counts))}"
        )
        details.append(f"hnswM={case.scenario.expected_hnsw_m}")

    print(
        f"PASS [{case.execution_path.label}] {report_label}: "
        + ", ".join(details)
    )


def _run_and_verify(
    case: EndToEndCase, context: PyLuceneContext
) -> tuple[IndexRun, tuple[float, ...]]:
    result = run_index_scenario(case.scenario, context)
    _assert_index_files_and_segments(case, result)
    _assert_index_metadata(case, result)
    _assert_execution_path(case, result)
    _assert_graph_configuration(case, result)
    recalls = _assert_search_results(case, result)
    _print_result(case, result, recalls)
    return result, recalls


@pytest.mark.parametrize("case", _case_parameters(SEGMENT_CASES))
def test_search_with_configured_segment_count(
    pylucene_context: PyLuceneContext, case: EndToEndCase
) -> None:
    """Exercise each execution path across configured segment topologies."""
    result, _ = _run_and_verify(case, pylucene_context)
    assert not result.document_ids_without_vectors
    assert not result.deleted_document_ids


@pytest.mark.parametrize("case", _case_parameters(SINGLE_LIVE_DOCUMENT_CASES))
def test_cagra_search_with_single_live_document(
    pylucene_context: PyLuceneContext, case: EndToEndCase
) -> None:
    """Search a warning-free CAGRA index after deleting all but one document."""
    result, _ = _run_and_verify(case, pylucene_context)
    assert result.live_document_count == 1
    assert len(result.searchable_vector_document_ids) == 1
    assert len(result.deleted_document_ids) == result.max_document_count - 1
    assert any(
        observation.query_document_state is QueryDocumentState.DELETED
        for observation in result.query_observations
    )


@pytest.mark.parametrize("case", _case_parameters(FORCE_MERGE_CASES))
def test_search_after_force_merge(
    pylucene_context: PyLuceneContext, case: EndToEndCase
) -> None:
    """Verify CPU and GPU search paths after representative force merges."""
    result, _ = _run_and_verify(case, pylucene_context)
    assert not result.document_ids_without_vectors
    assert not result.deleted_document_ids


@pytest.mark.parametrize("case", _case_parameters(HNSW_LAYER_CASES))
def test_cagra_built_hnsw_has_expected_layer_count(
    pylucene_context: PyLuceneContext, case: EndToEndCase
) -> None:
    """Verify CAGRA-built HNSW persists the requested three layers."""
    result, _ = _run_and_verify(case, pylucene_context)
    assert set(result.hnsw_layer_counts) == {case.expected_hnsw_layers}


@pytest.mark.parametrize(
    "case", _case_parameters(CAGRA_SEARCH_WIDTH_CASES)
)
def test_cagra_search_with_configured_search_width(
    pylucene_context: PyLuceneContext, case: EndToEndCase
) -> None:
    """Exercise GPU CAGRA search widths 1, 16, and 32."""
    _run_and_verify(case, pylucene_context)


@pytest.mark.parametrize(
    "case", _case_parameters(DELETED_DOCUMENT_CASES)
)
def test_deleted_documents_are_not_searchable(
    pylucene_context: PyLuceneContext, case: EndToEndCase
) -> None:
    """Verify GPU CAGRA search never returns a deleted vector document."""
    result, _ = _run_and_verify(case, pylucene_context)
    assert len(result.deleted_document_ids) == 1
    assert not result.document_ids_without_vectors
    assert any(
        observation.query_document_state is QueryDocumentState.DELETED
        for observation in result.query_observations
    )


@pytest.mark.parametrize(
    "case", _case_parameters(DOCUMENT_FILTER_CASES)
)
def test_vector_search_honors_selective_document_filter(
    pylucene_context: PyLuceneContext, case: EndToEndCase
) -> None:
    """Compare filtered CPU and GPU search results with brute force."""
    result, _ = _run_and_verify(case, pylucene_context)
    observation = result.filtered_query_observation
    assert observation is not None

    accepted_document_ids = tuple(
        document_id
        for document_id in result.searchable_vector_document_ids
        if document_id in case.scenario.document_ids_accepted_by_filter
    )
    accepted_document_id_set = set(accepted_document_ids)
    accepted_hit_ids = {
        f"doc-{document_id}" for document_id in accepted_document_ids
    }
    accepted_counts_by_segment = tuple(
        sum(
            document_id in accepted_document_id_set
            for document_id in segment_document_ids
        )
        for segment_document_ids in segment_document_id_ranges(
            case.scenario.document_count,
            case.scenario.segment_count,
        )
    )
    query_document_id = case.scenario.filter_query_document_id
    assert query_document_id is not None
    queried_document = f"doc-{query_document_id}"

    assert observation.query_document_id == query_document_id
    assert query_document_id not in (
        case.scenario.document_ids_accepted_by_filter
    )
    assert queried_document not in observation.hit_ids
    assert min(accepted_counts_by_segment) > case.scenario.top_k, (
        f"{case.selector}: every segment must retain more than topK="
        f"{case.scenario.top_k} accepted vectors; "
        f"acceptedPerSegment={accepted_counts_by_segment}"
    )

    rejected_hit_ids = tuple(
        hit_id
        for hit_id in observation.hit_ids
        if hit_id not in accepted_hit_ids
    )
    assert not rejected_hit_ids, (
        f"{case.selector}: filter-rejected documents were returned: "
        f"{rejected_hit_ids}"
    )
    expected_hit_count = min(
        case.scenario.top_k,
        len(accepted_document_ids),
    )
    assert len(observation.hit_ids) == expected_hit_count, (
        f"{case.selector}: expected {expected_hit_count} filtered hits, "
        f"got {observation.hit_ids}"
    )
    assert len(observation.hit_ids) == len(set(observation.hit_ids)), (
        f"{case.selector}: duplicate filtered hits returned: "
        f"{observation.hit_ids}"
    )

    expected_neighbors = _brute_force_neighbor_ids(
        case,
        result,
        query_document_id,
        accepted_document_ids,
    )
    recall = (
        len(set(observation.hit_ids) & set(expected_neighbors))
        / len(expected_neighbors)
    )
    assert recall >= case.min_recall, (
        f"{case.selector}: filtered query {queried_document} recall "
        f"{recall:.3f} is below floor {case.min_recall:.3f}; "
        f"expected={expected_neighbors}, actual={observation.hit_ids}"
    )
    print(
        f"FILTER [{case.execution_path.label}] {case.selector}: "
        f"acceptedPerSegment={min(accepted_counts_by_segment)}-"
        f"{max(accepted_counts_by_segment)}, "
        f"recall={recall:.3f}"
    )
