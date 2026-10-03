#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for case-scoped Lucene execution-path log evidence."""

import pytest

from _lucene_log_capture import (
    CPU_HNSW_FALLBACK_WARNING,
    case_used_cpu_hnsw_fallback,
)


def test_cpu_fallback_warning_is_attributed_only_to_its_named_case() -> None:
    fallback_case = (
        "test_lucene_integration.py::"
        "test_accelerated_hnsw_builds_on_gpu_and_searches_on_cpu"
    )
    gpu_cagra_case = (
        "test_lucene_integration.py::"
        "test_cagra_build_and_search_verify_persisted_index_and_search_behavior"
        "[gpu-cagra-aligned-dimensions]"
    )
    combined_log = "\n".join(
        (
            f"CUVS_BENCH_LUCENE_CASE_START {fallback_case}",
            "Sep 21, 2026 12:00:00 PM "
            "com.nvidia.cuvs.lucene.Lucene99AcceleratedHNSWVectorsFormat "
            "fieldsWriter",
            f"WARNING: {CPU_HNSW_FALLBACK_WARNING}",
            f"CUVS_BENCH_LUCENE_CASE_END {fallback_case}",
            f"CUVS_BENCH_LUCENE_CASE_START {gpu_cagra_case}",
            "Sep 21, 2026 12:00:01 PM "
            "com.nvidia.cuvs.lucene.CuVS2510GPUVectorsWriter mergeOneField",
            "INFO: Built CAGRA index",
            f"CUVS_BENCH_LUCENE_CASE_END {gpu_cagra_case}",
        )
    )

    assert case_used_cpu_hnsw_fallback(combined_log, fallback_case)
    assert not case_used_cpu_hnsw_fallback(combined_log, gpu_cagra_case)


def test_cpu_fallback_detection_fails_closed_when_case_markers_are_missing() -> (
    None
):
    with pytest.raises(ValueError, match="contains no complete case"):
        case_used_cpu_hnsw_fallback(
            f"WARNING: {CPU_HNSW_FALLBACK_WARNING}",
            "test_lucene_integration.py::test_unmarked_accelerated_hnsw_case",
        )
