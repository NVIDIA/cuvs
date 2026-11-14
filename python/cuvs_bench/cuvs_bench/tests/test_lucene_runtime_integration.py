# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live runtime contracts; select with --run-lucene-e2e after activating PyLucene.

GPU coverage also requires matching cuVS Java, thin Lucene JAR and native libraries,
resolved through the runtime's normal environment/configuration discovery.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cuvs_bench.backends._lucene_runtime import (
    CAGRA_CODEC,
    CPU_HNSW_CODEC,
    LuceneRuntime,
)
from cuvs_bench.backends._lucene_runtime_config import (
    resolve_lucene_runtime_config,
)

pytestmark = pytest.mark.lucene_e2e
_CHILD_PROCESS_ENV = "CUVS_LUCENE_RUNTIME_TEST_CHILD"


def _passed_in_fresh_process(request) -> bool:
    """Run this pytest case in its own JVM, isolated from other tests' VM state."""
    if os.environ.get(_CHILD_PROCESS_ENV) == "1":
        return False
    node = f"{Path(__file__).resolve()}::{request.node.name}"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-s", node, "--run-lucene-e2e"],
        env={**os.environ, _CHILD_PROCESS_ENV: "1"},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return True


@pytest.mark.parametrize(
    "num_candidates", [10, 100], ids=["equal-to-k", "above-document-count"]
)
def test_hnsw_search_returns_k_hits_with_a_sufficient_candidate_budget(
    tmp_path,
    request,
    num_candidates,
):
    if _passed_in_fresh_process(request):
        return
    runtime = LuceneRuntime.create({"requires_cuvs": False})
    vectors = (
        np.random.default_rng(3033)
        .standard_normal((64, 16))
        .astype(np.float32)
    )
    runtime.build_index(tmp_path, vectors, CPU_HNSW_CODEC)

    result = runtime.search_index(
        tmp_path,
        vectors[:1],
        k=10,
        num_candidates=num_candidates,
    )

    hits = result.hits[0]
    assert len(hits) == 10
    assert hits[0].document_id == 0
    assert len({hit.document_id for hit in hits}) == 10


def test_java_cagra_writer_matches_python_metadata_reader(tmp_path, request):
    if _passed_in_fresh_process(request):
        return
    runtime = LuceneRuntime.create(
        resolve_lucene_runtime_config(requires_cuvs=True)
    )
    # Enough vectors for the codec's graph parameters without native clamping.
    vectors = (
        np.random.default_rng(3033)
        .standard_normal((1024, 128))
        .astype(np.float32)
    )
    built = runtime.build_index(tmp_path, vectors, CAGRA_CODEC)

    verification = runtime.cagra_verifier.verify(
        tmp_path,
        expected_vector_count=1024,
        expected_dimensions=128,
    )

    assert built.segment_count == 1
    assert verification.metadata() == {
        "persisted_index_kind": "gpu_cagra_only",
        "segment_count": 1,
        "field_count": 1,
        "vector_count": 1024,
        "dimensions": 128,
    }
