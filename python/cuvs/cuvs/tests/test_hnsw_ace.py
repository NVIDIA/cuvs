# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import os
import tempfile

import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from cuvs.neighbors import hnsw
from cuvs.tests.ann_utils import calc_recall, generate_data


def run_hnsw_ace_build_search_test(
    n_rows=10000,
    n_cols=10,
    n_queries=100,
    k=10,
    dtype=np.float32,
    metric="sqeuclidean",
    npartitions=2,
    ef_construction=100,
    use_disk=False,
    hierarchy="gpu",
    expected_recall=0.9,
):
    """
    Test HNSW index build using ACE algorithm.

    - Build HNSW index using ACE via hnsw.build()
    - For disk mode: serialize -> deserialize -> search
    - For in-memory mode: search directly
    """
    dataset = generate_data((n_rows, n_cols), dtype)
    queries = generate_data((n_queries, n_cols), dtype)
    if metric == "inner_product":
        dataset = normalize(dataset, norm="l2", axis=1)
        queries = normalize(queries, norm="l2", axis=1)
        if dtype in [np.int8, np.uint8]:
            # Quantize the normalized data to the int8/uint8 range
            dtype_max = np.iinfo(dtype).max
            dataset = (dataset * dtype_max).astype(dtype)
            queries = (queries * dtype_max).astype(dtype)

    # Create a temporary directory for ACE build
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up ACE parameters
        ace_params = hnsw.AceParams(
            npartitions=npartitions,
            build_dir=temp_dir,
            use_disk=use_disk,
        )

        # Build parameters with ACE configuration
        index_params = hnsw.IndexParams(
            hierarchy=hierarchy,
            M=32,
            ef_construction=ef_construction,
            metric=metric,
            ace_params=ace_params,
        )

        # Build the HNSW index using ACE
        hnsw_index = hnsw.build(index_params, dataset)

        assert hnsw_index.trained

        if use_disk:
            # For disk mode, the index is serialized to disk by the build function
            # We need to deserialize it before searching
            hnsw_file = os.path.join(temp_dir, "hnsw_index.bin")
            assert os.path.exists(hnsw_file)

            # Deserialize from disk for searching
            deserialized_index = hnsw.load(
                index_params,
                hnsw_file,
                n_cols,
                dtype,
                metric=metric,
            )

            # Search the deserialized index
            search_params = hnsw.SearchParams(
                ef=max(ef_construction, k * 2), num_threads=1
            )
            out_dist, out_idx = hnsw.search(
                search_params, deserialized_index, queries, k
            )
        else:
            # For in-memory mode, search directly
            search_params = hnsw.SearchParams(
                ef=max(ef_construction, k * 2), num_threads=1
            )
            out_dist, out_idx = hnsw.search(
                search_params, hnsw_index, queries, k
            )

        # Calculate reference values with sklearn
        skl_metric = {
            "sqeuclidean": "sqeuclidean",
            "inner_product": "cosine",
            "euclidean": "euclidean",
        }[metric]
        nn_skl = NearestNeighbors(
            n_neighbors=k, algorithm="brute", metric=skl_metric
        )
        nn_skl.fit(dataset)
        skl_idx = nn_skl.kneighbors(queries, return_distance=False)

        recall = calc_recall(out_idx, skl_idx)
        assert recall >= expected_recall, (
            f"Recall {recall:.3f} is below expected {expected_recall}"
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int8, np.uint8])
@pytest.mark.parametrize("metric", ["sqeuclidean", "inner_product"])
@pytest.mark.parametrize("use_disk", [False, True])
def test_hnsw_ace_build_search(dtype, metric, use_disk):
    """Test HNSWACE with different data types and metrics."""
    run_hnsw_ace_build_search_test(
        dtype=dtype,
        metric=metric,
        use_disk=use_disk,
    )


@pytest.mark.parametrize("npartitions", [2, 3, 8])
def test_hnsw_ace_partitions(npartitions):
    """Test HNSW ACE with different partition sizes (disk mode only)."""
    run_hnsw_ace_build_search_test(
        use_disk=True,
        npartitions=npartitions,
    )


@pytest.mark.parametrize("ef_construction", [50, 100, 200])
def test_hnsw_ace_ef_construction(ef_construction):
    """Test HNSW ACE with different ef_construction values (disk mode only)."""
    run_hnsw_ace_build_search_test(
        use_disk=True,
        ef_construction=ef_construction,
    )


@pytest.mark.parametrize("hierarchy", ["none", "gpu"])
def test_hnsw_ace_hierarchy(hierarchy):
    """Test HNSW ACE with different hierarchy modes (disk mode only)."""
    run_hnsw_ace_build_search_test(
        use_disk=True,
        hierarchy=hierarchy,
    )


def test_hnsw_ace_disk_serialize_deserialize():
    """
    Test the full disk-based ACE workflow:
    build -> serialize -> deserialize -> search
    """
    n_rows = 10000
    n_cols = 10
    n_queries = 100
    k = 10
    dtype = np.float32
    metric = "sqeuclidean"

    dataset = generate_data((n_rows, n_cols), dtype)
    queries = generate_data((n_queries, n_cols), dtype)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create ACE params with disk mode enabled
        ace_params = hnsw.AceParams(
            npartitions=2,
            build_dir=temp_dir,
            use_disk=True,
        )

        # Create HNSW index params with ACE
        index_params = hnsw.IndexParams(
            hierarchy="gpu",
            M=32,
            ef_construction=120,
            metric=metric,
            ace_params=ace_params,
        )

        # Build the index using ACE
        hnsw_index = hnsw.build(index_params, dataset)
        assert hnsw_index.trained

        # Serialize to a specific file path
        hnsw_file = os.path.join(temp_dir, "test_hnsw_index.bin")
        hnsw.save(hnsw_file, hnsw_index)
        assert os.path.exists(hnsw_file)

        # Deserialize from disk
        loaded_index = hnsw.load(
            index_params,
            hnsw_file,
            n_cols,
            dtype,
            metric=metric,
        )

        # Search the loaded index
        search_params = hnsw.SearchParams(ef=200, num_threads=1)
        out_dist, out_idx = hnsw.search(
            search_params, loaded_index, queries, k
        )

        # Verify results against sklearn
        nn_skl = NearestNeighbors(
            n_neighbors=k, algorithm="brute", metric="sqeuclidean"
        )
        nn_skl.fit(dataset)
        skl_idx = nn_skl.kneighbors(queries, return_distance=False)

        recall = calc_recall(out_idx, skl_idx)
        assert recall >= 0.7, f"Recall {recall:.3f} is below expected 0.7"


def test_hnsw_ace_tiny_memory_limit_triggers_disk_mode():
    """Test that setting tiny memory limits triggers disk mode automatically."""
    n_rows = 5000
    n_cols = 64
    dtype = np.float32
    metric = "sqeuclidean"

    dataset = generate_data((n_rows, n_cols), dtype)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Set ACE parameters with memory limits slightly above the minimum required
        # This should force disk mode even though we didn't explicitly set use_disk=True
        ace_params = hnsw.AceParams(
            npartitions=2,
            build_dir=temp_dir,
            use_disk=False,  # Not explicitly requesting disk mode
            max_host_memory_gb=0.001,  # Tiny limit to force disk mode
            max_gpu_memory_gb=0.0,  # No GPU memory limit
        )

        # Create HNSW index params with ACE
        index_params = hnsw.IndexParams(
            hierarchy="gpu",
            M=32,
            ef_construction=100,
            metric=metric,
            ace_params=ace_params,
        )

        # Build the index using ACE - should automatically use disk mode
        hnsw_index = hnsw.build(index_params, dataset)
        assert hnsw_index.trained

        # In disk mode, the graph should be stored in the build directory
        # Check that the graph file was created
        graph_file = os.path.join(temp_dir, "cagra_graph.npy")
        reordered_file = os.path.join(temp_dir, "reordered_dataset.npy")

        assert os.path.exists(graph_file), (
            "Graph file should exist when disk mode is triggered"
        )
        assert os.path.exists(reordered_file), (
            "Reordered dataset file should exist when disk mode is triggered"
        )
