# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import os
import tempfile

import cupy as cp
import numpy as np
import pytest
from pylibraft.common import device_ndarray
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from cuvs.neighbors import cagra, hnsw
from cuvs.tests.ann_utils import calc_recall, generate_data


def run_cagra_ace_build_search_test(
    n_rows=10000,
    n_cols=10,
    n_queries=100,
    k=10,
    dtype=np.float32,
    metric="sqeuclidean",
    intermediate_graph_degree=128,
    graph_degree=64,
    npartitions=2,
    ef_construction=100,
    use_disk=False,
    hierarchy="gpu",
):
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
        ace_params = cagra.AceParams(
            npartitions=npartitions,
            ef_construction=ef_construction,
            build_dir=temp_dir,
            use_disk=use_disk,
        )

        # Build parameters
        build_params = cagra.IndexParams(
            metric=metric,
            intermediate_graph_degree=intermediate_graph_degree,
            graph_degree=graph_degree,
            build_algo="ace",
            ace_params=ace_params,
        )

        # Build the index with ACE (uses host memory)
        index = cagra.build(build_params, dataset)

        assert index.trained

        # For disk-based mode, we can't search directly
        # (would need HNSW conversion which is tested separately)
        if not use_disk:
            # For in-memory mode, we can search directly
            # But queries need to be on device
            search_params = cagra.SearchParams(itopk_size=64)

            # Transfer queries to device for search
            queries_device = device_ndarray(queries)

            out_dist, out_idx = cagra.search(
                search_params, index, queries_device, k
            )

            # Convert results back to host
            out_idx_host = out_idx.copy_to_host()

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

            recall = calc_recall(out_idx_host, skl_idx)
            assert recall > 0.7

            # test that we can get the cagra graph from the index
            graph = index.graph
            assert graph.shape == (n_rows, graph_degree)

            # make sure we can convert the graph to cupy, and access it
            cp_graph = cp.array(graph)
            assert cp_graph.shape == (n_rows, graph_degree)
        else:
            # For disk-based mode, verify that expected files were created
            assert os.path.exists(os.path.join(temp_dir, "cagra_graph.npy"))
            assert os.path.exists(
                os.path.join(temp_dir, "reordered_dataset.npy")
            )
            assert os.path.exists(
                os.path.join(temp_dir, "dataset_mapping.npy")
            )

            # Test HNSW conversion from disk-based ACE index
            hnsw_params = hnsw.IndexParams(hierarchy=hierarchy)
            hnsw_index_serialized = hnsw.from_cagra(hnsw_params, index)
            assert hnsw_index_serialized is not None
            assert os.path.exists(os.path.join(temp_dir, "hnsw_index.bin"))

            # Deserialize the HNSW index from disk for search
            hnsw_index = hnsw.load(
                hnsw_params,
                os.path.join(temp_dir, "hnsw_index.bin"),
                n_cols,
                dtype,
            )

            search_params = hnsw.SearchParams(ef=200, num_threads=1)
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
            skl_dist, skl_idx = nn_skl.kneighbors(
                queries, return_distance=True
            )

            recall = calc_recall(out_idx, skl_idx)
            assert recall > 0.7


@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int8, np.uint8])
@pytest.mark.parametrize("metric", ["sqeuclidean", "inner_product"])
@pytest.mark.parametrize("use_disk", [False, True])
def test_cagra_ace_dtypes_and_metrics(dtype, metric, use_disk):
    """Test ACE with different data types and metrics."""
    run_cagra_ace_build_search_test(
        dtype=dtype,
        metric=metric,
        use_disk=use_disk,
    )


@pytest.mark.parametrize("npartitions", [0, 1, 2, 3, 8])
def test_cagra_ace_partitions(npartitions):
    """Test ACE with different partition sizes (disk mode only)."""
    run_cagra_ace_build_search_test(
        use_disk=True,
        npartitions=npartitions,
    )


@pytest.mark.parametrize(
    ("npartitions", "message"),
    [
        (33, "cannot exceed dataset size"),
    ],
)
def test_cagra_ace_rejects_invalid_partition_count(npartitions, message):
    """ACE rejects partition counts that would make partition labeling invalid."""
    dataset = np.zeros((32, 8), dtype=np.float32)
    ace_params = cagra.AceParams(npartitions=npartitions, use_disk=False)
    build_params = cagra.IndexParams(
        build_algo="ace",
        ace_params=ace_params,
    )

    with pytest.raises(Exception, match=message):
        cagra.build(build_params, dataset)


@pytest.mark.parametrize("ef_construction", [50, 100, 200])
def test_cagra_ace_ef_construction(ef_construction):
    """Test ACE with different ef_construction values (disk mode only)."""
    run_cagra_ace_build_search_test(
        use_disk=True,
        ef_construction=ef_construction,
    )


@pytest.mark.parametrize("hierarchy", ["none", "gpu"])
def test_cagra_ace_hierarchy(hierarchy):
    """Test ACE with different hierarchy modes (disk mode only)."""
    run_cagra_ace_build_search_test(
        use_disk=True,
        hierarchy=hierarchy,
    )


def test_cagra_ace_tiny_memory_limit_triggers_disk_mode():
    """Test that setting tiny memory limits triggers disk mode automatically."""
    n_rows = 5000
    n_cols = 64
    dtype = np.float32
    metric = "sqeuclidean"

    dataset = generate_data((n_rows, n_cols), dtype)

    # Create a temporary directory for ACE build
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set ACE parameters with memory limits slightly above the minimum required
        # This should force disk mode even though we didn't explicitly set use_disk=True
        ace_params = cagra.AceParams(
            npartitions=2,
            ef_construction=100,
            build_dir=temp_dir,
            use_disk=False,  # Not explicitly requesting disk mode
            max_host_memory_gb=0.001,  # Tiny limit to force disk mode
            max_gpu_memory_gb=0.0,  # No GPU memory limit
        )

        # Build parameters
        build_params = cagra.IndexParams(
            metric=metric,
            intermediate_graph_degree=128,
            graph_degree=64,
            build_algo="ace",
            ace_params=ace_params,
        )

        # Build the index with ACE - should automatically use disk mode
        index = cagra.build(build_params, dataset)

        assert index.trained

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
