# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import os
import tempfile

import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from cuvs.common.exceptions import CuvsException
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
    Test HNSW index build using ACE via hnsw.build().
    ACE writes hnsw_index.bin; search uses the deserialized index.
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
        hnsw_file = os.path.join(temp_dir, "hnsw_index.bin")
        assert os.path.exists(hnsw_file)
        for cagra_artifact in (
            "cagra_graph.npy",
            "reordered_dataset.npy",
            "augmented_dataset.npy",
            "dataset_mapping.npy",
        ):
            assert not os.path.exists(os.path.join(temp_dir, cagra_artifact))

        deserialized_index = hnsw.load(
            index_params,
            hnsw_file,
            n_cols,
            dtype,
            metric=metric,
        )

        search_params = hnsw.SearchParams(
            ef=max(ef_construction, k * 2), num_threads=1
        )
        out_dist, out_idx = hnsw.search(
            search_params, deserialized_index, queries, k
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


def test_hnsw_ace_tiny_memory_limit_fails_before_staging():
    """A host/GPU cap below the minimum planned partition peak fails before staging."""
    n_rows = 5000
    n_cols = 64
    dtype = np.float32
    metric = "sqeuclidean"

    dataset = generate_data((n_rows, n_cols), dtype)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Cap below the minimum planned partition peak.
        ace_params = hnsw.AceParams(
            npartitions=2,
            build_dir=temp_dir,
            use_disk=False,
            max_host_memory_gb=0.001,  # Below the minimum planned partition peak
            max_gpu_memory_gb=0.0,
        )

        # Create HNSW index params with ACE
        index_params = hnsw.IndexParams(
            hierarchy="gpu",
            M=32,
            ef_construction=100,
            metric=metric,
            ace_params=ace_params,
        )

        with pytest.raises(CuvsException, match="cap|memory|peak"):
            hnsw.build(index_params, dataset)

        assert not os.path.exists(os.path.join(temp_dir, "hnsw_index.bin"))
        assert not os.path.exists(os.path.join(temp_dir, "cagra_graph.npy"))
        assert not os.path.exists(
            os.path.join(temp_dir, "reordered_dataset.npy")
        )
        assert not os.path.exists(
            os.path.join(temp_dir, "augmented_dataset.npy")
        )


def test_hnsw_ace_memmap_below_dataset_host_cap():
    """Build from a read-only memmap larger than the configured host cap."""
    n_rows = 20_000
    n_cols = 256
    n_queries = 32
    k = 10

    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_path = os.path.join(temp_dir, "dataset.dat")
        writable = np.memmap(
            dataset_path,
            dtype=np.float32,
            mode="w+",
            shape=(n_rows, n_cols),
        )
        writable[:] = generate_data((n_rows, n_cols), np.float32)
        writable.flush()
        del writable
        dataset = np.memmap(
            dataset_path,
            dtype=np.float32,
            mode="r",
            shape=(n_rows, n_cols),
        )
        assert dataset.nbytes > 18 * 1024**2
        queries = np.asarray(dataset[:n_queries]).copy()

        ace_params = hnsw.AceParams(
            npartitions=32,
            build_dir=temp_dir,
            use_disk=True,
            max_host_memory_gb=18 / 1024,
        )
        index_params = hnsw.IndexParams(
            hierarchy="none",
            M=16,
            ef_construction=100,
            metric="sqeuclidean",
            ace_params=ace_params,
        )
        built = hnsw.build(index_params, dataset)
        index_path = os.path.join(temp_dir, "hnsw_index.bin")
        assert built.trained
        assert os.path.exists(index_path)

        loaded = hnsw.load(
            index_params,
            index_path,
            n_cols,
            np.float32,
            metric="sqeuclidean",
        )
        distances, neighbors = hnsw.search(
            hnsw.SearchParams(ef=200, num_threads=1),
            loaded,
            queries,
            k,
        )
        assert distances.shape == (n_queries, k)
        assert neighbors.shape == (n_queries, k)
        assert (
            np.mean(
                np.any(
                    neighbors == np.arange(n_queries, dtype=np.int64)[:, None],
                    axis=1,
                )
            )
            >= 0.9
        )
