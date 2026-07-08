# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import os
import tempfile

import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from cuvs.common import MultiGpuResources
from cuvs.neighbors.mg import cagra as mg_cagra
from cuvs.tests.ann_utils import calc_recall, generate_data


MIN_ROWS_PER_SHARDED_GRAPH_DEGREE = 4


def get_gpu_count():
    """Return the number of visible CUDA devices."""
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount()
    except Exception:
        return 0


# Check if multi-GPU functionality is available
def has_multiple_gpus():
    """Check if system has multiple GPUs available."""
    return get_gpu_count() > 1


def _n_rows_for_distribution(
    n_rows, distribution_mode, graph_degree, intermediate_graph_degree
):
    """Keep sharded CAGRA graphs large enough as GPU count increases."""
    if distribution_mode != "sharded":
        return n_rows

    min_rows_per_shard = (
        max(graph_degree, intermediate_graph_degree)
        * MIN_ROWS_PER_SHARDED_GRAPH_DEGREE
    )
    return max(n_rows, max(1, get_gpu_count()) * min_rows_per_shard)


# Mark tests that require multiple GPUs
requires_multiple_gpus = pytest.mark.skipif(
    not has_multiple_gpus(), reason="Multi-GPU tests require multiple GPUs"
)


def run_mg_cagra_build_search_test(
    n_rows=10000,
    n_cols=10,
    n_queries=100,
    k=10,
    dtype=np.float32,
    metric="sqeuclidean",  # CAGRA only supports sqeuclidean and inner_product
    distribution_mode="sharded",
    search_mode="load_balancer",
    merge_mode="tree_merge",
    n_rows_per_batch=1000,
    compare=True,
    search_params=None,
    graph_degree=64,
    intermediate_graph_degree=128,
):
    """
    Run a multi-GPU CAGRA build and search test.

    Note: Multi-GPU CAGRA requires host memory arrays (NumPy), not device
    arrays.
    """
    n_rows = _n_rows_for_distribution(
        n_rows, distribution_mode, graph_degree, intermediate_graph_degree
    )

    # Generate host memory arrays (NumPy)
    dataset = generate_data((n_rows, n_cols), dtype)
    if metric == "inner_product":
        dataset = normalize(dataset, norm="l2", axis=1)

    queries = generate_data((n_queries, n_cols), dtype)
    if metric == "inner_product":
        queries = normalize(queries, norm="l2", axis=1)

    # Multi-GPU resources
    resources = MultiGpuResources()

    # Build parameters
    build_params = mg_cagra.IndexParams(
        metric=metric,
        distribution_mode=distribution_mode,
        graph_degree=graph_degree,
        intermediate_graph_degree=intermediate_graph_degree,
    )

    # Build index
    index = mg_cagra.build(build_params, dataset, resources=resources)
    assert index.trained

    # Search parameters
    search_params = dict(search_params or {})
    search_params_obj = mg_cagra.SearchParams(
        search_mode=search_mode,
        merge_mode=merge_mode,
        n_rows_per_batch=n_rows_per_batch,
        **search_params,
    )

    # Perform search
    distances, neighbors = mg_cagra.search(
        search_params_obj,
        index,
        queries,
        k,
        resources=resources,
    )

    # Verify results are in host memory (NumPy arrays)
    assert isinstance(distances, np.ndarray)
    assert isinstance(neighbors, np.ndarray)
    assert distances.shape == (n_queries, k)
    assert neighbors.shape == (n_queries, k)
    assert np.all(neighbors >= 0)
    assert np.all(neighbors < n_rows)
    assert np.all(np.isfinite(distances))

    if not compare:
        return distances, neighbors

    # Calculate reference values with sklearn
    skl_metric = {
        "sqeuclidean": "sqeuclidean",
        "inner_product": "cosine",
    }[metric]

    nn_skl = NearestNeighbors(
        n_neighbors=k, algorithm="brute", metric=skl_metric
    )
    nn_skl.fit(dataset)
    skl_idx = nn_skl.kneighbors(queries, return_distance=False)

    recall = calc_recall(neighbors, skl_idx)
    # Multi-GPU implementation may have lower recall due to data
    # distribution across GPUs
    # This is acceptable as long as the functionality works correctly
    assert recall > 0.3, (
        f"Recall too low: {recall:.3f} "
        f"(n_rows={n_rows}, distribution_mode={distribution_mode}, "
        f"graph_degree={graph_degree}, "
        f"intermediate_graph_degree={intermediate_graph_degree}, "
        f"num_gpus={get_gpu_count()})"
    )

    return distances, neighbors


@requires_multiple_gpus
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize(
    "metric", ["sqeuclidean"]
)  # Start with just sqeuclidean
@pytest.mark.parametrize(
    "distribution_mode", ["sharded"]
)  # Start with just sharded
def test_mg_cagra_basic(dtype, metric, distribution_mode):
    """Test basic multi-GPU CAGRA build and search functionality."""
    run_mg_cagra_build_search_test(
        n_rows=2000,  # Use smaller dataset for more reliable tests
        n_cols=8,
        n_queries=20,
        k=5,
        dtype=dtype,
        metric=metric,
        distribution_mode=distribution_mode,
        graph_degree=32,  # Smaller graph for faster tests
        intermediate_graph_degree=64,
    )


@requires_multiple_gpus
@pytest.mark.parametrize(
    "metric", ["inner_product"]
)  # Only test supported metrics
@pytest.mark.parametrize("distribution_mode", ["replicated"])
def test_mg_cagra_additional_metrics(metric, distribution_mode):
    """Test additional metrics and distribution modes."""
    run_mg_cagra_build_search_test(
        n_rows=2000,
        n_cols=8,
        n_queries=20,
        k=5,
        dtype=np.float32,
        metric=metric,
        distribution_mode=distribution_mode,
        graph_degree=32,
        intermediate_graph_degree=64,
    )


@requires_multiple_gpus
@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int8, np.uint8])
def test_mg_cagra_dtypes(dtype):
    """Test multi-GPU CAGRA with different data types."""
    run_mg_cagra_build_search_test(
        n_rows=1500,
        n_cols=8,
        n_queries=15,
        k=5,
        dtype=dtype,
        metric="sqeuclidean",
        graph_degree=32,
        intermediate_graph_degree=64,
    )


@requires_multiple_gpus
@pytest.mark.parametrize("distribution_mode", ["sharded", "replicated"])
def test_mg_cagra_distribution_modes(distribution_mode):
    """Test different distribution modes for multi-GPU CAGRA."""
    run_mg_cagra_build_search_test(
        n_rows=1500,
        n_cols=8,
        n_queries=15,
        k=5,
        distribution_mode=distribution_mode,
        graph_degree=32,
        intermediate_graph_degree=64,
    )


@requires_multiple_gpus
@pytest.mark.parametrize(
    "distribution_mode,search_mode,merge_mode,n_rows_per_batch",
    [
        ("replicated", "load_balancer", "tree_merge", 500),
        ("replicated", "round_robin", "tree_merge", 2000),
        ("sharded", "load_balancer", "merge_on_root_rank", 500),
        ("sharded", "load_balancer", "tree_merge", 500),
    ],
)
def test_mg_cagra_search_params(
    distribution_mode, search_mode, merge_mode, n_rows_per_batch
):
    """Test the relevant replicated and sharded search parameters."""
    run_mg_cagra_build_search_test(
        n_rows=1500,
        n_cols=8,
        n_queries=15,
        k=5,
        distribution_mode=distribution_mode,
        search_mode=search_mode,
        merge_mode=merge_mode,
        n_rows_per_batch=n_rows_per_batch,
        graph_degree=32,
        intermediate_graph_degree=64,
    )


@requires_multiple_gpus
@pytest.mark.parametrize(
    "metric", ["sqeuclidean"]
)  # Only test supported metrics
def test_mg_cagra_metrics(metric):
    """Test different distance metrics for multi-GPU CAGRA."""
    run_mg_cagra_build_search_test(
        n_rows=1500,
        n_cols=8,
        n_queries=15,
        k=5,
        metric=metric,
        graph_degree=32,
        intermediate_graph_degree=64,
    )


@requires_multiple_gpus
def test_mg_cagra_serialize():
    """Test save/load functionality for multi-GPU CAGRA."""
    graph_degree = 32
    intermediate_graph_degree = 64
    n_rows = _n_rows_for_distribution(
        2000, "sharded", graph_degree, intermediate_graph_degree
    )
    n_cols = 8
    k = 5

    # Generate data
    dataset = generate_data((n_rows, n_cols), np.float32)
    queries = generate_data((20, n_cols), np.float32)

    resources = MultiGpuResources()

    # Build original index
    build_params = mg_cagra.IndexParams(
        graph_degree=graph_degree,
        intermediate_graph_degree=intermediate_graph_degree,
    )
    original_index = mg_cagra.build(build_params, dataset, resources=resources)

    # Search with original index
    search_params = mg_cagra.SearchParams(itopk_size=32)
    orig_distances, orig_neighbors = mg_cagra.search(
        search_params, original_index, queries, k, resources=resources
    )

    # Save index to temporary file
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        temp_filename = f.name

    try:
        mg_cagra.save(original_index, temp_filename, resources=resources)

        # Load index from file
        loaded_index = mg_cagra.load(temp_filename, resources=resources)
        assert loaded_index.trained

        # Search with loaded index
        loaded_distances, loaded_neighbors = mg_cagra.search(
            search_params, loaded_index, queries, k, resources=resources
        )

        # Results should be identical
        np.testing.assert_array_equal(orig_neighbors, loaded_neighbors)
        np.testing.assert_allclose(orig_distances, loaded_distances, rtol=1e-6)

    finally:
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)


@requires_multiple_gpus
def test_mg_cagra_distribute():
    """Test distribute functionality for multi-GPU CAGRA."""
    # Note: Distribute is for replicating a single-GPU index across
    # multiple GPUs.
    # This test builds a single-GPU index, serializes it, then distributes it.

    n_rows, n_cols = 2000, 8
    k = 5

    # Generate data
    dataset = generate_data((n_rows, n_cols), np.float32)
    queries = generate_data((20, n_cols), np.float32)

    # Import single-GPU CAGRA to build and serialize a single-GPU index
    from cuvs.common import Resources
    from cuvs.neighbors import cagra

    # Build single-GPU index first
    single_gpu_resources = Resources()
    single_build_params = cagra.IndexParams(
        metric="sqeuclidean", graph_degree=32, intermediate_graph_degree=64
    )

    # Convert to device arrays for single-GPU build
    try:
        import cupy as cp

        device_dataset = cp.asarray(dataset)
        single_index = cagra.build(
            single_build_params, device_dataset, resources=single_gpu_resources
        )
    except ImportError:
        pytest.skip("CuPy not available for single-GPU index building")

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        temp_filename = f.name

    try:
        # Serialize single-GPU index
        cagra.save(temp_filename, single_index, resources=single_gpu_resources)

        # Now distribute the single-GPU index across multiple GPUs
        resources = MultiGpuResources()
        distributed_index = mg_cagra.distribute(
            temp_filename, resources=resources
        )
        assert distributed_index.trained

        # Search should work with distributed index (using host memory arrays)
        search_params = mg_cagra.SearchParams(itopk_size=32)
        distances, neighbors = mg_cagra.search(
            search_params, distributed_index, queries, k, resources=resources
        )

        assert distances.shape == (20, k)
        assert neighbors.shape == (20, k)
        assert np.all(neighbors >= 0)
        assert np.all(neighbors < n_rows)
        assert np.all(np.isfinite(distances))

    finally:
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)


def test_memory_location_validation():
    """Test that multi-GPU CAGRA validates memory locations correctly."""
    try:
        import cupy as cp
    except ImportError:
        pytest.skip("CuPy not available for memory location tests")

    graph_degree = 32
    intermediate_graph_degree = 64
    n_rows = _n_rows_for_distribution(
        1500, "sharded", graph_degree, intermediate_graph_degree
    )
    n_cols = 8

    # Create host and device arrays
    host_data = generate_data((n_rows, n_cols), np.float32)
    device_data = cp.asarray(host_data)

    resources = MultiGpuResources()
    build_params = mg_cagra.IndexParams(
        graph_degree=graph_degree,
        intermediate_graph_degree=intermediate_graph_degree,
    )

    # Test that device arrays are rejected for build
    with pytest.raises(ValueError, match="host memory"):
        mg_cagra.build(build_params, device_data, resources=resources)

    # Test that host arrays work for build
    index = mg_cagra.build(build_params, host_data, resources=resources)

    # Test that device arrays are rejected for search
    queries = generate_data((20, n_cols), np.float32)
    device_queries = cp.asarray(queries)
    search_params = mg_cagra.SearchParams(itopk_size=32)

    with pytest.raises(ValueError, match="host memory"):
        mg_cagra.search(
            search_params, index, device_queries, 5, resources=resources
        )

    # Test that host arrays work for search
    distances, neighbors = mg_cagra.search(
        search_params, index, queries, 5, resources=resources
    )
    assert isinstance(distances, np.ndarray)
    assert isinstance(neighbors, np.ndarray)
    assert np.all(neighbors >= 0)
    assert np.all(neighbors < n_rows)
    assert np.all(np.isfinite(distances))


def test_parameter_validation():
    """Test parameter validation for multi-GPU CAGRA."""
    # Test invalid distribution mode
    with pytest.raises(ValueError, match="distribution_mode must be"):
        mg_cagra.IndexParams(distribution_mode="invalid")

    # Test invalid search mode
    with pytest.raises(ValueError, match="search_mode must be"):
        mg_cagra.SearchParams(search_mode="invalid")

    # Test invalid merge mode
    with pytest.raises(ValueError, match="merge_mode must be"):
        mg_cagra.SearchParams(merge_mode="invalid")


def test_parameter_properties():
    """Test that parameters can be accessed via properties."""
    # Test IndexParams properties
    params = mg_cagra.IndexParams(distribution_mode="replicated")
    assert params.distribution_mode == "replicated"

    params = mg_cagra.IndexParams(distribution_mode="sharded")
    assert params.distribution_mode == "sharded"

    # Test SearchParams creation with different parameters
    mg_cagra.SearchParams(
        search_mode="round_robin",
        merge_mode="tree_merge",
        n_rows_per_batch=2000,
    )
    # These don't have properties exposed, but creation should work


def test_untrained_index_error():
    """Test that using an untrained index raises appropriate errors."""
    resources = MultiGpuResources()

    # Create untrained index
    index = mg_cagra.Index()
    assert not index.trained

    queries = generate_data((100, 10), np.float32)
    search_params = mg_cagra.SearchParams()

    # Test that search on untrained index fails
    with pytest.raises(ValueError, match="Index needs to be built"):
        mg_cagra.search(search_params, index, queries, 10, resources=resources)

    # Test that save on untrained index fails
    with pytest.raises(ValueError, match="Index needs to be built"):
        mg_cagra.save(index, "temp.bin", resources=resources)


@requires_multiple_gpus
def test_mg_cagra_with_prealloc_output():
    """Test multi-GPU CAGRA search with pre-allocated output arrays."""
    graph_degree = 32
    intermediate_graph_degree = 64
    n_rows = _n_rows_for_distribution(
        1500, "sharded", graph_degree, intermediate_graph_degree
    )
    n_cols = 8
    n_queries = 20
    k = 5

    # Generate data in host memory
    dataset = generate_data((n_rows, n_cols), np.float32)
    queries = generate_data((n_queries, n_cols), np.float32)

    resources = MultiGpuResources()

    # Build index
    build_params = mg_cagra.IndexParams(
        graph_degree=graph_degree,
        intermediate_graph_degree=intermediate_graph_degree,
    )
    index = mg_cagra.build(build_params, dataset, resources=resources)

    # Pre-allocate output arrays in host memory
    neighbors = np.empty((n_queries, k), dtype=np.int64)
    distances = np.empty((n_queries, k), dtype=np.float32)

    # Search with pre-allocated arrays
    search_params = mg_cagra.SearchParams(itopk_size=32)
    ret_distances, ret_neighbors = mg_cagra.search(
        search_params,
        index,
        queries,
        k,
        neighbors=neighbors,
        distances=distances,
        resources=resources,
    )

    # Should return the same arrays we passed in
    assert ret_distances is distances
    assert ret_neighbors is neighbors
    assert distances.shape == (n_queries, k)
    assert neighbors.shape == (n_queries, k)
    assert np.all(neighbors >= 0)
    assert np.all(neighbors < n_rows)
    assert np.all(np.isfinite(distances))


def test_index_repr():
    """Test string representation of Index."""
    index = mg_cagra.Index()
    assert repr(index) == "Index(type=MultiGpuCagra)"


def test_mg_cagra_simple():
    """Simple test to validate multi-GPU CAGRA works with very favorable
    parameters.
    """
    if not has_multiple_gpus():
        pytest.skip("Multi-GPU tests require multiple GPUs")

    # Use simple test case that should definitely work
    graph_degree = 16
    intermediate_graph_degree = 32
    n_rows = _n_rows_for_distribution(
        1000, "sharded", graph_degree, intermediate_graph_degree
    )
    n_cols = 8
    n_queries, k = 20, 5

    # Generate data
    dataset = generate_data((n_rows, n_cols), np.float32)
    queries = generate_data((n_queries, n_cols), np.float32)

    resources = MultiGpuResources()

    # Use small graph for reliable testing
    build_params = mg_cagra.IndexParams(
        metric="sqeuclidean",
        graph_degree=graph_degree,
        intermediate_graph_degree=intermediate_graph_degree,
    )

    # Build index
    index = mg_cagra.build(build_params, dataset, resources=resources)

    # Search with basic parameters
    search_params = mg_cagra.SearchParams(itopk_size=16)
    distances, neighbors = mg_cagra.search(
        search_params, index, queries, k, resources=resources
    )

    # Basic sanity checks
    assert distances.shape == (n_queries, k)
    assert neighbors.shape == (n_queries, k)
    assert isinstance(distances, np.ndarray)
    assert isinstance(neighbors, np.ndarray)

    # Check that we get valid neighbors
    assert np.all(neighbors >= 0)
    assert np.all(neighbors < n_rows)

    # Distances should be non-negative and sorted
    assert np.all(distances >= 0)
    for i in range(n_queries):
        assert np.all(distances[i, :-1] <= distances[i, 1:]), (
            f"Distances not sorted for query {i}"
        )


# Integration test with multiple operations
@requires_multiple_gpus
def test_mg_cagra_integration():
    """Integration test covering build, search, and serialization."""
    graph_degree = 32
    intermediate_graph_degree = 64
    n_rows = _n_rows_for_distribution(
        2000, "sharded", graph_degree, intermediate_graph_degree
    )
    n_cols = 8
    k = 5

    # Generate initial dataset
    dataset = generate_data((n_rows, n_cols), np.float32)
    queries = generate_data((20, n_cols), np.float32)

    resources = MultiGpuResources()

    # Build initial index
    build_params = mg_cagra.IndexParams(
        distribution_mode="sharded",
        metric="sqeuclidean",
        graph_degree=graph_degree,
        intermediate_graph_degree=intermediate_graph_degree,
    )
    index = mg_cagra.build(build_params, dataset, resources=resources)

    # Initial search
    search_params = mg_cagra.SearchParams(
        itopk_size=32,
        search_mode="load_balancer",
        merge_mode="merge_on_root_rank",
    )
    distances1, neighbors1 = mg_cagra.search(
        search_params, index, queries, k, resources=resources
    )

    # Save and reload
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        temp_filename = f.name

    try:
        mg_cagra.save(index, temp_filename, resources=resources)
        reloaded_index = mg_cagra.load(temp_filename, resources=resources)

        # Search with reloaded index
        distances2, neighbors2 = mg_cagra.search(
            search_params, reloaded_index, queries, k, resources=resources
        )

        # Results from reloaded index should match
        np.testing.assert_array_equal(neighbors1, neighbors2)
        np.testing.assert_allclose(distances1, distances2, rtol=1e-6)

    finally:
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)
