# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#


import cupy
import numpy as np
import pytest
from pylibraft.common import device_ndarray
from sklearn.datasets import make_blobs

from cuvs.common import MultiGpuResources, Resources
from cuvs.neighbors import all_neighbors, brute_force, ivf_pq, nn_descent
from cuvs.tests.ann_utils import calc_recall


def make_cosine(
    n_samples=100,
    n_features=2,
    x_range=(0, 2 * np.pi),
    noise=0.0,
    random_state=None,
):
    r = np.random.default_rng(random_state)
    x = r.uniform(x_range[0], x_range[1], n_samples)
    y = np.cos(x) + r.normal(0, noise, n_samples)
    X = (
        y.reshape(-1, 1)
        if n_features == 1
        else np.column_stack(
            (x, y, r.normal(size=(n_samples, max(0, n_features - 2))))
        )
    )
    return X, y


@pytest.mark.parametrize("algo", ["nn_descent", "brute_force", "ivf_pq"])
@pytest.mark.parametrize("cluster", ["single_cluster", "multi_cluster"])
@pytest.mark.parametrize(
    "metric",
    [
        "sqeuclidean",
        "l2",
        "cosine",
        "l1",
        "inner_product",
        "chebyshev",
        "canberra",
        "minkowski",
        "correlation",
        "jensenshannon",
    ],
)
@pytest.mark.parametrize(
    "output_location",
    ["host_arrays", "device_arrays", "return_on_host", "return_on_device"],
)
def test_all_neighbors_device_build_quality(
    algo, cluster, metric, output_location
):
    """Test device build with quality validation against brute force ground
    truth. Exercises all output placement paths:
      - host_arrays: pre-allocated numpy indices + distances
      - device_arrays: pre-allocated cupy indices + distances
      - return_on_host: auto-allocated via return_on_host=True
      - return_on_device: auto-allocated via return_on_host=False
    """
    n_rows, n_cols, k = 7151, 64, 16

    ivf_pq_valid_metrics = {"sqeuclidean"}
    nnd_valid_metrics = {"sqeuclidean", "l2", "cosine", "inner_product"}
    is_invalid = (algo == "ivf_pq" and metric not in ivf_pq_valid_metrics) or (
        algo == "nn_descent" and metric not in nnd_valid_metrics
    )

    if cluster == "single_cluster":
        overlap_factor = 0
    else:
        overlap_factor = 3

    np.random.seed(42)

    if metric == "cosine":
        X, _ = make_cosine(
            n_samples=n_rows, n_features=n_cols, random_state=42
        )
    elif metric == "jensenshannon":
        # Jensen-Shannon requires non-negative values representing probability distributions
        X, _ = make_blobs(
            n_samples=n_rows,
            n_features=n_cols,
            centers=10,
            cluster_std=1.0,
            center_box=(0.0, 10.0),  # Non-negative values only
            random_state=42,
        )
        # Normalize each row to sum to 1 (probability distribution)
        X = np.abs(X)  # Ensure non-negative
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        X = X / row_sums
    else:
        X, _ = make_blobs(
            n_samples=n_rows,
            n_features=n_cols,
            centers=10,
            cluster_std=1.0,
            center_box=(-10.0, 10.0),
            random_state=42,
        )
    X = X.astype(np.float32)
    X_device = device_ndarray(X)

    ivf_pq_params = None
    nn_descent_params = None
    if algo == "ivf_pq":
        ivf_pq_params = ivf_pq.IndexParams(
            metric=metric,
            n_lists=8 if cluster == "multi_cluster" else 4,
            pq_bits=8,
            pq_dim=0,
            add_data_on_build=True,
        )
    elif algo == "nn_descent":
        nn_descent_params = nn_descent.IndexParams(
            metric=metric,
            graph_degree=k,
            intermediate_graph_degree=k * 2,
            max_iterations=100,
            termination_threshold=0.001,
        )

    params = all_neighbors.AllNeighborsParams(
        algo=algo,
        overlap_factor=overlap_factor,
        n_clusters=1,
        metric=metric,
        ivf_pq_params=ivf_pq_params,
        nn_descent_params=nn_descent_params,
    )

    res = Resources()

    if is_invalid:
        with pytest.raises(Exception, match="Distance metric"):
            all_neighbors.build(
                X_device,
                k,
                params,
                distances=cupy.empty((n_rows, k), dtype=cupy.float32),
                resources=res,
            )
        return

    distances_result = None
    if output_location == "host_arrays":
        indices_arg = np.empty((n_rows, k), dtype="int64")
        distances_arg = np.empty((n_rows, k), dtype="float32")
        indices_result, distances_result = all_neighbors.build(
            X_device,
            k,
            params,
            indices=indices_arg,
            distances=distances_arg,
            resources=res,
        )
        assert isinstance(indices_result, np.ndarray)
        assert isinstance(distances_result, np.ndarray)
    elif output_location == "device_arrays":
        indices_arg = cupy.empty((n_rows, k), dtype=cupy.int64)
        distances_arg = cupy.empty((n_rows, k), dtype=cupy.float32)
        indices_result, distances_result = all_neighbors.build(
            X_device,
            k,
            params,
            indices=indices_arg,
            distances=distances_arg,
            resources=res,
        )
        assert hasattr(indices_result, "__cuda_array_interface__")
        assert hasattr(distances_result, "__cuda_array_interface__")
    elif output_location == "return_on_host":
        indices_result = all_neighbors.build(
            X_device,
            k,
            params,
            return_on_host=True,
            resources=res,
        )
        assert isinstance(indices_result, np.ndarray)
    elif output_location == "return_on_device":
        indices_result = all_neighbors.build(
            X_device,
            k,
            params,
            return_on_host=False,
            resources=res,
        )
        assert hasattr(indices_result, "__cuda_array_interface__")

    bf_index = brute_force.build(X_device, metric=metric)
    bf_distances, bf_indices = brute_force.search(bf_index, X_device, k=k)
    bf_indices_host = cupy.asnumpy(bf_indices)

    if isinstance(indices_result, np.ndarray):
        indices_host = indices_result
    else:
        indices_host = cupy.asnumpy(indices_result)

    assert indices_host.shape == (n_rows, k)
    assert indices_host.dtype == np.int64

    if distances_result is not None:
        if isinstance(distances_result, np.ndarray):
            distances_host = distances_result
        else:
            distances_host = cupy.asnumpy(distances_result)
        assert distances_host.shape == (n_rows, k)
        assert distances_host.dtype == np.float32

    recall = calc_recall(indices_host, bf_indices_host)
    assert recall > 0.85


@pytest.mark.parametrize("algo", ["nn_descent", "brute_force", "ivf_pq"])
@pytest.mark.parametrize("cluster", ["single_cluster", "multi_cluster"])
@pytest.mark.parametrize("snmg", [False, True])
@pytest.mark.parametrize(
    "output_location",
    ["host_arrays", "device_arrays", "return_on_host", "return_on_device"],
)
def test_all_neighbors_host_build_quality(
    algo, cluster, snmg, output_location
):
    """Test host build with quality validation against brute force ground
    truth. Exercises all output placement paths:
      - host_arrays: pre-allocated numpy indices + distances
      - device_arrays: pre-allocated cupy indices + distances
      - return_on_host: auto-allocated via return_on_host=True
      - return_on_device: auto-allocated via return_on_host=False
    """
    n_rows, n_cols, k = 7151, 64, 16

    if cluster == "single_cluster":
        n_clusters = 1
        overlap_factor = 0
    else:
        n_clusters = 8
        overlap_factor = 3

    np.random.seed(42)

    X_host, _ = make_blobs(
        n_samples=n_rows,
        n_features=n_cols,
        centers=10,
        cluster_std=1.0,
        center_box=(-10.0, 10.0),
        random_state=42,
    )
    X_host = X_host.astype(np.float32)
    X_device = device_ndarray(X_host)

    ivf_pq_params = None
    nn_descent_params = None

    if algo == "ivf_pq":
        ivf_pq_params = ivf_pq.IndexParams(
            metric="sqeuclidean",
            n_lists=8 if cluster == "multi_cluster" else 4,
            pq_bits=8,
            pq_dim=0,
            add_data_on_build=True,
        )
    elif algo == "nn_descent":
        nn_descent_params = nn_descent.IndexParams(
            metric="sqeuclidean",
            graph_degree=k,
            intermediate_graph_degree=k * 2,
            max_iterations=100,
            termination_threshold=0.001,
        )

    params = all_neighbors.AllNeighborsParams(
        algo=algo,
        overlap_factor=overlap_factor,
        n_clusters=n_clusters,
        metric="sqeuclidean",
        ivf_pq_params=ivf_pq_params,
        nn_descent_params=nn_descent_params,
    )

    if snmg:
        res = MultiGpuResources()
    else:
        res = Resources()

    distances_result = None
    if output_location == "host_arrays":
        indices_arg = np.empty((n_rows, k), dtype="int64")
        distances_arg = np.empty((n_rows, k), dtype="float32")
        indices_result, distances_result = all_neighbors.build(
            X_host,
            k,
            params,
            indices=indices_arg,
            distances=distances_arg,
            resources=res,
        )
        assert isinstance(indices_result, np.ndarray)
        assert isinstance(distances_result, np.ndarray)
    elif output_location == "device_arrays":
        indices_arg = cupy.empty((n_rows, k), dtype=cupy.int64)
        distances_arg = cupy.empty((n_rows, k), dtype=cupy.float32)
        indices_result, distances_result = all_neighbors.build(
            X_host,
            k,
            params,
            indices=indices_arg,
            distances=distances_arg,
            resources=res,
        )
        assert hasattr(indices_result, "__cuda_array_interface__")
        assert hasattr(distances_result, "__cuda_array_interface__")
    elif output_location == "return_on_host":
        indices_result = all_neighbors.build(
            X_host,
            k,
            params,
            return_on_host=True,
            resources=res,
        )
        assert isinstance(indices_result, np.ndarray)
    elif output_location == "return_on_device":
        indices_result = all_neighbors.build(
            X_host,
            k,
            params,
            return_on_host=False,
            resources=res,
        )
        assert hasattr(indices_result, "__cuda_array_interface__")

    bf_index = brute_force.build(X_device, metric="sqeuclidean")
    bf_distances, bf_indices = brute_force.search(bf_index, X_device, k=k)
    bf_indices_host = cupy.asnumpy(bf_indices)

    if isinstance(indices_result, np.ndarray):
        indices_host = indices_result
    else:
        indices_host = cupy.asnumpy(indices_result)

    assert indices_host.shape == (n_rows, k)
    assert indices_host.dtype == np.int64

    if distances_result is not None:
        if isinstance(distances_result, np.ndarray):
            distances_host = distances_result
        else:
            distances_host = cupy.asnumpy(distances_result)
        assert distances_host.shape == (n_rows, k)
        assert distances_host.dtype == np.float32

    recall = calc_recall(indices_host, bf_indices_host)
    assert recall > 0.85
