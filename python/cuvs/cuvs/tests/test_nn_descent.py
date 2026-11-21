# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest
import cupy as cp
from pylibraft.common import device_ndarray
from sklearn.datasets import make_blobs

from cuvs.neighbors import brute_force, nn_descent
from cuvs.tests.ann_utils import calc_recall


@pytest.mark.parametrize("n_rows", [1024, 2048])
@pytest.mark.parametrize("n_cols", [32, 64])
@pytest.mark.parametrize("device_memory", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("return_distances", [True, False])
def test_nn_descent(
    n_rows, n_cols, device_memory, dtype, inplace, return_distances
):
    # because of a limitation in the c++ api, we can't both return the
    # distances and have an inplace graph
    if inplace and return_distances:
        pytest.skip("Can't return distances with an inplace graph")

    metric = "sqeuclidean"
    graph_degree = 64

    input1 = np.random.random_sample((n_rows, n_cols)).astype(dtype)
    input1_device = device_ndarray(input1)
    graph = np.zeros((n_rows, graph_degree), dtype="uint32")

    params = nn_descent.IndexParams(
        metric=metric,
        graph_degree=graph_degree,
        return_distances=return_distances,
    )
    index = nn_descent.build(
        params,
        input1_device if device_memory else input1,
        graph=graph if inplace else None,
    )

    if not inplace:
        graph = index.graph

    bfknn_index = brute_force.build(input1_device, metric=metric)
    _, bfknn_graph = brute_force.search(
        bfknn_index, input1_device, k=graph_degree
    )
    bfknn_graph = bfknn_graph.copy_to_host()

    if return_distances:
        distances = index.distances
        assert distances.shape == graph.shape

    assert calc_recall(graph, bfknn_graph) > 0.9


@pytest.mark.parametrize("degree", [17, 32])
def test_nn_descent_edge_distances(degree):
    # Most candidates tie. Unscored slots can survive refinement, and the
    # final neighbor list can differ from the sorted list of candidates.
    x = np.zeros((128, 3), dtype=np.uint8)
    x[-1] = 1
    params = nn_descent.IndexParams(
        metric="sqeuclidean",
        graph_degree=degree,
        intermediate_graph_degree=64,
        max_iterations=1,
        return_distances=True,
    )
    index = nn_descent.build(params, x)
    points = x.astype(np.float32)
    expected = np.square(points[:, None] - points[index.graph]).sum(axis=-1)
    np.testing.assert_array_equal(index.distances, expected)


@pytest.mark.parametrize("n_cols", [2, 17, 32])
@pytest.mark.parametrize("dist_comp_dtype", ["auto", "fp32", "fp16"])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_nn_descent_dist_comp_dtype(n_cols, dist_comp_dtype, dtype):
    metric = "sqeuclidean"
    graph_degree = 32
    n_rows = 100_000

    X, _ = make_blobs(
        n_samples=n_rows, n_features=n_cols, centers=10, random_state=42
    )
    X = X.astype(dtype)

    params = nn_descent.IndexParams(
        metric=metric,
        graph_degree=graph_degree,
        return_distances=True,
        dist_comp_dtype=dist_comp_dtype,
    )

    index = nn_descent.build(params, X)
    nnd_indices = index.graph

    gpu_X = cp.asarray(X)
    index = brute_force.build(gpu_X, metric=metric)
    _, bf_indices = brute_force.search(index, gpu_X, k=graph_degree)
    bf_indices = bf_indices.copy_to_host()

    if n_cols <= 16 and dist_comp_dtype == "fp16" and dtype == np.float32:
        # for small dim, if data is fp32 but dist_comp_dtype is fp16, the recall will be low
        assert calc_recall(nnd_indices, bf_indices) < 0.7
    else:
        assert calc_recall(nnd_indices, bf_indices) > 0.9
