# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest
from pylibraft.common import device_ndarray

from cuvs.cluster.kmeans import (
    KMeansParams,
    cluster_cost,
    fit,
    predict,
)
from cuvs.distance import pairwise_distance


@pytest.mark.parametrize("n_rows", [100])
@pytest.mark.parametrize("n_cols", [5, 25])
@pytest.mark.parametrize("n_clusters", [5, 15])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("hierarchical", [True, False])
def test_kmeans_fit(n_rows, n_cols, n_clusters, dtype, hierarchical):
    if hierarchical and dtype == np.float64:
        pytest.skip("hierarchical kmeans doesn't support float64")

    # generate some random input points / centroids
    X_host = np.random.random_sample((n_rows, n_cols)).astype(dtype)
    centroids = device_ndarray(X_host[:n_clusters])
    X = device_ndarray(X_host)

    # compute the inertia, before fitting centroids
    original_inertia = cluster_cost(X, centroids)

    params = KMeansParams(n_clusters=n_clusters, hierarchical=hierarchical)

    # fit the centroids, make sure inertia has gone down
    centroids, inertia, n_iter = fit(params, X, centroids)
    assert n_iter >= 1

    # balanced kmeans doesn't return inertia
    if not hierarchical:
        assert inertia < original_inertia
        assert np.allclose(cluster_cost(X, centroids), inertia, rtol=1e-6)

    # make sure the prediction for each centroid is the centroid itself
    labels, inertia = predict(params, centroids, centroids)
    assert np.all(labels.copy_to_host() == np.arange(labels.shape[0]))


@pytest.mark.parametrize("n_rows", [100])
@pytest.mark.parametrize("n_cols", [5, 25])
@pytest.mark.parametrize("n_clusters", [4, 15])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cluster_cost(n_rows, n_cols, n_clusters, dtype):
    X = np.random.random_sample((n_rows, n_cols)).astype(dtype)
    X_device = device_ndarray(X)

    centroids = X[:n_clusters]
    centroids_device = device_ndarray(centroids)

    inertia = cluster_cost(X_device, centroids_device)

    # compute the nearest centroid to each sample
    distances = pairwise_distance(
        X_device, centroids_device, metric="sqeuclidean"
    ).copy_to_host()
    cluster_ids = np.argmin(distances, axis=1)

    cluster_distances = np.take_along_axis(
        distances, cluster_ids[:, None], axis=1
    )

    # need reduced tolerance for float32
    tol = 1e-3 if dtype == np.float32 else 1e-6
    assert np.allclose(inertia, sum(cluster_distances), rtol=tol, atol=tol)


@pytest.mark.parametrize("n_rows", [1000, 5000])
@pytest.mark.parametrize("n_cols", [10, 100])
@pytest.mark.parametrize("n_clusters", [8, 16])
@pytest.mark.parametrize("streaming_batch_size", [0, 100, 239, 500])
@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize("weighted", [False, True])
def test_fit_host_matches_fit_device(
    n_rows, n_cols, n_clusters, streaming_batch_size, dtype, weighted
):
    """
    Test that fit() with host (numpy) data produces the same centroids as
    fit() with device data, when given the same initial centroids.
    Optionally tests with non-uniform sample weights.
    """
    rng = np.random.default_rng(99)
    X_host = rng.random((n_rows, n_cols)).astype(dtype)

    centroid_indices = rng.choice(n_rows, size=n_clusters, replace=False)
    initial_centroids_host = X_host[centroid_indices].copy()

    if weighted:
        sample_weights_host = rng.uniform(0.5, 2.0, size=n_rows).astype(dtype)
        sample_weights_device = device_ndarray(sample_weights_host)
    else:
        sample_weights_host = None
        sample_weights_device = None

    params_device = KMeansParams(
        n_clusters=n_clusters,
        init_method="Array",
        max_iter=20,
        tol=1e-10,
    )
    centroids_regular, inertia_regular, _ = fit(
        params_device,
        device_ndarray(X_host),
        device_ndarray(initial_centroids_host.copy()),
        sample_weights=sample_weights_device,
    )
    centroids_regular = centroids_regular.copy_to_host()

    params_host = KMeansParams(
        n_clusters=n_clusters,
        init_method="Array",
        max_iter=20,
        tol=1e-10,
        streaming_batch_size=streaming_batch_size,
    )
    centroids_batched, inertia_batched, _ = fit(
        params_host,
        X_host,
        centroids=device_ndarray(initial_centroids_host.copy()),
        sample_weights=sample_weights_host,
    )
    centroids_batched = centroids_batched.copy_to_host()

    assert np.allclose(
        centroids_regular, centroids_batched, rtol=1e-3, atol=1e-3
    ), f"max diff: {np.max(np.abs(centroids_regular - centroids_batched))}"

    assert np.allclose(
        inertia_regular, inertia_batched, rtol=1e-3, atol=1e-3
    ), f"max diff: {np.max(np.abs(inertia_regular - inertia_batched))}"
