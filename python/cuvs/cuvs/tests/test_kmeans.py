# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from cuvs.common import Resources
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
@pytest.mark.parametrize("device_buffer_samples", [0, 100, 239, 500])
@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize("weighted", [False, True])
def test_fit_host_matches_fit_device(
    n_rows, n_cols, n_clusters, device_buffer_samples, dtype, weighted
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
        device_buffer_samples=device_buffer_samples,
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


def test_fit_host_with_stream_pool_setter():
    X = np.zeros((128, 2), dtype=np.float32)
    X[64:] = 10
    initial_centroids = device_ndarray(
        np.array([[0, 0], [10, 10]], dtype=np.float32)
    )
    params = KMeansParams(
        n_clusters=2,
        init_method="Array",
        device_buffer_samples=16,
        max_iter=10,
    )
    resources = Resources()
    resources.set_stream_pool(2)

    centroids, inertia, n_iter = fit(
        params, X, initial_centroids, resources=resources
    )
    resources.sync()

    assert n_iter >= 1
    assert np.isfinite(inertia)
    assert np.allclose(centroids.copy_to_host(), [[0, 0], [10, 10]], atol=1e-3)


@pytest.mark.parametrize("host_input", [False, True])
def test_ivf_all_probes_fit(host_input):
    from cuvs.cluster.kmeans import CentroidIvfParams
    from cuvs.neighbors import ivf_flat

    rng = np.random.default_rng(42)
    centers = np.arange(9, dtype=np.float32)[:, None] * 8
    x = (centers[np.arange(113) % 9] + rng.normal(0, 0.1, (113, 7))).astype(np.float32)
    initial = np.repeat(centers, 7, axis=1)
    data = x if host_input else device_ndarray(x)
    outputs = []
    for use_ivf in (False, True):
        # Inline temporaries exercise the lifetime of the borrowed C parameter bundle.
        params = KMeansParams(
            n_clusters=9, init_method="Array", max_iter=3,
            device_buffer_samples=23, batch_samples=11,
            centroid_ivf_params=CentroidIvfParams(
                build_params=ivf_flat.IndexParams(
                    n_lists=3, kmeans_n_iters=4, kmeans_trainset_fraction=1.0),
                search_params=ivf_flat.SearchParams(n_probes=3),
            ) if use_ivf else None,
        )
        if use_ivf:
            assert params.centroid_ivf_params.build_params.n_lists == 3
            assert params.centroid_ivf_params.search_params.n_probes == 3
        else:
            assert params.centroid_ivf_params is None
        out, inertia, _ = fit(params, data, device_ndarray(initial))
        outputs.append((out.copy_to_host(), inertia))
    np.testing.assert_allclose(outputs[0][0], outputs[1][0], rtol=4 * np.finfo(np.float32).eps)
    np.testing.assert_allclose(outputs[0][1], outputs[1][1], rtol=1e-3)


@pytest.mark.parametrize("build_options, probes", [
    ({"n_lists": 0}, 1),
    ({"n_lists": 4}, 5),
    ({"n_lists": 4}, 0),
    ({"kmeans_n_iters": 0}, 1),
    ({"kmeans_trainset_fraction": 0}, 1),
    ({"kmeans_trainset_fraction": 1.5}, 1),
    ({"add_data_on_build": False}, 1),
    ({"adaptive_centers": True}, 1),
    ({"metric": "cosine"}, 1),
])
def test_ivf_invalid_params(build_options, probes):
    from cuvs.cluster.kmeans import CentroidIvfParams
    from cuvs.neighbors import ivf_flat

    with pytest.raises(ValueError):
        CentroidIvfParams(
            build_params=ivf_flat.IndexParams(**build_options),
            search_params=ivf_flat.SearchParams(n_probes=probes),
        )


@pytest.mark.parametrize("options", [
    {"batch_samples": 0}, {"hierarchical": True}, {"metric": "cosine"},
])
def test_ivf_incompatible_kmeans_params(options):
    from cuvs.cluster.kmeans import CentroidIvfParams

    with pytest.raises(ValueError):
        KMeansParams(centroid_ivf_params=CentroidIvfParams(), **options)
