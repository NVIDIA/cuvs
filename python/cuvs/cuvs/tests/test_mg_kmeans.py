# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from cuvs.cluster.kmeans import KMeansParams
from cuvs.cluster.mg import kmeans as mg_kmeans
from cuvs.common import MultiGpuResources


def has_gpus(count=1):
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() >= count
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not has_gpus(1), reason="SNMG KMeans tests require at least one GPU"
)

requires_multiple_gpus = pytest.mark.skipif(
    not has_gpus(2),
    reason="SNMG KMeans multi-GPU smoke test requires two GPUs",
)


def make_inputs(dtype, n_rows=256, n_cols=8, n_clusters=4):
    rng = np.random.default_rng(123)
    labels = np.arange(n_rows) % n_clusters
    centers = rng.normal(
        loc=0.0, scale=10.0, size=(n_clusters, n_cols)
    ).astype(dtype)
    noise = rng.normal(loc=0.0, scale=0.01, size=(n_rows, n_cols)).astype(
        dtype
    )
    X = centers[labels] + noise
    centroids = X[np.arange(n_clusters)].copy()
    return np.ascontiguousarray(X), centroids


def make_sample_weights(dtype, n_rows):
    rng = np.random.default_rng(321)
    return rng.uniform(0.5, 2.0, size=n_rows).astype(dtype)


def predict_labels_host(X, centroids):
    distances = np.sum(
        (X[:, None, :] - centroids[None, :, :]) ** 2,
        axis=2,
    )
    labels = np.argmin(distances, axis=1)
    row_distances = distances[np.arange(X.shape[0]), labels]
    return labels, row_distances


def assert_same_label_partition(lhs, rhs):
    lhs = np.asarray(lhs)
    rhs = np.asarray(rhs)
    assert np.array_equal(
        np.equal.outer(lhs, lhs),
        np.equal.outer(rhs, rhs),
    )


def assert_valid_mg_fit_output(out, X, n_clusters):
    assert isinstance(out.centroids, np.ndarray)
    assert out.centroids.shape == (n_clusters, X.shape[1])
    assert out.centroids.dtype == X.dtype
    assert np.all(np.isfinite(out.centroids))
    assert np.isfinite(out.inertia)
    assert out.n_iter > 0


def assert_inertia_matches_centroids(out, X, sample_weights):
    _, row_distances = predict_labels_host(X, out.centroids)
    if sample_weights is not None:
        sample_weights = sample_weights * X.shape[0] / sample_weights.sum()
        row_distances = row_distances * sample_weights

    inertia_tol = 5e-2 if X.dtype == np.float32 else 1e-3
    assert np.allclose(
        out.inertia,
        np.sum(row_distances),
        rtol=inertia_tol,
        atol=inertia_tol,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("init_method", ["Array", "KMeansPlusPlus", "Random"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("device_buffer_prefetch", [False, True])
def test_mg_kmeans_fit_options(
    dtype, init_method, weighted, device_buffer_prefetch
):
    n_clusters = 4
    X, initial_centroids = make_inputs(dtype, n_clusters=n_clusters)
    resources = MultiGpuResources()

    if weighted:
        sample_weights = make_sample_weights(dtype, X.shape[0])
    else:
        sample_weights = None

    params = KMeansParams(
        n_clusters=n_clusters,
        init_method=init_method,
        max_iter=20,
        tol=1e-10,
        n_init=3 if init_method == "Random" else 1,
        init_size=X.shape[0],
        device_buffer_batch_size=37,
        device_buffer_prefetch=device_buffer_prefetch,
    )
    assert params.device_buffer_prefetch == device_buffer_prefetch
    centroids = initial_centroids.copy() if init_method == "Array" else None

    mg_out = mg_kmeans.fit(
        params,
        X,
        centroids=centroids,
        sample_weights=sample_weights,
        resources=resources,
    )
    resources.sync()

    assert_valid_mg_fit_output(mg_out, X, n_clusters)
    assert_inertia_matches_centroids(mg_out, X, sample_weights)

    labels, _ = predict_labels_host(X, mg_out.centroids)
    if init_method != "Random":
        expected_labels = np.arange(X.shape[0]) % n_clusters
        assert_same_label_partition(labels, expected_labels)
    assert len(np.unique(labels)) == n_clusters


def test_mg_kmeans_input_validation():
    import cupy as cp

    n_clusters = 4
    X, centroids = make_inputs(np.float32, n_clusters=n_clusters)
    resources = MultiGpuResources()
    params = KMeansParams(n_clusters=n_clusters, init_method="Array")

    with pytest.raises(ValueError, match="centroids must be provided"):
        mg_kmeans.fit(params, X, resources=resources)

    hierarchical_params = KMeansParams(
        n_clusters=n_clusters, hierarchical=True
    )
    with pytest.raises(ValueError, match="hierarchical"):
        mg_kmeans.fit(
            hierarchical_params,
            X,
            centroids=centroids.copy(),
            resources=resources,
        )

    with pytest.raises(ValueError, match="host memory"):
        mg_kmeans.fit(
            params,
            cp.asarray(X),
            centroids=centroids.copy(),
            resources=resources,
        )

    with pytest.raises(ValueError, match="host memory"):
        mg_kmeans.fit(
            params, X, centroids=cp.asarray(centroids), resources=resources
        )

    with pytest.raises(ValueError, match="C contiguous"):
        mg_kmeans.fit(
            params,
            np.asfortranarray(X),
            centroids=centroids.copy(),
            resources=resources,
        )

    with pytest.raises(TypeError, match="centroids dtype"):
        mg_kmeans.fit(
            params,
            X,
            centroids=centroids.astype(np.float64),
            resources=resources,
        )

    with pytest.raises(ValueError, match="Incorrect number of rows"):
        mg_kmeans.fit(
            params,
            X,
            centroids=centroids[: n_clusters - 1].copy(),
            resources=resources,
        )

    with pytest.raises(ValueError, match="sample_weights must be a 1D"):
        mg_kmeans.fit(
            params,
            X,
            centroids=centroids.copy(),
            sample_weights=np.ones((X.shape[0], 1), dtype=X.dtype),
            resources=resources,
        )

    with pytest.raises(TypeError, match="sample_weights dtype"):
        mg_kmeans.fit(
            params,
            X,
            centroids=centroids.copy(),
            sample_weights=np.ones(X.shape[0], dtype=np.float64),
            resources=resources,
        )

    with pytest.raises(ValueError, match="host memory"):
        mg_kmeans.fit(
            params,
            X,
            centroids=centroids.copy(),
            sample_weights=cp.ones(X.shape[0], dtype=cp.float32),
            resources=resources,
        )
