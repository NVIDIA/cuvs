# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest
from pylibraft.common import device_ndarray

from cuvs.cluster.soar import Params, predict

# Deliberately not the default of 1.0, so that the end-to-end test fails if
# lambda is dropped anywhere between Params and the C++ implementation.
LAMBDA = 2.0


def nearest_centroids(dataset, centroids):
    """Primary assignment, i.e. what k-means prediction would produce."""
    distances = ((dataset[:, None, :] - centroids[None, :, :]) ** 2).sum(
        axis=-1
    )
    return np.argmin(distances, axis=1)


def soar_scores(dataset, centroids, labels, lambda_):
    """
    ``||x - c||^2 + lambda * (dot(r / ||r||, x - c))^2`` for every row and
    centroid, computed in float64 as the reference for the device kernel.
    """
    dataset = dataset.astype(np.float64)
    centroids = centroids.astype(np.float64)

    residual = dataset - centroids[labels]
    unit_residual = residual / np.linalg.norm(residual, axis=1, keepdims=True)

    diff = dataset[:, None, :] - centroids[None, :, :]
    squared_distance = (diff**2).sum(axis=-1)
    projection = (diff * unit_residual[:, None, :]).sum(axis=-1)

    return squared_distance + lambda_ * projection**2


def make_inputs(n_rows=256, n_features=8, n_clusters=16, seed=0):
    rng = np.random.default_rng(seed)
    dataset = rng.uniform(-1, 1, (n_rows, n_features)).astype(np.float32)
    centroids = rng.uniform(-1, 1, (n_clusters, n_features)).astype(np.float32)
    labels = nearest_centroids(dataset, centroids)
    return dataset, centroids, labels


def test_params_defaults():
    assert Params().lambda_ == pytest.approx(1.0)


def test_params_custom():
    assert Params(lambda_=2.5).lambda_ == pytest.approx(2.5)


@pytest.mark.parametrize("n_rows", [256, 1000])
@pytest.mark.parametrize("n_features", [8, 37])
@pytest.mark.parametrize("n_clusters", [16, 100])
def test_predict_matches_host_reference(n_rows, n_features, n_clusters):
    dataset, centroids, labels = make_inputs(n_rows, n_features, n_clusters)
    labels = labels.astype(np.uint32)

    soar_labels = predict(
        Params(lambda_=LAMBDA),
        device_ndarray(dataset),
        device_ndarray(centroids),
        device_ndarray(labels),
    ).copy_to_host()

    assert soar_labels.dtype == labels.dtype
    assert soar_labels.shape == labels.shape
    assert np.all(soar_labels < centroids.shape[0])

    # Guards against the assertion below going vacuous: if the fixture ever
    # stopped spilling, simply echoing the primary labels would satisfy it.
    assert np.count_nonzero(soar_labels != labels) > 0

    # Compare losses rather than ids so the test is not fragile when two
    # centroids tie.
    scores = soar_scores(dataset, centroids, labels, LAMBDA)
    achieved = np.take_along_axis(
        scores, soar_labels.astype(np.int64)[:, None], axis=1
    ).squeeze(1)

    np.testing.assert_allclose(achieved, scores.min(axis=1), rtol=1e-4)


def test_predict_int32_matches_uint32():
    dataset, centroids, labels = make_inputs()

    def run(dtype):
        return predict(
            Params(lambda_=LAMBDA),
            device_ndarray(dataset),
            device_ndarray(centroids),
            device_ndarray(labels.astype(dtype)),
        ).copy_to_host()

    np.testing.assert_array_equal(
        run(np.uint32).astype(np.int64), run(np.int32).astype(np.int64)
    )


def test_predict_with_preallocated_output():
    dataset, centroids, labels = make_inputs()
    labels = labels.astype(np.uint32)

    out = device_ndarray(np.zeros(dataset.shape[0], dtype=np.uint32))
    returned = predict(
        Params(lambda_=LAMBDA),
        device_ndarray(dataset),
        device_ndarray(centroids),
        device_ndarray(labels),
        soar_labels=out,
    )

    assert returned is out
    expected = predict(
        Params(lambda_=LAMBDA),
        device_ndarray(dataset),
        device_ndarray(centroids),
        device_ndarray(labels),
    ).copy_to_host()
    np.testing.assert_array_equal(out.copy_to_host(), expected)


def test_predict_input_validation():
    dataset, centroids, labels = make_inputs()
    labels = labels.astype(np.uint32)
    n_rows = dataset.shape[0]

    params = Params()
    dataset_d = device_ndarray(dataset)
    centroids_d = device_ndarray(centroids)
    labels_d = device_ndarray(labels)

    with pytest.raises(TypeError, match="dtype float64"):
        predict(
            params,
            device_ndarray(dataset.astype(np.float64)),
            device_ndarray(centroids.astype(np.float64)),
            labels_d,
        )

    with pytest.raises(TypeError, match="dtype int64"):
        predict(
            params,
            dataset_d,
            centroids_d,
            device_ndarray(labels.astype(np.int64)),
        )

    with pytest.raises(ValueError, match="Incorrect number of columns"):
        predict(
            params,
            dataset_d,
            device_ndarray(np.ascontiguousarray(centroids[:, :-1])),
            labels_d,
        )

    with pytest.raises(ValueError, match="Incorrect number of rows"):
        predict(params, dataset_d, centroids_d, device_ndarray(labels[:-1]))

    with pytest.raises(ValueError, match="Incorrect number of rows"):
        predict(
            params,
            dataset_d,
            centroids_d,
            labels_d,
            soar_labels=device_ndarray(np.zeros(n_rows - 1, dtype=np.uint32)),
        )

    # int32 is a valid label dtype, but the output must match the dtype of
    # `labels`, so a uint32/int32 pair is still rejected.
    with pytest.raises(TypeError, match="dtype int32"):
        predict(
            params,
            dataset_d,
            centroids_d,
            labels_d,
            soar_labels=device_ndarray(np.zeros(n_rows, dtype=np.int32)),
        )

    with pytest.raises(ValueError, match="Row major"):
        predict(
            params,
            device_ndarray(np.asfortranarray(dataset)),
            centroids_d,
            labels_d,
        )
