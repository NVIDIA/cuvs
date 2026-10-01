# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from pylibraft.common import device_ndarray

from cuvs.neighbors import flowann


def _recall(found, expected):
    return np.mean(
        [
            len(set(row).intersection(truth)) / len(truth)
            for row, truth in zip(found, expected)
        ]
    )


def test_flowann_parameter_validation():
    with pytest.raises(ValueError, match="build_algo"):
        flowann.IndexParams(build_algo="invalid")
    with pytest.raises(ValueError, match="algo"):
        flowann.SearchParams(algo="invalid")
    with pytest.raises(ValueError, match="hashmap_mode"):
        flowann.SearchParams(hashmap_mode="invalid")


def test_flowann_build_search_and_serialize(tmp_path):
    rng = np.random.default_rng(7)
    dataset = rng.normal(size=(2048, 32)).astype(np.float32)
    queries = dataset[:32].copy()
    k = 10

    index = flowann.build(
        flowann.IndexParams(
            build_algo="nn_descent",
            intermediate_graph_degree=32,
            graph_degree=16,
            nn_descent_niter=20,
            device_graph_budget_bytes=2048 * 8 * np.dtype(np.uint32).itemsize,
            n_groups=4,
            training_rows=2048,
            assignment_batch_rows=512,
            kmeans_n_iters=10,
            validate=True,
            num_seeds=32,
            seed_training_rows=2048,
            seed=7,
        ),
        device_ndarray(dataset),
    )

    assert index.trained
    assert index.size == dataset.shape[0]
    assert index.dim == dataset.shape[1]
    assert index.graph_degree == 16

    search_params = flowann.SearchParams(
        algo="multi_cta",
        itopk_size=64,
        max_iterations=32,
        team_size=32,
        search_width=1,
        num_seeds=32,
        num_queues=2,
        empty_pause=0,
    )
    query_device = device_ndarray(queries)
    _, neighbors = flowann.search(search_params, index, query_device, k)
    # A second call exercises reuse of the index-owned search context and pollers.
    _, repeated_neighbors = flowann.search(
        search_params, index, query_device, k
    )

    expected = np.argsort(
        np.sum((queries[:, None, :] - dataset[None, :, :]) ** 2, axis=2),
        axis=1,
    )[:, :k]
    found = neighbors.copy_to_host()
    repeated = repeated_neighbors.copy_to_host()
    assert np.all(found < dataset.shape[0])
    assert np.all(repeated < dataset.shape[0])
    assert _recall(found, expected) >= 0.8
    assert _recall(repeated, expected) >= 0.8

    filename = tmp_path / "flowann.index"
    flowann.save(filename, index)
    loaded = flowann.load(filename, dtype=np.float32)
    assert loaded.trained
    assert (loaded.size, loaded.dim, loaded.graph_degree) == (
        index.size,
        index.dim,
        index.graph_degree,
    )
    _, loaded_neighbors = flowann.search(
        search_params, loaded, query_device, k
    )
    assert _recall(loaded_neighbors.copy_to_host(), expected) >= 0.8
