# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from pylibraft.common import device_ndarray

from cuvs.neighbors import cagra


def _recall(found, expected):
    return np.mean(
        [
            len(set(row).intersection(truth)) / len(truth)
            for row, truth in zip(found, expected)
        ]
    )


def test_cagra_tiered_parameter_validation():
    with pytest.raises(ValueError, match="build_algo"):
        cagra.IndexParams(build_algo="invalid")
    with pytest.raises(ValueError, match="algo"):
        cagra.SearchParams(algo="invalid")
    with pytest.raises(ValueError, match="hashmap_mode"):
        cagra.SearchParams(hashmap_mode="invalid")


def test_cagra_tiered_build_search_and_serialize(tmp_path):
    rng = np.random.default_rng(7)
    dataset = rng.normal(size=(2048, 32)).astype(np.float32)
    queries = dataset[:32].copy()
    k = 10

    index = cagra.build(
        cagra.IndexParams(
            build_algo="nn_descent",
            intermediate_graph_degree=32,
            graph_degree=16,
            nn_descent_niter=20,
            graph_storage="tiered",
            tiered=cagra.TieredGraphParams(
                device_graph_budget_bytes=2048
                * 8
                * np.dtype(np.uint32).itemsize,
                n_groups=4,
                training_rows=2048,
                assignment_batch_rows=512,
                kmeans_n_iters=10,
                validate=True,
                num_seeds=32,
                seed_training_rows=2048,
                seed=7,
            ),
        ),
        device_ndarray(dataset),
    )

    assert index.graph_storage == "tiered"
    assert index.trained
    assert len(index) == dataset.shape[0]
    assert index.dim == dataset.shape[1]
    assert index.graph_degree == 16

    search_params = cagra.SearchParams(
        algo="multi_cta",
        itopk_size=64,
        max_iterations=32,
        team_size=32,
        search_width=1,
        tiered=cagra.TieredSearchParams(
            num_seeds=32,
            num_queues=2,
            empty_pause=0,
        ),
    )
    query_device = device_ndarray(queries)
    _, neighbors = cagra.search(search_params, index, query_device, k)
    # A second call exercises reuse of the index-owned search context and pollers.
    _, repeated_neighbors = cagra.search(search_params, index, query_device, k)

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

    filename = tmp_path / "cagra.index"
    cagra.save(str(filename), index)
    loaded = cagra.Index()
    cagra.load(loaded, str(filename))
    assert loaded.trained
    assert (len(loaded), loaded.dim, loaded.graph_degree) == (
        len(index),
        index.dim,
        index.graph_degree,
    )
    _, loaded_neighbors = cagra.search(search_params, loaded, query_device, k)
    assert _recall(loaded_neighbors.copy_to_host(), expected) >= 0.8


def test_cagra_tiered_option_validation():
    with pytest.raises(ValueError, match="graph_storage"):
        cagra.IndexParams(graph_storage="unknown")
    with pytest.raises(ValueError, match="tiered"):
        cagra.IndexParams(tiered=cagra.TieredGraphParams())
    with pytest.raises(TypeError, match="TieredSearchParams"):
        cagra.SearchParams(tiered={"num_queues": 1})


def test_cagra_tiered_compressed_round_trip(tmp_path):
    from types import SimpleNamespace

    rng = np.random.default_rng(19)
    vectors = rng.normal(size=(512, 16)).astype(np.float32)
    params = cagra.IndexParams(
        build_algo="nn_descent",
        intermediate_graph_degree=16,
        graph_degree=8,
        graph_storage="tiered",
        tiered=cagra.TieredGraphParams(device_graph_budget_bytes=4096),
    )
    pq = SimpleNamespace(
        pq_bits=8,
        pq_dim=4,
        vq_n_centers=8,
        kmeans_n_iters=2,
        vq_kmeans_trainset_fraction=1.0,
        pq_kmeans_trainset_fraction=1.0,
    )
    index = cagra.build(params, vectors, compression=pq)
    assert index.graph_storage == "tiered"
    assert len(index) == len(vectors)
    queries = device_ndarray(vectors[:4].copy())
    search_params = cagra.SearchParams(
        algo="multi_cta", itopk_size=64, max_iterations=32
    )
    filename = tmp_path / "pq.index"
    cagra.save(str(filename), index)
    loaded = cagra.Index()
    cagra.load(loaded, str(filename))
    second = tmp_path / "pq-roundtrip.index"
    cagra.save(str(second), loaded)
    assert filename.read_bytes() == second.read_bytes()
    for idx in (index, loaded):
        _, ids = cagra.search(search_params, idx, queries, 16)
        found = ids.copy_to_host()
        assert np.all(found < len(vectors))
        assert all(i in found[i] for i in range(4))
