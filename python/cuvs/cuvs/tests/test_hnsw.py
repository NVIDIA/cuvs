# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from cuvs.common.exceptions import CuvsException
from cuvs.neighbors import cagra, hnsw
from cuvs.tests.ann_utils import calc_recall, generate_data


def run_hnsw_build_search_test(
    n_rows=10000,
    n_cols=10,
    n_queries=100,
    k=10,
    dtype=np.float32,
    metric="sqeuclidean",
    build_algo="ivf_pq",
    intermediate_graph_degree=128,
    graph_degree=64,
    hierarchy="none",
    search_params={},
    expected_recall=0.9,
):
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
    build_params = cagra.IndexParams(
        metric=metric,
        intermediate_graph_degree=intermediate_graph_degree,
        graph_degree=graph_degree,
        build_algo=build_algo,
    )

    index = cagra.build(build_params, dataset)

    assert index.trained

    hnsw_params = hnsw.IndexParams(hierarchy=hierarchy)
    hnsw_index = hnsw.from_cagra(hnsw_params, index)

    search_params = hnsw.SearchParams(**search_params)

    out_dist, out_idx = hnsw.search(search_params, hnsw_index, queries, k)

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
    skl_dist, skl_idx = nn_skl.kneighbors(queries, return_distance=True)

    recall = calc_recall(out_idx, skl_idx)
    assert recall >= expected_recall


@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int8, np.uint8])
@pytest.mark.parametrize("k", [10, 20])
@pytest.mark.parametrize("ef", [50, 150])
@pytest.mark.parametrize("num_threads", [2, 4])
@pytest.mark.parametrize("metric", ["sqeuclidean", "inner_product"])
@pytest.mark.parametrize("build_algo", ["ivf_pq", "nn_descent"])
@pytest.mark.parametrize("hierarchy", ["none", "cpu", "gpu"])
def test_hnsw(dtype, k, ef, num_threads, metric, build_algo, hierarchy):
    expected_recall = (
        0.9 if metric == "inner_product" and dtype == np.uint8 else 0.95
    )
    run_hnsw_build_search_test(
        dtype=dtype,
        k=k,
        metric=metric,
        build_algo=build_algo,
        hierarchy=hierarchy,
        search_params={"ef": ef, "num_threads": num_threads},
        expected_recall=expected_recall,
    )


def test_hnsw_build_hnsw_first_api(tmp_path):
    dataset = generate_data((2000, 16), np.float32)
    additional_dataset = generate_data((200, 16), np.float32)
    queries = generate_data((100, 16), np.float32)
    k = 10

    index_params = hnsw.IndexParams(
        hierarchy="gpu",
        M=16,
        ef_construction=120,
        metric="sqeuclidean",
    )
    hnsw_index = hnsw.build(index_params, dataset)

    assert hnsw_index.trained

    _out_dist, out_idx = hnsw.search(
        hnsw.SearchParams(ef=120, num_threads=1), hnsw_index, queries, k
    )

    nn_skl = NearestNeighbors(
        n_neighbors=k, algorithm="brute", metric="sqeuclidean"
    )
    nn_skl.fit(dataset)
    skl_idx = nn_skl.kneighbors(queries, return_distance=False)

    assert calc_recall(out_idx, skl_idx) >= 0.9

    with pytest.raises(CuvsException, match="dimensions"):
        hnsw.extend(
            hnsw.ExtendParams(),
            hnsw_index,
            generate_data((1, dataset.shape[1] - 1), np.float32),
        )

    hnsw.extend(
        hnsw.ExtendParams(num_threads=1), hnsw_index, additional_dataset
    )

    extended_dataset = np.vstack([dataset, additional_dataset])
    nn_skl.fit(extended_dataset)
    skl_idx = nn_skl.kneighbors(queries, return_distance=False)

    _extended_dist, extended_idx = hnsw.search(
        hnsw.SearchParams(ef=120, num_threads=1), hnsw_index, queries, k
    )

    assert calc_recall(extended_idx, skl_idx) >= 0.9

    with pytest.raises(CuvsException, match="k must not exceed"):
        hnsw.search(
            hnsw.SearchParams(),
            hnsw_index,
            queries[:1],
            extended_dataset.shape[0] + 1,
        )

    with pytest.raises(ValueError, match="ef must not be negative"):
        hnsw.SearchParams(ef=-1)

    # ef=0 keeps its historical meaning: the candidate list size defaults to k
    _zero_ef_dist, zero_ef_idx = hnsw.search(
        hnsw.SearchParams(ef=0, num_threads=1), hnsw_index, queries, k
    )
    assert zero_ef_idx.shape == (queries.shape[0], k)

    index_path = tmp_path / "hnsw_build_api.index"
    hnsw.save(str(index_path), hnsw_index)

    # an explicitly passed metric remains accepted and takes precedence
    explicit_metric_index = hnsw.load(
        index_params,
        str(index_path),
        dataset.shape[1],
        dataset.dtype,
        metric="sqeuclidean",
    )
    assert explicit_metric_index.trained

    # metric defaults to index_params.metric when not passed explicitly
    loaded_index = hnsw.load(
        index_params,
        str(index_path),
        dataset.shape[1],
        dataset.dtype,
    )

    _loaded_dist, loaded_idx = hnsw.search(
        hnsw.SearchParams(ef=120, num_threads=1), loaded_index, queries, k
    )

    assert calc_recall(loaded_idx, skl_idx) >= 0.9


def run_hnsw_extend_test(
    n_rows=10000,
    add_rows=2000,
    n_cols=10,
    n_queries=100,
    k=10,
    dtype=np.float32,
    metric="sqeuclidean",
    build_algo="ivf_pq",
    intermediate_graph_degree=128,
    graph_degree=64,
    search_params={},
    hierarchy="cpu",
):
    dataset = generate_data((n_rows, n_cols), dtype)
    add_dataset = generate_data((add_rows, n_cols), dtype)
    queries = generate_data((n_queries, n_cols), dtype)
    if metric == "inner_product":
        dataset = normalize(dataset, norm="l2", axis=1)
        add_dataset = normalize(add_dataset, norm="l2", axis=1)
        queries = normalize(queries, norm="l2", axis=1)
        if dtype in [np.int8, np.uint8]:
            # Quantize the normalized data to the int8/uint8 range
            dtype_max = np.iinfo(dtype).max
            dataset = (dataset * dtype_max).astype(dtype)
            add_dataset = (add_dataset * dtype_max).astype(dtype)
            queries = (queries * dtype_max).astype(dtype)
        if build_algo == "nn_descent":
            pytest.skip("inner_product metric is not supported for nn_descent")

    build_params = cagra.IndexParams(
        metric=metric,
        intermediate_graph_degree=intermediate_graph_degree,
        graph_degree=graph_degree,
        build_algo=build_algo,
    )

    index = cagra.build(build_params, dataset)

    assert index.trained

    hnsw_params = hnsw.IndexParams(hierarchy=hierarchy)
    hnsw_index = hnsw.from_cagra(hnsw_params, index)
    hnsw.extend(hnsw.ExtendParams(), hnsw_index, add_dataset)

    search_params = hnsw.SearchParams(**search_params)

    out_dist, out_idx = hnsw.search(search_params, hnsw_index, queries, k)

    # Calculate reference values with sklearn
    skl_metric = {
        "sqeuclidean": "sqeuclidean",
        "inner_product": "cosine",
        "euclidean": "euclidean",
    }[metric]
    nn_skl = NearestNeighbors(
        n_neighbors=k, algorithm="brute", metric=skl_metric
    )
    nn_skl.fit(np.vstack([dataset, add_dataset]))
    skl_dist, skl_idx = nn_skl.kneighbors(queries, return_distance=True)

    recall = calc_recall(out_idx, skl_idx)
    assert recall > 0.95


@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int8, np.uint8])
@pytest.mark.parametrize("k", [10, 20])
@pytest.mark.parametrize("ef", [30, 40])
@pytest.mark.parametrize("num_threads", [2, 4])
@pytest.mark.parametrize("metric", ["sqeuclidean"])
@pytest.mark.parametrize("build_algo", ["ivf_pq", "nn_descent"])
@pytest.mark.parametrize("hierarchy", ["cpu", "gpu"])
def test_hnsw_extend(dtype, k, ef, num_threads, metric, build_algo, hierarchy):
    # Note that inner_product tests use normalized input which we cannot
    # represent in int8, therefore we test only sqeuclidean metric here.
    run_hnsw_extend_test(
        dtype=dtype,
        k=k,
        metric=metric,
        build_algo=build_algo,
        search_params={"ef": ef, "num_threads": num_threads},
        hierarchy=hierarchy,
    )
