# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import tempfile

import cupy as cp
import numpy as np
import pytest
from pylibraft.common import device_ndarray
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from cuvs.neighbors import cagra, ivf_pq
from cuvs.tests.ann_utils import (
    calc_recall,
    generate_data,
    run_filtered_search_test,
)


def run_cagra_build_search_test(
    n_rows=10000,
    n_cols=10,
    n_queries=100,
    k=10,
    dtype=np.float32,
    metric="sqeuclidean",
    intermediate_graph_degree=128,
    graph_degree=64,
    build_algo="ivf_pq",
    array_type="device",
    compare=True,
    inplace=True,
    test_extend=False,
    search_params={},
    serialize=False,
):
    dataset = generate_data((n_rows, n_cols), dtype)
    if metric == "inner_product" or metric == "cosine":
        if dtype in [np.int8, np.uint8]:
            pytest.skip("skip normalization for int8/uint8 data")
        dataset = normalize(dataset, norm="l2", axis=1)
    dataset_device = device_ndarray(dataset)

    build_params = cagra.IndexParams(
        metric=metric,
        intermediate_graph_degree=intermediate_graph_degree,
        graph_degree=graph_degree,
        build_algo=build_algo,
    )

    if test_extend:
        dataset_1 = dataset[: n_rows // 2, :]
        dataset_2 = dataset[n_rows // 2 :, :]
        extend_params = cagra.ExtendParams()
        if array_type == "device":
            dataset_1_device = device_ndarray(dataset_1)
            dataset_2_device = device_ndarray(dataset_2)

            index = cagra.build(build_params, dataset_1_device)
            # Explicit caller-side preparation for padded-only extend contract:
            # 1) Ensure the index is device_padded before extend.
            # 2) Prepare a padded view for additional dataset via factories.
            extend_keepalive = []
            if cagra.get_dataset_view_kind(index.dataset) == "device_standard":
                base_padded_dataset = cagra.make_device_padded_dataset(
                    dataset_1_device
                )
                base_padded_view = cagra.make_view_from_owning_padded(
                    base_padded_dataset
                )
                cagra.attach_padded_dataset_for_search(index, base_padded_view)
                extend_keepalive.extend(
                    [base_padded_dataset, base_padded_view]
                )
            add_padded_dataset = cagra.make_device_padded_dataset(
                dataset_2_device
            )
            add_padded_view = cagra.make_view_from_owning_padded(
                add_padded_dataset
            )
            extended_dataset_owner = cagra.make_device_padded_dataset(
                device_ndarray(np.concatenate((dataset_1, dataset_2), axis=0))
            )
            extended_dataset_view = cagra.make_view_from_owning_padded(
                extended_dataset_owner
            )
            extend_keepalive.extend(
                [
                    add_padded_dataset,
                    add_padded_view,
                    extended_dataset_owner,
                    extended_dataset_view,
                ]
            )
            index = cagra.extend(
                extend_params, index, add_padded_view, extended_dataset_view
            )
        else:
            pytest.skip(
                "extend test path requires explicit device padded dataset view"
            )
    else:
        if array_type == "device":
            index = cagra.build(build_params, dataset_device)
        else:
            index = cagra.build(build_params, dataset)

    if not compare:
        return

    if serialize:
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            temp_filename = f.name
        cagra.save(temp_filename, index)
        layout = (
            "standard"
            if cagra.get_dataset_view_kind(
                dataset_device if array_type == "device" else dataset
            ).endswith("standard")
            else "padded"
        )
        index = cagra.Index()
        out_dataset = (
            cagra.StandardDataset()
            if layout == "standard"
            else cagra.PaddedDataset()
        )
        cagra.load(
            index,
            temp_filename,
            out_dataset=out_dataset,
        )
        if layout == "standard":
            padded_view = cagra.make_device_padded_dataset_view(dataset_device)
            cagra.attach_padded_dataset_for_search(index, padded_view)
            index_keepalive = [out_dataset, padded_view]
        else:
            index_keepalive = [out_dataset]
    else:
        index_keepalive = []
        view_kind = cagra.get_dataset_view_kind(
            dataset_device if array_type == "device" else dataset
        )
        if view_kind == "device_standard":
            padded_view = cagra.make_device_padded_dataset_view(dataset_device)
            cagra.attach_padded_dataset_for_search(index, padded_view)
            index_keepalive = [padded_view]
        elif view_kind == "host_padded":
            padded_view = cagra.make_device_padded_dataset_view(dataset_device)
            cagra.attach_device_dataset_on_host_index(index, padded_view)
            index_keepalive = [dataset_device, padded_view]
        elif view_kind == "host_standard":
            standard_view = cagra.make_device_standard_dataset_view(
                dataset_device
            )
            cagra.attach_device_dataset_on_host_index(index, standard_view)
            padded_view = cagra.make_device_padded_dataset_view(dataset_device)
            cagra.attach_padded_dataset_for_search(index, padded_view)
            index_keepalive = [
                dataset_device,
                standard_view,
                padded_view,
            ]

    assert index_keepalive is not None
    queries = generate_data((n_queries, n_cols), dtype)
    out_idx = np.zeros((n_queries, k), dtype=np.uint32)
    out_dist = np.zeros((n_queries, k), dtype=np.float32)

    queries_device = device_ndarray(queries)
    out_idx_device = device_ndarray(out_idx) if inplace else None
    out_dist_device = device_ndarray(out_dist) if inplace else None

    search_params = cagra.SearchParams(**search_params)

    ret_output = cagra.search(
        search_params,
        index,
        queries_device,
        k,
        neighbors=out_idx_device,
        distances=out_dist_device,
    )

    if not inplace:
        out_dist_device, out_idx_device = ret_output

    out_idx = out_idx_device.copy_to_host()
    out_dist = out_dist_device.copy_to_host()

    # Calculate reference values with sklearn
    skl_metric = {
        "sqeuclidean": "sqeuclidean",
        "inner_product": "cosine",
        "euclidean": "euclidean",
        "cosine": "cosine",
    }[metric]
    nn_skl = NearestNeighbors(
        n_neighbors=k, algorithm="brute", metric=skl_metric
    )
    nn_skl.fit(dataset)
    skl_idx = nn_skl.kneighbors(queries, return_distance=False)

    recall = calc_recall(out_idx, skl_idx)
    assert recall > 0.7

    # test that we can get the cagra graph from the index
    graph = index.graph
    assert graph.shape == (n_rows, graph_degree)

    # make sure we can convert the graph to cupy, and access it
    cp_graph = cp.array(graph)
    assert cp_graph.shape == (n_rows, graph_degree)

    # make sure we can get the dataset from the cagra index
    dataset_from_index = index.dataset
    logical_dim = dataset.shape[1]
    dataset_from_index_host = dataset_from_index.copy_to_host()
    # CAGRA may store padded rows internally; compare only logical columns.
    dataset_from_index_host_logical = dataset_from_index_host[:, :logical_dim]
    assert np.allclose(dataset, dataset_from_index_host_logical)

    # make sure we can reconstruct the index from the graph
    # Note that we can't actually use the dataset from the index itself
    # - since that is a strided matrix (and we expect non-strided inputs
    # in the C++ cagra::build api), so we are using the host version
    # which will have been copied into a non-strided layout
    reloaded_dataset_device = device_ndarray(dataset_from_index_host_logical)
    reloaded_index = cagra.from_graph(
        graph, reloaded_dataset_device, metric=metric
    )
    reloaded_keepalive = [reloaded_dataset_device]
    reloaded_kind = cagra.get_dataset_view_kind(reloaded_dataset_device)
    if reloaded_kind == "device_standard":
        reloaded_padded_view = cagra.make_device_padded_dataset_view(
            reloaded_dataset_device
        )
        cagra.attach_padded_dataset_for_search(
            reloaded_index, reloaded_padded_view
        )
        reloaded_keepalive = [
            reloaded_dataset_device,
            reloaded_padded_view,
        ]

    assert reloaded_keepalive is not None
    dist_device, idx_device = cagra.search(
        search_params, reloaded_index, queries_device, k
    )
    recall = calc_recall(idx_device.copy_to_host(), skl_idx)
    assert recall > 0.9


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int8, np.uint8])
@pytest.mark.parametrize("array_type", ["device"])
@pytest.mark.parametrize("build_algo", ["ivf_pq", "nn_descent"])
@pytest.mark.parametrize("metric", ["sqeuclidean", "inner_product", "cosine"])
@pytest.mark.parametrize("serialize", [True, False])
def test_cagra_dataset_dtype_host_device(
    dtype, array_type, inplace, build_algo, metric, serialize
):
    # Note that inner_product tests use normalized input which we cannot
    # represent in int8, therefore we test only sqeuclidean metric here.
    run_cagra_build_search_test(
        dtype=dtype,
        inplace=inplace,
        array_type=array_type,
        build_algo=build_algo,
        metric=metric,
        serialize=serialize,
    )


@pytest.mark.parametrize("sparsity", [0.2, 0.5, 0.7, 1.0])
def test_filtered_cagra(sparsity):
    run_filtered_search_test(cagra, sparsity)


@pytest.mark.parametrize(
    "params",
    [
        {
            "dtype": np.int8,
            "intermediate_graph_degree": 64,
            "graph_degree": 32,
            "test_extend": False,
            "k": 1,
            "metric": "sqeuclidean",
            "build_algo": "ivf_pq",
        },
        {
            "dtype": np.float32,
            "intermediate_graph_degree": 32,
            "graph_degree": 16,
            "test_extend": True,
            "k": 5,
            "metric": "sqeuclidean",
            "build_algo": "ivf_pq",
        },
        {
            "dtype": np.float32,
            "intermediate_graph_degree": 128,
            "graph_degree": 32,
            "test_extend": False,
            "k": 10,
            "metric": "inner_product",
            "build_algo": "nn_descent",
        },
        {
            "dtype": np.float16,
            "intermediate_graph_degree": 64,
            "graph_degree": 32,
            "test_extend": True,
            "k": 10,
            "metric": "cosine",
            "build_algo": "ivf_pq",
        },
    ],
)
def test_cagra_index_params(params):
    # Note that inner_product tests use normalized input which we cannot
    # represent in int8, therefore we test only sqeuclidean metric here.
    run_cagra_build_search_test(
        dtype=params["dtype"],
        test_extend=params["test_extend"],
        k=params["k"],
        metric=params["metric"],
        graph_degree=params["graph_degree"],
        intermediate_graph_degree=params["intermediate_graph_degree"],
        compare=False,
        build_algo=params["build_algo"],
    )


@pytest.mark.parametrize("internal_dtype", [np.float32, np.float16, np.uint8])
def test_cagra_ivf_pq(
    internal_dtype,
    n_rows=1000,
    n_cols=30,
    n_queries=20,
    k=5,
    dtype=np.float16,
    metric="inner_product",
    intermediate_graph_degree=32,
    graph_degree=16,
    build_algo="ivf_pq",
):
    dataset = generate_data((n_rows, n_cols), dtype)
    dataset = normalize(dataset, norm="l2", axis=1)
    dataset_device = device_ndarray(dataset)

    ivf_pq_params_build = ivf_pq.IndexParams(metric=metric, n_lists=10)
    ivf_pq_params_search = ivf_pq.SearchParams(
        n_probes=5,
        lut_dtype=internal_dtype,
        coarse_search_dtype=np.int8,
    )
    build_params = cagra.IndexParams(
        metric=metric,
        intermediate_graph_degree=intermediate_graph_degree,
        graph_degree=graph_degree,
        build_algo=build_algo,
        ivf_pq_build_params=ivf_pq_params_build,
        ivf_pq_search_params=ivf_pq_params_search,
        refinement_rate=1.2,
    )
    cudadtype_to_np = {np.float32: 0, np.float16: 2, np.int8: 3, np.uint8: 8}

    assert (
        build_params.ivf_pq_search_params.lut_dtype
        == cudadtype_to_np[internal_dtype]
    )
    assert (
        build_params.ivf_pq_search_params.coarse_search_dtype
        == cudadtype_to_np[np.int8]
    )
    assert np.isclose(build_params.refinement_rate, 1.2)
    index = cagra.build(build_params, dataset_device)
    keepalive = []
    if cagra.get_dataset_view_kind(dataset_device) == "device_standard":
        padded_view = cagra.make_device_padded_dataset_view(dataset_device)
        cagra.attach_padded_dataset_for_search(index, padded_view)
        keepalive = [padded_view]

    assert keepalive is not None
    queries = generate_data((n_queries, n_cols), dtype)
    queries_device = device_ndarray(queries)
    out_idx = np.zeros((n_queries, k), dtype=np.uint32)
    out_dist = np.zeros((n_queries, k), dtype=np.float32)
    out_idx_device = device_ndarray(out_idx)
    out_dist_device = device_ndarray(out_dist)

    cagra.search(
        cagra.SearchParams(),
        index,
        queries_device,
        k,
        neighbors=out_idx_device,
        distances=out_dist_device,
    )
    out_idx = out_idx_device.copy_to_host()

    skl_idx = (
        NearestNeighbors(n_neighbors=k, algorithm="brute", metric="cosine")
        .fit(dataset)
        .kneighbors(queries, return_distance=False)
    )

    recall = calc_recall(out_idx, skl_idx)

    assert recall > 0.9
