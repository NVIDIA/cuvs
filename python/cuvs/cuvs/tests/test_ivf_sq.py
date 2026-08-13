# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import os
import tempfile

import numpy as np
import pytest
from pylibraft.common import device_ndarray
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from cuvs.neighbors import ivf_sq
from cuvs.tests.ann_utils import (
    calc_recall,
    generate_data,
    run_filtered_search_test,
)


def run_ivf_sq_build_search_test(
    n_rows=10000,
    n_cols=10,
    n_queries=100,
    k=10,
    dtype=np.float32,
    add_data_on_build=True,
    metric="euclidean",
    compare=True,
    inplace=True,
    search_params={},
    serialize=False,
    extend_after_build=False,
    n_extend_rows=0,
):
    dataset = generate_data((n_rows, n_cols), dtype)
    if metric == "inner_product":
        dataset = normalize(dataset, norm="l2", axis=1)
    dataset_device = device_ndarray(dataset)

    build_params = ivf_sq.IndexParams(
        metric=metric,
        add_data_on_build=add_data_on_build,
    )

    index = ivf_sq.build(build_params, dataset_device)

    if serialize:
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            temp_filename = f.name
        ivf_sq.save(temp_filename, index)
        index = ivf_sq.load(temp_filename)

    if not add_data_on_build:
        dataset_1 = dataset[: n_rows // 2, :]
        dataset_2 = dataset[n_rows // 2 :, :]
        indices_1 = np.arange(n_rows // 2, dtype=np.int64)
        indices_2 = np.arange(n_rows // 2, n_rows, dtype=np.int64)

        dataset_1_device = device_ndarray(dataset_1)
        dataset_2_device = device_ndarray(dataset_2)
        indices_1_device = device_ndarray(indices_1)
        indices_2_device = device_ndarray(indices_2)
        index = ivf_sq.extend(index, dataset_1_device, indices_1_device)
        index = ivf_sq.extend(index, dataset_2_device, indices_2_device)
    elif extend_after_build:
        assert n_extend_rows > 0
        extend_data = generate_data((n_extend_rows, n_cols), dtype)
        if metric == "inner_product":
            extend_data = normalize(extend_data, norm="l2", axis=1)

        extend_indices = np.arange(
            n_rows, n_rows + n_extend_rows, dtype=np.int64
        )
        index = ivf_sq.extend(
            index,
            device_ndarray(extend_data),
            device_ndarray(extend_indices),
        )
        dataset = np.concatenate((dataset, extend_data), axis=0)
        n_rows += n_extend_rows

    queries = generate_data((n_queries, n_cols), dtype)
    out_idx = np.zeros((n_queries, k), dtype=np.int64)
    out_dist = np.zeros((n_queries, k), dtype=np.float32)

    queries_device = device_ndarray(queries)
    out_idx_device = device_ndarray(out_idx) if inplace else None
    out_dist_device = device_ndarray(out_dist) if inplace else None

    search_params = ivf_sq.SearchParams(**search_params)

    ret_output = ivf_sq.search(
        search_params,
        index,
        queries_device,
        k,
        neighbors=out_idx_device,
        distances=out_dist_device,
    )

    if not inplace:
        out_dist_device, out_idx_device = ret_output

    if not compare:
        return

    out_idx = out_idx_device.copy_to_host()
    out_dist = out_dist_device.copy_to_host()

    skl_metric = {
        "sqeuclidean": "sqeuclidean",
        "inner_product": "cosine",
        "cosine": "cosine",
        "euclidean": "euclidean",
    }[metric]
    nn_skl = NearestNeighbors(
        n_neighbors=k, algorithm="brute", metric=skl_metric
    )
    nn_skl.fit(dataset)
    skl_idx = nn_skl.kneighbors(queries, return_distance=False)

    recall = calc_recall(out_idx, skl_idx)
    assert recall > 0.7

    assert len(index) == n_rows
    assert index.dim == n_cols
    assert index.n_lists == build_params.n_lists

    centers = index.centers
    assert centers.shape[0] == build_params.n_lists
    assert centers.shape[1] == n_cols


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize(
    "metric", ["sqeuclidean", "inner_product", "euclidean", "cosine"]
)
def test_ivf_sq(inplace, dtype, metric):
    run_ivf_sq_build_search_test(
        dtype=dtype,
        inplace=inplace,
        metric=metric,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("serialize", [True, False])
def test_extend(dtype, serialize):
    run_ivf_sq_build_search_test(
        n_rows=10000,
        n_cols=10,
        n_queries=100,
        k=10,
        metric="sqeuclidean",
        dtype=dtype,
        add_data_on_build=False,
        serialize=serialize,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_extend_after_build_with_data(dtype):
    run_ivf_sq_build_search_test(
        n_rows=10000,
        n_cols=10,
        n_queries=100,
        k=10,
        metric="sqeuclidean",
        dtype=dtype,
        add_data_on_build=True,
        extend_after_build=True,
        n_extend_rows=2000,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_serialization(dtype):
    n_rows, n_cols = 5000, 16
    dataset = generate_data((n_rows, n_cols), dtype)
    index = ivf_sq.build(
        ivf_sq.IndexParams(metric="sqeuclidean"), device_ndarray(dataset)
    )

    expected_n_lists = index.n_lists
    expected_dim = index.dim
    expected_centers = index.centers.copy_to_host()

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        temp_filename = f.name
    try:
        ivf_sq.save(temp_filename, index)
        loaded = ivf_sq.load(temp_filename)
    finally:
        os.unlink(temp_filename)

    assert loaded.n_lists == expected_n_lists
    assert loaded.dim == expected_dim
    np.testing.assert_allclose(loaded.centers.copy_to_host(), expected_centers)


@pytest.mark.parametrize("sparsity", [0.5, 0.7, 1.0])
def test_filtered_ivf_sq(sparsity):
    run_filtered_search_test(ivf_sq, sparsity)


def test_untrained_index_accessors_raise():
    index = ivf_sq.Index()

    with pytest.raises(
        ValueError, match="Index needs to be built before getting n_lists"
    ):
        _ = index.n_lists

    with pytest.raises(
        ValueError, match="Index needs to be built before getting dim"
    ):
        _ = index.dim

    with pytest.raises(
        ValueError, match="Index needs to be built before getting len"
    ):
        len(index)

    with pytest.raises(
        ValueError, match="Index needs to be built before getting centers"
    ):
        _ = index.centers
