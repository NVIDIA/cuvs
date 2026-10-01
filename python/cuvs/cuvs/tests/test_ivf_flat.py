# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import tempfile

import numpy as np
import pytest
from pylibraft.common import device_ndarray
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from cuvs.neighbors import ivf_flat
from cuvs.tests.ann_utils import (
    calc_recall,
    generate_data,
    run_filtered_search_test,
)


def run_ivf_flat_build_search_test(
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
):
    dataset = generate_data((n_rows, n_cols), dtype)
    if metric == "inner_product":
        dataset = normalize(dataset, norm="l2", axis=1)
    dataset_device = device_ndarray(dataset)

    build_params = ivf_flat.IndexParams(
        metric=metric,
        add_data_on_build=add_data_on_build,
    )

    index = ivf_flat.build(build_params, dataset_device)

    if serialize:
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            temp_filename = f.name
        ivf_flat.save(temp_filename, index)
        index = ivf_flat.load(temp_filename)

    if not add_data_on_build:
        dataset_1 = dataset[: n_rows // 2, :]
        dataset_2 = dataset[n_rows // 2 :, :]
        indices_1 = np.arange(n_rows // 2, dtype=np.int64)
        indices_2 = np.arange(n_rows // 2, n_rows, dtype=np.int64)

        dataset_1_device = device_ndarray(dataset_1)
        dataset_2_device = device_ndarray(dataset_2)
        indices_1_device = device_ndarray(indices_1)
        indices_2_device = device_ndarray(indices_2)
        index = ivf_flat.extend(index, dataset_1_device, indices_1_device)
        index = ivf_flat.extend(index, dataset_2_device, indices_2_device)

    queries = generate_data((n_queries, n_cols), dtype)
    out_idx = np.zeros((n_queries, k), dtype=np.int64)
    out_dist = np.zeros((n_queries, k), dtype=np.float32)

    queries_device = device_ndarray(queries)
    out_idx_device = device_ndarray(out_idx) if inplace else None
    out_dist_device = device_ndarray(out_dist) if inplace else None

    search_params = ivf_flat.SearchParams(**search_params)

    ret_output = ivf_flat.search(
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

    # Calculate reference values with sklearn
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

    centers = index.centers
    assert centers.shape[0] == build_params.n_lists
    assert centers.shape[1] == n_cols


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("dtype", [np.float32])
@pytest.mark.parametrize(
    "metric", ["sqeuclidean", "inner_product", "euclidean", "cosine"]
)
def test_ivf_flat(inplace, dtype, metric):
    run_ivf_flat_build_search_test(
        dtype=dtype,
        inplace=inplace,
        metric=metric,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int8, np.uint8])
@pytest.mark.parametrize("serialize", [True, False])
def test_extend(dtype, serialize):
    run_ivf_flat_build_search_test(
        n_rows=10000,
        n_cols=10,
        n_queries=100,
        k=10,
        metric="sqeuclidean",
        dtype=dtype,
        add_data_on_build=False,
        serialize=serialize,
    )


@pytest.mark.parametrize("sparsity", [0.5, 0.7, 1.0])
def test_filtered_ivf_flat(sparsity):
    run_filtered_search_test(ivf_flat, sparsity)
