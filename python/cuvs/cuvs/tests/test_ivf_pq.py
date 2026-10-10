# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import tempfile

import numpy as np
import pytest
from pylibraft.common import device_ndarray
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from cuvs.neighbors import ivf_pq
from cuvs.tests.ann_utils import calc_recall, generate_data


def run_ivf_pq_build_search_test(
    n_rows=10000,
    n_cols=10,
    n_queries=100,
    k=10,
    dtype=np.float32,
    n_lists=100,
    metric="euclidean",
    pq_bits=8,
    pq_dim=0,
    codebook_kind="subspace",
    add_data_on_build=True,
    n_probes=100,
    lut_dtype=np.float32,
    internal_distance_dtype=np.float32,
    force_random_rotation=False,
    kmeans_trainset_fraction=1,
    kmeans_n_iters=20,
    compare=True,
    inplace=True,
    array_type="device",
    serialize=False,
):
    dataset = generate_data((n_rows, n_cols), dtype)
    if metric == "inner_product":
        dataset = normalize(dataset, norm="l2", axis=1)
    dataset_device = device_ndarray(dataset)

    build_params = ivf_pq.IndexParams(
        n_lists=n_lists,
        metric=metric,
        kmeans_n_iters=kmeans_n_iters,
        kmeans_trainset_fraction=kmeans_trainset_fraction,
        pq_bits=pq_bits,
        pq_dim=pq_dim,
        codebook_kind=codebook_kind,
        force_random_rotation=force_random_rotation,
        add_data_on_build=add_data_on_build,
    )

    if array_type == "device":
        index = ivf_pq.build(build_params, dataset_device)
    else:
        index = ivf_pq.build(build_params, dataset)

    if serialize:
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            temp_filename = f.name
        ivf_pq.save(temp_filename, index)
        index = ivf_pq.load(temp_filename)

    if not add_data_on_build:
        dataset_1 = dataset[: n_rows // 2, :]
        dataset_2 = dataset[n_rows // 2 :, :]
        indices_1 = np.arange(n_rows // 2, dtype=np.int64)
        indices_2 = np.arange(n_rows // 2, n_rows, dtype=np.int64)

        if array_type == "device":
            dataset_1_device = device_ndarray(dataset_1)
            dataset_2_device = device_ndarray(dataset_2)
            indices_1_device = device_ndarray(indices_1)
            indices_2_device = device_ndarray(indices_2)
            index = ivf_pq.extend(index, dataset_1_device, indices_1_device)
            index = ivf_pq.extend(index, dataset_2_device, indices_2_device)
        else:
            index = ivf_pq.extend(index, dataset_1, indices_1)
            index = ivf_pq.extend(index, dataset_2, indices_2)

    queries = generate_data((n_queries, n_cols), dtype)
    out_idx = np.zeros((n_queries, k), dtype=np.int64)
    out_dist = np.zeros((n_queries, k), dtype=np.float32)

    queries_device = device_ndarray(queries)
    out_idx_device = device_ndarray(out_idx) if inplace else None
    out_dist_device = device_ndarray(out_dist) if inplace else None

    search_params = ivf_pq.SearchParams(
        n_probes=n_probes,
        lut_dtype=lut_dtype,
        internal_distance_dtype=internal_distance_dtype,
    )

    ret_output = ivf_pq.search(
        search_params,
        index,
        queries_device,
        k,
        neighbors=out_idx_device,
        distances=out_dist_device,
    )

    if not inplace:
        out_dist_device, out_idx_device = ret_output

    centers = index.centers
    assert centers.shape[0] == n_lists
    assert centers.shape[1] == index.dim

    pq_centers = index.pq_centers
    assert len(pq_centers.shape) == 3
    assert pq_centers.shape[2] == 1 << pq_bits

    all_list_ids = set()
    for list_ids, list_data in index.lists():
        all_list_ids.update(list_ids.copy_to_host())
    assert all_list_ids == set(np.arange(n_rows))

    if not compare:
        return

    out_idx = out_idx_device.copy_to_host()
    out_dist = out_dist_device.copy_to_host()

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
    skl_idx = nn_skl.kneighbors(queries, return_distance=False)

    recall = calc_recall(out_idx, skl_idx)
    assert recall > 0.7


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("dtype", [np.float32])
@pytest.mark.parametrize(
    "metric", ["sqeuclidean", "inner_product", "euclidean"]
)
def test_ivf_pq(inplace, dtype, metric):
    # One dimension per PQ subspace. With the default pq_dim (8 for 10
    # columns), the inner-product recall on this data is only about 0.74 and
    # varies by about 0.01 from build to build, so `recall > 0.7` fails in
    # roughly 1 of 1000 runs.
    run_ivf_pq_build_search_test(
        n_cols=10,
        pq_dim=10,
        dtype=dtype,
        inplace=inplace,
        metric=metric,
    )


@pytest.mark.parametrize(
    "params",
    [
        {
            "k": 10,
            "n_probes": 100,
            "lut": np.float16,
            "idd": np.float32,
        },
        {
            "k": 10,
            "n_probes": 99,
            "lut": np.uint8,
            "idd": np.float32,
        },
        {
            "k": 10,
            "n_probes": 100,
            "lut": np.float16,
            "idd": np.float16,
        },
        {
            "k": 129,
            "n_probes": 100,
            "lut": np.float32,
            "idd": np.float32,
        },
    ],
)
def test_ivf_pq_search_params(params):
    run_ivf_pq_build_search_test(
        n_rows=10000,
        n_cols=16,
        n_queries=1000,
        k=params["k"],
        n_lists=100,
        n_probes=params["n_probes"],
        metric="sqeuclidean",
        dtype=np.float32,
        lut_dtype=params["lut"],
        internal_distance_dtype=params["idd"],
    )


@pytest.mark.parametrize("dtype", [np.float32])
@pytest.mark.parametrize("array_type", ["host", "device"])
def test_extend(dtype, array_type):
    run_ivf_pq_build_search_test(
        n_rows=10000,
        n_cols=10,
        n_queries=100,
        k=10,
        n_lists=100,
        metric="sqeuclidean",
        dtype=dtype,
        add_data_on_build=False,
        array_type=array_type,
    )


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int8, np.uint8])
@pytest.mark.parametrize("array_type", ["host", "device"])
@pytest.mark.parametrize("serialize", [True, False])
def test_ivf_pq_dtype(inplace, dtype, array_type, serialize):
    run_ivf_pq_build_search_test(
        dtype=dtype,
        inplace=inplace,
        array_type=array_type,
        serialize=serialize,
    )


@pytest.mark.parametrize("codebook_kind", ["subspace", "cluster"])
@pytest.mark.parametrize(
    "metric", ["sqeuclidean", "inner_product", "euclidean"]
)
def test_build_precomputed(codebook_kind, metric):
    n_rows = 5000
    n_cols = 32
    n_queries = 100
    k = 10
    n_lists = 50
    pq_bits = 8
    pq_dim = 8
    n_probes = 50
    dtype = np.float32

    # Generate dataset
    dataset = generate_data((n_rows, n_cols), dtype)
    if metric == "inner_product":
        dataset = normalize(dataset, norm="l2", axis=1)
    dataset_device = device_ndarray(dataset)

    # Build regular index with data
    build_params = ivf_pq.IndexParams(
        n_lists=n_lists,
        metric=metric,
        kmeans_n_iters=20,
        kmeans_trainset_fraction=1.0,
        pq_bits=pq_bits,
        pq_dim=pq_dim,
        codebook_kind=codebook_kind,
        force_random_rotation=False,
        add_data_on_build=True,
    )
    regular_index = ivf_pq.build(build_params, dataset_device)

    # Extract trained components from regular index
    # Use centers_padded which returns contiguous data suitable for build_precomputed
    pq_centers = regular_index.pq_centers
    centers = regular_index.centers_padded
    centers_rot = regular_index.centers_rot
    rotation_matrix = regular_index.rotation_matrix
    dim = regular_index.dim

    # Build precomputed index with extracted components
    precomputed_build_params = ivf_pq.IndexParams(
        n_lists=n_lists,
        metric=metric,
        pq_bits=pq_bits,
        pq_dim=pq_dim,
        codebook_kind=codebook_kind,
    )
    precomputed_index = ivf_pq.build_precomputed(
        precomputed_build_params,
        dim,
        pq_centers,
        centers,
        centers_rot,
        rotation_matrix,
    )

    # Extend precomputed index with the same data
    indices = np.arange(n_rows, dtype=np.int64)
    indices_device = device_ndarray(indices)
    precomputed_index = ivf_pq.extend(
        precomputed_index, dataset_device, indices_device
    )

    # Verify both indexes have the same size
    assert len(regular_index) == len(precomputed_index)
    assert len(regular_index) == n_rows

    # Generate queries
    queries = generate_data((n_queries, n_cols), dtype)
    queries_device = device_ndarray(queries)

    # Search both indexes for one extra neighbor, so that a distance tie
    # across the k-th position can be detected.
    search_params = ivf_pq.SearchParams(n_probes=n_probes)
    regular_dist, regular_idx = ivf_pq.search(
        search_params, regular_index, queries_device, k + 1
    )
    precomputed_dist, precomputed_idx = ivf_pq.search(
        search_params, precomputed_index, queries_device, k + 1
    )

    # Copy results to host for comparison
    regular_idx_host = regular_idx.copy_to_host()
    regular_dist_host = regular_dist.copy_to_host()
    precomputed_idx_host = precomputed_idx.copy_to_host()
    precomputed_dist_host = precomputed_dist.copy_to_host()

    # Both indexes hold the same centers and PQ codes, so the distances must
    # match exactly.
    np.testing.assert_array_equal(
        regular_dist_host[:, :k],
        precomputed_dist_host[:, :k],
        err_msg="Distances should match exactly",
    )

    # The order of the vectors within each list is not deterministic (extend
    # assigns the slots atomically), and the search breaks distance ties by
    # that order. Neighbors with bitwise-equal distances may therefore come
    # back in a different order, and a tie across the k-th position may
    # select different neighbors. Compare the indices of each group of equal
    # distances as a set, except for a group that crosses the k-th position.
    for row in range(n_queries):
        dist = regular_dist_host[row]
        start = 0
        while start < k:
            end = start + 1
            while end <= k and dist[end] == dist[start]:
                end += 1
            if end <= k:
                np.testing.assert_array_equal(
                    np.sort(regular_idx_host[row, start:end]),
                    np.sort(precomputed_idx_host[row, start:end]),
                    err_msg=f"Neighbor indices should match (query {row})",
                )
            start = end


@pytest.mark.parametrize("codebook_kind", ["subspace", "cluster"])
@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int8, np.uint8])
@pytest.mark.parametrize("n_rows", [500, 100000])
@pytest.mark.parametrize("pq_bits", [6, 8])
def test_transform(codebook_kind, dtype, n_rows, pq_bits):
    n_cols = 32
    n_lists = 50
    pq_dim = 8

    # build the ivf-pq index
    dataset = generate_data((n_rows, n_cols), dtype)
    dataset_device = device_ndarray(dataset)

    build_params = ivf_pq.IndexParams(
        n_lists=n_lists,
        pq_bits=pq_bits,
        pq_dim=pq_dim,
        codebook_kind=codebook_kind,
    )
    index = ivf_pq.build(build_params, dataset_device)

    # test out the ivf_pq.transform function - by transforming
    # the input dataset ,and making sure the transformed data
    # matches the encoded version stored with the index

    # collect the pq-encoded data from the index for the expected data
    labels_exp = np.zeros((n_rows), dtype="uint32")
    transformed_exp = np.zeros(
        (n_rows, int(np.ceil(index.pq_dim * index.pq_bits / 8))), dtype="uint8"
    )
    for cluster_id in range(n_lists):
        ids = index.list_indices(cluster_id).copy_to_host()
        labels_exp[ids] = cluster_id
        transformed_exp[ids] = index.list_data(cluster_id).copy_to_host()

    # call transform to get actual results, and make sure we're equal to expected values
    labels, transformed = ivf_pq.transform(index, dataset_device)

    np.testing.assert_array_equal(labels.copy_to_host(), labels_exp)
    np.testing.assert_array_equal(transformed.copy_to_host(), transformed_exp)
