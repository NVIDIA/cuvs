# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
from pylibraft.common import device_ndarray
from sklearn.neighbors import NearestNeighbors

from cuvs.neighbors import filters


def generate_data(shape, dtype):
    if dtype == np.byte:
        x = np.random.randint(-127, 128, size=shape, dtype=np.byte)
    elif dtype == np.ubyte:
        x = np.random.randint(0, 255, size=shape, dtype=np.ubyte)
    else:
        x = np.random.random_sample(shape).astype(dtype)

    return x


def calc_recall(ann_idx, true_nn_idx):
    assert ann_idx.shape == true_nn_idx.shape
    n = 0
    for i in range(ann_idx.shape[0]):
        n += np.intersect1d(ann_idx[i, :], true_nn_idx[i, :]).size
    recall = n / ann_idx.size
    return recall


def create_sparse_bitset(n_size, sparsity):
    bits_per_uint32 = 32
    num_bits = n_size
    num_uint32s = (num_bits + bits_per_uint32 - 1) // bits_per_uint32
    num_ones = int(num_bits * sparsity)

    array = np.zeros(num_uint32s, dtype=np.uint32)
    indices = np.random.choice(num_bits, num_ones, replace=False)

    for index in indices:
        i = index // bits_per_uint32
        bit_position = index % bits_per_uint32
        array[i] |= 1 << bit_position

    return array


def run_filtered_search_test(
    search_module,
    sparsity,
    n_rows=10000,
    n_cols=10,
    n_queries=10,
    k=10,
):
    dataset = generate_data((n_rows, n_cols), np.float32)
    queries = generate_data((n_queries, n_cols), np.float32)

    bitset = create_sparse_bitset(n_rows, sparsity)

    dataset_device = device_ndarray(dataset)
    queries_device = device_ndarray(queries)
    bitset_device = device_ndarray(bitset)

    module_name = str(search_module).lower()
    is_ivf = "ivf" in module_name
    is_cagra = "cagra" in module_name

    if is_ivf:
        # Adjust parameters based on sparsity to maximize recall for IVF
        # With higher sparsity (more filtered items), we need:
        # - Fewer clusters (lower n_lists) to ensure each cluster has enough
        #   unfiltered points
        # - More probes to search through enough clusters to find k neighbors

        n_lists = math.isqrt(n_rows) * sparsity
        n_probes = max(1, n_lists // (10 * sparsity))

        build_params = search_module.IndexParams(n_lists=n_lists)
        search_params = search_module.SearchParams(n_probes=n_probes)
    else:
        build_params = search_module.IndexParams()
        search_params = search_module.SearchParams()

    index = search_module.build(build_params, dataset_device)
    if is_cagra:
        view_kind = search_module.get_dataset_view_kind(dataset_device)
        if view_kind == "device_standard":
            padded_dataset = search_module.make_device_padded_dataset(
                dataset_device
            )
            padded_view = search_module.make_view_from_owning_padded(
                padded_dataset
            )
            search_module.attach_padded_dataset_for_search(index, padded_view)
            index._cagra_keepalive = [padded_dataset, padded_view]
    filter_ = filters.from_bitset(bitset_device)
    ret_distances, ret_indices = search_module.search(
        search_params,
        index,
        queries_device,
        k,
        filter=filter_,
    )

    # Convert bitset to bool array for validation
    bitset_as_uint8 = bitset.view(np.uint8)
    bool_filter = np.unpackbits(bitset_as_uint8)
    bool_filter = bool_filter.reshape(-1, 4, 8)
    bool_filter = np.flip(bool_filter, axis=2)
    bool_filter = bool_filter.reshape(-1)[:n_rows]
    bool_filter = np.logical_not(bool_filter)  # Flip so True means filtered

    # Get filtered dataset for reference calculation
    non_filtered_mask = ~bool_filter
    filtered_dataset = dataset[non_filtered_mask]

    nn_skl = NearestNeighbors(
        n_neighbors=k, algorithm="brute", metric="euclidean"
    )
    nn_skl.fit(filtered_dataset)
    skl_idx = nn_skl.kneighbors(queries, return_distance=False)

    actual_indices = ret_indices.copy_to_host()

    filtered_idx_map = (
        np.cumsum(~bool_filter) - 1
    )  # -1 because cumsum starts at 1

    # Map ANN indices to filtered space
    mapped_actual_indices = np.take(
        filtered_idx_map, actual_indices, mode="clip"
    )

    filtered_indices = np.where(bool_filter)[0]
    for i in range(n_queries):
        assert not np.intersect1d(filtered_indices, actual_indices[i]).size

    recall = calc_recall(mapped_actual_indices, skl_idx)

    assert recall > 0.7
