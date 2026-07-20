# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest
from pylibraft.common import device_ndarray

from cuvs.neighbors import brute_force, cagra, ivf_flat, ivf_pq
from cuvs.tests.ann_utils import (
    calc_recall,
    generate_data,
)


@pytest.mark.parametrize("dtype", [np.float32, np.int8, np.ubyte])
def test_save_load_ivf_flat(dtype):
    run_save_load(ivf_flat, dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.int8, np.ubyte])
def test_save_load_cagra(dtype):
    run_save_load(cagra, dtype)


def test_save_load_ivf_pq():
    run_save_load(ivf_pq, np.float32)


def test_save_load_brute_force():
    run_save_load(brute_force, np.float32)


def run_save_load(ann_module, dtype):
    n_rows = 10000
    n_cols = 50
    n_queries = 1000

    dataset = generate_data((n_rows, n_cols), dtype)
    dataset_device = device_ndarray(dataset)

    if ann_module == brute_force:
        index = ann_module.build(dataset_device)
    else:
        build_params = ann_module.IndexParams()
        index = ann_module.build(build_params, dataset_device)
        if ann_module == cagra:
            if (
                ann_module.get_dataset_view_kind(dataset_device)
                == "device_standard"
            ):
                padded_dataset = ann_module.make_device_padded_dataset(
                    dataset_device
                )
                padded_view = ann_module.make_view_from_owning_padded(
                    padded_dataset
                )
                ann_module.attach_padded_dataset_for_search(index, padded_view)
                index._cagra_keepalive = [padded_dataset, padded_view]

    assert index.trained
    filename = "my_index.bin"
    ann_module.save(filename, index)
    loaded_index = ann_module.load(filename)

    queries = generate_data((n_queries, n_cols), dtype)

    queries_device = device_ndarray(queries)
    k = 10
    if ann_module == brute_force:
        distance_dev, neighbors_dev = ann_module.search(
            index, queries_device, k
        )
    else:
        search_params = ann_module.SearchParams()
        distance_dev, neighbors_dev = ann_module.search(
            search_params, index, queries_device, k
        )

    neighbors = neighbors_dev.copy_to_host()
    dist = distance_dev.copy_to_host()
    del index

    if ann_module == brute_force:
        distance_dev, neighbors_dev = ann_module.search(
            loaded_index, queries_device, k
        )
    else:
        distance_dev, neighbors_dev = ann_module.search(
            search_params, loaded_index, queries_device, k
        )

    neighbors2 = neighbors_dev.copy_to_host()
    dist2 = distance_dev.copy_to_host()

    assert np.allclose(dist, dist2, rtol=1e-6)

    # Sort the neighbors to avoid ordering issues
    sorted_neighbors = np.argsort(neighbors, axis=-1)
    sorted_neighbors2 = np.argsort(neighbors2, axis=-1)
    neighbors = np.take_along_axis(neighbors, sorted_neighbors, axis=-1)
    neighbors2 = np.take_along_axis(neighbors2, sorted_neighbors2, axis=-1)
    all_match = np.all(neighbors == neighbors2)
    # If the neighbors are not the same, there might be a cutoff between the k
    # and k+1 neighbors at the same distance.
    # Calculate that the recall is at least 99.8%
    if not all_match:
        recall = calc_recall(neighbors, neighbors2)
        assert recall >= 0.998
