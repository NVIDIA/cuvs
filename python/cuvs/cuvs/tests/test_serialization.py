# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest
from pylibraft.common import device_ndarray

from cuvs.common import make_device_padded_dataset
from cuvs.neighbors import brute_force, cagra, ivf_flat, ivf_pq
from cuvs.tests.ann_utils import (
    calc_recall,
    generate_data,
)


@pytest.mark.parametrize("dtype", [np.float32, np.int8, np.ubyte])
def test_save_load_ivf_flat(dtype):
    run_save_load(ivf_flat, dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int8, np.ubyte])
def test_save_load_cagra(dtype, tmp_path):
    run_save_load(cagra, dtype, tmp_path / "cagra-index.bin")


def test_save_load_ivf_pq():
    run_save_load(ivf_pq, np.float32)


def test_save_load_brute_force():
    run_save_load(brute_force, np.float32)


def run_save_load(ann_module, dtype, filename="my_index.bin"):
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

    assert index.trained
    filename = str(filename)
    ann_module.save(filename, index)
    if ann_module == cagra:
        loaded_index = ann_module.Index()
        out_dataset = ann_module.Dataset()
        ann_module.load(
            loaded_index,
            filename,
            out_dataset=out_dataset,
        )
    else:
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


def test_cagra_graph_only_serialization(tmp_path):
    n_rows = 2048
    n_cols = 32
    dataset = generate_data((n_rows, n_cols), np.float32)
    dataset_device = device_ndarray(dataset)
    index = cagra.build(cagra.IndexParams(), dataset_device)

    graph_path = tmp_path / "cagra-graph.bin"
    cagra.save(str(graph_path), index, include_dataset=False)

    loaded = cagra.Index()
    cagra.load(loaded, str(graph_path))

    missing_dataset_index = cagra.Index()
    missing_dataset_owner = cagra.Dataset()
    with pytest.raises(Exception, match="no dataset"):
        cagra.load(
            missing_dataset_index,
            str(graph_path),
            out_dataset=missing_dataset_owner,
        )

    padded_dataset = make_device_padded_dataset(dataset)
    cagra.update_dataset(loaded, padded_dataset)
    queries = device_ndarray(generate_data((64, n_cols), np.float32))
    search_params = cagra.SearchParams()
    original_distances, original_neighbors = cagra.search(
        search_params, index, queries, 10
    )
    loaded_distances, loaded_neighbors = cagra.search(
        search_params, loaded, queries, 10
    )
    assert np.allclose(
        original_distances.copy_to_host(),
        loaded_distances.copy_to_host(),
    )
    assert np.array_equal(
        original_neighbors.copy_to_host(),
        loaded_neighbors.copy_to_host(),
    )


def test_cagra_deserialize_output_must_be_empty(tmp_path):
    dataset = device_ndarray(generate_data((2048, 32), np.float32))
    index = cagra.build(cagra.IndexParams(), dataset)
    path = tmp_path / "cagra-full.bin"
    cagra.save(str(path), index, include_dataset=True)
    output = cagra.Dataset()
    first_loaded = cagra.Index()
    cagra.load(first_loaded, str(path), out_dataset=output)

    assert output.memory_type == "device"
    assert output.layout == "padded"
    second_loaded = cagra.Index()
    with pytest.raises(Exception, match="must be null"):
        cagra.load(second_loaded, str(path), out_dataset=output)

    # Destroying the index and its separately owned dataset are independent.
    del first_loaded
    del output
