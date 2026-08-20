# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import cupy as cp
import numpy as np
import pytest
from pylibraft.common import device_ndarray

from cuvs.preprocessing.quantize import pq
from cuvs.neighbors import brute_force
from cuvs.tests.ann_utils import calc_recall, generate_data


@pytest.mark.parametrize("n_rows", [700, 1000])
@pytest.mark.parametrize("n_cols", [64, 128])
@pytest.mark.parametrize("pq_bits", [7, 9])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("pq_kmeans_type", ["kmeans", "kmeans_balanced"])
@pytest.mark.parametrize("train_coarse", [True, False])
@pytest.mark.parametrize("use_subspaces", [True, False])
@pytest.mark.parametrize("device_memory", [True, False])
def test_product_quantizer(
    n_rows,
    n_cols,
    pq_bits,
    inplace,
    pq_kmeans_type,
    train_coarse,
    use_subspaces,
    device_memory,
):
    pq_dim = 32
    vq_n_centers = 0
    dtype = np.float32
    input1 = np.random.random_sample((n_rows, n_cols)).astype(dtype)
    input1_device = device_ndarray(input1)

    params = pq.QuantizerParams(
        pq_bits=pq_bits,
        pq_dim=pq_dim,
        use_subspaces=use_subspaces,
        train_coarse=train_coarse,
        vq_n_centers=vq_n_centers,
        pq_kmeans_type=pq_kmeans_type,
    )
    if device_memory:
        quantizer = pq.build(params, input1_device)
    else:
        quantizer = pq.build(params, input1)

    output = (
        np.zeros((n_rows, quantizer.encoded_dim), dtype="uint8")
        if inplace
        else None
    )
    output_device = device_ndarray(output) if inplace else None
    vq_labels = (
        np.zeros((n_rows,), dtype="uint32")
        if inplace and train_coarse
        else None
    )
    vq_labels_device = (
        device_ndarray(vq_labels) if inplace and train_coarse else None
    )
    if device_memory:
        transformed, vq_labels_device = pq.transform(
            quantizer,
            input1_device,
            codes_output=output_device,
            vq_labels=vq_labels_device,
        )
    else:
        transformed, vq_labels_device = pq.transform(
            quantizer, input1, codes_output=output_device
        )
    actual = transformed if not inplace else output_device
    actual = actual.copy_to_host()

    # naive tests
    assert actual.any()  # Check that the output is not all zeros
    assert not np.isnan(actual).any()  # Check that the output has no NaNs
    assert cp.array(quantizer.pq_codebook).any()

    reconstructed = cp.empty((n_rows, n_cols), dtype=dtype)
    pq.inverse_transform(
        quantizer, transformed, reconstructed, vq_labels=vq_labels_device
    )
    reconstructed = cp.array(reconstructed)
    assert reconstructed.shape == input1.shape
    reconstruction_error = cp.linalg.norm(
        cp.array(input1_device) - reconstructed, axis=1
    )
    assert reconstruction_error.mean() < 1.5


def test_extreme_cases():
    n_samples = 5000
    n_features = 2048
    dataset = cp.random.random_sample(
        (n_samples, n_features), dtype=cp.float32
    )
    params = pq.QuantizerParams(pq_bits=8, pq_dim=2)
    quantizer = pq.build(params, dataset)
    pq.transform(quantizer, dataset)


@pytest.mark.parametrize("train_coarse", [True, False])
@pytest.mark.parametrize("use_subspaces", [True, False])
@pytest.mark.parametrize("pq_dim", [64, 128])
def test_recall(train_coarse, use_subspaces, pq_dim):
    n_samples = 5000
    n_queries = 150
    n_features = 256
    pq_bits = 8
    pq_kmeans_type = "kmeans_balanced"
    dataset = generate_data((n_samples, n_features), dtype=np.float32)
    queries = generate_data((n_queries, n_features), dtype=np.float32)
    queries_device = device_ndarray(queries)
    dataset_device = device_ndarray(dataset)
    params = pq.QuantizerParams(
        pq_bits=pq_bits,
        pq_dim=pq_dim,
        use_subspaces=use_subspaces,
        train_coarse=train_coarse,
        pq_kmeans_type=pq_kmeans_type,
    )
    quantizer = pq.build(params, dataset)
    transformed, vq_labels = pq.transform(quantizer, dataset)
    reconstructed = pq.inverse_transform(
        quantizer, transformed, vq_labels=vq_labels
    )

    index = brute_force.build(reconstructed)
    distances, indices = brute_force.search(index, queries_device, k=10)

    index_gt = brute_force.build(dataset_device)
    distances_gt, indices_gt = brute_force.search(
        index_gt, queries_device, k=10
    )
    indices_host = indices.copy_to_host()
    indices_gt_host = indices_gt.copy_to_host()
    recall = calc_recall(indices_host, indices_gt_host)
    expected_recall = 0.5 if pq_dim == 64 else 0.75
    assert recall > expected_recall
