# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# Regression tests for CAGRA search recall consistency across query batch sizes.
# See https://github.com/rapidsai/cuvs/issues/1187
#
# When the AUTO algorithm selector switches from MULTI_CTA to SINGLE_CTA
# at larger batch sizes, SINGLE_CTA computes max_iterations = itopk_size /
# search_width. With search_width > 1, this gives far fewer iterations than
# MULTI_CTA (which uses internal mc_search_width=1), causing a recall cliff.
# The fix prevents SINGLE_CTA selection when search_width > 1.

import numpy as np
import pytest
from pylibraft.common import device_ndarray

from cuvs.neighbors import cagra


def calc_recall(found_indices, ground_truth, k):
    """Calculate recall@k between found indices and ground truth."""
    n_queries = found_indices.shape[0]
    correct = 0
    total = n_queries * k
    for i in range(n_queries):
        correct += len(
            set(found_indices[i, :k].tolist())
            & set(ground_truth[i, :k].tolist())
        )
    return correct / total


def search_in_batches(search_params, index, queries, k, batch_size):
    """Search the index in fixed-size batches, return host neighbors array."""
    n_queries = queries.shape[0]
    all_neighbors = np.zeros((n_queries, k), dtype=np.uint32)
    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        q = device_ndarray(queries[start:end])
        _, neighbors = cagra.search(search_params, index, q, k=k)
        neighbors_host = neighbors.copy_to_host()
        all_neighbors[start:end] = neighbors_host
    return all_neighbors


@pytest.fixture(scope="module")
def cagra_test_data():
    """Build CAGRA index and brute-force ground truth once for all tests."""
    np.random.seed(42)
    n_samples = 50000
    n_queries = 512
    dim = 64
    k = 10

    dataset = np.random.randn(n_samples, dim).astype(np.float32)
    queries = np.random.randn(n_queries, dim).astype(np.float32)
    dataset_device = device_ndarray(dataset)

    build_params = cagra.IndexParams(
        graph_degree=32, intermediate_graph_degree=64
    )
    index = cagra.build(build_params, dataset_device)

    # Brute-force ground truth via NumPy (L2 distance), chunked to avoid OOM.
    # Each chunk creates a (chunk_size x n_samples x dim) array.
    # chunk_size=16 => 16 * 50000 * 64 * 4 bytes = ~200MB per chunk.
    gt_neighbors = np.zeros((n_queries, k), dtype=np.uint32)
    chunk_size = 16
    for start in range(0, n_queries, chunk_size):
        end = min(start + chunk_size, n_queries)
        q_chunk = queries[start:end]
        dists = np.sum(
            (q_chunk[:, None, :] - dataset[None, :, :]) ** 2, axis=2
        )
        gt_neighbors[start:end] = np.argsort(dists, axis=1)[:, :k].astype(
            np.uint32
        )
        del dists

    return {
        "index": index,
        "dataset_device": dataset_device,  # keep alive for index search
        "queries": queries,
        "gt_neighbors": gt_neighbors,
        "n_queries": n_queries,
        "k": k,
    }


@pytest.mark.parametrize("search_width", [1, 4, 8])
def test_cagra_batch_recall_consistency(cagra_test_data, search_width):
    """
    Recall must be consistent across batch sizes regardless of search_width.

    The AUTO selector may switch algorithms at larger batch sizes. This test
    verifies that the switch does not cause a recall cliff. Without the fix,
    search_width=8 drops from ~0.87 recall (MULTI_CTA) to ~0.65 (SINGLE_CTA)
    at batch_size >= 256 (on machines with >= 128 SMs, fewer for smaller GPUs).
    """
    data = cagra_test_data
    search_params = cagra.SearchParams(
        itopk_size=64, search_width=search_width
    )

    batch_sizes = [32, 64, 128, 256, 512]
    recalls = []

    for batch_size in batch_sizes:
        neighbors = search_in_batches(
            search_params,
            data["index"],
            data["queries"],
            data["k"],
            batch_size,
        )
        recall = calc_recall(neighbors, data["gt_neighbors"], data["k"])
        recalls.append(recall)

    recall_std = np.std(recalls)
    recall_range = max(recalls) - min(recalls)

    assert recall_std < 0.02, (
        f"search_width={search_width}: recall varies too much across batch "
        f"sizes (std={recall_std:.4f}). "
        + ", ".join(f"bs={bs}:{r:.4f}" for bs, r in zip(batch_sizes, recalls))
    )
    assert recall_range < 0.05, (
        f"search_width={search_width}: recall range too wide "
        f"({recall_range:.4f}). "
        + ", ".join(f"bs={bs}:{r:.4f}" for bs, r in zip(batch_sizes, recalls))
    )


@pytest.mark.parametrize(
    "search_width,min_expected_recall",
    [(1, 0.4), (4, 0.6), (8, 0.7)],
)
def test_cagra_search_width_recall_quality(
    cagra_test_data, search_width, min_expected_recall
):
    """
    Higher search_width must yield higher recall, not lower.

    SINGLE_CTA with search_width > 1 previously got fewer iterations,
    producing LOWER recall with HIGHER search_width — the opposite of
    expected behavior. This test catches that inversion.
    """
    data = cagra_test_data
    search_params = cagra.SearchParams(
        itopk_size=64, search_width=search_width
    )

    # Use batch_size=512 to trigger AUTO algorithm selection
    neighbors = search_in_batches(
        search_params,
        data["index"],
        data["queries"],
        data["k"],
        batch_size=512,
    )
    recall = calc_recall(neighbors, data["gt_neighbors"], data["k"])

    assert recall >= min_expected_recall, (
        f"search_width={search_width}: recall={recall:.4f} is below "
        f"minimum expected {min_expected_recall:.4f} at batch_size=512. "
        f"This suggests SINGLE_CTA may be selected with too few iterations."
    )


def test_cagra_search_width_monotonicity(cagra_test_data):
    """
    Recall must increase (or stay flat) as search_width increases.

    Wider search explores more neighbors per iteration. If recall decreases
    with higher search_width, the algorithm is getting fewer effective
    iterations — the exact bug this fix addresses.
    """
    data = cagra_test_data
    search_widths = [1, 2, 4, 8]
    recalls = []

    for sw in search_widths:
        search_params = cagra.SearchParams(itopk_size=64, search_width=sw)
        neighbors = search_in_batches(
            search_params,
            data["index"],
            data["queries"],
            data["k"],
            batch_size=512,
        )
        recall = calc_recall(neighbors, data["gt_neighbors"], data["k"])
        recalls.append(recall)

    # Each subsequent search_width should have recall >= previous - tolerance
    tolerance = 0.02
    for i in range(1, len(search_widths)):
        assert recalls[i] >= recalls[i - 1] - tolerance, (
            f"Recall decreased from sw={search_widths[i - 1]} "
            f"({recalls[i - 1]:.4f}) to sw={search_widths[i]} "
            f"({recalls[i]:.4f}) at batch_size=512. "
            f"Higher search_width should not reduce recall."
        )
