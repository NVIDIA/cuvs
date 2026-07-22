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


def test_ivf_flat_numba_cuda_mlir_ltoir_udf_matches_builtin_l2():
    cp = pytest.importorskip("cupy")
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import cuda

    if not cuda.is_available():
        pytest.skip("CUDA is not available to numba_cuda_mlir")

    with cp.cuda.Device(0):

        @ivf_flat.metric(
            order="min",
            initial=0.0,
            coarse_metric="sqeuclidean",
            symbol_name="cuvs_py_ivf_flat_l2_update_f32_test",
        )
        def l2_update(x, y, acc, ctx):
            d = x - y
            return acc + d * d

        dataset = cp.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.5, -0.5],
                [2.0, -1.0, 0.25],
                [-1.5, 2.0, 1.0],
                [3.0, 1.5, -2.0],
                [-2.0, -1.0, 2.5],
            ],
            dtype=cp.float32,
        )
        queries = cp.asarray(
            [[0.2, 0.1, -0.1], [2.1, -0.7, 0.4]],
            dtype=cp.float32,
        )

        index = ivf_flat.build(
            ivf_flat.IndexParams(n_lists=1, metric="sqeuclidean"),
            dataset,
        )
        builtin_distances, builtin_neighbors = ivf_flat.search(
            ivf_flat.SearchParams(n_probes=1),
            index,
            queries,
            3,
        )
        udf_distances, udf_neighbors = ivf_flat.search(
            ivf_flat.SearchParams(n_probes=1, metric=l2_update),
            index,
            queries,
            3,
        )

    cp.testing.assert_allclose(
        cp.asarray(udf_distances),
        cp.asarray(builtin_distances),
        rtol=1e-5,
        atol=1e-5,
    )
    cp.testing.assert_array_equal(
        cp.asarray(udf_neighbors), cp.asarray(builtin_neighbors)
    )


def test_ivf_flat_cuda_source_metric_matches_builtin_l2():
    cp = pytest.importorskip("cupy")

    with cp.cuda.Device(0):
        source = r"""
namespace cuvs::neighbors::ivf_flat::detail {
template <typename T, typename AccT, int Veclen>
__device__ __forceinline__ void compute_dist_udf_impl(
    AccT& acc, AccT x, AccT y)
{
  auto d = x - y;
  acc += d * d;
}
}
"""
        source_metric = ivf_flat.cuda_source_metric(
            source,
            symbol_name="cuvs_py_ivf_flat_cuda_source_l2_test",
        )

        dataset = cp.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.5, -0.5],
                [2.0, -1.0, 0.25],
                [-1.5, 2.0, 1.0],
                [3.0, 1.5, -2.0],
                [-2.0, -1.0, 2.5],
            ],
            dtype=cp.float32,
        )
        queries = cp.asarray(
            [[0.2, 0.1, -0.1], [2.1, -0.7, 0.4]],
            dtype=cp.float32,
        )

        index = ivf_flat.build(
            ivf_flat.IndexParams(n_lists=1, metric="sqeuclidean"),
            dataset,
        )
        builtin_distances, builtin_neighbors = ivf_flat.search(
            ivf_flat.SearchParams(n_probes=1),
            index,
            queries,
            3,
        )
        source_distances, source_neighbors = ivf_flat.search(
            ivf_flat.SearchParams(n_probes=1, metric=source_metric),
            index,
            queries,
            3,
        )

    cp.testing.assert_allclose(
        cp.asarray(source_distances),
        cp.asarray(builtin_distances),
        rtol=1e-5,
        atol=1e-5,
    )
    cp.testing.assert_array_equal(
        cp.asarray(source_neighbors), cp.asarray(builtin_neighbors)
    )


def test_ivf_flat_ltoir_weighted_l2_capture_matches_reference():
    cp = pytest.importorskip("cupy")
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import cuda

    if not cuda.is_available():
        pytest.skip("CUDA is not available to numba_cuda_mlir")

    with cp.cuda.Device(0):
        weights = cp.asarray([0.25, 1.5, 3.0, 0.75], dtype=cp.float32)

        @ivf_flat.metric(
            order="min",
            initial=0.0,
            coarse_metric="sqeuclidean",
            captures={"weights": weights},
            symbol_name="cuvs_py_ivf_flat_weighted_l2_update_f32_test",
        )
        def weighted_l2_update(x, y, acc, ctx):
            d = x - y
            return acc + ctx.weights[ctx.dim] * d * d

        dataset = cp.asarray(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.5, -0.5, 2.0],
                [2.0, -1.0, 0.25, -0.5],
                [-1.5, 2.0, 1.0, 0.75],
                [3.0, 1.5, -2.0, -1.0],
                [-2.0, -1.0, 2.5, 1.25],
            ],
            dtype=cp.float32,
        )
        queries = cp.asarray(
            [[0.2, 0.1, -0.1, 0.5], [2.1, -0.7, 0.4, -0.25]],
            dtype=cp.float32,
        )

        index = ivf_flat.build(
            ivf_flat.IndexParams(n_lists=1, metric="sqeuclidean"),
            dataset,
        )
        udf_distances, udf_neighbors = ivf_flat.search(
            ivf_flat.SearchParams(n_probes=1, metric=weighted_l2_update),
            index,
            queries,
            3,
        )

        diff = queries[:, None, :] - dataset[None, :, :]
        reference_distances = cp.sum(
            weights[None, None, :] * diff * diff, axis=2
        )
        reference_neighbors = cp.argsort(
            reference_distances, axis=1
        )[:, :3].astype(cp.int64)
        reference_top_distances = cp.take_along_axis(
            reference_distances, reference_neighbors, axis=1
        )

    cp.testing.assert_allclose(
        cp.asarray(udf_distances),
        reference_top_distances,
        rtol=1e-5,
        atol=1e-5,
    )
    cp.testing.assert_array_equal(
        cp.asarray(udf_neighbors), reference_neighbors
    )

