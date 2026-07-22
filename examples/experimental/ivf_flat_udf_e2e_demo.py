# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end IVF Flat custom metric UDF demo.

This PoC demonstrates two custom-metric paths:

1. A Python metric compiled to LTO-IR by numba-cuda-mlir, with a CuPy capture.
2. An expert CUDA/C++ source-string metric using the existing JIT/LTO UDF path.
"""

from __future__ import annotations

import importlib.util
import sys


def _require_module(name):
    if importlib.util.find_spec(name) is None:
        raise RuntimeError(f"required module {name!r} is not available")


def _print_banner(title):
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def _asnumpy(cp, value):
    return cp.asnumpy(cp.asarray(value))


def _format_array(cp, value):
    return str(_asnumpy(cp, value))


def _weighted_l2_reference(cp, queries, dataset, weights, k):
    diff = queries[:, None, :] - dataset[None, :, :]
    distances = cp.sum(weights[None, None, :] * diff * diff, axis=2)
    neighbors = cp.argsort(distances, axis=1)[:, :k].astype(cp.int64)
    top_distances = cp.take_along_axis(distances, neighbors, axis=1)
    return top_distances, neighbors


def run_python_metric_demo(cp, ivf_flat):
    _print_banner("Example 1: Python @ivf_flat.metric weighted L2 with a CuPy capture")

    weights = cp.asarray([0.25, 1.5, 3.0, 0.75], dtype=cp.float32)

    @ivf_flat.metric(
        order="min",
        initial=0.0,
        coarse_metric="sqeuclidean",
        captures={"weights": weights},
        symbol_name="cuvs_demo_weighted_l2_update_f32",
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
    reference_distances, reference_neighbors = _weighted_l2_reference(
        cp, queries, dataset, weights, 3
    )

    max_error = cp.max(cp.abs(cp.asarray(udf_distances) - reference_distances))
    distances_match = cp.allclose(
        cp.asarray(udf_distances), reference_distances, rtol=1e-5, atol=1e-5
    )
    neighbors_match = cp.array_equal(
        cp.asarray(udf_neighbors), reference_neighbors
    )
    passed = bool(distances_match and neighbors_match)

    print("weights:")
    print(_format_array(cp, weights))
    print("queries:")
    print(_format_array(cp, queries))
    print("dataset:")
    print(_format_array(cp, dataset))
    print("UDF neighbors:")
    print(_format_array(cp, udf_neighbors))
    print("reference neighbors:")
    print(_format_array(cp, reference_neighbors))
    print("UDF distances:")
    print(_format_array(cp, udf_distances))
    print("reference distances:")
    print(_format_array(cp, reference_distances))
    print(f"max abs distance error: {float(max_error):.8f}")
    print(f"neighbors match: {neighbors_match}")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")

    return passed


def run_cuda_source_metric_demo(cp, ivf_flat):
    _print_banner("Example 2: Expert CUDA/C++ source-string L2 metric")

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
        symbol_name="cuvs_demo_cuda_source_l2_metric",
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

    max_error = cp.max(cp.abs(cp.asarray(source_distances) - builtin_distances))
    distances_match = cp.allclose(
        cp.asarray(source_distances),
        cp.asarray(builtin_distances),
        rtol=1e-5,
        atol=1e-5,
    )
    neighbors_match = cp.array_equal(
        cp.asarray(source_neighbors), cp.asarray(builtin_neighbors)
    )
    passed = bool(distances_match and neighbors_match)

    print("queries:")
    print(_format_array(cp, queries))
    print("dataset:")
    print(_format_array(cp, dataset))
    print("CUDA source neighbors:")
    print(_format_array(cp, source_neighbors))
    print("built-in L2 neighbors:")
    print(_format_array(cp, builtin_neighbors))
    print("CUDA source distances:")
    print(_format_array(cp, source_distances))
    print("built-in L2 distances:")
    print(_format_array(cp, builtin_distances))
    print(f"max abs distance error: {float(max_error):.8f}")
    print(f"neighbors match: {neighbors_match}")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")

    return passed


def main():
    _require_module("cupy")
    _require_module("numba_cuda_mlir")

    import cupy as cp
    from cuvs.neighbors import ivf_flat
    from numba_cuda_mlir import cuda

    if not cuda.is_available():
        raise RuntimeError("CUDA is not available to numba_cuda_mlir")

    cp.set_printoptions(precision=4, suppress=True)

    with cp.cuda.Device(0):
        python_metric_ok = run_python_metric_demo(cp, ivf_flat)
        cuda_source_ok = run_cuda_source_metric_demo(cp, ivf_flat)

    return 0 if python_metric_ok and cuda_source_ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
