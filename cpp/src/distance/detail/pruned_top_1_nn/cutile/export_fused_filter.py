# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
"""Export the pruned 1-NN's fused first-filter cuTile kernel to cubin."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cuda.tile as ct
from cuda.tile.compilation import (
    ArrayConstraint,
    CallingConvention,
    ConstantConstraint,
    KernelSignature,
    ScalarConstraint,
    export_kernel,
)

# CI enables Python safe-path mode, so the script directory is not guaranteed
# to be importable even when this file is executed directly.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fused_filter_kernel import kernel_symbol, make_kernel  # noqa: E402


def _matrix():
    """float16, row-major, unit last stride, 16-byte aligned base and row pitch."""
    return ArrayConstraint(
        ct.float16,
        ndim=2,
        index_dtype=ct.int32,
        stride_lower_bound_incl=(0, None),
        alias_groups=("read_only_inputs",),
        may_alias_internally=False,
        stride_constant=(None, 1),
        stride_divisible_by=(8, 1),
        base_addr_divisible_by=16,
    )


def _vector(elem_dtype, alias_groups=()):
    return ArrayConstraint(
        elem_dtype,
        ndim=1,
        index_dtype=ct.int32,
        stride_lower_bound_incl=(None,),
        alias_groups=alias_groups,
        may_alias_internally=False,
        stride_constant=(1,),
        base_addr_divisible_by=16,
    )


def _segments():
    return ArrayConstraint(
        ct.int32,
        ndim=2,
        index_dtype=ct.int32,
        stride_lower_bound_incl=(0, None),
        alias_groups=(),
        may_alias_internally=False,
        stride_constant=(None, 1),
        base_addr_divisible_by=16,
    )


def _signature(data_type: str, tile_m: int, tile_n: int, tile_k: int):
    # Argument order of the C++ launch: each array as (pointer, shape..., stride...), then the
    # scalars; the tile sizes are compiled in.
    read_only = ("read_only_inputs",)
    return KernelSignature(
        parameters=[
            _matrix(),  # rows (n_rows, d)
            _matrix(),  # centroids (n_centroids, d)
            _vector(ct.float32, read_only),  # row_norms: ||row||^2 over d'
            _vector(ct.float32, read_only),  # centroid_norms: ||centroid||^2 over d'
            _vector(ct.float32, read_only),  # best: each row's current distance
            _vector(ct.int64),  # packed survivors
            _segments(),  # seg_count (n_rows, n_col_tiles)
            ScalarConstraint(ct.int32),  # n_rows
            ScalarConstraint(ct.int32),  # n_centroids
            ScalarConstraint(ct.float32),  # ratio
            ScalarConstraint(ct.int32),  # d'
            ConstantConstraint(tile_m),
            ConstantConstraint(tile_n),
            ConstantConstraint(tile_k),
        ],
        calling_convention=CallingConvention.cutile_python_v1(),
    ).with_symbol(kernel_symbol(data_type))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_file", type=Path)
    parser.add_argument("--format", choices=("cubin",), default="cubin")
    parser.add_argument("--data-type", choices=("half",), required=True)
    parser.add_argument("--metric", choices=("l2",), required=True)
    parser.add_argument("--index-type", choices=("int32",), required=True)
    parser.add_argument("--tile-m", type=int, required=True)
    parser.add_argument("--tile-n", type=int, required=True)
    parser.add_argument("--tile-k", type=int, required=True)
    parser.add_argument("--gpu-code", required=True)
    parser.add_argument("--occupancy", type=int)
    args = parser.parse_args()

    export_kernel(
        kernel=make_kernel(gpu_code=args.gpu_code, occupancy=args.occupancy),
        signatures=[
            _signature(args.data_type, args.tile_m, args.tile_n, args.tile_k)
        ],
        output_file=str(args.output_file),
        gpu_code=args.gpu_code,
        output_format=args.format,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
