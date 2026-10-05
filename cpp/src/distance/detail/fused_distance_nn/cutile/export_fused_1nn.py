# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export fused 1-NN cuTile kernels to cubin."""

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

from fused_1nn_kernel import (  # noqa: E402
    INDEX_TYPES,
    METRICS,
    _idx_dtype,
    index_abbrev,
    kernel_symbol,
    make_kernel,
)


def _dtype_for(data_type: str):
    if data_type == "half":
        return ct.float16
    if data_type == "float":
        return ct.float32
    raise ValueError(f"Unsupported data_type {data_type!r}")


def _data_abbrev(data_type: str) -> str:
    return {"half": "h", "float": "f"}[data_type]


def _elem_stride_divisible_for_tma(elem_dtype) -> tuple[int, int]:
    """Row stride (dim 0) divisible enough for 16-byte TMA access; last dim stride 1."""
    bytes_per_elem = 2 if elem_dtype == ct.float16 else 4
    return (16 // bytes_per_elem, 1)


def _elem_shape_divisible_for_ldgsts(elem_dtype) -> tuple[int, int]:
    """Matrix extent aligned to the same 16-byte row pitch enforced on strides."""
    bytes_per_elem = 2 if elem_dtype == ct.float16 else 4
    return (1, 16 // bytes_per_elem)


def _cuvs_matrix_constraint(
    elem_dtype,
    *,
    index_dtype=ct.int32,
    require_tma_friendly_pitch: bool = True,
    require_ldgsts_friendly_shape: bool = False,
):
    """Row-major device matrices for cuVS KMeans benchmarks.

    Assumes raft/cupy-style contiguous layout: stride[-1]==1, stride[0]==D,
    16-byte base alignment, and row pitch 16-byte aligned (float32 D%4==0,
    float16 D%8==0). Applies to both points and centroids matrices.

    SM80/SM86 strict exports also express the row-pitch guarantee as
    shape_divisible_by=(1, 4) for float32 or (1, 8) for float16. This
    duplicates the stride constraint intentionally so the compiler selects
    LDGSTS instead of LDG. Tail tiles remain masked in the kernel.

    Odd D or general layouts need a separate relaxed export profile.
    """
    return ArrayConstraint(
        elem_dtype,
        ndim=2,
        index_dtype=index_dtype,
        stride_lower_bound_incl=(0, None),
        # Dataset and centroid views are read-only and may legally share storage.
        alias_groups=("read_only_inputs",),
        may_alias_internally=False,
        stride_constant=(None, 1),
        stride_divisible_by=(
            _elem_stride_divisible_for_tma(elem_dtype)
            if require_tma_friendly_pitch
            else (1, 1)
        ),
        shape_divisible_by=(
            _elem_shape_divisible_for_ldgsts(elem_dtype)
            if require_ldgsts_friendly_shape
            else (1, 1)
        ),
        base_addr_divisible_by=16,
    )


def _cuvs_vector_constraint(
    elem_dtype, *, index_dtype=ct.int32, alias_groups=()
):
    """1-D device vectors: contiguous, 16-byte base. Length need not be divisible by 16."""
    return ArrayConstraint(
        elem_dtype,
        ndim=1,
        index_dtype=index_dtype,
        stride_lower_bound_incl=(None,),
        alias_groups=alias_groups,
        may_alias_internally=False,
        stride_constant=(1,),
        stride_divisible_by=(1,),
        shape_divisible_by=(1,),
        base_addr_divisible_by=16,
    )


def _relaxed_matrix_constraint(elem_dtype):
    """Deprecated alias for the arbitrary-row-pitch matrix constraint."""
    return _cuvs_matrix_constraint(
        elem_dtype, require_tma_friendly_pitch=False
    )


def _relaxed_vector_constraint(elem_dtype, *, tma_friendly: bool = False):
    """Deprecated alias; use _cuvs_vector_constraint."""
    del tma_friendly
    return _cuvs_vector_constraint(elem_dtype)


def _kernel_signature(
    data_type: str,
    metric: str,
    index_type: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    gpu_code: str,
    matrix_layout: str,
) -> KernelSignature:
    elem = _dtype_for(data_type)
    idx_dtype = _idx_dtype(index_type)
    matrix = _cuvs_matrix_constraint(
        elem,
        index_dtype=idx_dtype,
        require_tma_friendly_pitch=matrix_layout == "strict",
        require_ldgsts_friendly_shape=(
            matrix_layout == "strict" and gpu_code in ("sm_80", "sm_86")
        ),
    )
    norm_elem = ct.float32 if data_type == "half" else elem
    norm_array = _cuvs_vector_constraint(
        norm_elem,
        index_dtype=idx_dtype,
        alias_groups=("read_only_inputs",),
    )
    idx_array = _cuvs_vector_constraint(idx_dtype, index_dtype=idx_dtype)
    dist_array = _cuvs_vector_constraint(ct.float32, index_dtype=idx_dtype)

    abbrev = _data_abbrev(data_type)
    symbol = kernel_symbol(
        abbrev,
        index_abbrev(index_type),
        matrix_layout,
    )

    return KernelSignature(
        parameters=[
            matrix,
            matrix,
            norm_array,
            norm_array,
            idx_array,
            dist_array,
            ScalarConstraint(idx_dtype),
            ScalarConstraint(idx_dtype),
            ScalarConstraint(idx_dtype),
            ScalarConstraint(idx_dtype),
            ScalarConstraint(idx_dtype),
            ScalarConstraint(ct.int32),
            ConstantConstraint(tile_m),
            ConstantConstraint(tile_n),
            ConstantConstraint(tile_k),
        ],
        calling_convention=CallingConvention.cutile_python_v1(),
    ).with_symbol(symbol)


def export_binary(
    output_file: Path,
    *,
    data_type: str,
    metric: str,
    index_type: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    gpu_code: str,
    matrix_layout: str = "strict",
    occupancy: int | None = None,
) -> str:
    kernel = make_kernel(
        data_type,
        metric,
        tile_m,
        tile_n,
        tile_k,
        index_type=index_type,
        gpu_code=gpu_code,
        matrix_layout=matrix_layout,
        occupancy=occupancy,
    )
    signature = _kernel_signature(
        data_type,
        metric,
        index_type,
        tile_m,
        tile_n,
        tile_k,
        gpu_code,
        matrix_layout,
    )

    export_kernel(
        kernel=kernel,
        signatures=[signature],
        output_file=str(output_file),
        gpu_code=gpu_code,
        output_format="cubin",
    )

    return signature.symbol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_file", type=Path)
    parser.add_argument("--format", choices=("cubin",), default="cubin")
    parser.add_argument(
        "--data-type", choices=("half", "float"), required=True
    )
    parser.add_argument("--metric", choices=METRICS, required=True)
    parser.add_argument("--index-type", choices=INDEX_TYPES, required=True)
    parser.add_argument("--tile-m", type=int, required=True)
    parser.add_argument("--tile-n", type=int, required=True)
    parser.add_argument("--tile-k", type=int, required=True)
    parser.add_argument(
        "--gpu-code", required=True, help="Target SM for cubin export"
    )
    parser.add_argument(
        "--matrix-layout",
        choices=("strict", "relaxed"),
        default="strict",
    )
    parser.add_argument("--occupancy", type=int)
    args = parser.parse_args()

    export_binary(
        args.output_file,
        data_type=args.data_type,
        metric=args.metric,
        index_type=args.index_type,
        tile_m=args.tile_m,
        tile_n=args.tile_n,
        tile_k=args.tile_k,
        gpu_code=args.gpu_code,
        matrix_layout=args.matrix_layout,
        occupancy=args.occupancy,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
