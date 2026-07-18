# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Export fused 1-NN cuTile kernels to cubin or TileIR bytecode."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import cuda.tile as ct
from cuda.tile.compilation import (
    ArrayConstraint,
    CallingConvention,
    ConstantConstraint,
    KernelSignature,
    ScalarConstraint,
    export_kernel,
)

from fused_1nn_kernel import (
    INDEX_TYPES,
    METRICS,
    _idx_dtype,
    index_abbrev,
    kernel_symbol,
    make_kernel,
    metric_abbrev,
)

DEFAULT_TILEIR_BYTECODE_VERSION = "13.1"
# cuTile requires a gpu_code even for TileIR bytecode export: it selects the compilation
# target / feature set for lowering, not the runtime architecture (the driver JITs at load).
DEFAULT_TILEIR_EXPORT_GPU_CODE = "sm_80"


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


def _cuvs_matrix_constraint(
    elem_dtype,
    *,
    index_dtype=ct.int32,
    require_tma_friendly_pitch: bool = True,
):
    """Row-major device matrices for cuVS KMeans benchmarks.

      Assumes raft/cupy-style contiguous layout: stride[-1]==1, stride[0]==D,
      16-byte base alignment, and row pitch 16-byte aligned (float32 D%4==0,
      float16 D%8==0). Applies to both points and centroids matrices.

    shape_divisible_by is (1, 1); tail tiles are masked in the kernel.
      Odd D or general layouts need a separate relaxed export profile.
    """
    return ArrayConstraint(
        elem_dtype,
        ndim=2,
        index_dtype=index_dtype,
        stride_lower_bound_incl=(0, None),
        alias_groups=(),
        may_alias_internally=False,
        stride_constant=(None, 1),
        stride_divisible_by=(
            _elem_stride_divisible_for_tma(elem_dtype)
            if require_tma_friendly_pitch
            else (1, 1)
        ),
        shape_divisible_by=(1, 1),
        base_addr_divisible_by=16,
    )


def _cuvs_vector_constraint(elem_dtype, *, index_dtype=ct.int32):
    """1-D device vectors: contiguous, 16-byte base. Length need not be divisible by 16."""
    return ArrayConstraint(
        elem_dtype,
        ndim=1,
        index_dtype=index_dtype,
        stride_lower_bound_incl=(None,),
        alias_groups=(),
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
    matrix_layout: str,
) -> KernelSignature:
    elem = _dtype_for(data_type)
    idx_dtype = _idx_dtype(index_type)
    matrix = _cuvs_matrix_constraint(
        elem,
        index_dtype=idx_dtype,
        require_tma_friendly_pitch=matrix_layout == "strict",
    )
    norm_array = _cuvs_vector_constraint(elem, index_dtype=idx_dtype)
    idx_array = _cuvs_vector_constraint(idx_dtype, index_dtype=idx_dtype)
    dist_array = _cuvs_vector_constraint(elem, index_dtype=idx_dtype)

    abbrev = _data_abbrev(data_type)
    symbol = kernel_symbol(
        abbrev,
        metric_abbrev(metric),
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
            ConstantConstraint(tile_m),
            ConstantConstraint(tile_n),
            ConstantConstraint(tile_k),
        ],
        calling_convention=CallingConvention.cutile_python_v1(),
    ).with_symbol(symbol)


def export_binary(
    output_file: Path,
    *,
    output_format: Literal["cubin", "tileir_bytecode"],
    data_type: str,
    metric: str,
    index_type: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    gpu_code: str,
    matrix_layout: str = "strict",
    bytecode_version: str | None = None,
) -> str:
    kernel = make_kernel(
        data_type,
        metric,
        tile_m,
        tile_n,
        tile_k,
        index_type=index_type,
        gpu_code=gpu_code,
    )
    signature = _kernel_signature(
        data_type,
        metric,
        index_type,
        tile_m,
        tile_n,
        tile_k,
        matrix_layout,
    )

    export_kwargs = {
        "kernel": kernel,
        "signatures": [signature],
        "output_file": str(output_file),
        "gpu_code": gpu_code,
        "output_format": output_format,
    }
    if output_format == "tileir_bytecode":
        export_kwargs["bytecode_version"] = (
            bytecode_version or DEFAULT_TILEIR_BYTECODE_VERSION
        )

    export_kernel(**export_kwargs)

    return signature.symbol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_file", type=Path)
    parser.add_argument(
        "--format", choices=("cubin", "tileir_bytecode"), default="cubin"
    )
    parser.add_argument(
        "--data-type", choices=("half", "float"), required=True
    )
    parser.add_argument("--metric", choices=METRICS, required=True)
    parser.add_argument("--index-type", choices=INDEX_TYPES, required=True)
    parser.add_argument("--tile-m", type=int, required=True)
    parser.add_argument("--tile-n", type=int, required=True)
    parser.add_argument("--tile-k", type=int, required=True)
    parser.add_argument(
        "--gpu-code",
        default=DEFAULT_TILEIR_EXPORT_GPU_CODE,
        help="Target SM for cubin export, or compile hint for TileIR bytecode export",
    )
    parser.add_argument(
        "--matrix-layout",
        choices=("strict", "relaxed"),
        default="strict",
    )
    parser.add_argument(
        "--bytecode-version", default=DEFAULT_TILEIR_BYTECODE_VERSION
    )
    args = parser.parse_args()

    print(
        export_binary(
            args.output_file,
            output_format=args.format,
            data_type=args.data_type,
            metric=args.metric,
            index_type=args.index_type,
            tile_m=args.tile_m,
            tile_n=args.tile_n,
            tile_k=args.tile_k,
            gpu_code=args.gpu_code,
            matrix_layout=args.matrix_layout,
            bytecode_version=args.bytecode_version,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
