# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export fused physical-top-k cuTile kernels to cubin."""

from __future__ import annotations

import argparse
from dataclasses import replace
from functools import partial
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

# CI enables Python safe-path mode, so make the sibling kernel module explicit.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fused_topk_kernel import (  # noqa: E402
    INDEX_TYPES,
    METRICS,
    PHYSICAL_KS,
    _idx_dtype,
    kernel_symbol,
)


from fused_topk_pk16_stages import make_select_kernel, make_refine_kernel  # noqa: E402


def _matrix_constraint(
    *,
    gpu_code: str,
    require_tma_friendly_pitch: bool,
    data_type: str = "float",
):
    """Typed row-major input; descriptor indices remain int32 for both types."""
    if data_type not in ("float", "half"):
        raise ValueError(f"Unsupported data_type {data_type!r}")
    pitch = 8 if data_type == "half" else 4
    return ArrayConstraint(
        ct.float16 if data_type == "half" else ct.float32,
        ndim=2,
        index_dtype=ct.int32,
        stride_lower_bound_incl=(0, None),
        alias_groups=("read_only_inputs",),
        may_alias_internally=False,
        stride_constant=(None, 1),
        stride_divisible_by=(
            (pitch, 1) if require_tma_friendly_pitch else (1, 1)
        ),
        # SM80/SM86 need the redundant shape promise to select LDGSTS, just as Top1 does.
        shape_divisible_by=(
            (1, pitch)
            if require_tma_friendly_pitch and gpu_code in ("sm_80", "sm_86")
            else (1, 1)
        ),
        base_addr_divisible_by=16,
    )


def _output_matrix_constraint(elem_dtype):
    return ArrayConstraint(
        elem_dtype,
        ndim=2,
        index_dtype=ct.int32,
        stride_lower_bound_incl=(0, None),
        alias_groups=(),
        may_alias_internally=False,
        stride_constant=(None, 1),
        stride_divisible_by=(1, 1),
        shape_divisible_by=(1, 1),
        base_addr_divisible_by=16,
    )


def _kernel_signature(
    physical_k: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    gpu_code: str,
    matrix_layout: str,
    data_type: str = "float",
    index_type: str = "int32",
    *,
    stage: str = "select",
) -> KernelSignature:
    matrix = _matrix_constraint(
        gpu_code=gpu_code,
        require_tma_friendly_pitch=matrix_layout == "strict",
        data_type=data_type,
    )
    indices = _output_matrix_constraint(_idx_dtype(index_type))
    distances = _output_matrix_constraint(ct.float32)
    if (
        stage in ("refine", "refine_sortnet")
        and physical_k == 16
        and gpu_code == "sm_120"
        and matrix_layout == "strict"
        and index_type == "int32"
    ):
        # Native PK16 refinement uses physical-width scratch [Q, splits * 16]
        # before the logical-prefix merge. Keep arbitrary Q/N and input aliases.
        # The extra FP16 input-shape promise changes accumulation ordering on
        # near-tie graph rows. Keep its original input descriptor while still
        # describing the physical output width. FP32 retains the input promise.
        if data_type == "float":
            matrix = replace(matrix, shape_divisible_by=(1, 4))
        indices = replace(
            indices,
            shape_divisible_by=(1, physical_k),
            stride_divisible_by=(physical_k, 1),
        )
        distances = replace(
            distances,
            shape_divisible_by=(1, physical_k),
            stride_divisible_by=(physical_k, 1),
        )
    return KernelSignature(
        parameters=[
            matrix,
            matrix,
            indices,
            distances,
            ScalarConstraint(ct.int32),
            ScalarConstraint(ct.int32),
            ScalarConstraint(ct.int32),
            ScalarConstraint(ct.int32),
            ScalarConstraint(ct.int32),
            ConstantConstraint(tile_m),
            ConstantConstraint(tile_n),
            ConstantConstraint(tile_k),
            ConstantConstraint(physical_k),
        ],
        calling_convention=CallingConvention.cutile_python_v1(),
    ).with_symbol(
        kernel_symbol(physical_k, matrix_layout, data_type, index_type)
    )


def export_binary(
    output_file: Path,
    *,
    data_type: str,
    metric: str,
    index_type: str,
    physical_k: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    gpu_code: str,
    matrix_layout: str = "strict",
    occupancy: int | None = None,
    stage: str,
) -> str:
    if stage == "select_witness" and (
        data_type not in ("float", "half") or matrix_layout != "strict"
    ):
        raise ValueError(
            "Witness export is qualified only for strict float/half inputs"
        )
    if stage == "refine_sortnet" and (
        data_type != "float"
        or matrix_layout != "strict"
        or gpu_code != "sm_120"
    ):
        raise ValueError(
            "Sorting-network refinement requires strict SM120 FP32"
        )
    factory = {
        "select": make_select_kernel,
        "refine": make_refine_kernel,
        "refine_sortnet": partial(make_refine_kernel, sort_network=True),
        "select_witness": partial(
            make_select_kernel, first_candidate_witness=True
        ),
    }[stage]
    kernel = factory(
        data_type,
        metric,
        physical_k,
        tile_m,
        tile_n,
        tile_k,
        index_type=index_type,
        gpu_code=gpu_code,
        matrix_layout=matrix_layout,
        occupancy=occupancy,
    )
    signature = _kernel_signature(
        physical_k,
        tile_m,
        tile_n,
        tile_k,
        gpu_code,
        matrix_layout,
        data_type,
        index_type,
        stage=stage,
    )
    signature = signature.with_symbol(signature.symbol + "_" + stage)
    export_kernel(
        kernel=kernel,
        signatures=[signature],
        output_file=str(output_file),
        gpu_code=gpu_code,
        output_format="cubin",
    )
    return signature.symbol


def main(stage) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_file", type=Path)
    parser.add_argument("--format", choices=("cubin",), default="cubin")
    parser.add_argument(
        "--data-type", choices=("float", "half"), required=True
    )
    parser.add_argument("--metric", choices=METRICS, required=True)
    parser.add_argument("--index-type", choices=INDEX_TYPES, required=True)
    parser.add_argument(
        "--physical-k", type=int, choices=PHYSICAL_KS, required=True
    )
    parser.add_argument("--tile-m", type=int, required=True)
    parser.add_argument("--tile-n", type=int, required=True)
    parser.add_argument("--tile-k", type=int, required=True)
    parser.add_argument("--gpu-code", required=True)
    parser.add_argument(
        "--matrix-layout", choices=("strict", "relaxed"), default="strict"
    )
    parser.add_argument("--occupancy", type=int)
    args = parser.parse_args()

    export_binary(
        args.output_file,
        data_type=args.data_type,
        metric=args.metric,
        index_type=args.index_type,
        physical_k=args.physical_k,
        tile_m=args.tile_m,
        tile_n=args.tile_n,
        tile_k=args.tile_k,
        gpu_code=args.gpu_code,
        matrix_layout=args.matrix_layout,
        occupancy=args.occupancy,
        stage=stage,
    )
    return 0
