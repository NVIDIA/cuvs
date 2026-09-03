# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
"""Export the standalone cuTile embedding smoke kernel."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cuda.tile as ct
from cuda.tile.compilation import (
    ArrayConstraint,
    CallingConvention,
    KernelSignature,
    export_kernel,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from smoke_kernel import TILE_SIZE, cutile_smoke_add


def _array_constraint() -> ArrayConstraint:
    return ArrayConstraint(
        ct.float32,
        ndim=1,
        index_dtype=ct.int32,
        stride_lower_bound_incl=(None,),
        alias_groups=(),
        may_alias_internally=False,
        stride_constant=(1,),
        stride_divisible_by=(1,),
        shape_divisible_by=(TILE_SIZE,),
        base_addr_divisible_by=16,
    )


def _signature() -> KernelSignature:
    array = _array_constraint()
    return KernelSignature(
        parameters=[array, array, array],
        calling_convention=CallingConvention.cutile_python_v1(),
    ).with_symbol("cutile_smoke_add")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_file", type=Path)
    parser.add_argument("--format", choices=("cubin",), required=True)
    parser.add_argument("--data-type", choices=("float",), required=True)
    parser.add_argument("--metric", choices=("add",), required=True)
    parser.add_argument("--index-type", choices=("int32",), required=True)
    parser.add_argument("--tile-m", type=int, required=True)
    parser.add_argument("--tile-n", type=int, required=True)
    parser.add_argument("--tile-k", type=int, required=True)
    parser.add_argument("--gpu-code", required=True)
    args = parser.parse_args()

    if (args.tile_m, args.tile_n, args.tile_k) != (TILE_SIZE, 1, 1):
        raise ValueError("cutile smoke kernel requires a 256x1x1 tile")

    export_kernel(
        kernel=cutile_smoke_add,
        signatures=[_signature()],
        output_file=str(args.output_file),
        gpu_code=args.gpu_code,
        output_format=args.format,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
