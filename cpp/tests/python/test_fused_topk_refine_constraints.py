# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only guards for promises made by the native separate-refinement ABI."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("cuda.tile")
KERNEL = (
    Path(__file__).resolve().parents[2] / "src/neighbors/detail/cutile_topk"
)


def exporter(pk):
    name = "export_fused_topk" if pk == 8 else "export_fused_topk_pk16_stages"
    spec = importlib.util.spec_from_file_location(
        name, KERNEL / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("pk", [8, 16])
@pytest.mark.parametrize("dtype,pitch", [("float", 4), ("half", 8)])
def test_native_refinement_promises_physical_width_not_rows(pk, dtype, pitch):
    sig = exporter(pk)._kernel_signature(
        pk, 4, 128, 32, "sm_120", "strict", dtype, stage="refine"
    )
    for tensor in sig.parameters[:2]:
        assert tensor.shape_divisible_by == (1, pitch)
        assert tensor.stride_divisible_by == (pitch, 1)
        assert tensor.alias_groups == ("read_only_inputs",)
    for tensor in sig.parameters[2:4]:
        assert tensor.shape_divisible_by == (1, pk)
        assert tensor.stride_divisible_by == (pk, 1)
    for q in (1, 5, 33):
        for splits in (1, 3, 64):
            assert (splits * pk) % pk == 0
            assert q % sig.parameters[2].shape_divisible_by[0] == 0


@pytest.mark.parametrize("pk", [8, 16])
@pytest.mark.parametrize("dtype", ["float", "half"])
@pytest.mark.parametrize(
    "stage,arch,layout,index",
    [
        ("select", "sm_120", "strict", "int32"),
        ("refine", "sm_120", "relaxed", "int32"),
        ("refine", "sm_100", "strict", "int32"),
        ("refine", "sm_120", "strict", "int64"),
    ],
)
def test_other_signatures_keep_generic_output_width(
    pk, dtype, stage, arch, layout, index
):
    sig = exporter(pk)._kernel_signature(
        pk, 4, 128, 32, arch, layout, dtype, index, stage=stage
    )
    for tensor in sig.parameters[2:4]:
        assert tensor.shape_divisible_by == (1, 1)
        assert tensor.stride_divisible_by == (1, 1)


@pytest.mark.parametrize("pk", [2, 4, 8])
@pytest.mark.parametrize("dtype", ["float", "half"])
def test_fused_logical_prefix_keeps_generic_width(pk, dtype):
    sig = exporter(8)._kernel_signature(
        pk, 32, 128, 32, "sm_120", "strict", dtype, "int64"
    )
    for tensor in sig.parameters[2:4]:
        assert tensor.shape_divisible_by == (1, 1)
        assert tensor.stride_divisible_by == (1, 1)
