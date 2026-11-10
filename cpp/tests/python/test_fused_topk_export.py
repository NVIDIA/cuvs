# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of the optional fused Top-K cuTile exporter."""

import importlib.util
import json
from pathlib import Path
import runpy

import pytest

ct = pytest.importorskip(
    "cuda.tile", reason="requires the optional cuTile exporter"
)
from cuda.tile.compilation import (  # noqa: E402
    ArrayConstraint,
    ConstantConstraint,
    ScalarConstraint,
)

CPP = Path(__file__).resolve().parents[2]
KERNEL_DIR = CPP / "src/neighbors/detail/cutile_topk"
spec = importlib.util.spec_from_file_location(
    "fused_topk_export_under_test", KERNEL_DIR / "export_fused_topk.py"
)
assert spec is not None and spec.loader is not None
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)
import fused_topk_kernel as kernel  # noqa: E402


@pytest.fixture(autouse=True)
def forbid_device_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU exporter tests must not compile or launch kernels")

    monkeypatch.setattr(exporter, "export_kernel", forbidden)
    monkeypatch.setattr(ct, "launch", forbidden)


@pytest.mark.parametrize("physical_k", [2, 4])
def test_export_matrix_has_unique_typed_layouts(physical_k):
    iterator = runpy.run_path(
        str(CPP / "cmake/modules/compute_matrix_product.py")
    )["iterate_matrix_product"]
    rows = list(
        iterator(
            matrix=json.loads(
                (
                    KERNEL_DIR
                    / f"fused_topk_pk{physical_k}_cutile_matrix.json"
                ).read_text()
            )
        )
    )
    keys = [(r["data_type"], r["gpu_code"], r["matrix_layout"]) for r in rows]
    assert len(keys) == len(set(keys))
    assert {r["data_type"] for r in rows} == {"float", "half"}
    for row in rows:
        assert row["physical_k"] == physical_k
        assert row["index_type"] == "int32"
        assert row["metric"] == "runtime"
        assert row["matrix_layout"] in ("strict", "relaxed")
        for axis in ("tile_m", "tile_n", "tile_k"):
            assert row[axis] > 0 and row[axis] & (row[axis] - 1) == 0
        # Validate the real signature for every declared build configuration.
        exporter._kernel_signature(
            physical_k,
            row["tile_m"],
            row["tile_n"],
            row["tile_k"],
            row["gpu_code"],
            row["matrix_layout"],
            row["data_type"],
        )
    for data_type, gpu_code in {(r["data_type"], r["gpu_code"]) for r in rows}:
        assert {
            r["matrix_layout"]
            for r in rows
            if r["data_type"] == data_type and r["gpu_code"] == gpu_code
        } == {"strict", "relaxed"}


@pytest.mark.parametrize("data_type", ["float", "half"])
@pytest.mark.parametrize("layout", ["strict", "relaxed"])
@pytest.mark.parametrize("arch", ["sm_80", "sm_86", "sm_120"])
@pytest.mark.parametrize("physical_k", [2, 4])
def test_typed_signature_constraints(data_type, layout, arch, physical_k):
    signature = exporter._kernel_signature(
        physical_k, 64, 128, 32, arch, layout, data_type
    )
    pitch = 8 if data_type == "half" else 4
    matrix = ArrayConstraint(
        ct.float16 if data_type == "half" else ct.float32,
        ndim=2,
        index_dtype=ct.int32,
        stride_lower_bound_incl=(0, None),
        alias_groups=("read_only_inputs",),
        may_alias_internally=False,
        stride_constant=(None, 1),
        stride_divisible_by=(pitch, 1) if layout == "strict" else (1, 1),
        shape_divisible_by=(1, pitch)
        if layout == "strict" and arch in ("sm_80", "sm_86")
        else (1, 1),
        base_addr_divisible_by=16,
    )

    def output(element):
        return ArrayConstraint(
            element,
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

    assert signature.parameters == (
        matrix,
        matrix,
        output(ct.int32),
        output(ct.float32),
        *[ScalarConstraint(ct.int32)] * 5,
        *[ConstantConstraint(v) for v in (64, 128, 32, physical_k)],
    )
    assert (
        signature.calling_convention
        == exporter.CallingConvention.cutile_python_v1()
    )
    assert signature.symbol == kernel.kernel_symbol(
        physical_k, layout, data_type
    )


@pytest.mark.parametrize("data_type", ["double", "int8", "bfloat16"])
def test_unsupported_input_type_is_rejected(data_type):
    with pytest.raises(ValueError):
        exporter._kernel_signature(
            2, 64, 128, 32, "sm_120", "strict", data_type
        )
    with pytest.raises(ValueError):
        kernel.make_kernel(data_type, "runtime", 2, 64, 128, 32)
    with pytest.raises(ValueError):
        kernel.kernel_symbol(2, "strict", data_type)


def test_producer_rejects_unsupported_descriptor_index_width():
    with pytest.raises(ValueError):
        kernel.make_kernel(
            "half", "runtime", 2, 64, 128, 32, index_type="int64"
        )


EXPORT_CASES = [
    (data_type, physical_k, stage)
    for data_type in ("float", "half")
    for physical_k in (2, 4)
    for stage in ("fused", "select", "refine")
]
EXPORT_CASES.append(("half", 4, "witness"))


@pytest.mark.parametrize("data_type,physical_k,stage", EXPORT_CASES)
def test_export_forwards_typed_signature_without_execution(
    monkeypatch, tmp_path, data_type, physical_k, stage
):
    captured = {}
    sentinel = object()

    def make_kernel(*args, **kwargs):
        captured["factory_args"] = args
        captured["factory_kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(exporter, "make_kernel", make_kernel)
    monkeypatch.setattr(
        exporter, "export_kernel", lambda **kw: captured.update(kw)
    )
    output = tmp_path / "not-created.cubin"
    symbol = exporter.export_binary(
        output,
        data_type=data_type,
        metric="runtime",
        index_type="int32",
        physical_k=physical_k,
        tile_m=64,
        tile_n=128,
        tile_k=32,
        gpu_code="sm_120",
        matrix_layout="strict",
        occupancy=2,
        stage=stage,
    )
    expected = exporter._kernel_signature(
        physical_k, 64, 128, 32, "sm_120", "strict", data_type, stage
    )
    assert captured["kernel"] is sentinel
    assert captured["signatures"] == [expected]
    assert captured["output_file"] == str(output)
    assert captured["gpu_code"] == "sm_120"
    assert captured["output_format"] == "cubin"
    assert captured["factory_args"] == (
        data_type,
        "runtime",
        physical_k,
        64,
        128,
        32,
    )
    assert captured["factory_kwargs"] == {
        "index_type": "int32",
        "gpu_code": "sm_120",
        "matrix_layout": "strict",
        "occupancy": 2,
        "stage": "select" if stage == "witness" else stage,
        "witness": stage == "witness",
    }
    assert symbol == expected.symbol
    assert not output.exists()
