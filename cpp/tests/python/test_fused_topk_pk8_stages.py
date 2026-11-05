# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only PK8 split-stage ABI and retained-width tests."""

import importlib.util
import json
from pathlib import Path
import runpy

import pytest

ct = pytest.importorskip("cuda.tile")
CPP = Path(__file__).resolve().parents[2]
KERNEL = CPP / "src/neighbors/detail/cutile_topk"
spec = importlib.util.spec_from_file_location(
    "pk8_stage_export", KERNEL / "export_fused_topk.py"
)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)
import fused_topk_kernel as kernel  # noqa: E402


@pytest.fixture(autouse=True)
def no_device_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU stage tests may not compile or launch a kernel")

    monkeypatch.setattr(ct, "launch", forbidden)
    monkeypatch.setattr(exporter, "export_kernel", forbidden)


@pytest.mark.parametrize("dtype", ["float", "half"])
@pytest.mark.parametrize("stage,tm", [("select", 32), ("refine", 4)])
def test_factory_and_signature_keep_physical_eight(dtype, stage, tm):
    assert (
        kernel.make_kernel(dtype, "runtime", 8, tm, 128, 32, stage=stage)
        is not None
    )
    signature = exporter._kernel_signature(
        8, tm, 128, 32, "sm_120", "strict", dtype, stage=stage
    )
    assert (
        signature.symbol
        == f"fused_topk_pk8_{'f' if dtype == 'float' else 'h'}_i32_{stage}"
    )


@pytest.mark.parametrize("dtype", ["float", "half"])
@pytest.mark.parametrize(
    "override",
    [
        {"matrix_layout": "relaxed"},
        {"gpu_code": "sm_100"},
        {"index_type": "int64"},
        {"witness": True},
    ],
)
def test_unqualified_stage_variants_rejected(dtype, override):
    with pytest.raises(ValueError):
        kernel.make_kernel(
            dtype, "runtime", 8, 32, 128, 32, stage="select", **override
        )


@pytest.mark.parametrize("dtype", ["float", "half"])
def test_longscan_guard_supported_only_in_selection_stage(dtype):
    assert kernel.make_kernel(
        dtype, "runtime", 8, 32, 256, 32,
        stage="select", whole_tile_guard=True,
    ) is not None
    with pytest.raises(ValueError):
        kernel.make_kernel(
            dtype, "runtime", 8, 4, 128, 32,
            stage="refine", whole_tile_guard=True,
        )


@pytest.mark.parametrize("stage,tm", [("select", 32), ("refine", 4)])
def test_exported_stage_geometries(stage, tm):
    expand = runpy.run_path(
        str(CPP / "cmake/modules/compute_matrix_product.py")
    )["iterate_matrix_product"]
    matrix = json.loads(
        (KERNEL / f"fused_topk_{stage}_cutile_matrix.json").read_text()
    )
    rows = [r for r in expand(matrix=matrix) if r["physical_k"] == 8]
    assert len(rows) == 2 and {r["data_type"] for r in rows} == {
        "float",
        "half",
    }
    for row in rows:
        assert (row["tile_m"], row["tile_n"], row["tile_k"]) == (tm, 128, 32)
        assert row["index_type"] == "int32"
        assert row["matrix_layout"] == "strict" and row["gpu_code"] == "sm_120"
