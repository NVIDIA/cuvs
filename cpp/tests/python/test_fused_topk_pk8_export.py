# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only coverage of every ordinary PK8 export and direct-prefix ABI."""

import importlib.util
import json
from pathlib import Path
import runpy

import pytest

ct = pytest.importorskip(
    "cuda.tile", reason="requires the optional cuTile exporter"
)
from cuda.tile.compilation import ConstantConstraint  # noqa: E402

CPP = Path(__file__).resolve().parents[2]
KERNEL = CPP / "src/neighbors/detail/cutile_topk"
spec = importlib.util.spec_from_file_location(
    "pk8_export_test", KERNEL / "export_fused_topk.py"
)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)
import fused_topk_kernel as kernel  # noqa: E402

expand = runpy.run_path(str(CPP / "cmake/modules/compute_matrix_product.py"))[
    "iterate_matrix_product"
]
ROWS = list(
    expand(
        matrix=json.loads(
            (KERNEL / "fused_topk_pk8_cutile_matrix.json").read_text()
        )
    )
)


@pytest.fixture(autouse=True)
def forbid_compilation_and_launch(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU export tests may not compile or launch a kernel")

    monkeypatch.setattr(ct, "launch", forbidden)
    monkeypatch.setattr(exporter, "export_kernel", forbidden)


def test_matrix_has_unique_complete_typed_symbols():
    keys = [
        (r["gpu_code"], r["matrix_layout"], r["data_type"], r["index_type"])
        for r in ROWS
    ]
    assert len(keys) == len(set(keys)) == 16
    for row in ROWS:
        assert row["physical_k"] == 8 and row["metric"] == "runtime"
        assert row["index_type"] in ("int32", "int64")
        if row["index_type"] == "int64" or row["data_type"] == "half":
            assert row["gpu_code"] == "sm_120"
    for arch, _, dtype, index in keys:
        assert {
            layout
            for a, layout, d, i in keys
            if (a, d, i) == (arch, dtype, index)
        } == {"strict", "relaxed"}


@pytest.mark.parametrize("row", ROWS, ids=lambda r: r["artifact_basename"])
def test_every_export_keeps_physical_eight_and_logical_output_width(
    row, monkeypatch, tmp_path
):
    dtype, index, arch, layout = (
        row[k]
        for k in ("data_type", "index_type", "gpu_code", "matrix_layout")
    )
    tm, tn, tk = (row[k] for k in ("tile_m", "tile_n", "tile_k"))
    sig = exporter._kernel_signature(8, tm, tn, tk, arch, layout, dtype, index)
    assert sig.parameters[2].dtype == (
        ct.int64 if index == "int64" else ct.int32
    )
    assert sig.parameters[3].dtype == ct.float32
    assert sig.parameters[-1] == ConstantConstraint(8)
    for tensor in sig.parameters[2:4]:
        assert tensor.shape_divisible_by == (1, 1)
        assert tensor.stride_divisible_by == (1, 1)
    # Do not promise that logical K5/K6/K7 output is divisible by physical K8.
    assert (
        sig.parameters[0].alias_groups
        == sig.parameters[1].alias_groups
        == ("read_only_inputs",)
    )
    actual_kernel = kernel.make_kernel(
        dtype,
        "runtime",
        8,
        tm,
        tn,
        tk,
        gpu_code=arch,
        matrix_layout=layout,
        index_type=index,
        occupancy=row.get("occupancy"),
    )
    captured = {}
    monkeypatch.setattr(
        exporter, "make_kernel", lambda *args, **kwargs: actual_kernel
    )
    monkeypatch.setattr(
        exporter, "export_kernel", lambda **kwargs: captured.update(kwargs)
    )
    output = tmp_path / "not-created.cubin"
    symbol = exporter.export_binary(
        output,
        data_type=dtype,
        metric="runtime",
        index_type=index,
        physical_k=8,
        tile_m=tm,
        tile_n=tn,
        tile_k=tk,
        gpu_code=arch,
        matrix_layout=layout,
        occupancy=row.get("occupancy"),
    )
    assert captured["kernel"] is actual_kernel and captured["signatures"] == [
        sig
    ]
    assert (
        captured["gpu_code"] == arch and captured["output_format"] == "cubin"
    )
    assert captured["output_file"] == str(output) and symbol == sig.symbol
    assert not output.exists()
