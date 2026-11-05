# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU guards for explicit L2-only selection exports and metric isolation."""

import importlib.util
import json
from pathlib import Path
import runpy

import pytest

pytest.importorskip("cuda.tile")
from cuda.tile.compilation import ScalarConstraint

CPP = Path(__file__).resolve().parents[2]
KERNEL = CPP / "src/neighbors/detail/cutile_topk"
spec = importlib.util.spec_from_file_location(
    "l2_export_test", KERNEL / "export_fused_topk.py"
)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)
import fused_topk_kernel as kernel  # noqa: E402


@pytest.mark.parametrize("dtype", ["float", "half"])
def test_l2_export_preserves_scalar_abi_and_has_distinct_symbol(dtype):
    args = (8, 32, 128, 32, "sm_120", "strict", dtype, "int32")
    generic = exporter._kernel_signature(*args, stage="select")
    l2 = exporter._kernel_signature(*args, stage="select_l2")
    assert generic.parameters == l2.parameters
    assert generic.calling_convention == l2.calling_convention
    assert isinstance(l2.parameters[7], ScalarConstraint)
    assert l2.symbol == generic.symbol + "_l2"


@pytest.mark.parametrize(
    "stage,specialized", [("select", False), ("select_l2", True)]
)
@pytest.mark.parametrize("dtype", ["float", "half"])
def test_exporter_sets_specialization_only_explicitly(
    stage, specialized, dtype, monkeypatch, tmp_path
):
    seen = {}

    def fake_factory(*args, **kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(exporter, "make_kernel", fake_factory)
    monkeypatch.setattr(exporter, "export_kernel", lambda **kwargs: None)
    exporter.export_binary(
        tmp_path / "unused.cubin",
        data_type=dtype,
        metric="runtime",
        index_type="int32",
        physical_k=8,
        tile_m=32,
        tile_n=128,
        tile_k=32,
        gpu_code="sm_120",
        stage=stage,
    )
    assert (
        seen["stage"] == "select"
        and seen.get("specialize_l2", False) is specialized
    )


@pytest.mark.parametrize(
    "changes",
    [
        dict(physical_k=4),
        dict(stage="refine"),
        dict(matrix_layout="relaxed"),
        dict(index_type="int64"),
        dict(gpu_code="sm_100"),
    ],
)
def test_factory_rejects_unqualified_specializations(changes):
    args = dict(
        data_type="float",
        metric="runtime",
        physical_k=8,
        tile_m=32,
        tile_n=128,
        tile_k=32,
        stage="select",
        specialize_l2=True,
    )
    args.update(changes)
    with pytest.raises(ValueError, match="L2 specialization"):
        kernel.make_kernel(**args)


def test_two_images_only_and_native_metric_guard_with_fallback():
    expand = runpy.run_path(
        str(CPP / "cmake/modules/compute_matrix_product.py")
    )["iterate_matrix_product"]
    rows = list(
        expand(
            matrix=json.loads(
                (
                    KERNEL / "fused_topk_select_l2_cutile_matrix.json"
                ).read_text()
            )
        )
    )
    assert len(rows) == 2 and {r["data_type"] for r in rows} == {
        "float",
        "half",
    }
    assert all(
        r["physical_k"] == 8 and r["matrix_layout"] == "strict" for r in rows
    )
    source = (KERNEL / "fused_topk_tile.cu").read_text()
    block = source.split("} else if constexpr (PhysicalK == 8) {", 1)[1].split(
        "} else {", 1
    )[0]
    assert (
        "DistanceType::L2Expanded" in block
        and "DistanceType::L2SqrtExpanded" in block
    )
    assert (
        "DistanceType::InnerProduct" not in block
        and "DistanceType::CosineExpanded" not in block
    )
    assert block.index("l2_select.try_get_launcher()") < block.index(
        "if (!selection)"
    )
    assert "selection = select.try_get_launcher();" in block
