# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the optional wide-FP32 refinement image."""

import importlib.util
import json
from pathlib import Path
import runpy

import pytest

pytest.importorskip("cuda.tile")
CPP = Path(__file__).resolve().parents[2]
KERNEL = CPP / "src/neighbors/detail/cutile_topk"
spec = importlib.util.spec_from_file_location(
    "pk16_sortnet_export_test", KERNEL / "export_fused_topk_pk16_stages.py"
)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


@pytest.mark.parametrize(
    "stage,enabled", [("refine", False), ("refine_sortnet", True)]
)
def test_optional_image_keeps_abi_and_requires_explicit_factory_flag(
    stage, enabled, monkeypatch, tmp_path
):
    seen = {}

    def factory(*args, **kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(exporter, "make_refine_kernel", factory)
    monkeypatch.setattr(
        exporter,
        "export_kernel",
        lambda **kw: seen.update(signature=kw["signatures"][0]),
    )
    exporter.export_binary(
        tmp_path / "unused.cubin",
        data_type="float",
        metric="runtime",
        index_type="int32",
        physical_k=16,
        tile_m=4,
        tile_n=128,
        tile_k=32,
        gpu_code="sm_120",
        matrix_layout="strict",
        occupancy=2,
        stage=stage,
    )
    assert seen.get("sort_network", False) is enabled
    baseline = exporter._kernel_signature(
        16, 4, 128, 32, "sm_120", "strict", "float", "int32", stage="refine"
    )
    assert seen["signature"].parameters == baseline.parameters
    assert seen["signature"].symbol == baseline.symbol + "_" + stage


@pytest.mark.parametrize(
    "changes",
    [
        {"data_type": "half"},
        {"matrix_layout": "relaxed"},
        {"gpu_code": "sm_100"},
    ],
)
def test_unqualified_optional_image_rejected(changes, tmp_path):
    args = dict(
        data_type="float",
        metric="runtime",
        index_type="int32",
        physical_k=16,
        tile_m=4,
        tile_n=128,
        tile_k=32,
        gpu_code="sm_120",
        matrix_layout="strict",
        occupancy=2,
        stage="refine_sortnet",
    )
    args.update(changes)
    with pytest.raises(ValueError, match="Sorting-network refinement"):
        exporter.export_binary(tmp_path / "unused.cubin", **args)


def test_one_optional_image_and_pre_enqueue_fallback():
    expand = runpy.run_path(
        str(CPP / "cmake/modules/compute_matrix_product.py")
    )["iterate_matrix_product"]
    rows = list(
        expand(
            matrix=json.loads(
                (
                    KERNEL / "fused_topk_pk16_refine_sortnet_matrix.json"
                ).read_text()
            )
        )
    )
    assert len(rows) == 1
    assert (
        rows[0]["data_type"],
        rows[0]["matrix_layout"],
        rows[0]["physical_k"],
    ) == ("float", "strict", 16)
    source = (KERNEL / "fused_topk_tile.cu").read_text()
    assert source.index("network_planner.try_get_launcher()") < source.index(
        "dispatch(refiner, refine_tile_m)"
    )
    assert "if (network != nullptr)" in source
    assert "refiner = std::move(network);" in source
    policy = (
        (KERNEL / "fused_topk_tile.hpp")
        .read_text()
        .split(
            "constexpr bool fused_topk_pk16_sortnet_refinement_eligible", 1
        )[1]
        .split("}", 1)[0]
    )
    assert "dimensions >= 512 && splits > 1" in policy
