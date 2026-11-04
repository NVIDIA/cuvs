# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only PK2/PK4/PK8/PK16 typed export contract; no device execution."""

import ast
import importlib.util
import json
import re
from pathlib import Path
import sys

import pytest

ct = pytest.importorskip("cuda.tile")
from cuda.tile.compilation import (  # noqa: E402
    ArrayConstraint,
    ScalarConstraint,
    ConstantConstraint,
)

CPP = Path(__file__).resolve().parents[2]
KERNEL = CPP / "src/neighbors/detail/cutile_topk"
sys.path.insert(0, str(KERNEL))
import export_fused_topk as exporter  # noqa: E402
import fused_topk_kernel as kernel  # noqa: E402


@pytest.fixture(autouse=True)
def forbid_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail(
            "CPU contract test attempted compiler/export/device execution"
        )

    monkeypatch.setattr(exporter, "export_kernel", forbidden)
    monkeypatch.setattr(ct, "launch", forbidden)


def matrix_rows(matrix):
    spec = importlib.util.spec_from_file_location(
        "matrix_product", CPP / "cmake/modules/compute_matrix_product.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.iterate_matrix_product(matrix=matrix))


def test_witness_import_dependencies_cover_every_exported_dtype_and_abi():
    matrix = json.loads(
        (KERNEL / "fused_topk_pk16_witness_matrix.json").read_text()
    )
    expected = {
        (row["data_type"], row["matrix_layout"]) for row in matrix_rows(matrix)
    }
    branch = re.search(
        r'if\(stage STREQUAL "select_witness"\)(.*?)else\(\)',
        (CPP / "CMakeLists.txt").read_text(),
        flags=re.S,
    )
    assert branch is not None
    dtypes = re.search(r"set\(stage_dtypes\s+([^)]*)\)", branch.group(1))
    abis = re.search(r"set\(stage_abis\s+([^)]*)\)", branch.group(1))
    assert dtypes is not None and abis is not None
    actual = {
        (dtype, abi)
        for dtype in dtypes.group(1).split()
        for abi in abis.group(1).split()
    }
    assert actual == expected


@pytest.mark.parametrize("pk", [2, 4])
def test_ten_fp32_architecture_leaves_and_two_sm120_half_leaves(pk):
    path = KERNEL / f"fused_topk_pk{pk}_cutile_matrix.json"
    current = matrix_rows(json.loads(path.read_text()))
    floats = [row for row in current if row["data_type"] == "float"]
    assert len(floats) == 10 and len(current) == 12
    # Portable contract: do not depend on unpublished research commit IDs.
    assert [
        tuple(
            row.get(key)
            for key in (
                "gpu_code",
                "matrix_layout",
                "tile_m",
                "tile_n",
                "tile_k",
                "occupancy",
            )
        )
        for row in floats
    ] == [
        ("sm_80", "relaxed", 64, 128, 32, None),
        ("sm_86", "relaxed", 64, 128, 32, None),
        ("sm_80", "strict", 64, 128, 32, 2),
        ("sm_86", "strict", 64, 128, 32, 2),
        ("sm_90", "relaxed", 128, 128, 64, None),
        ("sm_90", "strict", 64, 256, 32, None),
        ("sm_100", "relaxed", 128, 256, 16, None),
        ("sm_100", "strict", 128, 128, 32, None),
        ("sm_120", "relaxed", 64, 128, 32, 2),
        ("sm_120", "strict", 64, 128, 32, 2),
    ]
    assert all(
        row["physical_k"] == pk
        and row["index_type"] == "int32"
        and row["metric"] == "runtime"
        for row in current
    )
    half = [row for row in current if row["data_type"] == "half"]
    assert {r["matrix_layout"] for r in half} == {"strict", "relaxed"}
    for row in half:
        assert (
            row["gpu_code"],
            row["physical_k"],
            row["index_type"],
            row["tile_m"],
            row["tile_n"],
            row["tile_k"],
            row["occupancy"],
        ) == ("sm_120", pk, "int32", 64, 128, 32, 2)
    assert (
        len(
            {
                (r["data_type"], r["gpu_code"], r["matrix_layout"])
                for r in current
            }
        )
        == 12
    )


@pytest.mark.parametrize("dtype", ["float", "half"])
@pytest.mark.parametrize("layout", ["strict", "relaxed"])
@pytest.mark.parametrize("arch", ["sm_80", "sm_120"])
@pytest.mark.parametrize("pk", [2, 4, 8, 16])
def test_actual_signature_full_constraints(dtype, layout, arch, pk):
    sig = exporter._kernel_signature(pk, 64, 128, 32, arch, layout, dtype)
    pitch = 8 if dtype == "half" else 4
    matrix = ArrayConstraint(
        ct.float16 if dtype == "half" else ct.float32,
        ndim=2,
        index_dtype=ct.int32,
        stride_lower_bound_incl=(0, None),
        alias_groups=("read_only_inputs",),
        may_alias_internally=False,
        stride_constant=(None, 1),
        stride_divisible_by=(pitch, 1) if layout == "strict" else (1, 1),
        shape_divisible_by=(1, pitch)
        if layout == "strict" and arch == "sm_80"
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

    assert sig.parameters == (
        matrix,
        matrix,
        output(ct.int32),
        output(ct.float32),
        *[ScalarConstraint(ct.int32)] * 5,
        *[ConstantConstraint(v) for v in (64, 128, 32, pk)],
    )
    assert (
        sig.calling_convention == exporter.CallingConvention.cutile_python_v1()
    )
    assert (
        sig.symbol
        == f"fused_topk_pk{pk}_{'h' if dtype == 'half' else 'f'}_i32"
        + ("_relaxed" if layout == "relaxed" else "")
    )
    # Four 2D int32 descriptors plus five runtime int32 scalars: actual ELF checked after build.
    assert (
        sum(
            5
            if isinstance(p, ArrayConstraint)
            else 1
            if isinstance(p, ScalarConstraint)
            else 0
            for p in sig.parameters
        )
        == 25
    )


@pytest.mark.parametrize("bad", ["double", "int8", "bfloat16"])
def test_reject_unsupported_input_type(bad):
    with pytest.raises(ValueError):
        exporter._kernel_signature(2, 64, 128, 32, "sm_120", "strict", bad)
    with pytest.raises(ValueError):
        kernel.make_kernel(bad, "runtime", 2, 64, 128, 32)
    with pytest.raises(ValueError):
        kernel.kernel_symbol(2, "strict", bad)


def test_producer_descriptor_width_is_not_public_index_width():
    with pytest.raises(ValueError):
        kernel.make_kernel(
            "half", "runtime", 2, 64, 128, 32, index_type="int64"
        )
    source = (KERNEL / "fused_topk_tile.cu").read_text()
    assert (
        "checked_tensor_bytes(query_rows, dimensions, sizeof(DataT))" in source
    )
    assert (
        "checked_tensor_bytes(corpus_rows, dimensions, sizeof(DataT))"
        in source
    )
    assert (
        "int shape_queries[2]" in source and "int stride_queries[2]" in source
    )


def test_fp32_widening_before_norm_and_refinement():
    source = (KERNEL / "fused_topk_kernel.py").read_text()
    assert (
        'mma_dtype = ct.float16 if data_type == "half" else ct.tfloat32'
        in source
    )
    assert "corpus_f32 = corpus_t.astype(ct.float32)" in source
    assert "corpus_f32 * corpus_f32" in source
    assignments = {
        target.id: node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and target.id in ("selected_values", "corpus_f32")
    }
    for value in assignments.values():
        assert isinstance(value, ast.Call)
        assert value.func.attr == "astype"
        assert ast.unparse(value.args[0]) == "ct.float32"
    assert set(assignments) == {"selected_values", "corpus_f32"}
    assert "delta = query_tile[:, None, :] - selected_values" in source
    assert "accumulator = ct.zeros((tm, tn), ct.float32)" in source


@pytest.mark.parametrize("dtype", ["float", "half"])
def test_export_passes_actual_typed_signature_without_execution(
    monkeypatch, tmp_path, dtype
):
    captured = {}
    sentinel = object()
    monkeypatch.setattr(exporter, "make_kernel", lambda *a, **kw: sentinel)
    monkeypatch.setattr(
        exporter, "export_kernel", lambda **kw: captured.update(kw)
    )
    result = exporter.export_binary(
        tmp_path / "never-created.cubin",
        data_type=dtype,
        metric="runtime",
        index_type="int32",
        physical_k=2,
        tile_m=64,
        tile_n=128,
        tile_k=32,
        gpu_code="sm_120",
        matrix_layout="strict",
        occupancy=2,
    )
    assert captured["kernel"] is sentinel
    assert captured["signatures"] == [
        exporter._kernel_signature(2, 64, 128, 32, "sm_120", "strict", dtype)
    ]
    assert result == captured["signatures"][0].symbol
    assert not (tmp_path / "never-created.cubin").exists()


def test_family_predicates_and_half_cache_are_explicit():
    public = (CPP / "src/neighbors/detail/knn_brute_force.cuh").read_text()
    for pk in (2, 4):
        assert f"is_fused_topk_pk{pk}_candidate<value_t>" in public
    planner = (KERNEL / "fused_topk_pk4_planner.hpp").read_text()
    assert "template <typename DataT, typename AbiTag>" in planner
    assert "if constexpr (std::is_same_v<DataT, half>)" in planner
    assert "fused_topk_pk4_h_i32" in planner


def test_exact_stored_half_surplus_fixture():
    import numpy as np

    values = np.array([4, 3, 2, 1], dtype=np.float16) / np.float16(16)
    square = values.astype(np.float32) ** 2
    approximate = np.float32(1024**2) + square
    assert np.all(approximate == np.float32(1024**2))
    assert np.argsort(square).tolist() == [3, 2, 1, 0]
    assert np.array_equal(
        square[[3, 2, 1]], np.array([1, 4, 9], np.float32) / 256
    )
    y = np.float32(np.float16(3 / 8192))
    assert np.float32(1) + y * y > np.float32(1)
    assert np.sqrt(np.float32(1) + y * y, dtype=np.float32) == np.float32(1)


def test_pk8_typed_matrix_and_planner():
    rows = matrix_rows(
        json.loads((KERNEL / "fused_topk_pk8_cutile_matrix.json").read_text())
    )
    assert len(rows) == 16
    assert sum(r["index_type"] == "int32" for r in rows) == 12
    assert sum(r["index_type"] == "int64" for r in rows) == 4
    assert all(r["physical_k"] == 8 for r in rows)
    assert all(
        r["gpu_code"] == "sm_120" for r in rows if r["index_type"] == "int64"
    )
    for dtype in ("float", "half"):
        sm120 = [
            r
            for r in rows
            if r["gpu_code"] == "sm_120" and r["data_type"] == dtype
        ]
        assert {r["matrix_layout"] for r in sm120} == {"strict", "relaxed"}
        assert all(
            (r["tile_m"], r["tile_n"], r["tile_k"], r["occupancy"])
            == (32, 128, 32, 2)
            for r in sm120
        )
    assert all(
        r["gpu_code"] == "sm_120" for r in rows if r["data_type"] == "half"
    )
    planner = (KERNEL / "fused_topk_pk8_planner.hpp").read_text()
    assert (
        "template <typename DataT, typename AbiTag, typename OutputIdxT = int>"
        in planner
    )
    assert "if constexpr (std::is_same_v<DataT, half>)" in planner
    public = (CPP / "src/neighbors/detail/knn_brute_force.cuh").read_text()
    assert "is_fused_topk_pk8_candidate<value_t>" in public


def test_pk8_exact_stored_half_surplus_fixture():
    import numpy as np

    values = np.arange(8, 0, -1, dtype=np.float16) / np.float16(32)
    squares = values.astype(np.float32) ** 2
    assert np.all(np.float32(1024**2) + squares == np.float32(1024**2))
    assert np.argsort(squares).tolist() == list(range(7, -1, -1))


@pytest.mark.parametrize("dtype", ["float", "half"])
@pytest.mark.parametrize("layout", ["strict", "relaxed"])
def test_pk8_i64_output_only_signature(dtype, layout):
    a = exporter._kernel_signature(
        8, 32, 128, 32, "sm_120", layout, dtype, "int32"
    )
    b = exporter._kernel_signature(
        8, 32, 128, 32, "sm_120", layout, dtype, "int64"
    )
    assert b.parameters[:2] == a.parameters[:2]
    assert b.parameters[3:] == a.parameters[3:]
    assert b.parameters[2] == exporter._output_matrix_constraint(ct.int64)
    assert b.symbol == a.symbol.replace("i32", "i64")
    source = (KERNEL / "fused_topk_kernel.py").read_text()
    assert 'idx_dtype = _idx_dtype("int32")' in source
    # ct.store widens to the descriptor dtype, after full physical refinement.
    stage = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "stage == 'select'"
    )
    final = ast.Module(body=stage.orelse, type_ignores=[])
    sort = next(
        node
        for node in ast.walk(final)
        if isinstance(node, ast.For) and ast.unparse(node.target) == "column"
    )
    stores = [
        node
        for node in ast.walk(final)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "ct.store"
    ]
    assert len(stores) == 2 and all(
        node.lineno > sort.end_lineno for node in stores
    )
    with pytest.raises(ValueError, match="strict SM120 int32"):
        kernel.make_kernel(
            dtype,
            "runtime",
            8,
            32,
            128,
            32,
            stage="select",
            matrix_layout="relaxed",
        )
    assert "sorted_indices.astype(ct.int64)" not in source


def test_pk16_matrix_retains_int32_producer_and_typed_sm120_baseline():
    rows = matrix_rows(
        json.loads((KERNEL / "fused_topk_pk16_cutile_matrix.json").read_text())
    )
    assert len(rows) == 12
    assert all(
        r["physical_k"] == 16 and r["index_type"] == "int32" for r in rows
    )
    assert sum(r["data_type"] == "float" for r in rows) == 10
    assert all(
        r["gpu_code"] == "sm_120" for r in rows if r["data_type"] == "half"
    )
    for r in rows:
        if r["gpu_code"] == "sm_120":
            assert (r["tile_m"], r["tile_n"], r["tile_k"], r["occupancy"]) == (
                32,
                128,
                32,
                2,
            )
    with pytest.raises(ValueError):
        kernel.make_kernel(
            "float",
            "runtime",
            16,
            64,
            128,
            32,
            index_type="int64",
            gpu_code="sm_120",
        )
    public = (CPP / "src/neighbors/detail/knn_brute_force.cuh").read_text()
    assert "is_fused_topk_pk16_candidate<value_t>" in public
    assert "k >= 5 && k <= 8 ? 32 : 64" in public


def test_pk16_exact_stored_half_surplus_fixture():
    import numpy as np

    values = np.arange(16, 0, -1, dtype=np.float16) / np.float16(64)
    squares = values.astype(np.float32) ** 2
    assert np.all(np.float32(1024**2) + squares == np.float32(1024**2))
    assert np.argsort(squares).tolist() == list(range(15, -1, -1))


@pytest.mark.parametrize("stage", ["stages", "refine"])
@pytest.mark.parametrize("dtype", ["float", "half"])
@pytest.mark.parametrize("layout", ["strict", "relaxed"])
def test_pk16_stage_geometry_by_dtype_and_layout(stage, dtype, layout):
    rows = matrix_rows(
        json.loads(
            (KERNEL / f"fused_topk_pk16_{stage}_matrix.json").read_text()
        )
    )
    assert len(rows) == 4
    selected = [
        r
        for r in rows
        if r["data_type"] == dtype and r["matrix_layout"] == layout
    ]
    assert len(selected) == 1
    row = selected[0]
    wide = stage == "stages" and dtype == "half" and layout == "strict"
    assert (row["tile_m"], row["tile_n"], row["tile_k"]) == (
        (64 if wide else 32) if stage == "stages" else 4,
        128,
        64 if wide else 32,
    )
    assert row["index_type"] == "int32" and row["physical_k"] == 16
    assert row["gpu_code"] == "sm_120"


def test_pk16_stages_share_corpus_partitions_not_feature_tiles():
    source = (KERNEL / "fused_topk_tile.cu").read_text()
    assert "tile_cfg.tile_n != refine_cfg.tile_n" in source
    assert "tile_cfg.tile_k != refine_cfg.tile_k" not in source
    assert "tile_cfg.tile_k <= 0" in source
    assert "refine_cfg.tile_k <= 0" in source
    planner = (KERNEL / "fused_topk_pk16_stages_planner.hpp").read_text()
    assert "Stage != 1 ? selector_tile_m : 4" in planner
    assert "Stage != 1 ? selector_tile_k : 32" in planner


def test_pk16_witness_qualified_dtype_images():
    rows = matrix_rows(
        json.loads(
            (KERNEL / "fused_topk_pk16_witness_matrix.json").read_text()
        )
    )
    assert len(rows) == 2
    assert {
        tuple(
            row[k]
            for k in (
                "data_type",
                "matrix_layout",
                "gpu_code",
                "tile_m",
                "tile_n",
                "tile_k",
                "occupancy",
            )
        )
        for row in rows
    } == {
        ("half", "strict", "sm_120", 64, 128, 64, 1),
        ("float", "strict", "sm_120", 32, 128, 32, 1),
    }


def test_pk16_witness_native_planner_and_launcher_enable_both_dtypes():
    for name in ("fused_topk_pk16_stages_planner.hpp", "fused_topk_tile.cu"):
        text = (KERNEL / name).read_text()
        assert (
            "std::is_same_v<DataT, half> || std::is_same_v<DataT, float>"
            in text
        )
