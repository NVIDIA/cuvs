# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only ABI, partition-policy and conservative guard contract tests."""

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest

ct = pytest.importorskip(
    "cuda.tile", reason="requires the optional cuTile exporter"
)

ROOT = Path(__file__).resolve().parents[3]
KERNEL = ROOT / "cpp/src/neighbors/detail/cutile_topk"
sys.path.insert(0, str(KERNEL))
import fused_topk_kernel as kernel  # noqa: E402
import export_fused_topk as exporter  # noqa: E402


@pytest.fixture(autouse=True)
def forbid_device_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail(
            "CPU contract test attempted compiler/export/device execution"
        )

    monkeypatch.setattr(exporter, "export_kernel", forbidden)
    monkeypatch.setattr(ct, "launch", forbidden)


@pytest.mark.parametrize("dtype", ["float", "half"])
@pytest.mark.parametrize("idx", ["int32", "int64"])
@pytest.mark.parametrize("layout", ["strict", "relaxed"])
def test_separate_symbol_same_abi(dtype, idx, layout):
    args = (8, 32, 128, 32, "sm_120", layout, dtype, idx)
    a = exporter._kernel_signature(*args)
    b = exporter._kernel_signature(*args, whole_tile_guard=True)
    assert a.symbol != b.symbol and "_longscan" in b.symbol
    assert (
        a.parameters == b.parameters
        and a.calling_convention == b.calling_convention
    )
    text = (KERNEL / "fused_topk_pk8_longscan_planner.hpp").read_text()
    assert b.symbol in text


@pytest.mark.parametrize(
    "pk,arch", [(2, "sm_120"), (4, "sm_120"), (8, "sm_90")]
)
def test_unsupported_guard_rejected_before_compilation(pk, arch):
    with pytest.raises(ValueError, match="Whole-tile guard"):
        kernel.make_kernel(
            "float",
            "runtime",
            pk,
            32,
            128,
            32,
            gpu_code=arch,
            whole_tile_guard=True,
        )


def test_eight_guard_matrix_leaves_and_distinct_registration():
    spec = importlib.util.spec_from_file_location(
        "matrix_product", ROOT / "cpp/cmake/modules/compute_matrix_product.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = list(
        module.iterate_matrix_product(
            matrix=json.loads(
                (KERNEL / "fused_topk_pk8_longscan_matrix.json").read_text()
            )
        )
    )
    assert len(rows) == 8
    assert {
        (r["data_type"], r["index_type"], r["matrix_layout"]) for r in rows
    } == {
        (d, i, layout)
        for d in ("float", "half")
        for i in ("int32", "int64")
        for layout in ("strict", "relaxed")
    }
    assert all(
        (
            r["gpu_code"],
            r["physical_k"],
            r["tile_m"],
            r["tile_n"],
            r["tile_k"],
            r["occupancy"],
        )
        == ("sm_120", 8, 32, 128, 32, 2)
        for r in rows
    )
    planner = (KERNEL / "fused_topk_pk8_longscan_planner.hpp").read_text()
    assert (
        "fragment_tag_fused_topk_pk8_longscan_cubin" in planner
        and "inline static" in planner
    )
    assert (ROOT / "cpp/CMakeLists.txt").read_text().count(
        "${cutile_fused_topk_pk8_longscan_files}"
    ) == 1


def test_actual_cpp_partition_rule(tmp_path):
    text = (KERNEL / "fused_topk_tile.hpp").read_text()
    begin = text.index("constexpr bool fused_topk_pk8_use_longscan(")
    end = text.index("\n}", begin) + 2
    fn = text[begin:end]
    code = (
        "#include <cassert>\n#include <limits>\n"
        + fn
        + """
int main(){
 for(int s: {1,2,3,7,32,128}) {
  int boundary=(512*s-1)*128;
  assert(!fused_topk_pk8_use_longscan(boundary,s,32,128,32));
  assert(fused_topk_pk8_use_longscan(boundary+1,s,32,128,32));
 }
 assert(!fused_topk_pk8_use_longscan(100000,32,32,128,32));
 assert(fused_topk_pk8_use_longscan(100000,1,32,128,32));
 assert(!fused_topk_pk8_use_longscan(100000,1,64,128,32));
 assert(!fused_topk_pk8_use_longscan(100000,0,32,128,32));
 assert(!fused_topk_pk8_use_longscan(-1,1,32,128,32));
 assert(fused_topk_pk8_use_longscan(std::numeric_limits<int>::max(),1,32,128,32));
}
"""
    )
    code = "#include <initializer_list>\n" + code
    p = tmp_path / "policy.cpp"
    p.write_text(code)
    exe = tmp_path / "policy"
    subprocess.run(
        ["g++", "-std=c++17", "-Wall", "-Werror", str(p), "-o", str(exe)],
        check=True,
    )
    subprocess.run([str(exe)], check=True)


def guard_model():
    import numpy as np
    import types

    tree = ast.parse((KERNEL / "fused_topk_kernel.py").read_text())
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "_pk8_tile_needs_selection"
    )
    fn.decorator_list = []
    ns = {"ct": types.SimpleNamespace(min=np.min, max=np.max, int32=np.int32)}
    exec(
        compile(ast.Module([fn], type_ignores=[]), "actual_guard", "exec"), ns
    )
    return ns[fn.name]


def test_actual_guard_safe_skips_and_ties():
    import numpy as np

    guard = guard_model()
    rng = np.random.default_rng(20261028)
    for rows in (1, 7, 31, 32):
        valid = np.arange(32) < rows
        for trial in range(100):
            retained = rng.normal(size=(32, 8)).astype(np.float32)
            scores = rng.normal(size=(32, 128)).astype(np.float32) + trial / 10
            expected = bool(
                np.any((scores.min(axis=1) <= retained.max(axis=1)) & valid)
            )
            assert bool(guard(scores, retained, valid)) == expected
            if not expected:
                assert all(
                    np.min(scores[r]) > np.max(retained[r])
                    for r in range(rows)
                )
    scores = np.full((32, 128), 100, np.float32)
    retained = np.zeros((32, 8), np.float32)
    valid = np.arange(32) < 1
    scores[1:] = -100
    assert not guard(scores, retained, valid), (
        "padded rows must not force admission"
    )
    scores[0, 0] = -0.0
    assert guard(scores, retained, valid), (
        "equal score must retain index tie path"
    )
    scores[0, 0] = 100
    retained[0, -1] = np.inf
    assert guard(scores, retained, valid), (
        "incomplete list must admit finite candidates"
    )
    scores[0] = np.inf
    assert guard(scores, retained, valid), (
        "infinity equality takes original path"
    )
