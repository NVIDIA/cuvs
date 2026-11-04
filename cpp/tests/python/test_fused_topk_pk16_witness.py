# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only exact-state checks of the production PK16 witness helper."""

import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
KERNEL = ROOT / "cpp/src/neighbors/detail/cutile_topk"
INVALID = np.iinfo(np.int32).max


def functions():
    calls = []

    def extract(a, index, shape):
        return a[
            tuple(slice(i * n, (i + 1) * n) for i, n in zip(index, shape))
        ]

    def reduce(values, axis, op, identity, keepdims):
        assert (
            axis == 1 and not keepdims and identity == (float("inf"), INVALID)
        )
        a, i = values
        s = np.full(len(a), np.inf, np.float32)
        j = np.full(len(a), INVALID, np.int32)
        for col in range(a.shape[1]):
            s, j = op(s, j, a[:, col], i[:, col])
        calls.append(1)
        return s, j

    ct = SimpleNamespace(
        extract=extract,
        cat=lambda a, axis: np.concatenate(a, axis),
        arange=np.arange,
        int32=np.int32,
        float32=np.float32,
        where=np.where,
        full=np.full,
        max=np.max,
        reduce=reduce,
    )
    scope = dict(
        ct=ct, ConstInt=int, _INF=float("inf"), _INVALID_INDEX=INVALID
    )
    for path, names in [
        (KERNEL / "fused_topk_kernel.py", {"_take_left", "_column"}),
        (
            KERNEL / "fused_topk_pk16_stages.py",
            {
                "_merge_stage16",
                "_merge_sorted16",
                "_extract_four",
                "_group4_continue",
                "_witness_update16",
                "_extract_group",
                "_front_witness_update16",
            },
        ),
    ]:
        nodes = [
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        assert {n.name for n in nodes} == names
        for n in nodes:
            n.decorator_list = []
        exec(
            compile(ast.Module(nodes, type_ignores=[]), str(path), "exec"),
            scope,
        )
    return scope, calls


def ordered(v, i, k=16):
    order = np.lexsort((i, v), axis=1)[:, :k]
    return np.take_along_axis(v, order, 1), np.take_along_axis(i, order, 1)


@pytest.mark.parametrize("front", [False, True])
def test_actual_witness_sequence_proof(front):
    helper = "_front_witness_update16" if front else "_witness_update16"
    stops = (1, 2, 4, 8, 12, 16) if front else (1, 4, 8, 12, 16)
    oracle = SimpleNamespace(
        functions=functions, ordered=ordered, INVALID=INVALID
    )
    f, calls = oracle.functions()
    rng = np.random.default_rng(20261028)
    counts, cases, updates = set(), 0, 0
    for rows in (1, 3, 32, 64):
        tm = 32 if rows <= 32 else 64
        valid = np.arange(tm) < rows
        for kind in (
            "random",
            "ties",
            "zero",
            "tail",
            "empty",
            "late",
            "skip_then_win",
            "one_row",
        ):
            best = np.full((tm, 16), np.inf, np.float32)
            ids = np.full((tm, 16), oracle.INVALID, np.int32)
            for tile in range(24):
                b = rng.normal(size=(tm, 128)).astype(np.float32)
                j = np.broadcast_to(
                    np.arange(tile * 128, (tile + 1) * 128, dtype=np.int32),
                    b.shape,
                ).copy()
                if kind == "ties":
                    b = np.round(b)
                if kind == "zero":
                    b[:] = np.where(j % 2, -0.0, 0.0)
                if kind == "tail":
                    b[:, tile % 17 :] = np.inf
                if kind == "empty":
                    b[:] = np.inf
                if kind == "late":
                    b[:] = 100 if tile < 22 else -100 - tile
                if kind == "skip_then_win":
                    b[:] = 0 if tile == 0 else (-100 if tile == 23 else 100)
                if kind == "one_row":
                    b[:] = 100 if tile else 0
                    b[rows - 1, 0] = -tile
                    if rows < tm:
                        b[rows:, :] = -10000
                j[b == np.inf] = oracle.INVALID
                expected = oracle.ordered(
                    np.concatenate((best, b), 1), np.concatenate((ids, j), 1)
                )
                start = len(calls)
                best, ids = f[helper](b, j, best, ids, valid, tm)
                count = len(calls) - start
                counts.add(count)
                assert count in stops, count
                np.testing.assert_array_equal(ids[:rows], expected[1][:rows])
                np.testing.assert_array_equal(
                    best[:rows].view(np.uint32),
                    expected[0][:rows].view(np.uint32),
                )
                updates += 1
            cases += 1
    # Every staged stopping count and the new one-reduction skip, with exact ties.
    for count in stops:
        tm = 32
        valid = np.arange(tm) < 3
        best = np.broadcast_to(
            np.arange(16, dtype=np.float32), (tm, 16)
        ).copy()
        ids = np.broadcast_to(np.arange(16, dtype=np.int32), (tm, 16)).copy()
        b = np.broadcast_to(
            np.arange(128, dtype=np.float32) + 100, (tm, 128)
        ).copy()
        if count > 1:
            b[:, : count - 1] = -100 + np.arange(count - 1, dtype=np.float32)
        j = np.broadcast_to(np.arange(128, dtype=np.int32) + 1000, b.shape)
        start = len(calls)
        f[helper](b, j, best, ids, valid, tm)
        assert len(calls) - start == count
        if count == 1:
            b[2, 0] = 15
            start = len(calls)
            f[helper](b, j, best, ids, valid, tm)
            assert len(calls) - start == (2 if front else 4), (
                "equality must continue"
            )
    print(
        dict(
            cpu_only=True,
            scenarios=cases,
            tile_updates=updates,
            reduction_counts=sorted(counts),
            exact_valid_score_bits=True,
            exact_valid_indices=True,
            gpu_compiled=False,
        )
    )
