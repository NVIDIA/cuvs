# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pathlib
import runpy
import sys


run_clang_compile_script = runpy.run_path(
    str(pathlib.Path(__file__).parent / "../../scripts/run-clang-compile.py")
)
run_clang_command = run_clang_compile_script["run_clang_command"]


def test_run_clang_command_does_not_invoke_a_shell(tmp_path):
    sentinel = tmp_path / "unexpected-shell-execution"

    status, output = run_clang_command(
        [sys.executable, "-c", "pass", f"; touch {sentinel}"], str(tmp_path)
    )

    assert status
    assert not sentinel.exists()
    assert f"touch {sentinel}" in output
