#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Case-scoped log evidence for live Lucene integration tests."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Iterator


CPU_HNSW_FALLBACK_WARNING = (
    "GPU based indexing not supported, falling back to using the "
    "Lucene99HnswVectorsWriter"
)

_CASE_START = "CUVS_BENCH_LUCENE_CASE_START "
_CASE_END = "CUVS_BENCH_LUCENE_CASE_END "


def _validate_case_name(case_name: str) -> None:
    if not case_name or "\n" in case_name or "\r" in case_name:
        raise ValueError("Lucene log case names must be nonempty single lines")


@contextmanager
def lucene_log_case(case_name: str) -> Iterator[None]:
    """Delimit one serial test case on the JVM logger's stderr stream."""
    _validate_case_name(case_name)
    print(f"{_CASE_START}{case_name}", file=sys.stderr, flush=True)
    try:
        yield
    finally:
        print(f"{_CASE_END}{case_name}", file=sys.stderr, flush=True)


def lucene_case_output(output: str, case_name: str) -> str:
    """Return only output enclosed by markers for ``case_name``."""
    _validate_case_name(case_name)
    sections: dict[str, list[list[str]]] = {}
    active_case: str | None = None
    active_lines: list[str] | None = None

    for line in output.splitlines():
        if line.startswith(_CASE_START):
            started_case = line.removeprefix(_CASE_START)
            _validate_case_name(started_case)
            if active_case is not None:
                raise ValueError(
                    "Nested Lucene log case markers are not supported: "
                    f"{active_case!r}, {started_case!r}"
                )
            active_case = started_case
            active_lines = []
            continue
        if line.startswith(_CASE_END):
            ended_case = line.removeprefix(_CASE_END)
            _validate_case_name(ended_case)
            if active_case != ended_case or active_lines is None:
                raise ValueError(
                    "Mismatched Lucene log case marker: "
                    f"expected {active_case!r}, found {ended_case!r}"
                )
            sections.setdefault(ended_case, []).append(active_lines)
            active_case = None
            active_lines = None
            continue
        if active_lines is not None:
            active_lines.append(line)

    if active_case is not None:
        raise ValueError(
            f"Unterminated Lucene log case marker: {active_case!r}"
        )
    try:
        case_sections = sections[case_name]
    except KeyError as error:
        raise ValueError(
            f"Lucene log contains no complete case named {case_name!r}"
        ) from error
    return "\n".join(line for section in case_sections for line in section)


def case_used_cpu_hnsw_fallback(output: str, case_name: str) -> bool:
    """Report whether one accelerated-HNSW case logged its CPU fallback."""
    return CPU_HNSW_FALLBACK_WARNING in lucene_case_output(output, case_name)
