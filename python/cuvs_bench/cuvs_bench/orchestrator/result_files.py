#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Persistence for benchmark backends that return in-process results."""

from __future__ import annotations

import json
import re
import stat
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union

from ..backends.base import BuildResult, SearchResult

BenchmarkResult = Union[BuildResult, SearchResult]
_SEARCH_SUFFIX_PATTERN = re.compile(r"k[1-9]\d*,bs[1-9]\d*")


@dataclass(frozen=True)
class ResultRecord:
    """Associate a backend result with its benchmark artifact identity."""

    result: BenchmarkResult
    index_name: str
    output_filename: str

    @property
    def phase(self) -> str:
        return "build" if isinstance(self.result, BuildResult) else "search"

    def to_json(self) -> Dict[str, Any]:
        row = self.result.to_json()
        row["name"] = f"{self.index_name}/{self.phase}"
        return row


def _validate_path_component(value: object, description: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Invalid {description}: {value!r}")
    path = Path(value)
    if not value or path.name != value or value in {".", ".."}:
        raise ValueError(f"Invalid {description}: {value!r}")
    return value


def validate_dataset_name(dataset: object) -> str:
    """Validate a dataset name before using it as a path component."""
    return _validate_path_component(dataset, "benchmark dataset name")


def _validate_output_filename(output_filename: str) -> None:
    _validate_path_component(output_filename, "benchmark result filename")


def validate_sweep_output_filenames(
    output_filenames: object,
) -> Tuple[str, str]:
    """Validate and normalize opt-in sweep result filename stems."""
    if (
        not isinstance(output_filenames, (list, tuple))
        or len(output_filenames) != 2
    ):
        raise ValueError(
            "Opt-in sweep persistence requires exactly two result filename "
            "stems: (build_stem, search_stem)"
        )

    build_stem, search_stem = output_filenames
    _validate_output_filename(build_stem)
    _validate_output_filename(search_stem)

    build_parts = build_stem.split(",")
    if len(build_parts) < 2 or any(not part for part in build_parts):
        raise ValueError(
            "Build result filename stem must contain at least "
            "'<algorithm>,<group>'"
        )

    expected_prefix = f"{build_stem},"
    search_suffix = (
        search_stem[len(expected_prefix) :]
        if search_stem.startswith(expected_prefix)
        else ""
    )
    if _SEARCH_SUFFIX_PATTERN.fullmatch(search_suffix) is None:
        raise ValueError(
            "Search result filename stem must be "
            "'<build_stem>,k<positive integer>,bs<positive integer>'"
        )

    return build_stem, search_stem


def _read_benchmarks(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    benchmarks = (
        payload.get("benchmarks") if isinstance(payload, dict) else None
    )
    if not isinstance(benchmarks, list):
        raise ValueError(
            f"Benchmark result file must contain a 'benchmarks' list: {path}"
        )
    seen_names = set()
    for position, row in enumerate(benchmarks):
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise ValueError(
                f"Benchmark row {position} must be an object with a string "
                f"'name': {path}"
            )
        if row["name"] in seen_names:
            raise ValueError(
                f"Benchmark result file contains duplicate name "
                f"{row['name']!r}: {path}"
            )
        seen_names.add(row["name"])
    return benchmarks


def _merge_build_rows(
    path: Path, records: List[ResultRecord]
) -> List[Dict[str, Any]]:
    rows_by_name = {row["name"]: row for row in _read_benchmarks(path)}
    for record in records:
        row = record.to_json()
        previous = rows_by_name.get(row["name"])
        if (
            record.result.success
            and record.result.metadata.get("skipped")
            and previous is not None
            and previous.get("success", True) is True
        ):
            continue
        rows_by_name[row["name"]] = row
    return list(rows_by_name.values())


def _write_payload(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temp_path = Path(file.name)
            json.dump({"benchmarks": rows}, file, indent=2, allow_nan=False)
            file.write("\n")
        temp_path.chmod(file_mode)
        temp_path.replace(path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _invalidate_derived_csvs(path: Path, phase: str) -> None:
    suffixes = (
        (".csv",)
        if phase == "build"
        else (",raw.csv", ",throughput.csv", ",latency.csv")
    )
    stem = path.with_suffix("")
    for suffix in suffixes:
        stem.with_name(f"{stem.name}{suffix}").unlink(missing_ok=True)


def _remove_result_artifacts(path: Path, phase: str) -> None:
    path.unlink(missing_ok=True)
    _invalidate_derived_csvs(path, phase)


def write_result_files(
    records: List[ResultRecord],
    dataset: str,
    dataset_path: str,
    *,
    stale_search_filenames: Iterable[str] = (),
) -> None:
    """Write grouped Google Benchmark-compatible JSON result files."""
    validate_dataset_name(dataset)
    grouped: Dict[Tuple[str, str], List[ResultRecord]] = defaultdict(list)
    for record in records:
        _validate_output_filename(record.output_filename)
        grouped[(record.phase, record.output_filename)].append(record)

    stale_search_filenames = set(stale_search_filenames)
    for output_filename in stale_search_filenames:
        _validate_output_filename(output_filename)

    result_root = Path(dataset_path) / dataset / "result"
    current_search_filenames = {
        output_filename
        for phase, output_filename in grouped
        if phase == "search"
    }
    for output_filename in stale_search_filenames - current_search_filenames:
        _remove_result_artifacts(
            result_root / "search" / f"{output_filename}.json",
            "search",
        )

    for (phase, output_filename), file_records in grouped.items():
        output_path = result_root / phase / f"{output_filename}.json"
        rows = (
            _merge_build_rows(output_path, file_records)
            if phase == "build"
            else [record.to_json() for record in file_records]
        )
        if rows:
            _write_payload(output_path, rows)
            _invalidate_derived_csvs(output_path, phase)


__all__ = [
    "ResultRecord",
    "validate_dataset_name",
    "validate_sweep_output_filenames",
    "write_result_files",
]
