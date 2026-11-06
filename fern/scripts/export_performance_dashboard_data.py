#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Export cuVS bench CSV results to JSON and TypeScript for the Performance dashboard."""

from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_DIR / "fern" / "data" / "results_cuvs_26_04.csv"
OUTPUT_JSON = REPO_DIR / "fern" / "assets" / "data" / "benchmark_results.json"
BENCHMARK_DATA_TS = (
    REPO_DIR / "fern" / "theme" / "nvidia" / "components" / "benchmarkData.ts"
)

NUMERIC_COLUMNS = {
    "Index Build Time (s)",
    "Search Batch Size",
    "TopK",
    "Mean Search Throughput (QPS)",
    "Mean Search Latency (ms)",
    "Mean Recall",
    "N Points in Bucket",
    "Total Vectors",
    "Dimensions",
}


def parse_row(row: dict[str, str]) -> dict:
    parsed: dict = {}
    for key, value in row.items():
        if key in NUMERIC_COLUMNS and value not in ("", "NA", None):
            try:
                parsed[key] = float(value) if "." in value else int(value)
            except ValueError:
                parsed[key] = value
        else:
            parsed[key] = value
    return parsed


def write_benchmark_data_ts(rows: list[dict], output_path: Path) -> None:
    output_path.write_text(
        """/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

export type BenchmarkRow = Record<string, string | number | null>;

export const BENCHMARK_ROWS: BenchmarkRow[] = """
        + json.dumps(rows, indent=2)
        + ";\n",
        encoding="utf-8",
    )


def export(csv_path: Path = DEFAULT_CSV, output_path: Path = OUTPUT_JSON) -> int:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Benchmark CSV not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = [parse_row(row) for row in csv.DictReader(handle, delimiter="\t")]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {output_path}")

    write_benchmark_data_ts(rows, BENCHMARK_DATA_TS)
    print(f"Wrote {len(rows)} rows to {BENCHMARK_DATA_TS}")

    return len(rows)


def main() -> int:
    export()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
