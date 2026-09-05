#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import runpy
from pathlib import Path

iterate_matrix_product = runpy.run_path(
    str(Path(__file__).with_name("compute_matrix_product.py"))
)["iterate_matrix_product"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--include", required=True)
    parser.add_argument("--alias-prefix", required=True)
    args = parser.parse_args()
    aliases = {}
    matrix = json.loads(args.matrix.read_text())
    for entry in iterate_matrix_product(matrix=matrix):
        tile = tuple(entry.get(key) for key in ("tile_m", "tile_n", "tile_k"))
        if any(value is None for value in tile):
            raise ValueError("missing cuTile tile geometry")
        suffix = (
            f"{entry['data_abbrev']}_"
            f"{entry.get('arch_tag', 'tileir')}_"
            f"{entry['abi_abbrev']}"
        )
        if suffix in aliases and aliases[suffix] != tile:
            raise ValueError(f"conflicting tile geometry for {suffix}")
        aliases[suffix] = tile
    lines = [
        "#pragma once",
        "",
        f"#include {args.include}",
        "",
        f"namespace {args.namespace} {{",
        "",
    ]
    for suffix, (m, n, k) in sorted(aliases.items()):
        lines.append(
            f"using {args.alias_prefix}_{suffix} = cutile_tile_config<{m}, {n}, {k}>;"
        )
    lines.extend(["", f"}}  // namespace {args.namespace}", ""])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
