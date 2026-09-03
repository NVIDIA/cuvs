#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--include", required=True)
    parser.add_argument("--alias-prefix", required=True)
    args = parser.parse_args()
    aliases = {}
    for entry in json.loads(args.matrix.read_text()):
        default_tile = entry.get("_tile", [{}])[0]
        for data in entry["_data"]:
            for abi in entry["_abi"]:
                tile = tuple(
                    abi.get(k, default_tile.get(k))
                    for k in ("tile_m", "tile_n", "tile_k")
                )
                if any(value is None for value in tile):
                    raise ValueError("missing cuTile tile geometry")
                for exported in entry["_export"]:
                    suffix = f"{data['data_abbrev']}_{exported.get('arch_tag', 'tileir')}_{abi['abi_abbrev']}"
                    if suffix in aliases and aliases[suffix] != tile:
                        raise ValueError(
                            f"conflicting tile geometry for {suffix}"
                        )
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
