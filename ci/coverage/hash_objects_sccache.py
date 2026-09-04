#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# PROOF OF CONCEPT -- see docs/cpp_test_selector.md §2.2 and
# docs/cpp_test_selector_rationale.md. Parses an sccache trace log
# (SCCACHE_SERVER_LOG=sccache::compiler::compiler=trace) into
# object_hashes.json. This relies on an undocumented internal sccache log
# line ("Hash key: ..."), not a stable CLI contract -- see the rationale
# doc for why, and for the upstream ask that would remove this dependency.
#
# Usage:
#   sccache --stop-server
#   SCCACHE_SERVER_LOG=sccache::compiler::compiler=trace \
#   SCCACHE_ERROR_LOG=$BUILD_DIR/sccache.log \
#     cmake --build --preset coverage
#   ci/coverage/hash_objects_sccache.py \
#     --sccache-log $BUILD_DIR/sccache.log \
#     --build-dir   $BUILD_DIR \
#     --output      object_hashes.json

import argparse
import json
import re
import sys
from pathlib import Path

HASH_RE = re.compile(r"\[(?P<object>[^\]]+)\]: Hash key: (?P<hash>\S+)")


def parse_log(log_path: Path) -> dict[str, str]:
    """Real ninja targets are always build-dir-relative in sccache's log
    (matches the `-o` argument ninja invokes with). nvcc internally forks
    sub-invocations (e.g. compiling to a .ltoir intermediate before the
    final fatbin link step) that sccache also wraps and logs separately,
    under an absolute /tmp/sccache/nvcc/.tmpXXXXX path with no stable
    correlation to a source file across builds -- these are noise, not
    real targets, and are dropped.
    """
    objects: dict[str, str] = {}
    with open(log_path, errors="replace") as f:
        for line in f:
            m = HASH_RE.search(line)
            if m:
                obj = m.group("object")
                if obj.startswith("/"):
                    continue
                objects[obj] = m.group("hash")
    return objects


def load_object_to_source(build_dir: Path) -> dict[str, str]:
    cc_path = build_dir / "compile_commands.json"
    entries = json.loads(cc_path.read_text())
    mapping = {}
    for e in entries:
        out = e.get("output")
        if not out:
            continue
        out_rel = str(Path(out).resolve().relative_to(build_dir.resolve()))
        mapping[out_rel] = e["file"]
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse an sccache trace log into object_hashes.json (PoC)."
    )
    parser.add_argument("--sccache-log", required=True, help="sccache trace log (SCCACHE_ERROR_LOG)")
    parser.add_argument("--build-dir", required=True, help="Coverage-preset CMake build directory")
    parser.add_argument("--output", default="object_hashes.json")
    args = parser.parse_args()

    log_path = Path(args.sccache_log)
    build_dir = Path(args.build_dir)
    if not log_path.exists():
        sys.exit(f"ERROR: sccache log not found: {log_path}")

    hashes = parse_log(log_path)
    obj_to_src = load_object_to_source(build_dir)

    objects = {}
    unmatched = 0
    for obj, h in hashes.items():
        src = obj_to_src.get(obj)
        if src is None:
            unmatched += 1
        objects[obj] = {"hash": h, "source": src}

    Path(args.output).write_text(json.dumps({"objects": objects}, indent=2))
    print(f"{len(objects)} objects parsed from {log_path}", flush=True)
    if unmatched:
        print(f"WARNING: {unmatched} object(s) had no compile_commands.json match", flush=True)
    print(f"written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
