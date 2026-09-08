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
# --update matters more here than in build_mapping.py: ninja only invokes
# the compiler (and therefore sccache) for objects it actually rebuilds,
# so an incremental build's trace log is silently INCOMPLETE for anything
# already up to date -- it contains no line at all for those objects, not
# a stale one. Without --update, re-running this after an incremental
# build would silently drop every untouched object from the map. Objects
# not rebuilt this run keep their previously-recorded hash, which is
# still correct by construction: if ninja didn't rebuild it, nothing about
# its inputs changed.
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


def load_object_to_source(build_dir: Path, repo_root: Path) -> dict[str, str]:
    """Source paths are normalized to match func2tests.json's convention
    exactly (collect_coverage.py: path relative to <repo_root>) -- required
    for affected_files/new_files to correlate with func2tests.json's
    file-level mapping in select_tests.py. A generated file under
    cpp/build/coverage/... normalizes to "cpp/build/coverage/...", same as
    func2tests.json already does for generated-file coverage entries.
    Relative to the repo root, not repo_root/"cpp": cuVS's C API library and
    its tests (../c, pulled into the cpp/ build via add_subdirectory) live
    in a sibling top-level directory, not under cpp/ -- normalizing against
    the repo root instead of hardcoding a "cpp" prefix covers both source
    trees uniformly.
    """
    cc_path = build_dir / "compile_commands.json"
    entries = json.loads(cc_path.read_text())
    mapping = {}
    for e in entries:
        out = e.get("output")
        if not out:
            continue
        out_rel = str(Path(out).resolve().relative_to(build_dir.resolve()))
        try:
            src_rel = str(Path(e["file"]).resolve().relative_to(repo_root))
        except ValueError:
            src_rel = None  # source lives outside the repo entirely -- no correlation possible
        mapping[out_rel] = src_rel
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse an sccache trace log into object_hashes.json (PoC)."
    )
    parser.add_argument("--sccache-log", required=True, help="sccache trace log (SCCACHE_ERROR_LOG)")
    parser.add_argument("--build-dir", required=True, help="Coverage-preset CMake build directory")
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root, for normalizing source paths to match func2tests.json's "
        "convention (default: auto-detect from --build-dir, assumes <repo-root>/cpp/build/...)",
    )
    parser.add_argument("--output", default="object_hashes.json")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Merge into an existing --output instead of overwriting it. Required for "
        "correctness after any incremental (non-clean) build -- see module docstring.",
    )
    args = parser.parse_args()

    log_path = Path(args.sccache_log)
    build_dir = Path(args.build_dir).resolve()
    if not log_path.exists():
        sys.exit(f"ERROR: sccache log not found: {log_path}")

    if args.repo_root:
        repo_root = Path(args.repo_root).resolve()
    else:
        # Coverage build dir is always <repo_root>/cpp/build/<preset>.
        repo_root = build_dir.parent.parent.parent
    if not (repo_root / "cpp").is_dir():
        sys.exit(
            f"ERROR: could not locate cpp/ under detected repo root {repo_root} "
            f"(from --build-dir {build_dir}); pass --repo-root explicitly"
        )

    hashes = parse_log(log_path)
    obj_to_src = load_object_to_source(build_dir, repo_root)

    output_path = Path(args.output)
    objects = {}
    if args.update and output_path.exists():
        objects = json.loads(output_path.read_text()).get("objects", {})

    new_count = 0
    unmatched = 0
    for obj, h in hashes.items():
        src = obj_to_src.get(obj)
        if src is None:
            unmatched += 1
        if obj not in objects:
            new_count += 1
        objects[obj] = {"hash": h, "source": src}

    output_path.write_text(json.dumps({"objects": objects}, indent=2))
    print(f"{len(hashes)} object(s) parsed from {log_path} ({new_count} new/updated)", flush=True)
    if unmatched:
        print(f"WARNING: {unmatched} object(s) had no compile_commands.json match", flush=True)
    print(f"{len(objects)} object(s) total written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
