#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# PROOF OF CONCEPT -- see docs/cpp_test_selector.md §2.3. Hashes every
# .so/.so.* linked into the given test binaries that lives outside the
# build directory -- third-party artifacts an object-hash diff can't see,
# since sccache only hashes what it compiles.
#
# Usage:
#   ci/coverage/hash_external_artifacts.py \
#     --build-dir cpp/build/coverage \
#     --output external_artifact_hashes.json \
#     cpp/build/coverage/gtests/SOME_TEST [more binaries ...]

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def resolve_libs(binary: Path) -> set[Path]:
    result = subprocess.run(["ldd", str(binary)], capture_output=True, text=True)
    libs = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if "=>" in line:
            target = line.split("=>", 1)[1].strip()
            path = target.split(" ")[0]
            if path and path != "not":
                libs.add(Path(path))
    return libs


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hash externally-linked .so files for PR change classification (PoC)."
    )
    parser.add_argument("--build-dir", required=True, help="Coverage-preset CMake build directory")
    parser.add_argument("--output", default="external_artifact_hashes.json")
    parser.add_argument("binaries", nargs="+", help="Built test binaries to run ldd against")
    args = parser.parse_args()

    build_dir = Path(args.build_dir).resolve()
    binaries = [Path(p) for p in args.binaries]
    for b in binaries:
        if not b.exists():
            sys.exit(f"ERROR: binary not found: {b}")

    all_libs: set[Path] = set()
    for b in binaries:
        all_libs |= resolve_libs(b)

    external = {p for p in all_libs if build_dir not in p.resolve().parents and p.exists()}

    result = {str(lib): sha256_file(lib) for lib in sorted(external)}

    Path(args.output).write_text(json.dumps({"libraries": result}, indent=2))
    print(f"{len(all_libs)} total linked libraries found across {len(binaries)} binaries", flush=True)
    print(f"{len(external)} external (outside build dir), hashed", flush=True)
    print(f"written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
