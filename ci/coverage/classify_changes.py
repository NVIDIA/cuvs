#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# PROOF OF CONCEPT -- see docs/cpp_test_selector.md §2.4 and §3.4. Diffs a
# PR's object_hashes.json / external_artifact_hashes.json (from
# hash_objects_sccache.py / hash_external_artifacts.py) against the
# nightly baseline's, and classifies the result:
#   FULL_SUITE -- an external artifact changed; its call graph is out of
#                 scope to reason about
#   NARROWED   -- new_files (ODR-safe, promote to Pass 1) and/or
#                 affected_files (feed into the file-level test mapping)
#
# Usage:
#   ci/coverage/classify_changes.py \
#     --baseline-objects  baseline/object_hashes.json \
#     --pr-objects        object_hashes.json \
#     --baseline-external baseline/external_artifact_hashes.json \
#     --pr-external       external_artifact_hashes.json

import argparse
import json
import sys
from pathlib import Path


def load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def classify(baseline_objects: dict, pr_objects: dict, baseline_external: dict, pr_external: dict):
    if any(baseline_external.get(lib) != h for lib, h in pr_external.items()):
        return "FULL_SUITE", [], []

    new_files, affected_files = [], []
    baseline_sources = {o["source"] for o in baseline_objects.values() if o.get("source")}
    for target, entry in pr_objects.items():
        src, h = entry.get("source"), entry["hash"]
        if target not in baseline_objects:
            if src and src not in baseline_sources:
                new_files.append(src)
            continue
        if baseline_objects[target]["hash"] != h:
            affected_files.append(src)
    return "NARROWED", sorted(set(new_files)), sorted(set(f for f in affected_files if f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a PR's compiled-object hash diff (PoC).")
    parser.add_argument("--baseline-objects", required=True)
    parser.add_argument("--pr-objects", required=True)
    parser.add_argument("--baseline-external", default=None)
    parser.add_argument("--pr-external", default=None)
    parser.add_argument("--new-files-out", default="new_source_files.txt")
    parser.add_argument("--affected-files-out", default="affected_files.txt")
    parser.add_argument("--result-out", default="classification_result.txt")
    args = parser.parse_args()

    baseline = load(args.baseline_objects).get("objects", {})
    pr = load(args.pr_objects).get("objects", {})
    baseline_ext = load(args.baseline_external).get("libraries", {}) if args.baseline_external else {}
    pr_ext = load(args.pr_external).get("libraries", {}) if args.pr_external else {}

    if not baseline or not pr:
        sys.exit("ERROR: baseline and PR object hash maps must both be non-empty")

    mode, new_files, affected_files = classify(baseline, pr, baseline_ext, pr_ext)

    Path(args.result_out).write_text(mode + "\n")
    Path(args.new_files_out).write_text("\n".join(new_files) + ("\n" if new_files else ""))
    Path(args.affected_files_out).write_text("\n".join(affected_files) + ("\n" if affected_files else ""))

    print(f"mode: {mode}", flush=True)
    print(f"new_files: {len(new_files)} -> {args.new_files_out}", flush=True)
    print(f"affected_files: {len(affected_files)} -> {args.affected_files_out}", flush=True)


if __name__ == "__main__":
    main()
