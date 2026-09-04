#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# Look up which CTest tests cover the functions and files changed in a PR.
# Writes selected_tests.txt (tests to run first) and remaining_tests.txt
# (all other registered tests).  The two files are disjoint; their union is
# the full registered test suite.
#
# Called by ci/coverage/select_tests.sh.

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# 4-strategy name matching (ctags name → gcov name)
# ---------------------------------------------------------------------------

def _suffix(name: str, n: int) -> str:
    """Last n components of a ::-separated name."""
    parts = name.split("::")
    return "::".join(parts[-n:]) if len(parts) >= n else name


def _strip_signature(name: str) -> str:
    """
    Remove a trailing (...) parameter list from a demangled gcov function
    name — e.g. "Foo::SetUp()" -> "Foo::SetUp", "Foo::Bar() const" ->
    "Foo::Bar". ctags names never include a parameter list, so leaving gcov's
    in place breaks suffix/bare-identifier comparison for every class member
    function (the trailing "()" survives template stripping since that only
    removes <...>).  Finds the last top-level "(...)" via a balanced
    backward scan (safe against nested parens inside template non-type
    arguments, since those already were removed by the caller's <...> strip)
    and splices it out, then drops common trailing qualifiers.
    """
    close = name.rfind(")")
    if close == -1:
        return name
    depth = 0
    open_idx = -1
    for i in range(close, -1, -1):
        if name[i] == ")":
            depth += 1
        elif name[i] == "(":
            depth -= 1
            if depth == 0:
                open_idx = i
                break
    if open_idx == -1:
        return name
    stripped = name[:open_idx] + name[close + 1:]
    return re.sub(r"\s*(const|noexcept|&&|&)\s*$", "", stripped).strip()


def find_tests_for_function(ctags_name: str, func_map: dict[str, list[str]]) -> set[str]:
    """
    Return the set of tests covering ctags_name by trying four strategies
    against every gcov function name in func_map.  Strategies are tried in
    order per gcov name; the first match for a given gcov name wins.
    """
    # Strip template parameter lists from the ctags name for strategies 3 & 4
    bare_ctags = re.sub(r"<[^>]*>", "", ctags_name)

    tests: set[str] = set()
    for gcov_name, test_list in func_map.items():
        bare_gcov = _strip_signature(re.sub(r"<[^>]*>", "", gcov_name))

        # Strategy 1: exact
        if ctags_name == gcov_name:
            tests.update(test_list)
            continue
        # Strategy 2: ctags name is a substring of the gcov name
        if ctags_name in gcov_name:
            tests.update(test_list)
            continue
        # Strategy 3: last two :: components match (template-stripped)
        if bare_ctags and _suffix(bare_ctags, 2) == _suffix(bare_gcov, 2):
            tests.update(test_list)
            continue
        # Strategy 4: bare identifier (final component) matches
        if bare_ctags and bare_ctags.rsplit("::", 1)[-1] == bare_gcov.rsplit("::", 1)[-1]:
            tests.update(test_list)

    return tests


# ---------------------------------------------------------------------------
# CTest helpers
# ---------------------------------------------------------------------------

def list_registered_tests(ctest_dir: Path, ctest_bin: str = "ctest") -> list[str]:
    """Return all test names registered with CTest in ctest_dir."""
    result = subprocess.run(
        [ctest_bin, "-N"],
        cwd=ctest_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    tests: list[str] = []
    for line in result.stdout.splitlines():
        m = re.match(r"\s+Test\s+#\d+:\s+(.+)", line)
        if m:
            tests.append(m.group(1).strip())
    return tests


def list_gtest_case_tests(
    ctest_dir: Path,
    ctest_bin: str = "ctest",
    refresh: bool = False,
) -> list[str]:
    """Return only gtest_case-labelled tests (fine-grained, one GTest case each).

    Results are cached in gtest_case_list_cache.txt inside ctest_dir and reused
    as long as the cache is newer than CTestTestfile.cmake (which CMake touches
    on every reconfigure).  Pass refresh=True to force a re-query regardless.
    """
    cache_path = ctest_dir / "gtest_case_list_cache.txt"

    if not refresh and cache_path.exists():
        ctestfile = ctest_dir / "CTestTestfile.cmake"
        if not ctestfile.exists() or cache_path.stat().st_mtime >= ctestfile.stat().st_mtime:
            tests = [l.strip() for l in cache_path.read_text().splitlines() if l.strip()]
            print(f"  Using cached test list ({len(tests)} tests) from {cache_path}", flush=True)
            return tests
        print(f"  Cache is stale (build reconfigured) — re-querying ctest", flush=True)

    result = subprocess.run(
        [ctest_bin, "-N", "-L", "gtest_case"],
        cwd=ctest_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    tests: list[str] = []
    for line in result.stdout.splitlines():
        m = re.match(r"\s+Test\s+#\d+:\s+(.+)", line)
        if m:
            tests.append(m.group(1).strip())

    cache_path.write_text("\n".join(tests) + "\n")
    print(f"  Cached {len(tests)} tests to {cache_path}", flush=True)

    return tests


# ---------------------------------------------------------------------------
# ctags processing
# ---------------------------------------------------------------------------

def extract_qualified_names(ctags_jsonl_path: Path) -> list[tuple[str, str]]:
    """
    Parse a universal-ctags JSONL file (one JSON object per line) and return
    a list of (fully-qualified function name, repo-relative source path) pairs.
    """
    names: list[tuple[str, str]] = []
    with open(ctags_jsonl_path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                tag = json.loads(line)
            except json.JSONDecodeError:
                continue
            if tag.get("_type") != "tag":
                continue
            kind = tag.get("kind", "")
            if kind not in ("function", "prototype", "method"):
                continue
            name = tag.get("name", "")
            scope = tag.get("scope", "")
            qualified = f"{scope}::{name}" if scope else name
            path = tag.get("path", "")
            names.append((qualified, path))
    return names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select CTest tests that cover changed functions/files and write "
            "selected_tests.txt + remaining_tests.txt."
        )
    )
    parser.add_argument(
        "--mapping",
        default="func2tests.json",
        help="Path to func2tests.json produced by build_mapping.py",
    )
    parser.add_argument(
        "--changed-files",
        required=True,
        help="File listing changed source paths (one per line, repo-relative)",
    )
    parser.add_argument(
        "--ctags-jsonl",
        default=None,
        help="universal-ctags JSONL output for changed source files (optional)",
    )
    parser.add_argument(
        "--base-tests",
        default=None,
        help="Path to base_tests.txt (tests always included in selected_tests.txt)",
    )
    parser.add_argument(
        "--ctest-dir",
        required=True,
        help="Directory from which to run ctest -N (build dir or installed gtests dir)",
    )
    parser.add_argument(
        "--ctest-bin",
        default="ctest",
        help="ctest executable to use (default: ctest)",
    )
    parser.add_argument(
        "--refresh-test-list",
        action="store_true",
        help="Force re-query of ctest -N even if a valid cache exists",
    )
    parser.add_argument(
        "--selected-output",
        default="selected_tests.txt",
        help="Output file for selected tests",
    )
    parser.add_argument(
        "--remaining-output",
        default="remaining_tests.txt",
        help="Output file for remaining tests",
    )
    parser.add_argument(
        "--force-full-suite",
        action="store_true",
        help="Skip classification entirely and select every registered test "
        "(§3.4 forced-full-suite branch -- selector-tooling changes or an "
        "external-dependency artifact change). All other selection flags "
        "are ignored when set.",
    )
    parser.add_argument(
        "--affected-files",
        default=None,
        help="Extra changed-file paths from object-hash classification (§2.2-2.4, "
        "PoC) -- unioned with --changed-files for ctags/file-level lookup",
    )
    parser.add_argument(
        "--new-files",
        default=None,
        help="Genuinely new source files from object-hash classification (§2.4, "
        "PoC). Presence of this file (even empty) promotes every currently "
        "unmapped registered test into Pass 1 -- see the ODR argument, §3.4.",
    )
    args = parser.parse_args()

    # Registered tests are needed for --force-full-suite too, so query first
    # and let force-full-suite short-circuit before touching the mapping.
    ctest_dir = Path(args.ctest_dir)
    print(f"Querying registered tests from {ctest_dir} ...", flush=True)
    try:
        registered = list_gtest_case_tests(ctest_dir, args.ctest_bin, args.refresh_test_list)
    except subprocess.CalledProcessError as e:
        sys.exit(f"ERROR: ctest -N failed: {e}")
    registered_set = set(registered)

    if args.force_full_suite:
        selected_sorted = sorted(registered_set)
        Path(args.selected_output).write_text(
            "\n".join(selected_sorted) + "\n" if selected_sorted else ""
        )
        Path(args.remaining_output).write_text("")
        print(
            f"Forced full suite: {len(selected_sorted):>5} tests → {args.selected_output} "
            f"(remaining_tests.txt empty)",
            flush=True,
        )
        return

    mapping_path = Path(args.mapping)
    if not mapping_path.exists():
        sys.exit(f"ERROR: mapping file not found: {mapping_path}")

    with open(mapping_path) as f:
        mapping = json.load(f)

    func_map: dict[str, list[str]] = mapping.get("functions", {})
    file_map: dict[str, list[str]] = mapping.get("files", {})

    # Changed files
    changed_files_path = Path(args.changed_files)
    if not changed_files_path.exists():
        sys.exit(f"ERROR: changed-files list not found: {changed_files_path}")
    changed_files = [
        l.strip() for l in changed_files_path.read_text().splitlines() if l.strip()
    ]

    # Object-hash-classified files (§2.4, PoC) are treated exactly like
    # changed source files for file-level lookup purposes -- they typically
    # have no ctags entries of their own (a CMakeLists.txt/dependency
    # change doesn't touch these files directly), so they flow straight
    # into the file-level fallback below.
    if args.affected_files:
        affected_path = Path(args.affected_files)
        if affected_path.exists():
            affected = [l.strip() for l in affected_path.read_text().splitlines() if l.strip()]
            if affected:
                print(f"  {len(affected)} file(s) affected via object-hash diff (§2.4)", flush=True)
            changed_files = list(dict.fromkeys(changed_files + affected))

    selected: set[str] = set()
    # Tracks which changed files had at least one function-level hit.
    # File-level lookup is used as a fallback only for files with no hits.
    files_with_function_hits: set[str] = set()

    # 1. Function-level lookup via ctags (preferred; more precise than file-level)
    if args.ctags_jsonl:
        ctags_path = Path(args.ctags_jsonl)
        if ctags_path.exists():
            qualified_names = extract_qualified_names(ctags_path)
            print(
                f"  {len(qualified_names)} function(s) extracted from ctags output",
                flush=True,
            )
            for qname, src_path in qualified_names:
                hits = find_tests_for_function(qname, func_map)
                if hits:
                    selected.update(hits)
                    if src_path:
                        files_with_function_hits.add(src_path)
        else:
            print(f"  WARNING: ctags JSONL not found: {ctags_path}", flush=True)

    # 2. File-level fallback — only for files where function-level found nothing.
    # This covers: new functions not yet in the mapping, non-function changes
    # (type definitions, macros, includes), and files ctags produced no tags for.
    fallback_files = [p for p in changed_files if p not in files_with_function_hits]
    if fallback_files:
        print(
            f"  {len(fallback_files)} file(s) with no function-level hits — "
            f"using file-level fallback",
            flush=True,
        )
    for path in fallback_files:
        for test in file_map.get(path, []):
            selected.add(test)

    # 3. BASE tests always run in Pass 1
    base_tests: list[str] = []
    if args.base_tests:
        base_path = Path(args.base_tests)
        if base_path.exists():
            base_tests = [
                l.strip() for l in base_path.read_text().splitlines()
                if l.strip() and not l.strip().startswith("#")
            ]
            selected.update(base_tests)

    # 4. New source files (§2.4/§3.4, PoC): a file that didn't exist before
    # this PR can't have affected any pre-existing test's coverage (the ODR
    # argument), so its own new/modified tests -- visible as CTest entries
    # not present anywhere in func2tests.json -- are safe to promote into
    # Pass 1 rather than waiting on Pass 2. This doesn't try to correlate
    # which specific new test came from which new object; any registered
    # test absent from the mapping entirely is treated as new/unmapped.
    if args.new_files is not None:
        mapped_tests: set[str] = set()
        for tests in func_map.values():
            mapped_tests.update(tests)
        for tests in file_map.values():
            mapped_tests.update(tests)
        unmapped = registered_set - mapped_tests
        if unmapped:
            print(
                f"  {len(unmapped)} registered test(s) absent from the mapping entirely "
                f"— promoting to Pass 1 (new-file ODR argument)",
                flush=True,
            )
        selected.update(unmapped)

    # 5. Intersect with live registered tests to guard against stale mapping entries
    stale = selected - registered_set
    if stale:
        print(
            f"  WARNING: {len(stale)} selected test(s) not in current CTest registry "
            f"(stale mapping entries) — dropping",
            flush=True,
        )
    selected = selected & registered_set

    remaining = sorted(registered_set - selected)
    selected_sorted = sorted(selected)

    # Write output files
    Path(args.selected_output).write_text("\n".join(selected_sorted) + "\n" if selected_sorted else "")
    Path(args.remaining_output).write_text("\n".join(remaining) + "\n" if remaining else "")

    print(
        f"Selected : {len(selected_sorted):>5} tests → {args.selected_output}",
        flush=True,
    )
    print(
        f"Remaining: {len(remaining):>5} tests → {args.remaining_output}",
        flush=True,
    )
    print(
        f"Total    : {len(registered_set):>5} registered gtest_case tests",
        flush=True,
    )


if __name__ == "__main__":
    main()
