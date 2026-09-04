#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# PR-time entry point: identify which tests cover the changes in a PR and
# write selected_tests.txt + remaining_tests.txt for two-pass CTest execution.
#
# Reproducible locally — all CI steps are driven through this script.
#
# Usage (CI):
#   ci/coverage/select_tests.sh --pr-number 1234 --mapping func2tests.json
#
# Usage (local):
#   ci/coverage/select_tests.sh --base-ref main --mapping func2tests.json
#
# Options:
#   --pr-number N       GitHub PR number; uses `gh pr diff` to get changed files
#   --base-ref REF      Git ref to diff against; uses `git diff` for local runs
#   --mapping PATH      func2tests.json produced by collect_and_map.sh
#                       (default: <repo>/func2tests.json)
#   --object-hashes PATH    Baseline object_hashes.json (PoC, §2.2). Enables
#                           non-source-change classification when paired with
#                           --external-hashes; requires a fresh sccache trace
#                           log already at <ctest-dir>/sccache.log from the
#                           PR's own coverage-preset build.
#   --external-hashes PATH  Baseline external_artifact_hashes.json (PoC, §2.3)
#   --ctest-dir PATH    Directory from which to run ctest -N
#                       (default: cpp/build/coverage if it exists, otherwise
#                        $CONDA_PREFIX/bin/gtests/libcuvs)
#   --ctest-bin PATH    ctest executable to use (default: ctest)
#   --base-tests PATH   base_tests.txt to always include (default: auto-detected)
#   --selected PATH     Output file for selected tests  (default: selected_tests.txt)
#   --remaining PATH    Output file for remaining tests (default: remaining_tests.txt)
#   --work-dir PATH     Scratch directory for intermediate files (default: /tmp/cuvs-select)
#   --refresh-test-list Force re-query of ctest -N even if a valid cache exists
#   --run               After selecting, run Pass 1 (selected tests) with ctest
#   -j N                Parallelism passed to ctest when --run is active (default: 8)

set -euo pipefail

# Universal ctags is required for --output-format=json and CUDA language support.
# On Ubuntu, apt install universal-ctags installs the binary as ctags-universal
# alongside exuberant-ctags (which owns the plain ctags name).
if command -v ctags-universal &>/dev/null; then
  CTAGS_BIN="ctags-universal"
elif command -v ctags &>/dev/null; then
  CTAGS_BIN="ctags"
else
  echo "ERROR: universal ctags is required but not found." >&2
  echo "  Install: sudo apt-get install universal-ctags" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PR_NUMBER=""
BASE_REF=""
MAPPING="$REPO_ROOT/func2tests.json"
OBJECT_HASHES=""
EXTERNAL_HASHES=""
CTEST_DIR=""
CTEST_BIN=""
REFRESH_TEST_LIST=0
BASE_TESTS="$SCRIPT_DIR/base_tests.txt"
SELECTED_OUT="selected_tests.txt"
REMAINING_OUT="remaining_tests.txt"
WORK_DIR="/tmp/cuvs-select-$$"
RUN=0
JOBS=8

while [[ $# -gt 0 ]]; do
  case $1 in
    --pr-number)   PR_NUMBER="$2";   shift 2 ;;
    --base-ref)    BASE_REF="$2";    shift 2 ;;
    --mapping)     MAPPING="$2";     shift 2 ;;
    --object-hashes)      OBJECT_HASHES="$2";   shift 2 ;;
    --external-hashes)    EXTERNAL_HASHES="$2"; shift 2 ;;
    --ctest-dir)   CTEST_DIR="$2";   shift 2 ;;
    --ctest-bin)          CTEST_BIN="$2"; shift 2 ;;
    --refresh-test-list)  REFRESH_TEST_LIST=1; shift ;;
    --base-tests)         BASE_TESTS="$2"; shift 2 ;;
    --selected)    SELECTED_OUT="$2"; shift 2 ;;
    --remaining)   REMAINING_OUT="$2"; shift 2 ;;
    --work-dir)    WORK_DIR="$2";    shift 2 ;;
    --run)         RUN=1;            shift   ;;
    -j)            JOBS="$2";        shift 2 ;;
    -j*)           JOBS="${1#-j}";   shift   ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$PR_NUMBER" && -z "$BASE_REF" ]]; then
  echo "ERROR: one of --pr-number or --base-ref is required" >&2
  exit 1
fi

# Auto-detect ctest directory
if [[ -z "$CTEST_DIR" ]]; then
  if [[ -f "$REPO_ROOT/cpp/build/coverage/CTestTestfile.cmake" ]]; then
    CTEST_DIR="$REPO_ROOT/cpp/build/coverage"
  elif [[ -d "${CONDA_PREFIX:-}/bin/gtests/libcuvs" ]]; then
    CTEST_DIR="$CONDA_PREFIX/bin/gtests/libcuvs"
  else
    echo "ERROR: cannot locate ctest directory; pass --ctest-dir explicitly" >&2
    exit 1
  fi
fi

mkdir -p "$WORK_DIR"
trap 'rm -rf "$WORK_DIR"' EXIT

CHANGED_FILES="$WORK_DIR/changed_files.txt"
NON_SOURCE_CHANGED="$WORK_DIR/non_source_changed.txt"
CTAGS_JSONL="$WORK_DIR/changed_functions.jsonl"

# ── Step 1: Get changed files ─────────────────────────────────────────────────
echo "==> Getting changed files ..."
if [[ -n "$PR_NUMBER" ]]; then
  gh pr diff "$PR_NUMBER" --name-only > "$WORK_DIR/all_changed.txt"
else
  # Committed changes on this branch vs base, plus any staged/unstaged edits
  # not yet committed.  Using separate commands and deduplicating so that a
  # local work-in-progress edit is included without pulling in every prior
  # commit on the branch when diffing against a common ancestor.
  { git -C "$REPO_ROOT" diff --name-only "${BASE_REF}...HEAD"
    git -C "$REPO_ROOT" diff --name-only HEAD
  } | sort -u > "$WORK_DIR/all_changed.txt"
fi

# Filter to C/C++/CUDA source files; everything else is classified in Step 2b.
grep -E '\.(c|cc|cpp|cu|cuh|hpp|h)$' "$WORK_DIR/all_changed.txt" > "$CHANGED_FILES" || true
grep -vE '\.(c|cc|cpp|cu|cuh|hpp|h)$' "$WORK_DIR/all_changed.txt" > "$NON_SOURCE_CHANGED" || true

N_ALL=$(wc -l < "$WORK_DIR/all_changed.txt")
N_SRC=$(wc -l < "$CHANGED_FILES")
echo "  $N_ALL file(s) changed; $N_SRC are C/C++/CUDA"
if [[ -s "$CHANGED_FILES" ]]; then
  while IFS= read -r f; do echo "    $f"; done < "$CHANGED_FILES"
fi

if [[ ! -s "$CHANGED_FILES" ]]; then
  echo "  No C/C++/CUDA files changed — selecting base tests only."
fi

# ── Step 2: Extract changed function names via ctags ─────────────────────────
if [[ -s "$CHANGED_FILES" ]]; then
  echo "==> Running ctags on changed source files ..."

  # Build list of files that actually exist (deleted files are skipped).
  # Kept repo-relative (not $REPO_ROOT/$f) and run from $REPO_ROOT below, so
  # ctags' own "path" field matches $CHANGED_FILES' format exactly —
  # select_tests.py compares them directly to track per-file function hits.
  EXISTING_FILES=()
  while IFS= read -r f; do
    [[ -f "$REPO_ROOT/$f" ]] && EXISTING_FILES+=("$f")
  done < "$CHANGED_FILES"

  if [[ ${#EXISTING_FILES[@]} -gt 0 ]]; then
    (cd "$REPO_ROOT" && "$CTAGS_BIN" \
      --output-format=json \
      --fields=+nKs \
      --kinds-C++=f+p \
      --kinds-CUDA=f \
      --langmap=C++:+.cuh \
      -f - \
      "${EXISTING_FILES[@]}") \
      > "$CTAGS_JSONL" 2>"$WORK_DIR/ctags_stderr.txt" || true
    N_TAGS=$(wc -l < "$CTAGS_JSONL")
    echo "  $N_TAGS tag(s) extracted"
    if [[ "$N_TAGS" -eq 0 && -s "$WORK_DIR/ctags_stderr.txt" ]]; then
      echo "  ctags stderr:" >&2
      cat "$WORK_DIR/ctags_stderr.txt" >&2
    fi
  else
    touch "$CTAGS_JSONL"
  fi
fi

# ── Step 2b: Classify non-source changes (§2.2-2.4, §3.4 — PoC) ──────────────
# Every non-source changed file gets an explicit classification instead of
# being silently dropped. Forced-full-suite and "can't classify, fail safe"
# both take priority over the object-hash diff itself.
FORCE_FULL_SUITE=0
AFFECTED_FILES=""
NEW_FILES=""

if [[ -s "$NON_SOURCE_CHANGED" ]]; then
  echo "==> Classifying non-source changes ..."

  # Selector-tooling changes: a modified selector can't validate itself.
  if grep -qE '^ci/coverage/|^cpp/CMakePresets\.json$' "$NON_SOURCE_CHANGED"; then
    echo "  Selector tooling changed — forcing full suite."
    FORCE_FULL_SUITE=1
  elif [[ -z "$OBJECT_HASHES" || -z "$EXTERNAL_HASHES" ]]; then
    echo "  Non-source change(s) present but no --object-hashes/--external-hashes" >&2
    echo "  baseline provided — cannot classify; forcing full suite (fail safe, not" >&2
    echo "  an empty selection)." >&2
    FORCE_FULL_SUITE=1
  elif [[ ! -f "$CTEST_DIR/sccache.log" ]]; then
    echo "  Non-source change(s) present but no sccache trace log at" >&2
    echo "  $CTEST_DIR/sccache.log (expected from the PR's own coverage-preset" >&2
    echo "  build) — cannot classify; forcing full suite (fail safe)." >&2
    FORCE_FULL_SUITE=1
  else
    PR_OBJECT_HASHES="$WORK_DIR/pr_object_hashes.json"
    PR_EXTERNAL_HASHES="$WORK_DIR/pr_external_artifact_hashes.json"

    python3 "$SCRIPT_DIR/hash_objects_sccache.py" \
      --sccache-log "$CTEST_DIR/sccache.log" \
      --build-dir   "$CTEST_DIR" \
      --output      "$PR_OBJECT_HASHES"

    shopt -s nullglob
    BINARIES=("$CTEST_DIR"/gtests/*)
    shopt -u nullglob
    if [[ ${#BINARIES[@]} -gt 0 ]]; then
      python3 "$SCRIPT_DIR/hash_external_artifacts.py" \
        --build-dir "$CTEST_DIR" \
        --output    "$PR_EXTERNAL_HASHES" \
        "${BINARIES[@]}"
    else
      echo "{}" > "$PR_EXTERNAL_HASHES"
    fi

    python3 "$SCRIPT_DIR/classify_changes.py" \
      --baseline-objects  "$OBJECT_HASHES"        \
      --pr-objects        "$PR_OBJECT_HASHES"      \
      --baseline-external "$EXTERNAL_HASHES"       \
      --pr-external       "$PR_EXTERNAL_HASHES"    \
      --new-files-out      "$WORK_DIR/new_source_files.txt" \
      --affected-files-out "$WORK_DIR/affected_files.txt"   \
      --result-out         "$WORK_DIR/classification_result.txt"

    RESULT=$(cat "$WORK_DIR/classification_result.txt")
    if [[ "$RESULT" == "FULL_SUITE" ]]; then
      echo "  External dependency artifact changed — forcing full suite."
      FORCE_FULL_SUITE=1
    else
      AFFECTED_FILES="$WORK_DIR/affected_files.txt"
      NEW_FILES="$WORK_DIR/new_source_files.txt"
    fi
  fi
fi

# ── Step 3: Select tests ──────────────────────────────────────────────────────
echo "==> Selecting tests ..."

if [[ "$FORCE_FULL_SUITE" -eq 1 ]]; then
  CTEST_BIN_ARG=""
  [[ -n "$CTEST_BIN" ]] && CTEST_BIN_ARG="--ctest-bin $CTEST_BIN"
  REFRESH_TEST_LIST_ARG=""
  [[ "$REFRESH_TEST_LIST" -eq 1 ]] && REFRESH_TEST_LIST_ARG="--refresh-test-list"

  python3 "$SCRIPT_DIR/select_tests.py" \
    --force-full-suite \
    --changed-files /dev/null \
    --ctest-dir "$CTEST_DIR" \
    --selected-output  "$SELECTED_OUT"  \
    --remaining-output "$REMAINING_OUT" \
    $CTEST_BIN_ARG \
    $REFRESH_TEST_LIST_ARG
else
  BASE_TESTS_ARG=""
  [[ -f "$BASE_TESTS" ]] && BASE_TESTS_ARG="--base-tests $BASE_TESTS"

  CTAGS_ARG=""
  [[ -f "$CTAGS_JSONL" ]] && CTAGS_ARG="--ctags-jsonl $CTAGS_JSONL"

  CTEST_BIN_ARG=""
  [[ -n "$CTEST_BIN" ]] && CTEST_BIN_ARG="--ctest-bin $CTEST_BIN"

  REFRESH_TEST_LIST_ARG=""
  [[ "$REFRESH_TEST_LIST" -eq 1 ]] && REFRESH_TEST_LIST_ARG="--refresh-test-list"

  AFFECTED_FILES_ARG=""
  [[ -n "$AFFECTED_FILES" ]] && AFFECTED_FILES_ARG="--affected-files $AFFECTED_FILES"

  NEW_FILES_ARG=""
  [[ -n "$NEW_FILES" ]] && NEW_FILES_ARG="--new-files $NEW_FILES"

  python3 "$SCRIPT_DIR/select_tests.py" \
    --mapping       "$MAPPING"        \
    --changed-files "$CHANGED_FILES"  \
    --ctest-dir     "$CTEST_DIR"      \
    --selected-output  "$SELECTED_OUT"  \
    --remaining-output "$REMAINING_OUT" \
    $BASE_TESTS_ARG \
    $CTAGS_ARG \
    $CTEST_BIN_ARG \
    $REFRESH_TEST_LIST_ARG \
    $AFFECTED_FILES_ARG \
    $NEW_FILES_ARG
fi

echo "==> Done."
echo "    Pass 1: ${CTEST_BIN:-ctest} --tests-from-file $SELECTED_OUT"
echo "    Pass 2: ${CTEST_BIN:-ctest} --tests-from-file $REMAINING_OUT"

if [[ "$RUN" -eq 1 ]]; then
  if [[ -s "$SELECTED_OUT" ]]; then
    echo "==> Running Pass 1 ..."
    "${CTEST_BIN:-ctest}" \
      --test-dir  "$CTEST_DIR" \
      -j "$JOBS"  \
      --output-on-failure \
      --tests-from-file "$SELECTED_OUT"
  else
    echo "==> Pass 1: no tests selected — skipping run."
  fi
fi
