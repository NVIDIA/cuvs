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
#                       (default: <repo>/func2tests.json). If passed explicitly,
#                       PATH must exist -- this errors immediately, it does not
#                       silently defer to a later failure.
#   --object-hashes PATH    Baseline object_hashes.json (PoC, §2.2). Enables
#                           non-source-change classification when paired with
#                           --external-hashes; requires a fresh sccache trace
#                           log already at <ctest-dir>/sccache.log from the
#                           PR's own coverage-preset build. If passed
#                           explicitly, PATH must exist -- errors immediately.
#   --external-hashes PATH  Baseline external_artifact_hashes.json (PoC, §2.3).
#                           If passed explicitly, PATH must exist -- errors
#                           immediately.
#   --mapping-dir DIR   Convenience for local use: scans DIR for
#                       func2tests.json/object_hashes.json/external_artifact_hashes.json
#                       and enables whichever of --mapping/--object-hashes/
#                       --external-hashes have a matching file present, without
#                       having to name all three individually. Unlike the
#                       explicit flags above, a missing file under DIR is NOT
#                       an error -- that feature is just left disabled, and
#                       what got enabled/disabled is always printed. An
#                       explicit --mapping/--object-hashes/--external-hashes
#                       always takes precedence over what --mapping-dir would
#                       have picked for that same file.
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
MAPPING_SET=0
OBJECT_HASHES=""
OBJECT_HASHES_SET=0
EXTERNAL_HASHES=""
EXTERNAL_HASHES_SET=0
MAPPING_DIR=""
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
    --mapping)     MAPPING="$2";     MAPPING_SET=1;        shift 2 ;;
    --object-hashes)      OBJECT_HASHES="$2";   OBJECT_HASHES_SET=1;   shift 2 ;;
    --external-hashes)    EXTERNAL_HASHES="$2"; EXTERNAL_HASHES_SET=1; shift 2 ;;
    --mapping-dir) MAPPING_DIR="$2"; shift 2 ;;
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

# ── --mapping-dir: convenience resolution of the three baseline files ────────
# Only fills in a flag that wasn't already explicitly passed; a missing file
# under --mapping-dir just leaves that feature disabled (not an error) --
# unlike explicit --mapping/--object-hashes/--external-hashes, validated below.
if [[ -n "$MAPPING_DIR" ]]; then
  echo "==> --mapping-dir $MAPPING_DIR:"

  if [[ "$MAPPING_SET" -eq 0 ]]; then
    if [[ -f "$MAPPING_DIR/func2tests.json" ]]; then
      MAPPING="$MAPPING_DIR/func2tests.json"
      MAPPING_SET=1
    else
      echo "    func2tests.json: not found"
    fi
  fi

  HAVE_OBJECT_HASHES=0
  if [[ "$OBJECT_HASHES_SET" -eq 0 ]]; then
    if [[ -f "$MAPPING_DIR/object_hashes.json" ]]; then
      OBJECT_HASHES="$MAPPING_DIR/object_hashes.json"
      OBJECT_HASHES_SET=1
      HAVE_OBJECT_HASHES=1
    else
      echo "    object_hashes.json: not found"
    fi
  else
    HAVE_OBJECT_HASHES=1
  fi

  HAVE_EXTERNAL_HASHES=0
  if [[ "$EXTERNAL_HASHES_SET" -eq 0 ]]; then
    if [[ -f "$MAPPING_DIR/external_artifact_hashes.json" ]]; then
      EXTERNAL_HASHES="$MAPPING_DIR/external_artifact_hashes.json"
      EXTERNAL_HASHES_SET=1
      HAVE_EXTERNAL_HASHES=1
    else
      echo "    external_artifact_hashes.json: not found"
    fi
  else
    HAVE_EXTERNAL_HASHES=1
  fi

  # Summary -- the one place that states what's actually enabled.
  if [[ "$MAPPING_SET" -eq 1 ]]; then
    echo "    func2tests.json found — ctags/gcov function-file selection enabled"
  else
    echo "    ctags/gcov function-file selection NOT enabled (no func2tests.json)"
  fi
  if [[ "$HAVE_OBJECT_HASHES" -eq 1 && "$HAVE_EXTERNAL_HASHES" -eq 1 ]]; then
    echo "    object_hashes.json + external_artifact_hashes.json found — object-hash classification enabled"
  elif [[ "$HAVE_OBJECT_HASHES" -eq 1 || "$HAVE_EXTERNAL_HASHES" -eq 1 ]]; then
    echo "    object-hash classification NOT enabled (both files are required; only one was found)"
  else
    echo "    object-hash classification NOT enabled (neither file found)"
  fi
fi

# ── Explicit --mapping/--object-hashes/--external-hashes must exist ─────────
# Unlike --mapping-dir, using one of these directly is a statement that the
# file should be there -- fail fast and clearly rather than deferring to
# whatever downstream script happens to touch the path first.
if [[ "$MAPPING_SET" -eq 1 && ! -f "$MAPPING" ]]; then
  echo "ERROR: --mapping file not found: $MAPPING" >&2
  exit 1
fi
if [[ "$OBJECT_HASHES_SET" -eq 1 && ! -f "$OBJECT_HASHES" ]]; then
  echo "ERROR: --object-hashes file not found: $OBJECT_HASHES" >&2
  exit 1
fi
if [[ "$EXTERNAL_HASHES_SET" -eq 1 && ! -f "$EXTERNAL_HASHES" ]]; then
  echo "ERROR: --external-hashes file not found: $EXTERNAL_HASHES" >&2
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
    # cpp/'s own tests land in $CTEST_DIR/gtests; the C API library and its
    # tests (../c, pulled in via add_subdirectory) build into $CTEST_DIR/c,
    # so its test binaries land in $CTEST_DIR/c/gtests -- a separate glob.
    BINARIES=("$CTEST_DIR"/gtests/* "$CTEST_DIR"/c/gtests/*)
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
