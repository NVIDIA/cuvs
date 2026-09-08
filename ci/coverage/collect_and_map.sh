#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# Nightly entry point: configure a gcov-instrumented cuVS build, run every
# GTest case individually to capture per-test coverage, and produce
# func2tests.json mapping source functions to the tests that cover them.
# Also produces object_hashes.json / external_artifact_hashes.json (PoC,
# see docs/cpp_test_selector.md §2.2-2.3 in the sibling
# rapids_dynamic_test_harness repo) for PR-time non-source-change
# classification.
#
# Reproducible locally — all CI steps are driven through this script.
#
# Usage:
#   ci/coverage/collect_and_map.sh [OPTIONS]
#
# Options:
#   --skip-build          Reuse an existing coverage build; skip cmake/ninja
#   --skip-hashing         Skip object_hashes.json/external_artifact_hashes.json (PoC)
#   --skip-tests           Skip per-test gcov collection + func2tests.json mapping
#   --build-dir PATH      CMake build directory      (default: <repo>/cpp/build/coverage)
#   --coverage-dir PATH   Per-test .cov.json/.jit.log output directory
#                         (default: <build-dir>/per_test_coverage)
#   --output PATH         func2tests.json output path (default: <repo>/func2tests.json)
#   --object-hashes-output PATH    (default: <repo>/object_hashes.json)
#   --external-hashes-output PATH  (default: <repo>/external_artifact_hashes.json)
#   -j N                  Parallel collection workers (default: number of GPUs detected by nvidia-smi)
#   --post-process-slots N  Concurrent gcov post-processing slots per worker (default: 2)
#
# --skip-hashing and --skip-tests are independent — either, both, or
# neither may be passed. Both skipped plus --skip-build leaves only the
# (no-op) script scaffolding, which is a valid but useless invocation.
#
# Prerequisites:
#   cmake, ninja, gcov (bundled with GCC), python3, and a GPU-capable host with
#   the cuVS build dependencies available (see cpp/CMakePresets.json for the
#   coverage preset). sccache on PATH if hashing is not skipped.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

BUILD_DIR="$REPO_ROOT/cpp/build/coverage"
COVERAGE_DIR=""
MAPPING_OUT="$REPO_ROOT/func2tests.json"
OBJECT_HASHES_OUT="$REPO_ROOT/object_hashes.json"
EXTERNAL_HASHES_OUT="$REPO_ROOT/external_artifact_hashes.json"
SKIP_BUILD=0
SKIP_HASHING=0
SKIP_TESTS=0
JOBS=""
POST_PROCESS_SLOTS=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-build)              SKIP_BUILD=1; shift ;;
    --skip-hashing)             SKIP_HASHING=1; shift ;;
    --skip-tests)                SKIP_TESTS=1; shift ;;
    --build-dir)                BUILD_DIR="$2"; shift 2 ;;
    --coverage-dir)              COVERAGE_DIR="$2"; shift 2 ;;
    --output)                    MAPPING_OUT="$2"; shift 2 ;;
    --object-hashes-output)      OBJECT_HASHES_OUT="$2"; shift 2 ;;
    --external-hashes-output)    EXTERNAL_HASHES_OUT="$2"; shift 2 ;;
    -j)                          JOBS="$2"; shift 2 ;;
    -j*)                         JOBS="${1#-j}"; shift ;;
    --post-process-slots)        POST_PROCESS_SLOTS="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# Default coverage dir lives inside the build tree so it is never committed
[[ -z "$COVERAGE_DIR" ]] && COVERAGE_DIR="$BUILD_DIR/per_test_coverage"

SCCACHE_LOG="$BUILD_DIR/sccache.log"

if [[ $SKIP_HASHING -eq 0 ]] && ! command -v sccache &>/dev/null; then
  echo "==> WARNING: --skip-hashing not set but sccache is not on PATH — skipping hashing." >&2
  SKIP_HASHING=1
fi

# ── Step 1: gcov-instrumented build ──────────────────────────────────────────
if [[ $SKIP_BUILD -eq 0 ]]; then
  echo "==> Configuring coverage build (cmake --preset coverage) ..."
  cmake --preset coverage -S "$REPO_ROOT/cpp"

  echo "==> Building ..."
  if [[ $SKIP_HASHING -eq 0 ]]; then
    # A fresh trace log is required for a complete object_hashes.json this
    # run: ninja only invokes sccache for objects it actually rebuilds, so
    # the log only ever reflects THIS invocation's work, never the full
    # object set on an incremental build — hence --update below, not a
    # fresh log alone. The server must be restarted for new
    # SCCACHE_SERVER_LOG/SCCACHE_ERROR_LOG values to take effect; a
    # server already running under different logging settings would
    # otherwise silently keep using them.
    sccache --stop-server 2>/dev/null || true
    export SCCACHE_SERVER_LOG="sccache::compiler::compiler=trace"
    export SCCACHE_ERROR_LOG="$SCCACHE_LOG"
    rm -f "$SCCACHE_LOG"
  fi
  (cd "$REPO_ROOT/cpp" && cmake --build --preset coverage)
else
  echo "==> Skipping build (--skip-build)"
fi

# ── Step 2: Object/external-artifact hashing (PoC) ───────────────────────────
if [[ $SKIP_HASHING -eq 0 ]]; then
  if [[ ! -f "$SCCACHE_LOG" ]]; then
    echo "==> WARNING: no sccache trace log at $SCCACHE_LOG (build was skipped without" >&2
    echo "    a prior hashed run?) — skipping hashing for this invocation." >&2
  else
    echo "==> Hashing compiled objects ..."
    UPDATE_FLAG=""
    [[ -f "$OBJECT_HASHES_OUT" ]] && UPDATE_FLAG="--update"
    python3 "$SCRIPT_DIR/hash_objects_sccache.py" \
      --sccache-log "$SCCACHE_LOG" \
      --build-dir   "$BUILD_DIR"   \
      --output      "$OBJECT_HASHES_OUT" \
      $UPDATE_FLAG

    echo "==> Hashing external artifacts ..."
    shopt -s nullglob
    # cpp/'s own tests land in $BUILD_DIR/gtests; the C API library and its
    # tests (../c, pulled in via add_subdirectory) build into $BUILD_DIR/c,
    # so its test binaries land in $BUILD_DIR/c/gtests -- a separate glob.
    BINARIES=("$BUILD_DIR"/gtests/* "$BUILD_DIR"/c/gtests/*)
    shopt -u nullglob
    if [[ ${#BINARIES[@]} -eq 0 ]]; then
      echo "==> WARNING: no test binaries found under $BUILD_DIR/gtests or" >&2
      echo "    $BUILD_DIR/c/gtests — skipping external-artifact hashing." >&2
    else
      python3 "$SCRIPT_DIR/hash_external_artifacts.py" \
        --build-dir "$BUILD_DIR" \
        --output    "$EXTERNAL_HASHES_OUT" \
        "${BINARIES[@]}"
    fi
  fi
else
  echo "==> Skipping hashing (--skip-hashing)"
fi

# ── Step 3: Per-test gcov collection ─────────────────────────────────────────
# collect_coverage.py invokes ctest exactly once (ctest --show-only=json-v1) to
# discover every gtest_case test's exact command/cwd/env/timeout, then executes
# each test binary directly for the rest of the run. This avoids paying
# gtest_discover_tests(DISCOVERY_MODE PRE_TEST)'s full-binary GTest rescan on
# every individual test — see the module docstring in collect_coverage.py.
if [[ $SKIP_TESTS -eq 0 ]]; then
  echo "==> Collecting per-test coverage ..."
  JOBS_ARG=""
  [[ -n "$JOBS" ]] && JOBS_ARG="-j $JOBS"
  SLOTS_ARG=""
  [[ -n "$POST_PROCESS_SLOTS" ]] && SLOTS_ARG="--post-process-slots $POST_PROCESS_SLOTS"

  python3 "$SCRIPT_DIR/collect_coverage.py" \
    --build-dir    "$BUILD_DIR"    \
    --coverage-dir "$COVERAGE_DIR" \
    --repo-root    "$REPO_ROOT"    \
    --continue-on-failure          \
    --resume                       \
    $JOBS_ARG $SLOTS_ARG

  # ── Step 4: Build function→test mapping ────────────────────────────────────
  MAPPING_SCRIPT="$SCRIPT_DIR/build_mapping.py"
  if [[ -f "$MAPPING_SCRIPT" ]]; then
    echo "==> Building func2tests.json ..."
    UPDATE_FLAG=""
    [[ -f "$MAPPING_OUT" ]] && UPDATE_FLAG="--update"
    python3 "$MAPPING_SCRIPT" \
      --coverage-dir  "$COVERAGE_DIR" \
      --jit-sources   "$BUILD_DIR/jit_lto_sources.json" \
      --repo-root     "$REPO_ROOT" \
      --output        "$MAPPING_OUT" \
      $UPDATE_FLAG
    echo "==> Mapping written to: $MAPPING_OUT"
  else
    echo "==> build_mapping.py not yet present (Phase 2 deliverable). Skipping."
    echo "    Coverage .cov.json files are in: $COVERAGE_DIR"
  fi
else
  echo "==> Skipping per-test gcov collection + func2tests.json mapping (--skip-tests)"
fi

echo "==> Done."
