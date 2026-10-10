#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Usage: ci/test_cpp.sh [SHARD NUM_SHARDS]
# With SHARD and NUM_SHARDS (1 <= SHARD <= NUM_SHARDS), only every NUM_SHARDS-th libcuvs test is
# run, starting from test number SHARD, so the tests can be split across several CI jobs.
SHARD=${1:-1}
NUM_SHARDS=${2:-1}

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libcuvs cuvs --cuda "$RAPIDS_CUDA_VERSION")")

rapids-logger "Generate C++ testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_cpp \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" \
  --prepend-channel "${CPP_CHANNEL}" \
  | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"

# CI provides CUDA_CACHE_PATH through the reusable workflow's cache-environment input.
# So that we can re-use the CUDA driver's on-disk JIT cache between runs.
CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-.cache/cuda-jit}"
if [[ "${CUDA_CACHE_PATH}" != /* ]]; then
  CUDA_CACHE_PATH="$(realpath -m "${CUDA_CACHE_PATH}")"
fi
export CUDA_CACHE_PATH
mkdir -p "${CUDA_CACHE_PATH}"

rapids-print-env

rapids-logger "Check GPU usage"
nvidia-smi

# RAPIDS_DATASET_ROOT_DIR is used by test scripts
RAPIDS_DATASET_ROOT_DIR=${RAPIDS_TESTS_DIR}/dataset
export RAPIDS_DATASET_ROOT_DIR
./ci/get_test_data.sh --NEIGHBORS_ANN_VAMANA_TEST

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

# Run Python build utilities tests (once, in the first shard)
if [[ "${SHARD}" == "1" ]]; then
  rapids-logger "Run libcuvs Python build utilities tests"
  pytest cpp/tests/python
fi

# Run libcuvs gtests from libcuvs-tests package
# [DO NOT MERGE] Reproduce the flaky silhouetteScore batched tests (#2406).
pushd "$CONDA_PREFIX"/bin/gtests/libcuvs
rapids-logger "Run full STATS_TEST once (shard ${SHARD} of ${NUM_SHARDS} ignored)"
./STATS_TEST

SIL_LOGS="${RAPIDS_TESTS_DIR}/silhouette"
mkdir -p "${SIL_LOGS}"
NPROC=8
# run_mode <name> <reps> <alias(0/1)>: run NPROC concurrent STATS_TEST processes looping the batched
# silhouette tests, then count failing iterations per test.
run_mode() {
  local name=$1 reps=$2 alias=$3
  rapids-logger "silhouette mode=${name}: ${NPROC} processes x ${reps} repeats (alias=${alias})"
  local start=${SECONDS}
  for p in $(seq 1 ${NPROC}); do
    if [[ "${alias}" == "1" ]]; then
      CUVS_DIAG_SILHOUETTE_ALIAS=1 timeout -v 40m ./STATS_TEST --gtest_filter='silhouetteScore.Batched*' \
        --gtest_repeat="${reps}" --gtest_brief=1 > "${SIL_LOGS}/${name}_${p}.log" 2>&1 &
    else
      timeout -v 40m ./STATS_TEST --gtest_filter='silhouetteScore.Batched*' \
        --gtest_repeat="${reps}" --gtest_brief=1 > "${SIL_LOGS}/${name}_${p}.log" 2>&1 &
    fi
  done
  wait
  local elapsed=$((SECONDS - start))
  local fails
  fails=$(cat "${SIL_LOGS}/${name}"_*.log | grep -cE '^\[  FAILED  \] silhouetteScore\.[A-Za-z]+ \([0-9]+ ms\)' || true)
  echo "SILHOUETTE_SUMMARY mode=${name} processes=${NPROC} repeats_per_process=${reps} total_iterations=$((NPROC * reps)) failed_test_iterations=${fails} elapsed_s=${elapsed}"
  cat "${SIL_LOGS}/${name}"_*.log | grep -oE '^\[  FAILED  \] silhouetteScore\.[A-Za-z]+ ' | sort | uniq -c || true
  cat "${SIL_LOGS}/${name}"_*.log | grep -A3 -E 'silhouette_score.cu:[0-9]+: Failure' | head -40 || true
  for p in $(seq 1 ${NPROC}); do tail -n 3 "${SIL_LOGS}/${name}_${p}.log"; done
  if [[ "${alias}" == "0" && "${fails}" != "0" ]]; then EXITCODE=1; fi
}
nvidia-smi
run_mode aliased 1000 1
run_mode fixed 3000 0
run_mode aliased2 1000 1
popd

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
