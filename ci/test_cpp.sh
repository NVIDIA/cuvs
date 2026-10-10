#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Usage: ci/test_cpp.sh [SHARD NUM_SHARDS]
# With SHARD and NUM_SHARDS (1 <= SHARD <= NUM_SHARDS), only every NUM_SHARDS-th libcuvs test is
# run, starting from test number SHARD, so the tests can be split across several CI jobs.
# shellcheck disable=SC2034
SHARD=${1:-1}
# shellcheck disable=SC2034
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

# [CI ONLY] Repeat the IVF-RaBitQ tests to reproduce a flaky failure (dim = 1, host input) and
# to verify the empty-list fix.
pushd "$CONDA_PREFIX"/bin/gtests/libcuvs
rapids-logger "Run the IVF-RaBitQ empty-list regression tests repeatedly"
timeout -v --signal=SIGINT --kill-after=60s 15m ./NEIGHBORS_ANN_IVF_RABITQ_TEST \
  --gtest_filter='*empty_lists*' --gtest_brief=1 --gtest_repeat=50 > empty_lists.log 2>&1
rapids-logger "empty lists exit code: $?"
grep -c "PASSED" empty_lists.log || true
grep -B2 -A20 "FAILED\|Failure" empty_lists.log | head -200 || true
rapids-logger "Run NEIGHBORS_ANN_IVF_RABITQ_TEST small-dim cases repeatedly"
timeout -v --signal=SIGINT --kill-after=60s 20m ./NEIGHBORS_ANN_IVF_RABITQ_TEST \
  --gtest_filter='IvfRabitq/f32_f32_i64.*/1:IvfRabitq/f32_f32_i64.*/2:IvfRabitq/f32_f32_i64.*/3' \
  --gtest_brief=1 --gtest_repeat=1000 > small_dims.log 2>&1
rapids-logger "small dims exit code: $?"
grep -c "PASSED" small_dims.log || true
grep -B2 -A40 "FAILED\|Failure\|DEGENERATE\|before subtract" small_dims.log | head -400 || true
grep -oE "dim=[0-9]+ n_lists=[0-9]+ streaming=[0-9] residency=[0-9] on_device=[0-9] min_list=[0-9]+ empty_lists=[0-9]+" small_dims.log | sed -E 's/ n_lists=[0-9]+//; s/ on_device=[0-9]//' | sort | uniq -c | sort -k2 | head -100 || true
rapids-logger "Run full NEIGHBORS_ANN_IVF_RABITQ_TEST repeatedly"
timeout -v --signal=SIGINT --kill-after=60s 30m ./NEIGHBORS_ANN_IVF_RABITQ_TEST \
  --gtest_filter='-*empty_lists*' --gtest_brief=1 --gtest_repeat=30 > full.log 2>&1
rapids-logger "full exit code: $?"
grep -c "PASSED" full.log || true
grep -B2 -A40 "FAILED\|Failure\|DEGENERATE\|before subtract" full.log | head -400 || true
popd

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
