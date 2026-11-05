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
echo "CUDA JIT cache ${CUDA_CACHE_PATH}: $(find "${CUDA_CACHE_PATH}" -type f | wc -l) files before the tests"

rapids-print-env

rapids-logger "Check GPU usage"
nvidia-smi

# RAPIDS_DATASET_ROOT_DIR is used by test scripts
RAPIDS_DATASET_ROOT_DIR=${RAPIDS_TESTS_DIR}/dataset
export RAPIDS_DATASET_ROOT_DIR
./ci/get_test_data.sh --NEIGHBORS_ANN_VAMANA_TEST

rapids-logger "DIAG: system"
uname -r; nproc; free -g; lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Socket|NUMA node\(s\)'
/opt/conda/bin/python -I ci/cuda_latency_probe.py
cd "$CONDA_PREFIX"/bin/gtests/libcuvs
set +e
diag() {
  local name=$1
  shift
  rapids-logger "DIAG: ${name}"
  local t0=${SECONDS}
  "$@" 2>&1 | grep -E '^\[ +(OK|FAILED) +\]|^\[==========\]|rror'
  echo "DIAG-TIME ${name} $((SECONDS - t0)) s"
}
first_tests() {
  "./$1" --gtest_list_tests | awk '/^[^ ]/{s=$1} /^  /{print s $1}' | head -n "$2" | paste -sd:
}
F_SQ=$(first_tests NEIGHBORS_ANN_IVF_SQ_TEST 12)
F_FLAT=$(first_tests NEIGHBORS_ANN_IVF_FLAT_TEST 12)
F_PQ='IvfPq/f32_f32_i64.build_host_input_search/1:IvfPq/f32_f32_i64.build_host_input_search/2:IvfPq/f32_f32_i64.build_host_input_search/19'
diag sq-async timeout 20m ./NEIGHBORS_ANN_IVF_SQ_TEST --gtest_filter="${F_SQ}"
diag flat-async timeout 20m ./NEIGHBORS_ANN_IVF_FLAT_TEST --gtest_filter="${F_FLAT}"
diag pq-async timeout 20m ./NEIGHBORS_ANN_IVF_PQ_TEST --gtest_filter="${F_PQ}"
exit 1

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

# Run Python build utilities tests (once, in the first shard)
if [[ "${SHARD}" == "1" ]]; then
  rapids-logger "Run libcuvs Python build utilities tests"
  pytest cpp/tests/python
fi

# Run libcuvs gtests from libcuvs-tests package
rapids-logger "Run libcuvs tests (shard ${SHARD} of ${NUM_SHARDS})"
pushd "$CONDA_PREFIX"/bin/gtests/libcuvs
timeout -v --signal=SIGINT --kill-after=60s 100m ctest -j8 --output-on-failure -I "${SHARD},,${NUM_SHARDS}"
popd

# Every kernel that was not in the restored cache and had to be linked adds files.
echo "CUDA JIT cache ${CUDA_CACHE_PATH}: $(find "${CUDA_CACHE_PATH}" -type f | wc -l) files ($(du -sh "${CUDA_CACHE_PATH}" | cut -f1)) after the tests"

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
