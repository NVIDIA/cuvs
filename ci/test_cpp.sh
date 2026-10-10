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
# skipped for the reproducer: ./ci/get_test_data.sh --NEIGHBORS_ANN_VAMANA_TEST

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

# Flaky spectral clustering reproducer (DO NOT MERGE)
echo "Ignoring shard ${SHARD} of ${NUM_SHARDS}"
pushd "$CONDA_PREFIX"/bin/gtests/libcuvs
LOGDIR="${RAPIDS_TESTS_DIR}/spectral"
mkdir -p "${LOGDIR}"

rapids-logger "Full CLUSTER_TEST binary in fresh processes"
FULL_RUNS=${FULL_RUNS:-5}
full_fail=0
for i in $(seq 1 "${FULL_RUNS}"); do
  if ! ./CLUSTER_TEST --gtest_filter='-SpectralClusteringDiag*' > "${LOGDIR}/full_${i}.log" 2>&1; then
    full_fail=$((full_fail + 1))
    grep -E "FAILED|Score|Failure|actual|Expected|Actual|eigensolver" "${LOGDIR}/full_${i}.log" || true
  fi
done
echo "SUMMARY full CLUSTER_TEST: ${full_fail}/${FULL_RUNS} runs failed"

rapids-logger "Spectral clustering diagnostics"
SPECTRAL_DIAG_REPEATS=${SPECTRAL_DIAG_REPEATS:-300} ./CLUSTER_TEST --gtest_filter='SpectralClusteringDiag*' 2>&1 | tee "${LOGDIR}/diag.log" | grep -v "graph diff" || true

rapids-logger "Spectral clustering tests repeated in one process"
./CLUSTER_TEST --gtest_filter='SpectralClusteringTest*' --gtest_repeat=300 --gtest_brief=1 > "${LOGDIR}/repeat.log" 2>&1 || true
grep -E "FAILED|Score|actual" "${LOGDIR}/repeat.log" | sort | uniq -c | sort -rn | head -50 || true
echo "SUMMARY in-process repeat: $(grep -c '^\[  FAILED  \] .*ms)$' "${LOGDIR}/repeat.log" || true) failed test instances (300 repeats x 22 tests)"

rapids-logger "SpectralClusteringTestF.Result/7 in fresh processes"
FRESH_RUNS=${FRESH_RUNS:-200}
fresh_fail=0
for i in $(seq 1 "${FRESH_RUNS}"); do
  if ! ./CLUSTER_TEST --gtest_filter='SpectralClusteringTests/SpectralClusteringTestF.Result/7' --gtest_brief=1 > "${LOGDIR}/fresh_${i}.log" 2>&1; then
    fresh_fail=$((fresh_fail + 1))
    grep -E "Score|actual|eigensolver" "${LOGDIR}/fresh_${i}.log" || true
  fi
done
echo "SUMMARY fresh-process Result/7: ${fresh_fail}/${FRESH_RUNS} runs failed"

rapids-logger "Distance and 1-NN tests"
other_fail=0
./DISTANCE_TEST --gtest_brief=1 || other_fail=$((other_fail + 1))
./NEIGHBORS_TEST --gtest_filter='NNTest*:Top1nn*' --gtest_brief=1 || other_fail=$((other_fail + 1))
echo "SUMMARY distance / 1-NN test binaries failed: ${other_fail}"
popd

if [[ ${full_fail} -gt 0 || ${fresh_fail} -gt 0 || ${other_fail} -gt 0 ]] || grep -q '^\[  FAILED  \]' "${LOGDIR}/repeat.log"; then
  EXITCODE=1
fi

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
