#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail


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

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

# Run the k-means test binaries once in full.
rapids-logger "Run k-means test binaries"
pushd "$CONDA_PREFIX"/bin/gtests/libcuvs
timeout -v --signal=SIGINT --kill-after=60s 60m ctest -j4 --output-on-failure -R "^(CLUSTER_TEST|CLUSTER_KMEANS_MG_TEST|CLUSTER_KMEANS_MNMG_TEST|KMEANS_C_TEST|KMEANS_MG_C_TEST)$"
popd

# Repeat the k-means fit tests to measure their failure rate.
rapids-logger "Run k-means fit tests repeatedly"
pushd "$CONDA_PREFIX"/bin/gtests/libcuvs
KMEANS_FILTER="${KMEANS_FILTER:-KmeansFitBatchedTests/*}"
./CLUSTER_TEST --gtest_filter="${KMEANS_FILTER}" --gtest_repeat=100 > kmeans_repeat.log 2>&1
popd
LOG="$CONDA_PREFIX"/bin/gtests/libcuvs/kmeans_repeat.log
grep -E "^\[  FAILED  \]|Value of|Actual:|Expected|mismatch|n_iter|Failure" "${LOG}" | head -n 400 || true
rapids-logger "Failure counts per test (out of 100 repetitions)"
grep -E "^\[  FAILED  \] .* \([0-9]+ ms\)$" "${LOG}" | sed -E 's/, where GetParam.*//; s/ \([0-9]+ ms\)//' | sort | uniq -c || true
rapids-logger "Pass counts per test"
grep -E "^\[       OK \]" "${LOG}" | sed -E 's/ \([0-9]+ ms\)//' | sort | uniq -c || true
rapids-logger "n_iter summary"
grep -E "^KMEANS_DIAG" "${LOG}" | sort | uniq -c | sort -rn | head -n 100 || true

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
