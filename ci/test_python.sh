#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libcuvs cuvs --cuda "$RAPIDS_CUDA_VERSION")")
PYTHON_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_python cuvs cuvs --stable --cuda "$RAPIDS_CUDA_VERSION")")

rapids-logger "Generate Python testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_python \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
  --prepend-channel "${CPP_CHANNEL}" \
  --prepend-channel "${PYTHON_CHANNEL}" \
  | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
RAPIDS_COVERAGE_DIR=${RAPIDS_COVERAGE_DIR:-"${PWD}/coverage-results"}
mkdir -p "${RAPIDS_TESTS_DIR}" "${RAPIDS_COVERAGE_DIR}"

rapids-print-env

rapids-logger "Check GPU usage"
nvidia-smi

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "pytest cuvs"
pushd python/cuvs/cuvs
pytest \
 --cache-clear \
 --junitxml="${RAPIDS_TESTS_DIR}/junit-cuvs.xml" \
 --cov-config=../.coveragerc \
 --cov=cuvs \
 --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cuvs-coverage.xml" \
 --cov-report=term \
 tests

rapids-logger "pytest cuvs-bench"
popd
pushd python/cuvs_bench/cuvs_bench
pytest \
 --cache-clear \
 --junitxml="${RAPIDS_TESTS_DIR}/junit-cuvs.xml" \
 --cov-config=../.coveragerc \
 --cov=cuvs \
 --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cuvs-bench-coverage.xml" \
 --cov-report=term \
 tests

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
