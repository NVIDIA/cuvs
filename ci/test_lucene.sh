#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "Check GPU usage"
nvidia-smi

rapids-logger "Run cuvs-lucene build and tests"

RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
export RAPIDS_CUDA_MAJOR
# This CI entry point requires critical GPU suites to execute; do not allow JNI, allocator, or
# native-library failures to turn the job into an all-skipped success.
export CUVS_LUCENE_REQUIRE_GPU_TESTS=1

# Forwards the name of the cuvs-java artifact given by the workflow.
# TODO: switch to installing pre-built artifacts instead of rebuilding in test jobs
#       ref: https://github.com/rapidsai/cuvs/issues/868
ci/build_lucene.sh "$@" --run-java-tests

rapids-logger "Verify required cuvs-lucene GPU suites executed without skips"
REQUIRED_GPU_TESTS=(
  TestCagraHnswBulkIndexWriter
  TestBulkFbinIndexStorage
  TestBulkFbinMultiSegmentStorage
  TestAcceleratedThreeLayerRoundTrip
)
for test_class in "${REQUIRED_GPU_TESTS[@]}"; do
  report="java/cuvs-lucene/target/surefire-reports/TEST-com.nvidia.cuvs.lucene.${test_class}.xml"
  if [[ ! -f "${report}" ]]; then
    rapids-logger "Missing required GPU test report: ${report}"
    EXITCODE=1
    continue
  fi
  if ! grep -Eq '<testsuite[^>]*tests="[1-9][0-9]*"' "${report}"; then
    rapids-logger "Required GPU suite executed no tests: ${test_class}"
    EXITCODE=1
  fi
  if grep -Eq '<testsuite[^>]*skipped="[1-9][0-9]*"' "${report}"; then
    rapids-logger "Required GPU suite skipped tests: ${test_class}"
    EXITCODE=1
  fi
done

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
