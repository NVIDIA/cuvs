#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
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

# Forwards the name of the cuvs-java artifact given by the workflow.
# TODO: switch to installing pre-built artifacts instead of rebuilding in test jobs
#       ref: https://github.com/rapidsai/cuvs/issues/868
ci/build_lucene.sh "$@" --run-java-tests

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
