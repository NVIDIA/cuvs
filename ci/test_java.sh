#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "Check GPU usage"
nvidia-smi

rapids-logger "Run Java build and tests"

RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
export RAPIDS_CUDA_MAJOR

# TODO: switch to installing pre-built artifacts instead of rebuilding in test jobs
#       ref: https://github.com/nvidia/cuvs/issues/868
ci/build_java.sh --run-java-tests

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
