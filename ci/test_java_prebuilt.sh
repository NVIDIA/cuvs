#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Runs cuvs-java's own IT test suite against the classes from a prior amd64 build job,
# without recompiling (or running jextract) on this host. cuvs-java is always built on
# amd64; this verifies that the resulting jextract-generated Panama bindings also work
# correctly against a native libcuvs_c.so on the given (non-amd64) architecture.
#
# Takes the target architecture (as used by the GitHub Actions job matrix, e.g. "arm64")
# and the name of the cuvs-java artifact uploaded by the amd64 Java job.

ARCH="${1:?Usage: $0 <arch> <cuvs-java-artifact-name>}"
CUVS_JAVA_ARTIFACT="${2:?Usage: $0 <arch> <cuvs-java-artifact-name>}"

rapids-logger "Testing the amd64-built cuvs-java artifact on ${ARCH}"

case "${ARCH}" in
  amd64) CONDA_ARCH="x86_64" ;;
  arm64) CONDA_ARCH="aarch64" ;;
  *) CONDA_ARCH="${ARCH}" ;;
esac

if [ -e "/opt/conda/etc/profile.d/conda.sh" ]; then
  . /opt/conda/etc/profile.d/conda.sh
fi

rapids-logger "Check GPU usage"
nvidia-smi

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libcuvs cuvs --cuda "$RAPIDS_CUDA_VERSION")")
CUVS_JAVA_DIR=$(rapids-download-from-github "${CUVS_JAVA_ARTIFACT}")

rapids-logger "Generate Java testing dependencies"

ENV_YAML_DIR="$(mktemp -d)"

rapids-dependency-file-generator \
  --output conda \
  --file-key java \
  --prepend-channel "${CPP_CHANNEL}" \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=${CONDA_ARCH}" | tee "${ENV_YAML_DIR}/env.yaml"

rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n java

# Temporarily allow unbound variables for conda activation.
set +u
conda activate java
set -u

rapids-print-env

# libcuvs comes from the conda environment here, matching the architecture this script is
# running on.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

rapids-logger "Restore the amd64-built target/ directory so Maven can run the tests without recompiling"

rm -rf java/cuvs-java/target
mkdir -p java/cuvs-java/target
cp -a "${CUVS_JAVA_DIR}/." java/cuvs-java/target/

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "Run cuvs-java IT tests against the amd64-built classes"

# Invoking the failsafe plugin's goals directly (instead of through the "integration-test"
# lifecycle phase) runs only those goals -- it does not trigger compile/test-compile first,
# so this executes the already-compiled amd64 test classes as-is.
pushd java/cuvs-java
mvn --batch-mode failsafe:integration-test failsafe:verify
popd

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
