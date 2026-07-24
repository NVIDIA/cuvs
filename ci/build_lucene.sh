#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# TODO: Remove this argument-handling when build and test workflows are separated,
#       and test_lucene.sh no longer calls build_lucene.sh
#       ref: https://github.com/rapidsai/cuvs/issues/868
EXTRA_BUILD_ARGS=()
if [[ "${1:-}" == "--run-java-tests" ]]; then
  EXTRA_BUILD_ARGS+=("--run-java-tests")
fi

if [ -e "/opt/conda/etc/profile.d/conda.sh" ]; then
  . /opt/conda/etc/profile.d/conda.sh
fi

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-logger "Downloading artifacts from previous jobs"
# libcuvs C++ conda package (provides libcuvs.so / libcuvs_c.so used at test time)
CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libcuvs cuvs --cuda "$RAPIDS_CUDA_VERSION")")
# cuvs-java jar built by the java-build job (avoids rebuilding cuvs-java here)
CUVS_JAVA_ARTIFACT_DIR=$(rapids-download-from-github "cuvs-java-cuda${RAPIDS_CUDA_VERSION}")

rapids-logger "Generate Java testing dependencies"

ENV_YAML_DIR="$(mktemp -d)"

rapids-dependency-file-generator \
  --output conda \
  --file-key java \
  --prepend-channel "${CPP_CHANNEL}" \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee "${ENV_YAML_DIR}/env.yaml"

rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n java

# Temporarily allow unbound variables for conda activation.
set +u
conda activate java
set -u

rapids-print-env

# cuvs-lucene depends on the plain cuvs-java jar (no bundled native libraries), so it loads
# libcuvs through the system loader. Make the conda-provided libcuvs discoverable at test time.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Install the cuvs-java artifact built by the java-build job into the local Maven repository
# so that cuvs-lucene resolves com.nvidia.cuvs:cuvs-java without rebuilding it. The GAV is
# taken from the artifact's own pom.xml (bundled alongside the jar by java/build.sh).
VERSION=$(grep -oP 'VERSION="\K[^"]+' java/build.sh)
CUVS_JAVA_JAR=$(find "${CUVS_JAVA_ARTIFACT_DIR}" -name "cuvs-java-${VERSION}.jar" | head -1)
CUVS_JAVA_POM=$(find "${CUVS_JAVA_ARTIFACT_DIR}" -name "pom.xml" | head -1)
if [[ -z "${CUVS_JAVA_JAR}" || -z "${CUVS_JAVA_POM}" ]]; then
  echo "Error: could not locate cuvs-java-${VERSION}.jar and/or pom.xml in ${CUVS_JAVA_ARTIFACT_DIR}"
  exit 1
fi
rapids-logger "Installing cuvs-java ${VERSION} from ${CUVS_JAVA_JAR}"
mvn install:install-file -Dfile="${CUVS_JAVA_JAR}" -DpomFile="${CUVS_JAVA_POM}"

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "Run cuVS Lucene build"

RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
export RAPIDS_CUDA_MAJOR

bash ./build.sh lucene "${EXTRA_BUILD_ARGS[@]}"

rapids-logger "Build script exiting with value: $EXITCODE"
exit ${EXITCODE}
