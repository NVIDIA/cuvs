#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-configure-sccache

export SCCACHE_S3_KEY_PREFIX="cuvs-lucene/${RAPIDS_CONDA_ARCH}/cuda${RAPIDS_CUDA_VERSION%%.*}/maven/objects-cache"
export SCCACHE_S3_PREPROCESSOR_CACHE_KEY_PREFIX="cuvs-lucene/${RAPIDS_CONDA_ARCH}/cuda${RAPIDS_CUDA_VERSION%%.*}/maven/preprocessor-cache"
export SCCACHE_S3_USE_PREPROCESSOR_CACHE_MODE=true

# Takes the name of the cuvs-java artifact uploaded by the Java job, plus an optional
# --run-java-tests flag.
# TODO: Remove the flag handling when build and test workflows are separated,
#       and test_lucene.sh no longer calls build_lucene.sh
#       ref: https://github.com/rapidsai/cuvs/issues/868
EXTRA_BUILD_ARGS=()
CUVS_JAVA_ARTIFACT=""
for arg in "$@"; do
  case "${arg}" in
    --run-java-tests) EXTRA_BUILD_ARGS+=("${arg}") ;;
    *) CUVS_JAVA_ARTIFACT="${arg}" ;;
  esac
done

if [ -z "${CUVS_JAVA_ARTIFACT}" ]; then
  echo "Error: name of the cuvs-java artifact is missing" >&2
  exit 1
fi

if [ -e "/opt/conda/etc/profile.d/conda.sh" ]; then
  . /opt/conda/etc/profile.d/conda.sh
fi

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
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee "${ENV_YAML_DIR}/env.yaml"

rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n java

# Temporarily allow unbound variables for conda activation.
set +u
conda activate java
set -u

rapids-print-env

# libcuvs comes from the conda environment here. cuvs-lucene depends on the plain cuvs-java jar,
# which bundles no native libraries, so the JVM resolves libcuvs_c.so through the dynamic loader
# and the environment's lib directory has to be on LD_LIBRARY_PATH.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

rapids-logger "Install the cuvs-java artifact into the local Maven repository"

# cuvs-lucene resolves cuvs-java from the local Maven repository. Rather than rebuilding the Java
# bindings here, install the jar built by the Java job. Its pom.xml travels with the artifact and
# supplies the coordinates, so no version needs to be hardcoded.
CUVS_JAVA_POM="${CUVS_JAVA_DIR}/pom.xml"
if [ ! -f "${CUVS_JAVA_POM}" ]; then
  echo "Could not find pom.xml in the cuvs-java artifact at ${CUVS_JAVA_DIR}" >&2
  exit 1
fi

# The artifact also carries the per-architecture native jar and the sources/javadoc/test jars;
# cuvs-lucene depends on the plain one.
mapfile -t CUVS_JAVA_JARS < <(find "${CUVS_JAVA_DIR}" -maxdepth 1 -name 'cuvs-java-*.jar' \
  ! -name '*-sources.jar' ! -name '*-javadoc.jar' ! -name '*-tests.jar' ! -name '*-cuda*.jar')
if [ "${#CUVS_JAVA_JARS[@]}" -ne 1 ]; then
  echo "Expected exactly one cuvs-java jar in ${CUVS_JAVA_DIR}, found: ${CUVS_JAVA_JARS[*]:-none}" >&2
  exit 1
fi

mvn --batch-mode install:install-file -Dfile="${CUVS_JAVA_JARS[0]}" -DpomFile="${CUVS_JAVA_POM}"

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "Run cuvs-lucene build"

RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
export RAPIDS_CUDA_MAJOR

bash ./build.sh lucene "${EXTRA_BUILD_ARGS[@]}"

sccache --show-adv-stats
sccache --stop-server >/dev/null 2>&1 || true

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
