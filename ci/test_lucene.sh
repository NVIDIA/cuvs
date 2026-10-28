#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Runs cuvs-lucene's test suite against the classes from a prior amd64 build job, without
# recompiling on this host. cuvs-lucene is always built on amd64, against the amd64-built
# cuvs-java jar; this verifies that pairing also works correctly on this host's
# architecture. Since cuvs-lucene depends on the plain (no bundled natives) cuvs-java jar,
# this doubles as a cross-arch check that the jextract-generated Panama bindings baked
# into the amd64 jar work unmodified against a native libcuvs_c.so on that architecture.
#
# With --min-jdk, it instead runs the tests on the oldest JDK cuvs-lucene supports, set by the
# java_min_jdk dependencies in dependencies.yaml, without libcuvs or a GPU. That JDK is older
# than the 22 that cuvs-java needs to reach the GPU: cuvs-lucene is compiled with a lower
# --release so that applications on it (Solr, for example) can load it. There the GPU tests skip
# themselves, and what runs checks that the codecs still load, read existing indexes and fall back
# to the CPU. None of that depends on the GPU or on the Lucene release, so this mode runs on a CPU
# node and tests only the newest Lucene module.

MIN_JDK=false
while [ $# -gt 0 ]; do
  case "$1" in
    --min-jdk)
      MIN_JDK=true
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

CUVS_JAVA_ARTIFACT="cuvs-java-cuda${RAPIDS_CUDA_VERSION}"
CUVS_LUCENE_ARTIFACT="cuvs-lucene-cuda${RAPIDS_CUDA_VERSION}"

if [ "${MIN_JDK}" = true ]; then
  rapids-logger "Testing the amd64-built cuvs-lucene artifacts on the oldest supported JDK, without a GPU"
else
  rapids-logger "Testing the amd64-built cuvs-java/cuvs-lucene artifacts on $(arch)"
fi

if [ -e "/opt/conda/etc/profile.d/conda.sh" ]; then
  . /opt/conda/etc/profile.d/conda.sh
fi

if [ "${MIN_JDK}" = false ]; then
  rapids-logger "Check GPU usage"
  nvidia-smi
fi

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-logger "Downloading artifacts from previous jobs"
CUVS_JAVA_DIR=$(rapids-download-from-github "${CUVS_JAVA_ARTIFACT}")
CUVS_LUCENE_DIR=$(rapids-download-from-github "${CUVS_LUCENE_ARTIFACT}")

rapids-logger "Generate Java testing dependencies"

ENV_YAML_DIR="$(mktemp -d)"

if [ "${MIN_JDK}" = true ]; then
  # Just Maven and the JDK: there is no GPU to use, and cuvs-java can't on this JDK.
  rapids-dependency-file-generator \
    --output conda \
    --file-key java_min_jdk \
    --matrix "" | tee "${ENV_YAML_DIR}/env.yaml"
else
  CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libcuvs cuvs --cuda "$RAPIDS_CUDA_VERSION")")
  rapids-dependency-file-generator \
    --output conda \
    --file-key java \
    --prepend-channel "${CPP_CHANNEL}" \
    --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee "${ENV_YAML_DIR}/env.yaml"
fi

rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n java

# Temporarily allow unbound variables for conda activation.
set +u
conda activate java
set -u

rapids-print-env

if [ "${MIN_JDK}" = true ]; then
  # Maven, and the JVM surefire forks for the tests, run on the JDK of JAVA_HOME rather than the
  # java on PATH, so check that the one Maven reports is the one this environment installed.
  JDK_VERSION="$("${CONDA_PREFIX}/bin/java" -XshowSettings:properties -version 2>&1 | sed -n 's/^ *java\.specification\.version = //p')"
  MAVEN_JAVA_VERSION="$(cd java/cuvs-lucene && mvn --batch-mode -v | sed -n 's/^Java version: \([0-9]*\).*/\1/p')"
  if [ -z "${JDK_VERSION}" ] || [ "${MAVEN_JAVA_VERSION}" != "${JDK_VERSION}" ]; then
    echo "Expected Maven to run on the environment's JDK '${JDK_VERSION}', but it runs on JDK '${MAVEN_JAVA_VERSION}'" >&2
    exit 1
  fi
else
  # libcuvs comes from the conda environment here, matching the architecture this script is
  # running on. cuvs-lucene depends on the plain cuvs-java jar, which bundles no native
  # libraries, so the JVM resolves libcuvs_c.so through the dynamic loader.
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

rapids-logger "Install the amd64-built cuvs-java artifact into the local Maven repository"

# cuvs-lucene resolves cuvs-java from the local Maven repository. Its pom.xml travels with
# the artifact and supplies the coordinates, so no version needs to be hardcoded.
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

# Install cuvs-java jar into .m2, cd is needed to pick up pom.xml in order to avoid rate limit of main maven repo
pushd java/cuvs-lucene
mvn --batch-mode install:install-file -Dfile="${CUVS_JAVA_JARS[0]}" -DpomFile="${CUVS_JAVA_POM}"
popd

rapids-logger "Restore the amd64-built cuvs-lucene target/ directories so Maven can run the tests without recompiling"

# The artifact holds one <module>/target/ per module.
rm -rf java/cuvs-lucene/*/target
cp -a "${CUVS_LUCENE_DIR}/." java/cuvs-lucene/

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "Run cuvs-lucene tests against the amd64-built classes"

# -Dskip.compile activates the pom's "skip-compile" profile, which disables all
# compilation for this run, forcing test to use amd64-compiled jar instead of
# local code. --fail-at-end tests every module even if an earlier one fails.
pushd java/cuvs-lucene
MAVEN_TEST_ARGS=(--fail-at-end -Dskip.compile=true)
if [ "${MIN_JDK}" = true ]; then
  NEWEST_MODULE="$(find . -maxdepth 1 -type d -name 'lucene-*' -printf '%f\n' | sort -V | tail -1)"
  rapids-logger "Testing ${NEWEST_MODULE} on JDK ${JDK_VERSION}"
  MAVEN_TEST_ARGS+=(-pl "${NEWEST_MODULE}")
fi
mvn --batch-mode test "${MAVEN_TEST_ARGS[@]}"
popd

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
